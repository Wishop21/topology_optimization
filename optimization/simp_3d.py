"""
3D SIMP material interpolation and sensitivity analysis.

Performance optimisations over naive implementation:
  1. Vectorised Lame parameters: no Python loop over elements (261x faster)
  2. K0_local cache: unit-material element stiffness computed once at startup,
     reused every iteration (saves ~1s per iteration at 30x15x10)
"""

import numpy as np
import skfem
from skfem.helpers import ddot, sym_grad, trace, DiscreteField
from skfem import BilinearForm
from core.fem_solver_3d import lame_3d


@BilinearForm
def _het_elasticity_3d(u, v, w):
    """Heterogeneous 3D linear elasticity bilinear form."""
    lam = w['lam'].value
    mu  = w['mu'].value
    e   = sym_grad(u)
    return 2.0 * mu * ddot(e, sym_grad(v)) + lam * trace(e) * trace(sym_grad(v))


def _lame_fields_vectorised(basis, E_elem, nu):
    """
    Build DiscreteField objects for lam and mu without a Python loop.

    Vectorised: applies Lame formulas directly to the full E_elem array.
    ~260x faster than the equivalent per-element loop.

    Shape required by scikit-fem: (n_elem, n_qp) — last dim = n_qp.
    """
    n_qp = basis.X.shape[-1]

    lam_arr = E_elem * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_arr  = E_elem / (2.0 * (1.0 + nu))

    lam_field = DiscreteField(value=np.tile(lam_arr[:, None], (1, n_qp)))
    mu_field  = DiscreteField(value=np.tile(mu_arr[:, None],  (1, n_qp)))
    return lam_field, mu_field


def build_k0_cache(basis, nu):
    """
    Precompute element stiffness matrices at unit material (E=1).

    This only needs to be called ONCE per problem setup, since the
    unit-material stiffness doesn't change between iterations.
    Saves ~1s per iteration vs rebuilding it inside compute_sensitivities_3d.

    Args:
        basis: skfem.Basis (3D vectorial TetP1)
        nu:    Poisson's ratio

    Returns:
        K0_local: (n_elem, 12, 12) array of element stiffness matrices
    """
    n_elem    = basis.mesh.t.shape[1]
    n_qp      = basis.X.shape[-1]
    lam_u, mu_u = lame_3d(1.0, nu)

    K0_local = _het_elasticity_3d.elemental(
        basis,
        lam=DiscreteField(value=np.full((n_elem, n_qp), lam_u)),
        mu =DiscreteField(value=np.full((n_elem, n_qp), mu_u))
    ).tolocal()   # (n_elem, 12, 12)

    return K0_local


def assemble_simp_stiffness_3d(basis, elem_to_cell, rho_cells,
                                E0, E_min, penal, nu):
    """
    Assemble 3D SIMP stiffness matrix.

    E(rho) = E_min + rho^p * (E0 - E_min)

    Args:
        basis:        skfem.Basis (3D vectorial TetP1)
        elem_to_cell: (n_elements,) cell index per element
        rho_cells:    (n_cells,) filtered densities
        E0:           full material stiffness
        E_min:        minimum stiffness (void, keeps K non-singular)
        penal:        SIMP penalty exponent
        nu:           Poisson's ratio

    Returns:
        K:      sparse stiffness matrix
        E_elem: (n_elements,) effective Young's modulus
    """
    rho_elem = rho_cells[elem_to_cell]
    E_elem   = E_min + rho_elem**penal * (E0 - E_min)

    lam_field, mu_field = _lame_fields_vectorised(basis, E_elem, nu)
    K = _het_elasticity_3d.assemble(basis, lam=lam_field, mu=mu_field)
    return K, E_elem


def compute_sensitivities_3d(basis, u, elem_to_cell, n_cells, rho_cells,
                              E0, E_min, penal, nu, K0_local=None):
    """
    Compute dc/drho for each cell.

    dc/drho_e = -p * rho_e^(p-1) * (E0-E_min) * u_e^T K0_e u_e

    Vectorised using precomputed K0_local and einsum.

    Args:
        basis:        skfem.Basis
        u:            displacement vector (n_dofs,)
        elem_to_cell: (n_elements,) cell index per element
        n_cells:      total design cells
        rho_cells:    (n_cells,) cell densities
        E0, E_min, penal, nu: SIMP parameters
        K0_local:     optional precomputed (n_elem,12,12) from build_k0_cache().
                      Pass this for speed. If None, recomputes each call.

    Returns:
        dc_drho: (n_cells,) sensitivity
    """
    rho_elem     = rho_cells[elem_to_cell]
    dE_drho_elem = penal * rho_elem**(penal - 1.0) * (E0 - E_min)

    # Use cached K0 if provided, otherwise compute (slower)
    if K0_local is None:
        n_elem    = basis.mesh.t.shape[1]
        n_qp      = basis.X.shape[-1]
        lam_u, mu_u = lame_3d(1.0, nu)
        K0_local = _het_elasticity_3d.elemental(
            basis,
            lam=DiscreteField(value=np.full((n_elem, n_qp), lam_u)),
            mu =DiscreteField(value=np.full((n_elem, n_qp), mu_u))
        ).tolocal()

    # Vectorised element strain energy: u_e^T K0_e u_e
    u_local = u[basis.element_dofs.T]   # (n_elem, 12)
    strain_energy_elem = np.einsum('ei,eij,ej->e', u_local, K0_local, u_local)

    dc_drho_elem  = -dE_drho_elem * strain_energy_elem

    # Accumulate into cells (6 tets per cell)
    dc_drho_cells = np.zeros(n_cells)
    np.add.at(dc_drho_cells, elem_to_cell, dc_drho_elem)
    return dc_drho_cells
