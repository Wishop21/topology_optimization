"""
SIMP material interpolation and sensitivity analysis for hex elements.

Identical algorithm to simp_3d.py but adapted for Q1 hex:
- Vectorised Lame parameters (same optimisation)
- elem_to_cell is trivial: element i IS cell i (no mapping needed)
- K0_local has shape (n_cells, 24, 24)  [24 dofs per hex element]
- Quadrature: 8 points per element (intorder=2, not 64)
"""

import numpy as np
import skfem
from skfem.helpers import ddot, sym_grad, trace, DiscreteField
from skfem import BilinearForm
from core.fem_solver_hex import lame_3d
from core.mesh_hex import create_hex_basis


@BilinearForm
def _het_elasticity_hex(u, v, w):
    """Heterogeneous 3D linear elasticity for hex elements."""
    lam = w['lam'].value
    mu  = w['mu'].value
    e   = sym_grad(u)
    return 2.0 * mu * ddot(e, sym_grad(v)) + lam * trace(e) * trace(sym_grad(v))


def _lame_fields_vectorised(basis, E_elem, nu):
    """
    Vectorised Lame field construction.
    Shape: (n_elem, n_qp) where n_qp=8 for hex with intorder=2.
    """
    n_qp    = basis.X.shape[-1]
    lam_arr = E_elem * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_arr  = E_elem / (2.0 * (1.0 + nu))
    lam_f = DiscreteField(value=np.tile(lam_arr[:, None], (1, n_qp)))
    mu_f  = DiscreteField(value=np.tile(mu_arr[:, None],  (1, n_qp)))
    return lam_f, mu_f


def build_k0_cache_hex(basis, nu):
    """
    Precompute unit-material element stiffness matrices for hex.

    Returns:
        K0_local: (n_elem, 24, 24) — one matrix per hex element
    """
    n_elem    = basis.mesh.t.shape[1]
    n_qp      = basis.X.shape[-1]
    lam_u, mu_u = lame_3d(1.0, nu)

    K0_local = _het_elasticity_hex.elemental(
        basis,
        lam=DiscreteField(value=np.full((n_elem, n_qp), lam_u)),
        mu =DiscreteField(value=np.full((n_elem, n_qp), mu_u))
    ).tolocal()   # (n_elem, 24, 24)

    return K0_local


def assemble_simp_stiffness_hex(basis, rho_cells, E0, E_min, penal, nu):
    """
    Assemble SIMP stiffness for hex mesh.

    For hex, elem_to_cell is trivial: element i = cell i.
    No elem_to_cell argument needed.

    E(rho) = E_min + rho^p * (E0 - E_min)

    Args:
        basis:     skfem.Basis (hex, intorder=2)
        rho_cells: (n_cells,) filtered densities — also (n_elements,) since 1:1
        E0:        full material stiffness
        E_min:     void stiffness
        penal:     SIMP exponent
        nu:        Poisson's ratio

    Returns:
        K:      sparse stiffness matrix
        E_elem: (n_elements,) effective Young's modulus
    """
    E_elem   = E_min + rho_cells**penal * (E0 - E_min)
    lam_f, mu_f = _lame_fields_vectorised(basis, E_elem, nu)
    K = _het_elasticity_hex.assemble(basis, lam=lam_f, mu=mu_f)
    return K, E_elem


def compute_sensitivities_hex(basis, u, n_cells, rho_cells,
                               E0, E_min, penal, nu, K0_local=None):
    """
    Compute dc/drho for each cell.

    Hex: elem_to_cell is trivial (element i = cell i), so
    accumulation is a direct assignment not np.add.at.

    Args:
        basis:      skfem.Basis (hex)
        u:          displacement vector (n_dofs,)
        n_cells:    number of design cells (= n_elements for hex)
        rho_cells:  (n_cells,) filtered densities
        E0, E_min, penal, nu: SIMP parameters
        K0_local:   optional precomputed (n_elem,24,24) from build_k0_cache_hex

    Returns:
        dc_drho: (n_cells,) sensitivity
    """
    dE_drho_elem = penal * rho_cells**(penal - 1.0) * (E0 - E_min)

    if K0_local is None:
        K0_local = build_k0_cache_hex(basis, nu)

    # Vectorised element strain energy
    u_local = u[basis.element_dofs.T]   # (n_elem, 24)
    strain_energy_elem = np.einsum('ei,eij,ej->e', u_local, K0_local, u_local)

    # For hex: no accumulation loop — direct 1:1 mapping
    dc_drho = -dE_drho_elem * strain_energy_elem

    return dc_drho
