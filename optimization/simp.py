"""
SIMP density interpolation and sensitivity analysis.

SIMP interpolates material stiffness from density:
    E(rho) = E_min + rho^p * (E0 - E_min)

Sensitivity of compliance w.r.t. cell density:
    dc/drho_i = -p * rho_i^(p-1) * (E0 - E_min) * u_i^T * K0_i * u_i

where K0_i is element stiffness at unit material (E=1).
"""

import numpy as np
import skfem
from skfem.models import lame_parameters
from skfem.helpers import ddot, sym_grad, trace, DiscreteField
from skfem import BilinearForm


def plane_stress_lame(E, nu):
    """
    Plane stress Lamé parameters.
    lambda* = E*nu/(1-nu^2),  mu = E/(2*(1+nu))
    """
    lam_3d = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu     = E / (2.0 * (1.0 + nu))
    lam_ps = 2.0 * lam_3d * mu / (lam_3d + 2.0 * mu)
    return lam_ps, mu


def build_elem_to_cell(mesh, nx, ny):
    """
    Map each triangle element to its parent rectangular cell index.

    Args:
        mesh: skfem.MeshTri from init_tensor
        nx, ny: number of cells in x and y

    Returns:
        elem_to_cell: int array (n_elements,)
    """
    cx = mesh.p[0, mesh.t].mean(axis=0)
    cy = mesh.p[1, mesh.t].mean(axis=0)

    dx = mesh.p[0].max() / nx
    dy = mesh.p[1].max() / ny

    ix = np.floor(cx / dx).astype(int).clip(0, nx - 1)
    iy = np.floor(cy / dy).astype(int).clip(0, ny - 1)

    return iy * nx + ix


@BilinearForm
def _het_elasticity(u, v, w):
    """Heterogeneous linear elasticity bilinear form."""
    lam = w['lam'].value
    mu  = w['mu'].value
    e   = sym_grad(u)
    return 2.0 * mu * ddot(e, sym_grad(v)) + lam * trace(e) * trace(sym_grad(v))


def _make_lame_fields(basis, E_elem, nu):
    """
    Build DiscreteField objects for lam and mu.

    scikit-fem requires DiscreteField.shape[-1] == n_qp (number of
    quadrature points per element, which is 3 for P1 triangles).
    We store per-element constants and tile across quadrature points.

    Shape: (n_elem, n_qp)
    """
    n_elem = len(E_elem)
    n_qp   = basis.X.shape[-1]

    lam_arr = np.zeros(n_elem)
    mu_arr  = np.zeros(n_elem)
    for i, E_i in enumerate(E_elem):
        lam_arr[i], mu_arr[i] = plane_stress_lame(E_i, nu)

    lam_field = DiscreteField(value=np.tile(lam_arr[:, None], (1, n_qp)))
    mu_field  = DiscreteField(value=np.tile(mu_arr[:, None],  (1, n_qp)))

    return lam_field, mu_field


def assemble_simp_stiffness(basis, elem_to_cell, rho_cells, E0, E_min, penal, nu):
    """
    Assemble global stiffness with SIMP material interpolation.

    E(rho) = E_min + rho^p * (E0 - E_min)

    Args:
        basis: skfem.Basis (vectorial P1)
        elem_to_cell: (n_elements,) cell index per element
        rho_cells: (n_cells,) filtered cell densities in [0, 1]
        E0: full material stiffness
        E_min: minimum stiffness (void, keeps K non-singular)
        penal: SIMP penalty exponent
        nu: Poisson's ratio

    Returns:
        K: sparse stiffness matrix
        E_elem: (n_elements,) effective Young's modulus per element
    """
    rho_elem = rho_cells[elem_to_cell]
    E_elem   = E_min + rho_elem**penal * (E0 - E_min)

    lam_field, mu_field = _make_lame_fields(basis, E_elem, nu)
    K = _het_elasticity.assemble(basis, lam=lam_field, mu=mu_field)
    return K, E_elem


def compute_sensitivities(basis, u, elem_to_cell, n_cells, rho_cells,
                           E0, E_min, penal, nu):
    """
    Compute dc/drho for each cell.

    dc/drho_e = -p * rho_e^(p-1) * (E0-E_min) * u_e^T K0_e u_e

    Vectorised: uses einsum over the (n_elem, dofs, dofs) local matrix array.

    Args:
        basis: skfem.Basis
        u: displacement vector (n_dofs,)
        elem_to_cell: (n_elements,) cell index per element
        n_cells: total design cells
        rho_cells: (n_cells,) cell densities
        E0, E_min, penal, nu: SIMP parameters

    Returns:
        dc_drho: (n_cells,) sensitivity of compliance w.r.t. cell density
    """
    rho_elem     = rho_cells[elem_to_cell]
    dE_drho_elem = penal * rho_elem**(penal - 1.0) * (E0 - E_min)

    # Assemble element stiffness matrices at unit material (E=1)
    n_elem    = basis.mesh.t.shape[1]
    n_qp      = basis.X.shape[-1]
    lam_u, mu_u = plane_stress_lame(1.0, nu)

    K0_local = _het_elasticity.elemental(
        basis,
        lam=DiscreteField(value=np.full((n_elem, n_qp), lam_u)),
        mu =DiscreteField(value=np.full((n_elem, n_qp), mu_u))
    ).tolocal()   # (n_elem, dofs_per_elem, dofs_per_elem)

    # Vectorised element strain energy: u_e^T K0_e u_e
    # basis.element_dofs: (dofs_per_elem, n_elem)
    u_local = u[basis.element_dofs.T]   # (n_elem, dofs_per_elem)
    strain_energy_elem = np.einsum('ei,eij,ej->e', u_local, K0_local, u_local)

    dc_drho_elem = -dE_drho_elem * strain_energy_elem

    # Accumulate element sensitivities into cells (2 elements per cell → sum)
    dc_drho_cells = np.zeros(n_cells)
    np.add.at(dc_drho_cells, elem_to_cell, dc_drho_elem)

    return dc_drho_cells
