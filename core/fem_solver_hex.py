"""
3D linear elasticity solver using hexahedral (Q1) elements.

Identical physics to fem_solver_3d.py but using:
- ElementHex1 instead of ElementTetP1
- intorder=2 (8 quadrature points vs default 64)
- 3 DOFs per node, 8 nodes per element = 24 DOFs per element

Uses AMG-preconditioned CG for the linear solve.
"""

import numpy as np
import skfem
from skfem.models import linear_elasticity
from scipy.sparse.linalg import spsolve, cg
from core.mesh_hex import create_hex_basis


def lame_3d(E, nu):
    """Full 3D Lamé parameters (identical to fem_solver_3d)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    return lam, mu


def _solve_amg_pcg(K, f, tol=1e-8, maxiter=500):
    """AMG-preconditioned CG. Falls back to spsolve if pyamg unavailable."""
    try:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(K)
        M  = ml.aspreconditioner(cycle='V')
        u, info = cg(K, f, M=M, rtol=tol, maxiter=maxiter)
        if info != 0:
            u = spsolve(K, f)
        return u
    except ImportError:
        return spsolve(K, f)


class LinearElasticitySolverHex:
    """
    3D linear elasticity solver using Q1 hexahedral elements.

    Advantages over P1 tet:
    - No volumetric locking
    - Same DOF count = same memory / solve time
    - Fewer elements needed for equivalent accuracy (~6x fewer)
    - Much faster sensitivity computation (6x fewer elements)
    - Trivial elem_to_cell mapping (1:1 — each element IS a cell)

    DOF layout: node k → [3k, 3k+1, 3k+2]  (same as tet solver)
    """

    def __init__(self, mesh, E=1.0, nu=0.3):
        """
        Args:
            mesh: skfem.MeshHex object (from create_hex_mesh)
            E:    Young's modulus
            nu:   Poisson's ratio
        """
        if not (0 < nu < 0.5):
            raise ValueError(f"Poisson's ratio must be in (0, 0.5), got {nu}")
        if E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {E}")

        self.mesh = mesh
        self.E    = E
        self.nu   = nu

        # Use intorder=2 (8 qp) — exact for Q1 hex, not the wasteful default 64
        self.basis = create_hex_basis(mesh)
        self.lam, self.mu = lame_3d(E, nu)

    def assemble_stiffness(self):
        """Assemble global stiffness matrix."""
        return linear_elasticity(self.lam, self.mu).assemble(self.basis)

    def solve(self, K, f, fixed_dofs):
        """Apply BCs and solve K u = f using AMG-PCG."""
        K_cond, f_cond, _, I_free = skfem.condense(
            K, f, D=fixed_dofs, expand=True)
        u = np.zeros(self.basis.N)
        u[I_free] = _solve_amg_pcg(K_cond, f_cond)
        return u


def node_dofs_hex(node_indices, direction='all'):
    """
    Convert node indices to DOF indices (identical to tet version).
    DOF layout: node k → [3k, 3k+1, 3k+2]
    """
    node_indices = np.asarray(node_indices)
    if direction == 'all':
        dofs = np.concatenate([3*node_indices, 3*node_indices+1, 3*node_indices+2])
    elif direction == 'x':
        dofs = 3 * node_indices
    elif direction == 'y':
        dofs = 3 * node_indices + 1
    elif direction == 'z':
        dofs = 3 * node_indices + 2
    else:
        raise ValueError(f"direction must be 'all','x','y','z', got '{direction}'")
    return dofs.astype(int)


def build_force_vector_hex(n_dofs, node_indices, fx=0.0, fy=0.0, fz=0.0):
    """Distribute (fx,fy,fz) equally across node_indices (identical to tet version)."""
    f = np.zeros(n_dofs)
    n = len(node_indices)
    if n == 0:
        raise ValueError("node_indices is empty")
    for node in node_indices:
        f[3*node]   += fx / n
        f[3*node+1] += fy / n
        f[3*node+2] += fz / n
    return f
