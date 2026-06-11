"""
3D linear elasticity FEA solver.

Uses full 3D constitutive law (no plane stress correction).
Element type: P1 tetrahedra (4 nodes, 3 DOFs per node = 12 DOFs per element).

Solver: AMG-preconditioned Conjugate Gradient (AMG-PCG).
  - PyAMG builds a multigrid hierarchy from the stiffness matrix
  - Used as preconditioner for scipy CG
  - ~4-5x faster than direct spsolve on 3D topology opt problems
  - Falls back to spsolve if pyamg is not installed
"""

import numpy as np
import skfem
from skfem.models import linear_elasticity
from scipy.sparse.linalg import spsolve, cg


def lame_3d(E, nu):
    """
    Full 3D Lamé parameters.

    Args:
        E:  Young's modulus (Pa)
        nu: Poisson's ratio

    Returns:
        lam: first Lamé parameter (Pa)
        mu:  shear modulus (Pa)
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    return lam, mu


def _solve_amg_pcg(K, f, tol=1e-8, maxiter=500):
    """
    Solve K u = f using AMG-preconditioned Conjugate Gradient.

    PyAMG builds a smoothed aggregation multigrid hierarchy and uses
    it as a preconditioner for scipy CG. This is 4-5x faster than
    direct spsolve for 3D FEA systems.

    Falls back to spsolve if pyamg is unavailable or CG fails.
    """
    try:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(K)
        M  = ml.aspreconditioner(cycle='V')
        u, info = cg(K, f, M=M, rtol=tol, maxiter=maxiter)
        if info != 0:
            # CG did not converge — fall back to direct
            u = spsolve(K, f)
        return u
    except ImportError:
        # pyamg not available — use direct solver
        return spsolve(K, f)


class LinearElasticitySolver3D:
    """
    3D linear elasticity solver using P1 tetrahedral elements.

    Solves: K u = f
    3 DOFs per node: u_x, u_y, u_z
    DOF layout: node k → DOFs [3k, 3k+1, 3k+2]

    Uses AMG-PCG for the linear solve (4-5x faster than direct).
    """

    def __init__(self, mesh, E=1.0, nu=0.3):
        """
        Args:
            mesh: skfem.MeshTet object
            E:    Young's modulus
            nu:   Poisson's ratio (must be in (0, 0.5))
        """
        if not (0 < nu < 0.5):
            raise ValueError(f"Poisson's ratio must be in (0, 0.5), got {nu}")
        if E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {E}")

        self.mesh = mesh
        self.E    = E
        self.nu   = nu

        # Vectorial P1 basis: 3 DOFs per node
        self.basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementTetP1()))
        self.lam, self.mu = lame_3d(E, nu)

    def assemble_stiffness(self):
        """
        Assemble global stiffness matrix.

        Returns:
            K: sparse CSR matrix (n_dofs, n_dofs)
        """
        return linear_elasticity(self.lam, self.mu).assemble(self.basis)

    def solve(self, K, f, fixed_dofs):
        """
        Apply Dirichlet BCs and solve K u = f using AMG-PCG.

        Args:
            K:          global stiffness matrix
            f:          force vector (n_dofs,)
            fixed_dofs: DOF indices to set to zero

        Returns:
            u: displacement vector (n_dofs,), fixed DOFs = 0
        """
        K_cond, f_cond, _, I_free = skfem.condense(
            K, f, D=fixed_dofs, expand=True)

        u_free = _solve_amg_pcg(K_cond, f_cond)

        u = np.zeros(self.basis.N)
        u[I_free] = u_free
        return u


def node_dofs_3d(node_indices, direction='all'):
    """
    Convert node indices to DOF indices for a 3D vectorial basis.

    DOF layout: node k → [3k, 3k+1, 3k+2] = [u_x, u_y, u_z]

    Args:
        node_indices: array of node indices
        direction: 'all' (x,y,z), 'x', 'y', or 'z'

    Returns:
        dofs: array of DOF indices
    """
    node_indices = np.asarray(node_indices)

    if direction == 'all':
        dofs = np.concatenate([
            3 * node_indices,
            3 * node_indices + 1,
            3 * node_indices + 2
        ])
    elif direction == 'x':
        dofs = 3 * node_indices
    elif direction == 'y':
        dofs = 3 * node_indices + 1
    elif direction == 'z':
        dofs = 3 * node_indices + 2
    else:
        raise ValueError(f"direction must be 'all','x','y','z', got '{direction}'")

    return dofs.astype(int)


def build_force_vector_3d(n_dofs, node_indices, fx=0.0, fy=0.0, fz=0.0):
    """
    Build a nodal force vector, distributing (fx, fy, fz) across nodes.

    Args:
        n_dofs:       total number of DOFs
        node_indices: nodes where force is applied
        fx, fy, fz:   total force components to distribute (N)

    Returns:
        f: force vector (n_dofs,)
    """
    f = np.zeros(n_dofs)
    n = len(node_indices)
    if n == 0:
        raise ValueError("node_indices is empty")

    for node in node_indices:
        f[3 * node]     += fx / n
        f[3 * node + 1] += fy / n
        f[3 * node + 2] += fz / n

    return f
