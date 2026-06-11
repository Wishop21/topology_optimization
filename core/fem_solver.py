"""
Finite Element Analysis solver for 2D linear elasticity.

This is a thin wrapper around scikit-fem. The library handles:
- Element stiffness matrix assembly (via BilinearForm)
- Gaussian quadrature integration
- Sparse matrix construction

We handle:
- Plane stress material parameters (NOT 3D Lame - this matters)
- Boundary condition application
- Force vector construction
"""

import numpy as np
import skfem
from skfem.models import linear_elasticity, lame_parameters
from scipy.sparse.linalg import spsolve


def plane_stress_lame(E, nu):
    """
    Compute Lamé parameters for 2D plane stress.

    This is NOT the same as 3D Lamé parameters. Plane stress assumes
    sigma_z = 0 (thin plate), which modifies the effective lambda:

        3D:           lambda = E*nu / ((1+nu)(1-2nu))
        Plane stress: lambda* = E*nu / (1 - nu^2)

    Derivation: eliminate epsilon_z from 3D constitutive law using
    sigma_z = 0 constraint → lambda* = 2*lambda*mu / (lambda + 2*mu)

    Args:
        E: Young's modulus (Pa)
        nu: Poisson's ratio

    Returns:
        lam: Plane stress lambda (Pa)
        mu:  Shear modulus (Pa), unchanged from 3D
    """
    lam_3d = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu     = E / (2.0 * (1.0 + nu))
    # Plane stress correction
    lam_ps = 2.0 * lam_3d * mu / (lam_3d + 2.0 * mu)
    return lam_ps, mu


class LinearElasticitySolver:
    """
    2D plane stress linear elasticity solver.

    Solves: Ku = f
    where K is the global stiffness matrix, u is displacement, f is force vector.

    Uses scikit-fem's linear_elasticity BilinearForm for stiffness assembly.
    Element type: P1 triangles (piecewise linear, 3 nodes per element).
    """

    def __init__(self, mesh, E=1.0, nu=0.3):
        """
        Initialize solver with mesh and material properties.

        Args:
            mesh: skfem.MeshTri object
            E: Young's modulus (Pa)
            nu: Poisson's ratio (dimensionless, must be in (0, 0.5))
        """
        if not (0 < nu < 0.5):
            raise ValueError(f"Poisson's ratio must be in (0, 0.5), got {nu}")
        if E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {E}")

        self.mesh = mesh
        self.E = E
        self.nu = nu

        # Vectorial P1 basis: 2 DOFs per node (u_x, u_y)
        self.basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementTriP1()))

        # Precompute plane stress Lamé parameters
        self.lam, self.mu = plane_stress_lame(E, nu)

    def assemble_stiffness(self):
        """
        Assemble global stiffness matrix K.

        Returns:
            K: scipy.sparse.csr_matrix of shape (n_dofs, n_dofs)

        Uses scikit-fem's linear_elasticity form:
            a(u,v) = integral of  sigma(u) : epsilon(v)  dV
        where sigma = C : epsilon is the stress-strain relation.
        """
        K = linear_elasticity(self.lam, self.mu).assemble(self.basis)
        return K

    def solve(self, K, f, fixed_dofs):
        """
        Apply boundary conditions and solve the linear system.

        Args:
            K: Global stiffness matrix from assemble_stiffness()
            f: Global force vector (n_dofs,)
            fixed_dofs: Array of DOF indices to set to zero (Dirichlet BCs)

        Returns:
            u: Full displacement vector (n_dofs,), fixed DOFs = 0

        Note:
            K is passed in (not assembled inside) so topology optimization
            can modify K externally without reassembling from scratch.
        """
        # condense() removes fixed DOFs from the system.
        # expand=True returns the free DOF index array so we can reconstruct u.
        K_cond, f_cond, _, I_free = skfem.condense(K, f, D=fixed_dofs, expand=True)

        # Solve reduced system (UMFPACK/SuperLU via scipy)
        u_free = spsolve(K_cond, f_cond)

        # Reconstruct full displacement vector; fixed DOFs stay at zero
        u = np.zeros(self.basis.N)
        u[I_free] = u_free
        return u


def node_dofs(node_indices, direction='both'):
    """
    Convert node indices to DOF indices for a 2D vectorial basis.

    DOF layout (interleaved per node):
        node 0 -> DOFs [0, 1]     (u_x0, u_y0)
        node 1 -> DOFs [2, 3]     (u_x1, u_y1)
        node k -> DOFs [2k, 2k+1]

    Args:
        node_indices: Array of node indices
        direction: 'both' (x and y), 'x' only, or 'y' only

    Returns:
        dofs: Array of DOF indices
    """
    node_indices = np.asarray(node_indices)

    if direction == 'both':
        dofs = np.concatenate([2 * node_indices, 2 * node_indices + 1])
    elif direction == 'x':
        dofs = 2 * node_indices
    elif direction == 'y':
        dofs = 2 * node_indices + 1
    else:
        raise ValueError(f"direction must be 'both', 'x', or 'y', got '{direction}'")

    return dofs.astype(int)


def build_force_vector(n_dofs, node_indices, fx=0.0, fy=0.0):
    """
    Build a nodal force vector.

    Distributes (fx, fy) equally across the given nodes.

    Args:
        n_dofs: Total number of DOFs in the system
        node_indices: Nodes where force is applied
        fx: Total x-force to distribute (N)
        fy: Total y-force to distribute (N)

    Returns:
        f: Force vector (n_dofs,)

    Example:
        # 1000 N downward on right edge
        f = build_force_vector(basis.N, right_nodes, fy=-1000.0)
    """
    f = np.zeros(n_dofs)
    n = len(node_indices)
    if n == 0:
        raise ValueError("node_indices is empty")

    for node in node_indices:
        f[2 * node]     += fx / n
        f[2 * node + 1] += fy / n

    return f


if __name__ == "__main__":
    from mesh import create_rectangle_mesh, identify_boundary_nodes

    print("Quick solver test...")
    mesh = create_rectangle_mesh(10.0, 2.0, 50, 10)
    solver = LinearElasticitySolver(mesh, E=210e9, nu=0.3)
    print(f"DOFs: {solver.basis.N}")

    K = solver.assemble_stiffness()
    print(f"K: {K.shape}, nnz={K.nnz}")
    print(f"Symmetric: {np.allclose((K - K.T).data, 0, atol=1e-6)}")
