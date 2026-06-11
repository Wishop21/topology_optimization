"""
Post-processing and analysis of FEA results.

Handles:
- Stress recovery from displacements
- Von Mises stress calculation
- Compliance (strain energy) calculation
- Validation against analytical solutions
"""

import numpy as np
import skfem
from skfem.helpers import sym_grad


def compute_strain(basis, u):
    """
    Compute strain field from displacement field.
    
    Args:
        basis: skfem.Basis object (vectorial)
        u: Displacement vector (n_dofs,)
    
    Returns:
        strain: DiscreteField object with strain at quadrature points
                Shape: (3, n_qpts, n_elements)
                [ε_xx, ε_yy, 2*ε_xy] in Voigt notation
    """
    # Project displacement to basis for interpolation
    uh = basis.interpolate(u)
    
    # Compute strain: ε = sym_grad(u) = (∇u + ∇u^T) / 2
    @skfem.Functional
    def strain_functional(w):
        # Get displacement field at quadrature points
        u_field = w['u']
        return sym_grad(u_field)
    
    # Evaluate at quadrature points
    strain = strain_functional.elemental(basis, u=uh)
    
    return strain


def compute_stress(strain, E, nu):
    """
    Compute stress from strain using constitutive relation.
    
    Args:
        strain: Strain field from compute_strain
        E: Young's modulus
        nu: Poisson's ratio
    
    Returns:
        stress: Stress field (3, n_qpts, n_elements)
                [σ_xx, σ_yy, σ_xy] in Voigt notation
    """
    # Elasticity tensor for plane stress
    factor = E / (1 - nu**2)
    C = factor * np.array([
        [1, self.nu, 0],
        [self.nu, 1, 0],
        [0, 0, (1 - self.nu) / 2]
    ])
    
    # σ = C:ε
    # Broadcasting: C is (3,3), strain is (3, n_qpts, n_elems)
    stress = np.einsum('ij,j...->i...', C, strain)
    
    return stress


def compute_von_mises_stress(stress):
    """
    Compute Von Mises stress from stress tensor.
    
    Args:
        stress: Stress field (3, n_qpts, n_elements)
                [σ_xx, σ_yy, σ_xy]
    
    Returns:
        von_mises: Von Mises stress (n_qpts, n_elements)
    
    Formula for 2D plane stress:
        σ_vm = √(σ_xx² - σ_xx*σ_yy + σ_yy² + 3*σ_xy²)
    """
    sigma_xx = stress[0]
    sigma_yy = stress[1]
    sigma_xy = stress[2]
    
    von_mises = np.sqrt(
        sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2
    )
    
    return von_mises


def compute_compliance(K, u):
    """
    Compute compliance (strain energy).
    
    Args:
        K: Global stiffness matrix
        u: Displacement vector
    
    Returns:
        compliance: Scalar value = u^T K u
    
    This is the objective function for topology optimization.
    Minimizing compliance = maximizing stiffness.
    """
    compliance = u @ (K @ u)
    return compliance


def average_stress_to_nodes(basis, stress_field):
    """
    Average element-wise stress to node values for plotting.
    
    Args:
        basis: skfem.Basis object
        stress_field: Stress at quadrature points (n_qpts, n_elements)
    
    Returns:
        nodal_stress: Stress averaged to nodes (n_nodes,)
    
    Note:
        Simple averaging. For better accuracy, use L2 projection.
    """
    mesh = basis.mesh
    n_nodes = mesh.p.shape[1]
    n_elements = mesh.t.shape[1]
    
    nodal_stress = np.zeros(n_nodes)
    node_count = np.zeros(n_nodes)
    
    # Average stress from all elements connected to each node
    for elem_idx in range(n_elements):
        # Get nodes of this element
        elem_nodes = mesh.t[:, elem_idx]
        
        # Average quadrature points for this element
        elem_stress = stress_field[:, elem_idx].mean()
        
        # Accumulate to nodes
        for node in elem_nodes:
            nodal_stress[node] += elem_stress
            node_count[node] += 1
    
    # Average
    nodal_stress /= np.maximum(node_count, 1)
    
    return nodal_stress


class AnalyticalSolutions:
    """
    Analytical solutions for validation.
    
    These are exact solutions for simple geometries that we can
    compare our FEA results against.
    """
    
    @staticmethod
    def cantilever_tip_deflection(length, height, E, I, force):
        """
        Tip deflection of cantilever beam with end load.
        
        Args:
            length: Beam length
            height: Beam height
            E: Young's modulus
            I: Second moment of area = b*h³/12
            force: Point load at tip (downward = negative)
        
        Returns:
            deflection: Tip deflection (negative = downward)
        
        Formula: δ = -F*L³ / (3*E*I)
        """
        deflection = force * length**3 / (3 * E * I)
        return deflection
    
    @staticmethod
    def cantilever_max_stress(height, force, I, y):
        """
        Maximum bending stress in cantilever at fixed end.
        
        Args:
            height: Beam height
            force: Applied force at tip
            I: Second moment of area
            y: Distance from neutral axis (height/2 for max stress)
        
        Returns:
            stress: Maximum normal stress
        
        Formula: σ_max = M*y / I, where M = F*L at fixed end
        """
        # Note: This requires moment M, which depends on position
        # For comparison, need to evaluate at specific location
        raise NotImplementedError("Use with specific loading case")
    
    @staticmethod
    def simple_tension_stress(force, area):
        """
        Stress in simple tension.
        
        Args:
            force: Applied tensile force
            area: Cross-sectional area
        
        Returns:
            stress: Normal stress = F/A
        """
        return force / area
    
    @staticmethod
    def simple_tension_strain(stress, E):
        """
        Strain in simple tension.
        
        Args:
            stress: Normal stress
            E: Young's modulus
        
        Returns:
            strain: Normal strain = σ/E
        """
        return stress / E


if __name__ == "__main__":
    # Test analytical solutions
    print("Cantilever beam example:")
    L = 10.0  # m
    h = 2.0   # m
    b = 1.0   # thickness (for plane stress)
    E = 210e9  # Pa (steel)
    I = b * h**3 / 12
    F = -1000.0  # N (downward)
    
    delta = AnalyticalSolutions.cantilever_tip_deflection(L, h, E, I, F)
    print(f"  Analytical tip deflection: {delta*1000:.3f} mm")
    
    print("\nSimple tension example:")
    A = h * b  # Cross-sectional area
    F_tension = 10000.0  # N
    sigma = AnalyticalSolutions.simple_tension_stress(F_tension, A)
    epsilon = AnalyticalSolutions.simple_tension_strain(sigma, E)
    print(f"  Stress: {sigma/1e6:.2f} MPa")
    print(f"  Strain: {epsilon*1e6:.2f} µε")
