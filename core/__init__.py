"""
Core FEA components for topology optimization.

Modules:
    mesh: Mesh generation and loading
    fem_solver: Linear elasticity FEA solver (scikit-fem wrapper)
    analysis: Stress recovery, compliance calculation
"""

__all__ = ['mesh', 'fem_solver', 'analysis']
