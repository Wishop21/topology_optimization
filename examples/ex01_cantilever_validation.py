"""
Example 01: Cantilever Beam Validation

Solves a cantilever beam and validates tip deflection against
Euler-Bernoulli beam theory. Produces a 4-panel plot.

Run:
    python ex01_cantilever_validation.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

from core.mesh import create_rectangle_mesh, identify_boundary_nodes, get_mesh_info
from core.fem_solver import LinearElasticitySolver, node_dofs, build_force_vector


def main():
    print("\n" + "=" * 70)
    print("CANTILEVER BEAM - FEM Validation")
    print("=" * 70)

    # ---- Problem definition ----
    L, h, b = 10.0, 2.0, 1.0   # m
    E, nu   = 210e9, 0.3        # Pa, -
    F       = -1000.0           # N (downward at tip)

    print(f"\nGeometry : {L} m x {h} m x {b} m")
    print(f"Material : E = {E/1e9:.0f} GPa,  nu = {nu}")
    print(f"Load     : {abs(F):.0f} N downward distributed on right edge")

    # ---- Mesh ----
    mesh = create_rectangle_mesh(L, h, nx=100, ny=20)
    info = get_mesh_info(mesh)
    bounds = identify_boundary_nodes(mesh)

    print(f"\nMesh     : {info['n_elements']} elements, {info['n_nodes']} nodes")

    # ---- Solver ----
    solver = LinearElasticitySolver(mesh, E=E, nu=nu)
    print(f"DOFs     : {solver.basis.N}")

    # ---- Boundary conditions ----
    fixed = node_dofs(bounds['left'], direction='both')
    f_vec = build_force_vector(solver.basis.N, bounds['right'], fy=F)

    # ---- Solve ----
    print("\nAssembling stiffness matrix...")
    K = solver.assemble_stiffness()
    print(f"  K: {K.shape}, {K.nnz} non-zeros")

    print("Solving...")
    u = solver.solve(K, f_vec, fixed)

    # ---- Extract results ----
    u_x = u[0::2]
    u_y = u[1::2]
    tip_deflection = u[2 * bounds['right'] + 1].mean()

    print(f"\nMax x-displacement : {u_x.max()*1000:.4f} mm")
    print(f"Max y-displacement : {abs(u_y.min())*1000:.4f} mm")
    print(f"Tip deflection     : {abs(tip_deflection)*1000:.4f} mm")

    # ---- Validate ----
    I = b * h**3 / 12.0
    tip_theory = F * L**3 / (3.0 * E * I)
    error = abs(tip_deflection - tip_theory) / abs(tip_theory) * 100

    print(f"\nValidation:")
    print(f"  FEM    : {tip_deflection*1000:.4f} mm")
    print(f"  Theory : {tip_theory*1000:.4f} mm")
    print(f"  Error  : {error:.2f}%  ->  {'PASS' if error < 5 else 'FAIL'}")

    # ---- Plot ----
    x = mesh.p[0, :]
    y = mesh.p[1, :]
    tri = mesh.t.T
    scale = 300

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Mesh with BCs
    ax = axes[0, 0]
    ax.triplot(x, y, tri, 'k-', lw=0.3)
    ax.plot(x[bounds['left']], y[bounds['left']], 'bs', ms=3, label='Fixed')
    ax.plot(x[bounds['right']], y[bounds['right']], 'r^', ms=3, label='Loaded')
    ax.set_aspect('equal')
    ax.set_title(f"Mesh ({info['n_elements']} elements)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Deformed shape (exaggerated)
    ax = axes[0, 1]
    ax.triplot(x, y, tri, 'k-', lw=0.2, alpha=0.3, label='Undeformed')
    ax.triplot(x + scale*u_x, y + scale*u_y, tri, 'r-', lw=0.4, alpha=0.7,
               label=f'Deformed x{scale}')
    ax.set_aspect('equal')
    ax.set_title('Deformed shape (exaggerated)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. X-displacement
    ax = axes[1, 0]
    cf = ax.tricontourf(x, y, tri, u_x * 1000, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='mm')
    ax.set_aspect('equal')
    ax.set_title('X-displacement (mm)')

    # 4. Y-displacement
    ax = axes[1, 1]
    cf = ax.tricontourf(x, y, tri, u_y * 1000, levels=20, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='mm')
    ax.set_aspect('equal')
    ax.set_title('Y-displacement (mm)')

    plt.tight_layout()
    fname = 'cantilever_results.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {fname}")
    plt.show()


if __name__ == "__main__":
    main()
