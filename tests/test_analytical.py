"""
Validation tests against analytical solutions.

Two tests:
  1. Cantilever beam tip deflection  (target: < 5% error)
  2. Simple tension bar stress       (target: < 1% error)

Run:
    python test_analytical.py          # text output only
    python test_analytical.py --plot   # also saves PNG plots
"""

import sys
import argparse
import numpy as np
import skfem

sys.path.insert(0, '..')

from core.mesh import create_rectangle_mesh, identify_boundary_nodes
from core.fem_solver import LinearElasticitySolver, node_dofs, build_force_vector


# ---------------------------------------------------------------------------
# Analytical reference solutions
# ---------------------------------------------------------------------------

def cantilever_tip_deflection(L, h, b, E, F):
    """
    Euler-Bernoulli tip deflection for cantilever with end load.

    delta = F * L^3 / (3 * E * I)
    I = b * h^3 / 12
    """
    I = b * h**3 / 12.0
    return F * L**3 / (3.0 * E * I)


def tension_stress(F, h, b):
    """Uniform stress in simple tension: sigma = F / A."""
    return F / (h * b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cantilever(plot=False):
    print("\n" + "=" * 60)
    print("TEST 1: Cantilever Beam Tip Deflection")
    print("=" * 60)

    # Problem definition
    L, h, b = 10.0, 2.0, 1.0   # m
    E, nu   = 210e9, 0.3        # Pa, -
    F       = -1000.0           # N (downward)

    print(f"  Geometry : {L} m x {h} m, thickness {b} m")
    print(f"  Material : E = {E/1e9:.0f} GPa, nu = {nu}")
    print(f"  Load     : {abs(F):.0f} N downward at tip")

    # Mesh
    mesh = create_rectangle_mesh(L, h, nx=100, ny=20)
    bounds = identify_boundary_nodes(mesh)
    print(f"  Mesh     : {mesh.t.shape[1]} elements, {mesh.p.shape[1]} nodes")

    # Solver
    solver = LinearElasticitySolver(mesh, E=E, nu=nu)
    K = solver.assemble_stiffness()
    print(f"  DOFs     : {solver.basis.N}")

    # BCs: fix left edge fully
    fixed = node_dofs(bounds['left'], direction='both')

    # Force: distribute F_y across right edge nodes
    f = build_force_vector(solver.basis.N, bounds['right'], fy=F)

    # Solve
    u = solver.solve(K, f, fixed)

    # Extract average tip deflection (y-displacement on right edge)
    right_y_dofs = 2 * bounds['right'] + 1
    tip_fem = u[right_y_dofs].mean()

    # Reference
    tip_ref = cantilever_tip_deflection(L, h, b, E, F)

    error = abs(tip_fem - tip_ref) / abs(tip_ref) * 100
    print(f"\n  FEM tip deflection : {tip_fem*1000:.4f} mm")
    print(f"  Analytical         : {tip_ref*1000:.4f} mm")
    print(f"  Error              : {error:.2f}%")

    passed = error < 5.0
    print(f"  Result             : {'PASS' if passed else 'FAIL'}")

    if plot:
        import matplotlib.pyplot as plt

        x = mesh.p[0, :]
        y = mesh.p[1, :]
        tri = mesh.t.T
        u_x = u[0::2]
        u_y = u[1::2]
        scale = 300

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].triplot(x, y, tri, 'k-', lw=0.3, alpha=0.4, label='Undeformed')
        axes[0].triplot(x + scale*u_x, y + scale*u_y, tri, 'r-', lw=0.4, alpha=0.6, label=f'Deformed x{scale}')
        axes[0].set_aspect('equal'); axes[0].set_title('Deformed shape'); axes[0].legend()

        cf = axes[1].tricontourf(x, y, tri, u_y * 1000, levels=20, cmap='RdBu_r')
        plt.colorbar(cf, ax=axes[1], label='mm')
        axes[1].set_aspect('equal'); axes[1].set_title('Y-displacement (mm)')

        plt.tight_layout()
        fname = 'cantilever_validation.png'
        plt.savefig(fname, dpi=150)
        print(f"  Plot saved: {fname}")
        plt.show()

    return passed


def test_simple_tension(plot=False):
    print("\n" + "=" * 60)
    print("TEST 2: Simple Tension Bar")
    print("=" * 60)

    L, h, b = 1.0, 0.1, 1.0   # m
    E, nu   = 210e9, 0.3
    F       = 10000.0          # N (tensile, x-direction)

    print(f"  Geometry : {L} m x {h} m")
    print(f"  Load     : {F:.0f} N tensile (x-direction)")

    mesh = create_rectangle_mesh(L, h, nx=20, ny=4)
    bounds = identify_boundary_nodes(mesh)
    print(f"  Mesh     : {mesh.t.shape[1]} elements")

    solver = LinearElasticitySolver(mesh, E=E, nu=nu)
    K = solver.assemble_stiffness()

    # Fix x-DOFs on left edge; pin one y-DOF to remove rigid body rotation
    fixed_x = node_dofs(bounds['left'], direction='x')
    fixed_y0 = np.array([2 * bounds['left'][0] + 1], dtype=int)
    fixed = np.concatenate([fixed_x, fixed_y0])

    f = build_force_vector(solver.basis.N, bounds['right'], fx=F)
    u = solver.solve(K, f, fixed)

    # Average x-displacement on each edge, compute strain
    u_left_x  = u[2 * bounds['left']].mean()
    u_right_x = u[2 * bounds['right']].mean()
    strain_fem = (u_right_x - u_left_x) / L
    stress_fem = E * strain_fem

    stress_ref = tension_stress(F, h, b)
    error = abs(stress_fem - stress_ref) / stress_ref * 100

    print(f"\n  FEM stress  : {stress_fem/1e6:.3f} MPa")
    print(f"  Analytical  : {stress_ref/1e6:.3f} MPa")
    print(f"  Error       : {error:.3f}%")

    passed = error < 1.0
    print(f"  Result      : {'PASS' if passed else 'FAIL'}")

    return passed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(plot=False):
    print("\n" + "=" * 60)
    print("RUNNING ALL VALIDATION TESTS")
    print("=" * 60)

    results = {}

    for name, test_fn in [('cantilever', test_cantilever),
                           ('tension',    test_simple_tension)]:
        try:
            results[name] = test_fn(plot=plot)
        except Exception as e:
            import traceback
            print(f"\nEXCEPTION in {name} test:")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:20s}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print("\n" + ("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"))
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    ok = run_all(plot=args.plot)
    sys.exit(0 if ok else 1)
