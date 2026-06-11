"""
Example 04: 3D SIMP Topology Optimization with Hex Elements

Demonstrates hex elements vs tet on the same problems.

Usage:
    python ex04_hex_optimization.py                  # hex cantilever
    python ex04_hex_optimization.py --edgeload       # edge load
    python ex04_hex_optimization.py --mbb            # MBB beam
    python ex04_hex_optimization.py --torsion        # torsion beam
    python ex04_hex_optimization.py --multiload      # two load cases
    python ex04_hex_optimization.py --compare        # hex vs tet benchmark
    python ex04_hex_optimization.py --no-pyvista     # skip 3D viewer
"""

import sys
import argparse
import time
sys.path.insert(0, '..')

from problems.problems_hex import (
    CantileverHex, CantileverHexEdgeLoad, MBBBeamHex,
    TorsionBeamHex, CantileverHexMultiLoad, LBracketHex)
from topology_optimizer_hex import TopologyOptimizerHex


def run_cantilever(pyvista=True):
    print("\n" + "=" * 65)
    print("3D CANTILEVER — Hex Elements")
    print("=" * 65)
    problem = CantileverHex(nx=30, ny=15, nz=10, vol_frac=0.3)
    opt = TopologyOptimizerHex(problem)
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)
    opt.save_slices('cantilever_hex_slices.png')
    if pyvista:
        opt.show_result(threshold=0.4, save_path='cantilever_hex_surface.png')
    return opt  # return optimizer so caller can export


def run_edgeload(pyvista=True):
    print("\n" + "=" * 65)
    print("3D CANTILEVER EDGE LOAD — Hex Elements")
    print("=" * 65)
    problem = CantileverHexEdgeLoad(nx=30, ny=15, nz=10, vol_frac=0.3)
    opt = TopologyOptimizerHex(problem)
    rho = opt.run(max_iter=400, tol=0.01, r_min=2.0, sens_avg=3)
    opt.save_slices('cantilever_edge_hex_slices.png')
    if pyvista:
        opt.show_result(threshold=0.4, save_path='cantilever_edge_hex_surface.png')
    return rho


def run_mbb(pyvista=True):
    print("\n" + "=" * 65)
    print("3D MBB BEAM — Hex Elements")
    print("=" * 65)
    problem = MBBBeamHex(nx=30, ny=10, nz=10, vol_frac=0.5)
    opt = TopologyOptimizerHex(problem)
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)
    opt.save_slices('mbb_hex_slices.png')
    if pyvista:
        opt.show_result(threshold=0.4, save_path='mbb_hex_surface.png')
    return rho


def run_torsion(pyvista=True):
    print("\n" + "=" * 65)
    print("3D TORSION BEAM — Hex Elements")
    print("=" * 65)
    problem = TorsionBeamHex(nx=30, ny=10, nz=10, vol_frac=0.3)
    opt = TopologyOptimizerHex(problem)
    rho = opt.run(max_iter=400, tol=0.01, r_min=2.0, sens_avg=3)
    opt.save_slices('torsion_hex_slices.png')
    if pyvista:
        opt.show_result(threshold=0.3, save_path='torsion_hex_surface.png')
    return rho


def run_multiload(pyvista=True):
    print("\n" + "=" * 65)
    print("3D CANTILEVER MULTI-LOAD — Hex Elements")
    print("=" * 65)
    problem = CantileverHexMultiLoad(nx=24, ny=12, nz=12, vol_frac=0.3)
    opt = TopologyOptimizerHex(problem)
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)
    opt.save_slices('multiload_hex_slices.png')
    if pyvista:
        opt.show_result(threshold=0.3, save_path='multiload_hex_surface.png')
    return rho


def run_comparison():
    """
    Benchmark hex vs tet on identical mesh resolution.
    Shows accuracy and timing improvement from hex elements.
    """
    print("\n" + "=" * 65)
    print("HEX vs TET COMPARISON — 30x15x10 mesh")
    print("=" * 65)
    import numpy as np
    import skfem
    from skfem.models import linear_elasticity
    from scipy.sparse.linalg import spsolve

    L, h, w = 4.0, 1.0, 1.0
    E, nu = 210e9, 0.3
    F = -1000.0
    I = w * h**3 / 12
    tip_theory = F * L**3 / (3 * E * I)
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))

    for label, MeshCls, ElemCls, intorder in [
        ("Tet P1  ", skfem.MeshTet, skfem.ElementTetP1, None),
        ("Hex Q1  ", skfem.MeshHex, skfem.ElementHex1,  2),
    ]:
        mesh = MeshCls.init_tensor(
            np.linspace(0,L,31), np.linspace(0,h,16), np.linspace(0,w,11))
        basis = (skfem.Basis(mesh, skfem.ElementVector(ElemCls()), intorder=intorder)
                 if intorder else
                 skfem.Basis(mesh, skfem.ElementVector(ElemCls())))

        t0 = time.time()
        K = linear_elasticity(lam, mu).assemble(basis)
        t_assemble = time.time()-t0

        x = mesh.p[0]; tol = 1e-10
        left  = np.where(np.abs(x-x.min())<tol)[0]
        right = np.where(np.abs(x-x.max())<tol)[0]
        fixed = np.concatenate([3*left, 3*left+1, 3*left+2])
        fv = np.zeros(basis.N)
        for n in right: fv[3*n+1] = F/len(right)
        K_c, f_c, _, I_free = skfem.condense(K, fv, D=fixed, expand=True)
        t0 = time.time()
        u = np.zeros(basis.N); u[I_free] = spsolve(K_c, f_c)
        t_solve = time.time()-t0

        tip = u[3*right+1].mean()
        err = abs(tip-tip_theory)/abs(tip_theory)*100
        n_elem = mesh.t.shape[1]
        n_qp   = basis.X.shape[-1]

        print(f"\n{label}: {n_elem:6d} elements, {basis.N:6d} DOFs, {n_qp} qp/elem")
        print(f"  Assembly: {t_assemble:.3f}s   Solve: {t_solve:.3f}s")
        print(f"  Tip deflection: {tip*1000:.4f} mm  (theory: {tip_theory*1000:.4f} mm)")
        print(f"  Error: {err:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Hex element topology optimization')
    parser.add_argument('--edgeload',  action='store_true')
    parser.add_argument('--mbb',       action='store_true')
    parser.add_argument('--torsion',   action='store_true')
    parser.add_argument('--multiload', action='store_true')
    parser.add_argument('--compare',   action='store_true',
                        help='Benchmark hex vs tet accuracy and timing')
    parser.add_argument('--no-pyvista', action='store_true')
    parser.add_argument('--export', action='store_true',
                        help='Export STL and OBJ after optimization')
    args = parser.parse_args()

    pyvista = not args.no_pyvista

    if args.compare:
        run_comparison()
    elif args.edgeload:
        result = run_edgeload(pyvista)
    elif args.mbb:
        result = run_mbb(pyvista)
    elif args.torsion:
        result = run_torsion(pyvista)
    elif args.multiload:
        result = run_multiload(pyvista)
    else:
        result = run_cantilever(pyvista)

    # Export CAD files if requested
    if args.export and not args.compare:
        if hasattr(result, 'export_cad'):
            print("\nExporting CAD files...")
            files = result.export_cad(
                basename='topology_export',
                threshold=0.4,
                smooth_iter=50,
                formats=('stl', 'obj')
            )
            print(f"\nExported files:")
            for fmt, path in files.items():
                print(f"  {fmt.upper()}: {path}")
        else:
            print("Re-run without --compare to export.")


if __name__ == "__main__":
    main()
