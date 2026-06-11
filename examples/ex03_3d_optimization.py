"""
Example 03: 3D SIMP Topology Optimization

Runs 3D problems with live slice monitoring and PyVista final result.

Usage:
    python ex03_3d_optimization.py                  # 3D cantilever (default)
    python ex03_3d_optimization.py --mbb            # 3D MBB beam
    python ex03_3d_optimization.py --lbracket       # 3D L-bracket
    python ex03_3d_optimization.py --multiload      # cantilever with 2 load cases
    python ex03_3d_optimization.py --fine           # finer mesh (slower)
    python ex03_3d_optimization.py --no-pyvista     # skip PyVista, slices only
"""

import sys
import argparse
sys.path.insert(0, '..')

from problems.problems_3d import (
    Cantilever3D, MBBBeam3D, LBracket3D, Cantilever3DMultiLoad,
    Cantilever3DEdgeLoad, TorsionBeam3D)
from topology_optimizer_3d import TopologyOptimizer3D


def run_cantilever(fine=False, pyvista=True):
    print("\n" + "=" * 65)
    print("3D CANTILEVER BEAM — SIMP Optimization")
    print("=" * 65)

    nx, ny, nz = (40, 20, 15) if fine else (30, 15, 10)

    problem = Cantilever3D(
        nx=nx, ny=ny, nz=nz,
        width=2.0, height=1.0, depth=1.0,
        E=1.0, nu=0.3,
        vol_frac=0.3,
        load_direction='y'
    )

    opt = TopologyOptimizer3D(problem)
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)

    opt.save_slices('cantilever_3d_slices.png')
    if pyvista:
        print("\nLaunching PyVista viewer (close window to exit)...")
        opt.show_result(threshold=0.3, save_path='cantilever_3d_surface.png')
    return rho


def run_mbb(fine=False, pyvista=True):
    print("\n" + "=" * 65)
    print("3D MBB BEAM (half model) — SIMP Optimization")
    print("=" * 65)

    nx, ny, nz = (40, 15, 15) if fine else (30, 10, 10)

    problem = MBBBeam3D(
        nx=nx, ny=ny, nz=nz,
        E=1.0, nu=0.3,
        vol_frac=0.5
    )

    opt = TopologyOptimizer3D(problem)
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)

    opt.save_slices('mbb_3d_slices.png')
    if pyvista:
        print("\nLaunching PyVista viewer (close window to exit)...")
        opt.show_result(threshold=0.4, save_path='mbb_3d_surface.png')
    return rho


def run_lbracket(fine=False, pyvista=True):
    print("\n" + "=" * 65)
    print("3D L-BRACKET — SIMP Optimization")
    print("=" * 65)

    nx, ny, nz = (40, 40, 15) if fine else (30, 30, 10)

    problem = LBracket3D(
        nx=nx, ny=ny, nz=nz,
        E=1.0, nu=0.3,
        vol_frac=0.4
    )

    opt = TopologyOptimizer3D(problem)

    # For L-bracket: force void cells to stay empty throughout optimization
    # by clamping them in the OC update via very low initial density
    import numpy as np
    # Pre-zero void cells so they stay near rho_min
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)

    opt.save_slices('lbracket_3d_slices.png')
    if pyvista:
        print("\nLaunching PyVista viewer (close window to exit)...")
        opt.show_result(threshold=0.3, save_path='lbracket_3d_surface.png')
    return rho


def run_multiload(fine=False, pyvista=True):
    print("\n" + "=" * 65)
    print("3D CANTILEVER — Multi-Load (downward + lateral)")
    print("=" * 65)

    nx, ny, nz = (30, 15, 15) if fine else (24, 12, 12)

    problem = Cantilever3DMultiLoad(
        nx=nx, ny=ny, nz=nz,
        E=1.0, nu=0.3,
        vol_frac=0.3
    )

    print("Load cases:")
    print("  Case 1: -1.0 N downward (y-direction)")
    print("  Case 2: -1.0 N lateral  (z-direction)")
    print("Expected: symmetric diagonal bracing (not planar)")

    opt = TopologyOptimizer3D(problem)
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)

    opt.save_slices('multiload_3d_slices.png')
    if pyvista:
        print("\nLaunching PyVista viewer (close window to exit)...")
        opt.show_result(threshold=0.3, save_path='multiload_3d_surface.png')
    return rho



def run_edge_load(fine=False, pyvista=True):
    print("\n" + "=" * 65)
    print("3D CANTILEVER (EDGE LOAD) — More interesting topology")
    print("=" * 65)
    print("Load on bottom edge of right face → clear tension/compression members")

    nx, ny, nz = (40, 20, 15) if fine else (30, 15, 10)
    problem = Cantilever3DEdgeLoad(nx=nx, ny=ny, nz=nz, vol_frac=0.3)

    opt = TopologyOptimizer3D(problem)
    # Edge load benefits from larger filter (r_min=2.0) to reduce oscillation
    rho = opt.run(max_iter=400, tol=0.01, r_min=2.0, sens_avg=3)
    opt.save_slices('cantilever_edge_3d_slices.png')
    if pyvista:
        opt.show_result(threshold=0.4, save_path='cantilever_edge_3d_surface.png')
    return rho


def run_torsion(fine=False, pyvista=True):
    print("\n" + "=" * 65)
    print("3D TORSION BEAM — Genuinely 3D topology")
    print("=" * 65)
    print("Opposing forces create torsion → helical bracing pattern")

    nx, ny, nz = (40, 15, 15) if fine else (30, 10, 10)
    problem = TorsionBeam3D(nx=nx, ny=ny, nz=nz, vol_frac=0.3)

    opt = TopologyOptimizer3D(problem)
    rho = opt.run(max_iter=400, tol=0.01, r_min=2.0, sens_avg=3)
    opt.save_slices('torsion_3d_slices.png')
    if pyvista:
        opt.show_result(threshold=0.3, save_path='torsion_3d_surface.png')
    return rho


def main():
    parser = argparse.ArgumentParser(description='3D SIMP Topology Optimization')
    parser.add_argument('--mbb',       action='store_true')
    parser.add_argument('--lbracket',  action='store_true')
    parser.add_argument('--multiload', action='store_true')
    parser.add_argument('--fine',      action='store_true')
    parser.add_argument('--edgeload',  action='store_true', help='Edge-load cantilever (clearer topology)')
    parser.add_argument('--torsion',   action='store_true', help='Torsion beam (genuinely 3D topology)')
    parser.add_argument('--no-pyvista', action='store_true',
                        help='Skip PyVista isosurface viewer')
    args = parser.parse_args()

    pyvista = not args.no_pyvista

    if args.edgeload:
        run_edge_load(args.fine, pyvista)
    elif args.torsion:
        run_torsion(args.fine, pyvista)
    elif args.mbb:
        run_mbb(args.fine, pyvista)
    elif args.lbracket:
        run_lbracket(args.fine, pyvista)
    elif args.multiload:
        run_multiload(args.fine, pyvista)
    else:
        run_cantilever(args.fine, pyvista)


if __name__ == "__main__":
    main()
