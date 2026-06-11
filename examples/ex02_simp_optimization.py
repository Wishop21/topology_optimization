"""
Example 02: SIMP Topology Optimization

Runs three classic benchmark problems in sequence. Each shows a live
matplotlib window updating each iteration (or every N for large meshes).

Run:
    python ex02_simp_optimization.py             # runs cantilever (fast)
    python ex02_simp_optimization.py --mbb       # MBB beam benchmark
    python ex02_simp_optimization.py --bridge    # bridge structure
    python ex02_simp_optimization.py --all       # all three in sequence
    python ex02_simp_optimization.py --fine      # finer mesh (slower)
"""

import sys
import argparse
sys.path.insert(0, '..')

from problems.problems_2d import CantileverBeam2D, MBBBeam2D, BridgeStructure2D
from topology_optimizer import TopologyOptimizer


def run_cantilever(fine=False):
    print("\n" + "=" * 60)
    print("CANTILEVER BEAM - SIMP Optimization")
    print("=" * 60)

    nx, ny = (80, 40) if fine else (40, 20)
    r_min = 2.5 if fine else 1.5  # finer mesh needs larger filter for convergence

    problem = CantileverBeam2D(
        nx=nx, ny=ny,
        width=2.0, height=1.0,
        E=1.0, nu=0.3,
        vol_frac=0.5,
        force=-1.0
    )

    print(f"Mesh    : {nx} x {ny} cells ({nx*ny} design variables)")
    print(f"DOFs    : {2 * (nx+1) * (ny+1)}")
    print(f"Target  : 50% volume fraction")
    print(f"Expected: diagonal bracing pattern\n")

    opt = TopologyOptimizer(problem)
    print(f"Plot update interval: every {opt.update_interval} iteration(s)")

    rho = opt.run(max_iter=350, tol=0.01, r_min=r_min)
    opt.save_result('cantilever_optimized.png')
    return rho


def run_mbb(fine=False):
    print("\n" + "=" * 60)
    print("MBB BEAM - SIMP Optimization (half model)")
    print("=" * 60)

    nx, ny = (90, 30) if fine else (60, 20)

    problem = MBBBeam2D(
        nx=nx, ny=ny,
        E=1.0, nu=0.3,
        vol_frac=0.5,
        force=-1.0
    )

    print(f"Mesh    : {nx} x {ny} cells ({nx*ny} design variables)")
    print(f"Target  : 50% volume fraction")
    print(f"Expected: classic arch/truss (compare to Sigmund 2001)\n")

    opt = TopologyOptimizer(problem)
    print(f"Plot update interval: every {opt.update_interval} iteration(s)")

    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)
    opt.save_result('mbb_optimized.png')
    return rho


def run_bridge(fine=False):
    print("\n" + "=" * 60)
    print("BRIDGE STRUCTURE - SIMP Optimization")
    print("=" * 60)

    nx, ny = (80, 40) if fine else (60, 30)

    problem = BridgeStructure2D(
        nx=nx, ny=ny,
        E=1.0, nu=0.3,
        vol_frac=0.4,
        force=-1.0
    )

    print(f"Mesh    : {nx} x {ny} cells ({nx*ny} design variables)")
    print(f"Target  : 40% volume fraction")
    print(f"Expected: arch-and-cable truss\n")

    opt = TopologyOptimizer(problem)
    print(f"Plot update interval: every {opt.update_interval} iteration(s)")

    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5)
    opt.save_result('bridge_optimized.png')
    return rho


def main():
    parser = argparse.ArgumentParser(description='SIMP Topology Optimization')
    parser.add_argument('--mbb',    action='store_true', help='Run MBB beam')
    parser.add_argument('--bridge', action='store_true', help='Run bridge')
    parser.add_argument('--all',    action='store_true', help='Run all problems')
    parser.add_argument('--fine',   action='store_true', help='Use finer mesh')
    args = parser.parse_args()

    if args.all:
        run_cantilever(args.fine)
        run_mbb(args.fine)
        run_bridge(args.fine)
    elif args.mbb:
        run_mbb(args.fine)
    elif args.bridge:
        run_bridge(args.fine)
    else:
        run_cantilever(args.fine)


if __name__ == "__main__":
    main()
