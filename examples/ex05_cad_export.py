"""
Example 05: CAD Export

Export topology optimization results to STL and OBJ files for:
  - 3D printing (STL)
  - Import into Autodesk Inventor as mesh body (OBJ or STL)
  - Viewing in MeshLab, Blender, etc.

Usage:
    # Export from a fresh optimization run:
    python ex05_cad_export.py

    # Export with finer mesh for better surface quality:
    python ex05_cad_export.py --fine

    # Adjust threshold (lower = include more grey material):
    python ex05_cad_export.py --threshold 0.35

    # More smoothing:
    python ex05_cad_export.py --smooth 100

    # Also try STEP (requires pythonOCC):
    python ex05_cad_export.py --step

About the output files:
    topology_result.stl  - Binary STL, opens in any CAD/slicer tool
    topology_result.obj  - Wavefront OBJ, Inventor 2019+ imports as mesh body

About thresholds:
    threshold=0.5  - strict: only clearly solid cells (sharpest boundaries)
    threshold=0.3  - relaxed: includes some grey cells (fuller structure)
    threshold=0.4  - good compromise for most results

About Inventor import:
    1. Open Inventor
    2. File → Import → choose your .stl or .obj file
    3. It imports as a mesh body (not an editable solid)
    4. You can measure, section, and view it but not add features
    For editable solid: use Mesh Enabler add-in or manual reconstruction
"""

import sys
import argparse
sys.path.insert(0, '..')

from problems.problems_hex import CantileverHex, MBBBeamHex, CantileverHexEdgeLoad
from topology_optimizer_hex import TopologyOptimizerHex
from export.cad_export import export_all, density_to_mesh


def run_and_export(problem_name='cantilever', fine=False,
                   threshold=0.4, smooth_iter=50, include_step=False):
    """Run optimization then export results."""

    # ── Choose problem ────────────────────────────────────────────────────
    if problem_name == 'cantilever':
        nx, ny, nz = (40, 20, 15) if fine else (30, 15, 10)
        problem = CantileverHex(nx=nx, ny=ny, nz=nz, vol_frac=0.3)
        basename = 'cantilever_export'

    elif problem_name == 'edgeload':
        nx, ny, nz = (40, 20, 15) if fine else (30, 15, 10)
        problem = CantileverHexEdgeLoad(nx=nx, ny=ny, nz=nz, vol_frac=0.3)
        basename = 'edgeload_export'

    elif problem_name == 'mbb':
        nx, ny, nz = (40, 15, 15) if fine else (30, 10, 10)
        problem = MBBBeamHex(nx=nx, ny=ny, nz=nz, vol_frac=0.5)
        basename = 'mbb_export'

    else:
        raise ValueError(f"Unknown problem: {problem_name}")

    # ── Optimize ──────────────────────────────────────────────────────────
    print(f"\nOptimizing {problem.name}...")
    opt = TopologyOptimizerHex(problem)
    rho = opt.run(max_iter=300, tol=0.01, r_min=1.5, verbose=True)

    # ── Show result ───────────────────────────────────────────────────────
    print("\nOpening PyVista viewer...")
    opt.show_result(threshold=threshold)

    # ── Export ────────────────────────────────────────────────────────────
    formats = ['stl', 'obj']
    if include_step:
        formats.append('step')

    print(f"\nExporting at threshold={threshold}, smooth={smooth_iter} iters...")
    outputs = opt.export_cad(
        basename=basename,
        threshold=threshold,
        smooth_iter=smooth_iter,
        formats=tuple(formats)
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("EXPORT COMPLETE")
    print("=" * 55)
    for fmt, path in outputs.items():
        import os
        size_kb = os.path.getsize(path) / 1024
        print(f"  {fmt.upper():5s}: {path}  ({size_kb:.0f} KB)")

    print("\nTo open in Inventor:")
    if 'obj' in outputs:
        print(f"  File → Open → {outputs['obj']}")
        print("  Select 'OBJ Files' in the file type dropdown")
    if 'stl' in outputs:
        print(f"  Or: File → Import → {outputs['stl']}")

    print("\nTo open in MeshLab / Blender / Fusion 360:")
    print(f"  Open {outputs.get('stl', outputs.get('obj'))}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description='Export topology results to CAD')
    parser.add_argument('--problem', default='cantilever',
                        choices=['cantilever', 'edgeload', 'mbb'],
                        help='Which problem to run and export')
    parser.add_argument('--fine',      action='store_true',
                        help='Use finer mesh (better surface quality, slower)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Density isosurface threshold (default: 0.4)')
    parser.add_argument('--smooth',    type=int,   default=50,
                        help='Smoothing iterations (default: 50)')
    parser.add_argument('--step',      action='store_true',
                        help='Also export STEP (requires pythonOCC)')
    args = parser.parse_args()

    run_and_export(
        problem_name=args.problem,
        fine=args.fine,
        threshold=args.threshold,
        smooth_iter=args.smooth,
        include_step=args.step
    )


if __name__ == "__main__":
    main()
