"""
Compare OC vs MMA convergence on the two problems that showed
oscillation under OC: edge-load cantilever and torsion beam.

Runs both optimizers on both problems (4 runs total), each with
show_monitor=False (no blocking plot windows), then plots compliance
and max-density-change history side by side.

Usage:
    python compare_oc_mma.py

Notes:
    - Uses a SMALLER mesh (nx=20,ny=10,nz=10 ish) than your usual runs
      to keep total runtime reasonable for 4 full optimizations. If you
      want production-resolution comparisons, increase nx/ny/nz but
      expect ~4x longer per doubling of resolution (roughly -- depends
      on solver scaling).
    - max_iter is capped at 150 with tol=0.005 (tighter than the usual
      0.01) so that if one method oscillates and never converges, you
      still get a bounded, comparable run rather than waiting for
      max_iter=300 every time.
    - sens_avg=1 for MMA runs per the earlier recommendation (raw
      sensitivity signal, no smoothing that could mask oscillation).
      sens_avg=3 (default) for OC runs, matching your existing usage.
"""

import numpy as np
import matplotlib.pyplot as plt

from topology_optimizer_hex import TopologyOptimizerHex
from problems.problems_hex import CantileverHexEdgeLoad, TorsionBeamHex


def run_case(problem_factory, optimizer, sens_avg, label):
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    problem = problem_factory()
    opt = TopologyOptimizerHex(problem)
    opt.run(max_iter=150, tol=0.005, sens_avg=sens_avg,
            optimizer=optimizer, verbose=True, show_monitor=False)
    return opt.history


def main():
    # Smaller meshes for fast back-to-back comparison runs.
    edge_load_factory = lambda: CantileverHexEdgeLoad(nx=20, ny=10, nz=10)
    torsion_factory    = lambda: TorsionBeamHex(nx=20, ny=10, nz=10)

    results = {}
    results['edge_oc']  = run_case(edge_load_factory, 'oc',  3, "Edge-load OC")
    results['edge_mma'] = run_case(edge_load_factory, 'mma', 1, "Edge-load MMA")
    results['tors_oc']  = run_case(torsion_factory,    'oc',  3, "Torsion OC")
    results['tors_mma'] = run_case(torsion_factory,    'mma', 1, "Torsion MMA")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Edge-load: compliance ---
    ax = axes[0, 0]
    ax.plot(results['edge_oc']['compliance'],  label='OC',  color='C0')
    ax.plot(results['edge_mma']['compliance'], label='MMA', color='C1')
    ax.set_title('Edge-Load: Compliance')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Compliance')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Edge-load: change ---
    ax = axes[0, 1]
    ax.plot(results['edge_oc']['change'],  label='OC',  color='C0')
    ax.plot(results['edge_mma']['change'], label='MMA', color='C1')
    ax.axhline(0.005, color='k', ls=':', lw=1, label='tol')
    ax.set_title('Edge-Load: Max Density Change')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Change')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Torsion: compliance ---
    ax = axes[1, 0]
    ax.plot(results['tors_oc']['compliance'],  label='OC',  color='C0')
    ax.plot(results['tors_mma']['compliance'], label='MMA', color='C1')
    ax.set_title('Torsion: Compliance')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Compliance')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Torsion: change ---
    ax = axes[1, 1]
    ax.plot(results['tors_oc']['change'],  label='OC',  color='C0')
    ax.plot(results['tors_mma']['change'], label='MMA', color='C1')
    ax.axhline(0.005, color='k', ls=':', lw=1, label='tol')
    ax.set_title('Torsion: Max Density Change')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Change')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('OC vs MMA Convergence Comparison', fontweight='bold')
    plt.tight_layout()
    plt.savefig('oc_vs_mma_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: oc_vs_mma_comparison.png")

    # Summary stats
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for key, hist in results.items():
        n_iter = len(hist['change'])
        final_compliance = hist['compliance'][-1]
        final_change = hist['change'][-1]
        # Crude oscillation metric: count sign changes in the
        # iteration-to-iteration delta of 'change' itself (i.e. is
        # the convergence metric oscillating up/down rather than
        # monotonically decreasing?)
        change_arr = np.array(hist['change'])
        deltas = np.diff(change_arr)
        sign_changes = np.sum(np.diff(np.sign(deltas)) != 0)
        print(f"{key:12s}: iters={n_iter:4d}  final_compliance={final_compliance:10.4f}  "
              f"final_change={final_change:.5f}  sign_changes_in_change={sign_changes}")

    plt.show()


if __name__ == "__main__":
    main()
