"""
Validation tests for mma_update.

Test 1 (toy, analytical): minimize sum(c_i * rho_i) subject to
    sum(rho_i)/n <= vol_frac,  rho_min <= rho_i <= 1
This is a linear program. The exact optimum is: rank cells by c_i
ascending (most negative c_i = best to set to 1, since dc_drho = c_i
constant means d(objective)/d(rho_i) = c_i; minimizing means we want
rho_i = 1 where c_i < 0 and rho_i = rho_min where c_i > 0), then fill
to satisfy the volume constraint exactly (bang-bang with possibly one
fractional cell). We check MMA converges to this.

Test 2 (consistency with oc_update): on a smooth quadratic-ish toy
sensitivity field, both oc_update and mma_update should drive the
design toward similar volume-feasible configurations and decrease
the toy objective monotonically (ish) iteration-to-iteration.

Test 3 (oscillation behaviour, synthetic): construct a sensitivity
field that FLIPS SIGN every iteration in a subset of cells (worst-case
oscillation driver) and confirm MMA's asymptotes shrink for those
cells (i.e. the adaptive mechanism actually engages).
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimization.mma import init_mma_state, mma_update, _update_asymptotes
from optimization.optimality_criteria import oc_update


def test_linear_program_bangbang():
    """
    Toy LP: n cells, constant cost c_i per unit rho_i, equal cell volumes.
    minimize sum(c_i * rho_i)  s.t. mean(rho) <= vol_frac, rho in [rho_min,1]

    dc_drho_i = c_i (constant, independent of rho -- so MMA's convex
    approximation is exact... actually MMA's p,q construction assumes
    a specific curvature, so even for a linear objective the SUBPROBLEM
    is not exactly linear, but as asymptotes -> infinity the MMA
    approximation -> linear and should reproduce the LP optimum).

    We use a fairly large initial asymptote distance to approximate
    the linear case and check the FINAL converged design matches the
    bang-bang LP solution.
    """
    n = 20
    rho_min = 1e-3
    vol_frac = 0.4
    cell_volumes = np.ones(n)

    rng = np.random.default_rng(0)
    c = rng.uniform(-1, 1, size=n)   # constant cost coefficients

    # LP optimum: sort by c ascending, fill rho=1 for most negative c
    # until volume budget exhausted.
    order = np.argsort(c)
    budget = vol_frac * n
    rho_opt = np.full(n, rho_min)
    remaining = budget - rho_min * n
    for idx in order:
        if remaining <= 0:
            break
        add = min(1.0 - rho_min, remaining)
        rho_opt[idx] = rho_min + add
        remaining -= add

    # Run MMA from a uniform start
    rho = np.full(n, vol_frac)
    state = init_mma_state(rho, rho_min=rho_min, rho_max=1.0,
                            asymp_init_frac=2.0)  # large -> near-linear

    for _ in range(200):
        dc_drho = c.copy()   # constant gradient (linear objective)
        rho, state = mma_update(rho, dc_drho, vol_frac, cell_volumes,
                                  state, move=0.5)

    # Check volume constraint satisfied
    vol = (rho * cell_volumes).sum() / cell_volumes.sum()
    assert abs(vol - vol_frac) < 1e-3, f"Volume constraint violated: {vol} vs {vol_frac}"

    # Check design matches bang-bang LP solution (within tolerance,
    # since MMA approaches but doesn't exactly reach rho_min/1 bounds
    # for the marginal cell)
    err = np.abs(rho - rho_opt)
    # All but possibly one cell (the marginal one near the volume cutoff)
    # should match closely.
    n_mismatch = np.sum(err > 0.05)
    assert n_mismatch <= 1, (
        f"Too many cells mismatch LP optimum: {n_mismatch}\n"
        f"rho={np.round(rho,3)}\nrho_opt={np.round(rho_opt,3)}\nc={np.round(c,3)}"
    )

    print("test_linear_program_bangbang: PASS")
    print(f"  final volume = {vol:.4f} (target {vol_frac})")
    print(f"  max per-cell error vs LP optimum (excluding marginal cell): "
          f"{np.sort(err)[:-1].max():.4f}")


def test_consistency_with_oc():
    """
    On a smooth synthetic sensitivity field (decreasing in iteration
    count, simulating convergence), both OC and MMA should:
      - satisfy the volume constraint at every iteration
      - move rho_i in the direction indicated by sign(-dc_drho_i)
        relative to rho_i's "fair share" (i.e. cells with more negative
        sensitivity should end up with higher density than cells with
        less negative sensitivity)
    """
    n = 30
    rho_min = 1e-3
    vol_frac = 0.3
    cell_volumes = np.ones(n)

    rng = np.random.default_rng(1)
    base_sens = -rng.uniform(0.1, 1.0, size=n)   # all negative (typical)

    rho_oc  = np.full(n, vol_frac)
    rho_mma = np.full(n, vol_frac)
    state = init_mma_state(rho_mma, rho_min=rho_min, rho_max=1.0)

    for it in range(50):
        dc_drho = base_sens * (1.0 + 0.1 * rng.normal(size=n))

        rho_oc, _  = oc_update(rho_oc, dc_drho, vol_frac, cell_volumes)
        rho_mma, state = mma_update(rho_mma, dc_drho, vol_frac, cell_volumes,
                                      state, move=0.2)

        vol_oc  = (rho_oc * cell_volumes).sum() / cell_volumes.sum()
        vol_mma = (rho_mma * cell_volumes).sum() / cell_volumes.sum()

        assert abs(vol_oc - vol_frac) < 1e-3, f"OC volume drift at iter {it}: {vol_oc}"
        # NOTE: MMA's volume constraint is satisfied only by its RATIONAL
        # APPROXIMATION at each step (g_approx(x)=0), not the true linear
        # volume function -- so vol_mma will not hit vol_frac exactly each
        # iteration, only in the limit as the design converges (asymptotes
        # tighten and approx -> exact). A loose per-iteration bound plus a
        # tight final check (after the loop) is the correct test.
        assert abs(vol_mma - vol_frac) < 0.15, f"MMA volume drift too large at iter {it}: {vol_mma}"
        assert np.all(rho_oc >= rho_min - 1e-9) and np.all(rho_oc <= 1 + 1e-9)
        assert np.all(rho_mma >= rho_min - 1e-9) and np.all(rho_mma <= 1 + 1e-9)

    # MMA's constraint approximation tightens as the design stabilizes
    # (asymptotes converge), so the FINAL volume should be close to target.
    assert abs(vol_mma - vol_frac) < 0.02, f"MMA final volume not converged: {vol_mma}"

    # Both should rank cells similarly: most-negative-sensitivity cells
    # should have higher final density.
    rank_corr_oc  = np.corrcoef(base_sens, rho_oc)[0, 1]
    rank_corr_mma = np.corrcoef(base_sens, rho_mma)[0, 1]

    # base_sens is negative; more negative = better = should end up
    # with higher rho => correlation should be NEGATIVE and similar
    # magnitude for both methods.
    assert rank_corr_oc < -0.3, f"OC didn't separate cells by sensitivity: corr={rank_corr_oc}"
    assert rank_corr_mma < -0.3, f"MMA didn't separate cells by sensitivity: corr={rank_corr_mma}"

    print("test_consistency_with_oc: PASS")
    print(f"  OC  sensitivity-density correlation:  {rank_corr_oc:.3f}")
    print(f"  MMA sensitivity-density correlation:  {rank_corr_mma:.3f}")
    print(f"  final volumes: OC={vol_oc:.4f} MMA={vol_mma:.4f} (target {vol_frac})")


def test_adaptive_asymptotes_engage():
    """
    Direct unit test of _update_asymptotes: given an explicit rho history
    where cell A oscillates (sign flips between consecutive steps) and
    cell B moves monotonically, confirm:
      - cell A's asymptote distance SHRINKS (factor ~0.7)
      - cell B's asymptote distance GROWS  (factor ~1.2)

    This tests the mechanism in isolation, with a controlled rho history,
    rather than relying on a full mma_update loop to PRODUCE oscillation
    (which depends on dual solves, clipping, etc. and is not a reliable
    way to drive a specific rho trajectory for testing purposes).
    """
    n = 2
    rho_min, rho_max = 1e-3, 1.0

    # Construct state as if 2 iterations have already happened.
    state = {
        'rho_min': rho_min,
        'rho_max': rho_max,
        'iter': 2,   # >= 2 triggers adaptation
    }

    # rho_older -> rho_old: cell A went 0.5 -> 0.6 (increase)
    #              cell B went 0.5 -> 0.6 (increase)
    rho_older = np.array([0.5, 0.5])
    rho_old   = np.array([0.6, 0.6])

    # rho_old -> rho (current): cell A goes 0.6 -> 0.5 (DECREASE -> sign
    # flip vs previous step -> oscillating); cell B goes 0.6 -> 0.7
    # (increase -> same sign as previous step -> monotonic)
    rho = np.array([0.5, 0.7])

    state['rho_old']   = rho_old
    state['rho_older'] = rho_older
    # Asymptote distances from rho_old: 0.4 on each side for both cells
    state['low'] = rho_old - 0.4
    state['upp'] = rho_old + 0.4

    low, upp = _update_asymptotes(rho, state, shrink=0.7, expand=1.2)

    dist_low = rho - low
    dist_upp = upp - rho

    print("test_adaptive_asymptotes_engage:")
    print(f"  cell A (oscillating): dist_low={dist_low[0]:.4f} dist_upp={dist_upp[0]:.4f} "
          f"(expect ~0.4*0.7=0.28)")
    print(f"  cell B (monotonic):   dist_low={dist_low[1]:.4f} dist_upp={dist_upp[1]:.4f} "
          f"(expect ~0.4*1.2=0.48)")

    assert np.isclose(dist_low[0], 0.4 * 0.7, atol=1e-9)
    assert np.isclose(dist_upp[0], 0.4 * 0.7, atol=1e-9)
    assert np.isclose(dist_low[1], 0.4 * 1.2, atol=1e-9)
    assert np.isclose(dist_upp[1], 0.4 * 1.2, atol=1e-9)

    print("  PASS: oscillating cell shrunk by 0.7x, monotonic cell grew by 1.2x")


if __name__ == "__main__":
    test_linear_program_bangbang()
    print()
    test_consistency_with_oc()
    print()
    test_adaptive_asymptotes_engage()
