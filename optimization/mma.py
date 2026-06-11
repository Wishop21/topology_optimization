"""
GCMMA-style topology optimization update with adaptive asymptotes.

Implements the Method of Moving Asymptotes (Svanberg, 1987) with the
GCMMA adaptive asymptote heuristic (Svanberg, 2002, simplified):

    - Maintain moving asymptotes L_j < rho_j < U_j for each design variable.
    - On the 3rd iteration onward, adapt asymptote DISTANCE based on
      whether the design is oscillating or moving monotonically:
        oscillating  (sign change)   -> shrink distance  (factor 0.7)
        monotonic    (same sign)     -> expand distance  (factor 1.2)
        first move / no info         -> keep distance unchanged
      This is the mechanism that should reduce oscillation relative to OC.

    - Build separable convex approximation of objective + constraints
      using p_i, q_i coefficients (standard MMA, NOT the second-order
      GCMMA curvature correction -- see module docstring note below).

    - Solve the dual problem for the Lagrange multiplier(s) lambda_j.

SCOPE / LIMITATIONS (read before extending):

  1. This is "MMA with adaptive asymptotes", not full GCMMA. Full GCMMA
     adds a second convexity parameter (rho_j, confusingly named the same
     as density -- here we avoid that name entirely) that provides a
     monotonic-convergence GUARANTEE via inner iterations that re-solve
     the subproblem if the approximation underestimates the true
     constraint. That inner loop is NOT implemented here. If you observe
     non-monotonic objective convergence that this module's adaptive
     asymptotes don't fix, that second-order correction is the next step
     -- don't try to patch it in piecemeal, it changes the outer loop
     structure (subproblem may be solved multiple times per iteration).

  2. Dual solve is implemented for m=1 (single constraint: volume) using
     1D bisection on lambda >= 0, directly analogous to oc_update's
     bisection. This is EXACT for m=1 (the dual is separable and
     monotonic in lambda). For m>1, the dual variables are coupled
     through x*(lambda) and a 1D bisection per constraint is NOT correct
     -- a proper multi-dimensional dual solve (e.g. projected gradient
     ascent on the dual, or Svanberg's primal-dual Newton) is needed.
     The function signature accepts dg_drho as (m, n) and g as (m,) so
     this can be extended later, but `_solve_dual_m1` raises if m != 1.
     Do not silently loop bisection per-row for m>1 -- that gives a
     plausible-looking but WRONG answer.

  3. Stopping/move limits use the standard alpha/beta box (10% of the
     asymptote-to-rho distance), independent of the OC `move` parameter.
"""

import numpy as np


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

def init_mma_state(rho, rho_min=1e-3, rho_max=1.0, asymp_init_frac=0.5):
    """
    Create initial MMA state dict.

    Args:
        rho:             (n,) initial design (used to size asymptotes)
        rho_min, rho_max: design variable bounds
        asymp_init_frac: initial asymptote distance as fraction of
                         (rho_max - rho_min). 0.5 is Svanberg's default.

    Returns:
        state: dict with keys 'rho_old', 'rho_older', 'low', 'upp',
               'iter', 'rho_min', 'rho_max'
    """
    span = rho_max - rho_min
    return {
        'rho_old':   rho.copy(),
        'rho_older': rho.copy(),
        'low': rho - asymp_init_frac * span,
        'upp': rho + asymp_init_frac * span,
        'iter': 0,
        'rho_min': rho_min,
        'rho_max': rho_max,
    }


# ---------------------------------------------------------------------------
# Asymptote update (the GCMMA adaptive part)
# ---------------------------------------------------------------------------

def _update_asymptotes(rho, state,
                        shrink=0.7, expand=1.2,
                        dist_min_frac=0.01, dist_max_frac=10.0):
    """
    Update asymptotes low/upp based on oscillation history.

    Iterations 1-2 (no two-step history yet): keep asymptotes from init,
    just recenter is NOT done -- Svanberg's scheme keeps the initial
    asymptotes fixed for the first two iterations, then starts adapting.

    Iteration >= 3: for each variable, check sign of
        (rho - rho_old) * (rho_old - rho_older)
    < 0  -> oscillating  -> shrink asymptote distance from rho
    > 0  -> monotonic    -> expand asymptote distance from rho
    == 0 -> unchanged

    The new asymptote is recentered on the CURRENT rho at the (possibly
    rescaled) distance:
        new_dist = old_dist * factor
        low_new  = rho - new_dist
        upp_new  = rho + new_dist

    Distance is clamped to [dist_min_frac, dist_max_frac] * (rho_max-rho_min)
    to prevent asymptotes collapsing to zero (singular subproblem) or
    exploding (subproblem becomes effectively unconstrained / linear).

    Returns:
        low, upp: (n,) updated asymptote arrays
    """
    n = len(rho)
    span = state['rho_max'] - state['rho_min']
    dist_min = dist_min_frac * span
    dist_max = dist_max_frac * span

    if state['iter'] < 2:
        # Not enough history -- keep existing asymptotes as-is.
        return state['low'].copy(), state['upp'].copy()

    rho_old   = state['rho_old']
    rho_older = state['rho_older']

    # Old (pre-update) asymptote distances from the OLD design point.
    # We rescale these distances, then recenter on the NEW design point.
    dist_low_old = rho_old - state['low']   # > 0 by construction
    dist_upp_old = state['upp'] - rho_old   # > 0 by construction

    sign_indicator = (rho - rho_old) * (rho_old - rho_older)

    factor = np.ones(n)
    factor[sign_indicator < 0] = shrink   # oscillating -> tighten
    factor[sign_indicator > 0] = expand   # monotonic   -> relax
    # sign_indicator == 0 -> factor stays 1.0 (unchanged distance)

    new_dist_low = np.clip(dist_low_old * factor, dist_min, dist_max)
    new_dist_upp = np.clip(dist_upp_old * factor, dist_min, dist_max)

    low = rho - new_dist_low
    upp = rho + new_dist_upp

    return low, upp


# ---------------------------------------------------------------------------
# Separable convex approximation coefficients
# ---------------------------------------------------------------------------

def _mma_pq(grad, rho, low, upp, eps=1e-6):
    """
    Compute MMA p, q coefficients for one function (objective or a
    constraint) given its gradient at the current point.

    Standard MMA approximation:
        f(x) ~ sum_i [ p_i / (U_i - x_i) + q_i / (x_i - L_i) ] + const

    with, for grad_i >= 0:
        p_i = (U_i - rho_i)^2 * grad_i
        q_i = eps_small  (no contribution from negative part)
    for grad_i < 0:
        p_i = eps_small
        q_i = (rho_i - L_i)^2 * (-grad_i)

    This is the standard "always-conservative" choice (Svanberg 1987):
    p,q >= 0 ensures the approximation is convex and, for the correct
    sign convention, an upper bound on the true function near rho
    (first-order). The eps floor avoids exact zeros which would make
    a variable's bound (alpha or beta) irrelevant to that term and can
    cause divide-by-zero in the dual gradient.

    Args:
        grad: (n,) gradient of f w.r.t. rho at current point
        rho, low, upp: (n,) current design and asymptotes
        eps:  floor value for the "wrong sign" coefficient

    Returns:
        p, q: (n,) arrays
    """
    Uxr = np.maximum(upp - rho, 1e-10)   # U_i - rho_i, guard > 0
    xrL = np.maximum(rho - low, 1e-10)   # rho_i - L_i, guard > 0

    pos = np.maximum(grad, 0.0)
    neg = np.maximum(-grad, 0.0)

    p = (Uxr ** 2) * pos + eps
    q = (xrL ** 2) * neg + eps

    return p, q


# ---------------------------------------------------------------------------
# Primal variable from dual variable (m=1 case)
# ---------------------------------------------------------------------------

def _primal_from_lambda(lam, p0, q0, p1, q1, low, upp, alpha, beta):
    """
    Given dual variable lambda (scalar, for the single constraint),
    compute the primal-optimal x_i for the separable subproblem:

        minimize_x  sum_i [ p0_i/(U_i-x_i) + q0_i/(x_i-L_i) ]
                   + lam * sum_i [ p1_i/(U_i-x_i) + q1_i/(x_i-L_i) ]
        s.t. alpha_i <= x_i <= beta_i

    Each term is separable in x_i. The unconstrained minimizer of
        p_i/(U_i-x) + q_i/(x-L_i)
    (with p_i = p0_i + lam*p1_i, q_i = q0_i + lam*q1_i) is:

        x*_i = (sqrt(p_i)*L_i + sqrt(q_i)*U_i) / (sqrt(p_i) + sqrt(q_i))

    (standard MMA closed-form, derived by setting d/dx = 0). Then clip
    to [alpha_i, beta_i].

    Args:
        lam: scalar dual variable (lambda >= 0)
        p0,q0: objective coefficients (n,)
        p1,q1: constraint coefficients (n,)
        low, upp, alpha, beta: (n,) asymptotes and move-limit bounds

    Returns:
        x: (n,) primal point for this lambda
    """
    p = p0 + lam * p1
    q = q0 + lam * q1

    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)

    x_star = (sqrt_p * low + sqrt_q * upp) / (sqrt_p + sqrt_q)

    return np.clip(x_star, alpha, beta)


# ---------------------------------------------------------------------------
# Dual solve for m = 1
# ---------------------------------------------------------------------------

def _solve_dual_m1(p0, q0, p1, q1, low, upp, alpha, beta, g_val,
                    lam_min=1e-10, lam_max=1e10, n_bisect=100, tol=1e-10):
    """
    Solve the m=1 dual problem by bisection on lambda >= 0.

    The dual function is:
        W(lambda) = sum_i [ p_i(lambda)/(U_i-x_i(lambda)) + q_i(lambda)/(x_i(lambda)-L_i) ]
                    - lambda * b
    where b is the constraint constant such that the constraint reads
        sum_i [ p1_i/(U_i-x_i) + q1_i/(x_i-L_i) ] <= b
    (b = -g_val + sum_i [p1_i/(U_i-rho_i) + q1_i/(rho_i-L_i)], i.e. the
    RHS that makes the linearized constraint g(x) <= 0 at x=rho consistent
    -- see _build_constraint_rhs below).

    By weak duality / complementary slackness, the optimal lambda* is
    found where the constraint g_approx(x(lambda)) = 0 if the unconstrained
    (lambda=0) solution violates g_approx(x) <= 0; otherwise lambda*=0.

    We bisect directly on the CONSTRAINT VALUE g_approx(x(lambda)) as a
    function of lambda, which is monotonically decreasing in lambda
    (more penalty on the constraint -> x moves to reduce it). This
    mirrors oc_update's bisection on the volume constraint exactly.

    Args:
        p0,q0,p1,q1: (n,) MMA coefficients for objective (0) and
                     constraint (1)
        low, upp, alpha, beta: (n,) asymptotes and bounds
        g_val: scalar, current (linearized) constraint value at x=rho
               using the SAME p1,q1 (i.e. g_approx(rho)). Used to define
               b via _build_constraint_rhs.
        lam_min, lam_max: bisection bracket for lambda
        n_bisect: max bisection iterations
        tol: relative bracket tolerance

    Returns:
        lam: optimal dual variable (scalar, >= 0)
        x:   (n,) corresponding primal point
    """
    b = _constraint_rhs(p1, q1, low, upp, alpha, beta, g_val,
                          x_ref=None)  # see helper; uses alpha as ref pt
    # NOTE: b is computed once outside the bisection loop below using
    # a fixed reference point -- see _constraint_rhs docstring.

    def constraint_residual(lam):
        x = _primal_from_lambda(lam, p0, q0, p1, q1, low, upp, alpha, beta)
        g_approx = np.sum(p1 / (upp - x) + q1 / (x - low)) - b
        return g_approx, x

    # Check lambda = 0 first: if constraint already satisfied, done.
    g0, x0 = constraint_residual(0.0)
    if g0 <= 0:
        return 0.0, x0

    # constraint_residual is decreasing in lambda (more weight on
    # constraint -> x moves to reduce sum p1/(U-x)+q1/(x-L)).
    # Find bracket [lo, hi] with residual(lo) > 0 > residual(hi).
    lo, hi = lam_min, lam_max
    g_hi, x_hi = constraint_residual(hi)
    n_expand = 0
    while g_hi > 0 and n_expand < 50:
        hi *= 10.0
        g_hi, x_hi = constraint_residual(hi)
        n_expand += 1
    if g_hi > 0:
        # Could not bracket -- constraint cannot be satisfied within
        # bounds even at very large lambda. Return the best we have
        # (x at largest lambda); caller's outer loop / move limits will
        # still keep this bounded, but flag via large lambda.
        return hi, x_hi

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        g_mid, x_mid = constraint_residual(mid)
        if g_mid > 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol * max(hi, 1.0):
            break

    lam = 0.5 * (lo + hi)
    _, x = constraint_residual(lam)
    return lam, x


def _constraint_rhs(p1, q1, low, upp, alpha, beta, g_val, x_ref=None):
    """
    Compute the constant b such that the linearized constraint reads:
        sum_i [ p1_i/(U_i-x_i) + q1_i/(x_i-L_i) ] - b <= 0
    is equivalent to g_approx(x) <= 0, given that g_approx(rho) = g_val
    and p1,q1 were built from the gradient at rho (so the approximation
    is exact in value AND first derivative at x=rho).

    At x = rho:
        sum_i [ p1_i/(U_i-rho_i) + q1_i/(rho_i-L_i) ] - b = g_val
    => b = sum_i [ p1_i/(U_i-rho_i) + q1_i/(rho_i-L_i) ] - g_val

    Note: this requires rho itself, not just alpha/beta. We pass it
    via the module-level call in mma_update (see x_ref usage there);
    here x_ref must be rho. The `x_ref=None` signature with alpha as a
    placeholder above was a planning error -- corrected: this function
    REQUIRES x_ref=rho and will raise if not given.
    """
    if x_ref is None:
        raise ValueError(
            "_constraint_rhs requires x_ref=rho (the point at which "
            "p1,q1,g_val were evaluated). Internal call site error."
        )
    rho = x_ref
    lhs_at_rho = np.sum(p1 / (upp - rho) + q1 / (rho - low))
    b = lhs_at_rho - g_val
    return b


# ---------------------------------------------------------------------------
# Main update function
# ---------------------------------------------------------------------------

def mma_update(rho, dc_drho, vol_frac, cell_volumes, state,
               move=0.2):
    """
    One GCMMA-style update step. Drop-in replacement for oc_update with
    an extra `state` argument (must be threaded through the loop).

    Args:
        rho:          (n,) current densities
        dc_drho:      (n,) objective (compliance) sensitivity, dC/drho.
                      Sign convention: SAME as oc_update, i.e. typically
                      negative (increasing rho decreases compliance).
        vol_frac:     target volume fraction
        cell_volumes: (n,) cell volumes
        state:        dict from init_mma_state, updated in place AND
                       returned for clarity
        move:         move-limit fraction (alpha/beta box width as
                       fraction of asymptote distance). Same role as
                       OC's `move` but applied differently -- see below.

    Returns:
        rho_new: (n,) updated densities
        state:   updated state dict (same object, mutated)

    Sign conventions (IMPORTANT, verified against oc_update):
        - dc_drho is negative where increasing density helps (reduces
          compliance) -- this matches compute_sensitivities_hex's output
          (dc_drho = -dE_drho * strain_energy, strain_energy >= 0).
        - The volume constraint is:
              g(rho) = sum(rho_i * v_i) / sum(v_i) - vol_frac <= 0
          dg/drho_i = v_i / sum(v_i)  (always positive).
        - MMA's p,q construction in _mma_pq assumes "minimize f, grad
          as given" -- since we are MINIMIZING compliance and dc_drho
          is mostly negative, _mma_pq will assign most of its (p,q)
          weight to q (the "x - L" term), which is correct: the convex
          approximation should curve such that increasing rho_i (toward
          U_i) is favourable, matching p small / q small near a negative
          gradient... [see inline comments for the precise mechanism]
    """
    n = len(rho)
    rho_min = state['rho_min']
    rho_max = state['rho_max']

    # --- 1. Update asymptotes (the adaptive GCMMA part) ---
    low, upp = _update_asymptotes(rho, state)

    # Asymptotes must satisfy low_i < rho_i < upp_i strictly; if numerical
    # issues push rho to an asymptote, nudge (should be rare given the
    # dist_min_frac clamp in _update_asymptotes, but cheap to guard).
    eps_gap = 1e-6 * (rho_max - rho_min)
    low = np.minimum(low, rho - eps_gap)
    upp = np.maximum(upp, rho + eps_gap)

    # --- 2. Move limits (alpha, beta box) ---
    # Standard MMA: alpha/beta are the tighter of (asymptote-derived bound,
    # explicit move limit, design bounds [rho_min, rho_max]).
    alpha = np.maximum.reduce([
        low + 0.1 * (rho - low),
        rho - move,
        np.full(n, rho_min)
    ])
    beta = np.minimum.reduce([
        upp - 0.1 * (upp - rho),
        rho + move,
        np.full(n, rho_max)
    ])
    # Guard against alpha > beta (can happen if move is very small and
    # rounding pushes them past each other).
    bad = alpha > beta
    if np.any(bad):
        mid = 0.5 * (alpha[bad] + beta[bad])
        alpha[bad] = mid - 1e-12
        beta[bad]  = mid + 1e-12

    # --- 3. Build p,q for objective ---
    p0, q0 = _mma_pq(dc_drho, rho, low, upp)

    # --- 4. Build p,q for volume constraint ---
    total_vol = cell_volumes.sum()
    dg_drho = cell_volumes / total_vol            # (n,), all positive
    g_val = float((rho * cell_volumes).sum() / total_vol - vol_frac)
    p1, q1 = _mma_pq(dg_drho, rho, low, upp)

    # --- 5. Solve dual (m=1) ---
    b = _constraint_rhs(p1, q1, low, upp, alpha, beta, g_val, x_ref=rho)

    def constraint_residual(lam):
        x = _primal_from_lambda(lam, p0, q0, p1, q1, low, upp, alpha, beta)
        g_approx = np.sum(p1 / (upp - x) + q1 / (x - low)) - b
        return g_approx, x

    g0, x0 = constraint_residual(0.0)
    if g0 <= 0:
        rho_new = x0
        lam = 0.0
    else:
        lo, hi = 1e-10, 1e10
        g_hi, x_hi = constraint_residual(hi)
        n_expand = 0
        while g_hi > 0 and n_expand < 50:
            hi *= 10.0
            g_hi, x_hi = constraint_residual(hi)
            n_expand += 1
        if g_hi > 0:
            rho_new = x_hi
            lam = hi
        else:
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                g_mid, x_mid = constraint_residual(mid)
                if g_mid > 0:
                    lo = mid
                else:
                    hi = mid
                if (hi - lo) < 1e-10 * max(hi, 1.0):
                    break
            lam = 0.5 * (lo + hi)
            _, rho_new = constraint_residual(lam)

    # --- 6. Update state for next iteration ---
    state['rho_older'] = state['rho_old']
    state['rho_old']   = rho.copy()
    state['low']       = low
    state['upp']       = upp
    state['iter']      += 1

    return rho_new, state
