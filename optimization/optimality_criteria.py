"""
Optimality Criteria (OC) density update for SIMP topology optimization.

The OC method is a heuristic update rule derived from the KKT conditions
of the optimization problem:

    minimize    c(rho) = u^T K u         (compliance = 1/stiffness)
    subject to  sum(rho_i * v_i) = V*    (volume constraint)
                0 < rho_min <= rho <= 1  (density bounds)

The KKT condition at optimum requires:
    -dc/drho_i = lambda * dV/drho_i  for active cells (0 < rho < 1)

where lambda is the Lagrange multiplier for the volume constraint.

This motivates the update rule:
    rho_new = rho * (B_i)^eta

where:
    B_i  = -dc/drho_i / (lambda * v_i)   (optimality indicator)
    eta  = damping exponent (0.5 is standard)
    lambda is found by bisection to satisfy the volume constraint

With move limits to prevent oscillation:
    rho_new = clip(B_scaled, rho - move, rho + move)
    rho_new = clip(rho_new, rho_min, 1.0)
"""

import numpy as np


def oc_update(rho, dc_drho, vol_frac, cell_volumes,
              move=0.2, eta=0.5, rho_min=1e-3):
    """
    Perform one OC density update step.

    Args:
        rho:          (n_cells,) current densities
        dc_drho:      (n_cells,) compliance sensitivities (negative values)
        vol_frac:     target volume fraction (e.g. 0.5)
        cell_volumes: (n_cells,) volume of each cell (for volume constraint)
        move:         maximum density change per iteration (move limit)
        eta:          OC damping exponent (0.5 is standard)
        rho_min:      minimum density to keep K non-singular

    Returns:
        rho_new: (n_cells,) updated densities
        lambda_: Lagrange multiplier found by bisection
    """
    V_target = vol_frac * cell_volumes.sum()

    # Bisection on lambda to find the multiplier that satisfies V = V_target
    # B_i = (-dc/drho_i) / (lambda * v_i)
    # rho_new_i = clip(rho * B^eta, move limits, [rho_min, 1])
    # Constraint: sum(rho_new_i * v_i) = V_target

    l1, l2 = 1e-9, 1e9

    for _ in range(200):   # bisection converges in ~60 iterations to 1e-9 tolerance
        lmid = 0.5 * (l1 + l2)
        B = (-dc_drho) / (lmid * cell_volumes)

        # Apply update with move limits and density bounds
        rho_candidate = np.maximum(
            rho_min,
            np.maximum(
                rho - move,
                np.minimum(
                    1.0,
                    np.minimum(
                        rho + move,
                        rho * B**eta
                    )
                )
            )
        )

        # Check volume constraint
        V_current = rho_candidate @ cell_volumes
        if V_current > V_target:
            l1 = lmid
        else:
            l2 = lmid

        if l2 - l1 < 1e-9 * lmid:
            break

    return rho_candidate, lmid


def check_convergence(rho_new, rho_old, tol=0.01):
    """
    Check if density field has converged.

    Uses the maximum change in density across all cells.
    Standard tolerance for SIMP is 0.01 (1% max change).

    Args:
        rho_new: updated densities
        rho_old: previous densities
        tol:     convergence tolerance

    Returns:
        converged: bool
        change:    max absolute density change (for logging)
    """
    change = np.max(np.abs(rho_new - rho_old))
    return change < tol, change
