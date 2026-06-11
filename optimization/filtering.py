"""
Density filtering for topology optimization.

Without filtering, SIMP optimization tends to produce checkerboard patterns:
alternating solid/void elements in a checkerboard. This happens because
the FEM basis functions make a checkerboard artificially stiffer than
it physically should be.

The density filter replaces each cell density with a weighted average
of its neighbours:

    rho_filtered[i] = sum_j(w_ij * rho[j]) / sum_j(w_ij)

where w_ij = max(0, r_min - dist(i, j)) is a linear hat function
with radius r_min. Cells further than r_min apart have zero weight.

This is implemented as a sparse matrix multiply: rho_f = H @ rho / Hs
where H[i,j] = w_ij and Hs[i] = sum_j(H[i,j]).

The filter radius r_min controls minimum feature size:
    r_min = 1.5 * cell_size  →  no features smaller than ~1.5 cells
    r_min = 3.0 * cell_size  →  smoother but more blurred result
"""

import numpy as np
from scipy.sparse import coo_matrix


def build_filter_matrix(nx, ny, r_min):
    """
    Build the sparse filter weight matrix H and row sums Hs.

    Args:
        nx: number of cells in x-direction
        ny: number of cells in y-direction
        r_min: filter radius in cell units (e.g. 1.5 means 1.5 cell widths)

    Returns:
        H:  scipy sparse matrix (n_cells, n_cells), unnormalized weights
        Hs: (n_cells,) array of row sums for normalization

    Usage:
        rho_filtered = (H @ rho) / Hs
    """
    n_cells = nx * ny

    rows = []
    cols = []
    vals = []

    for iy in range(ny):
        for ix in range(nx):
            i = iy * nx + ix   # flat index of center cell

            # Only visit cells within the bounding box of the filter radius
            ix_lo = max(0,  int(ix - np.ceil(r_min)))
            ix_hi = min(nx, int(ix + np.ceil(r_min)) + 1)
            iy_lo = max(0,  int(iy - np.ceil(r_min)))
            iy_hi = min(ny, int(iy + np.ceil(r_min)) + 1)

            for jy in range(iy_lo, iy_hi):
                for jx in range(ix_lo, ix_hi):
                    dist = np.sqrt((ix - jx)**2 + (iy - jy)**2)
                    w = r_min - dist
                    if w > 0:
                        j = jy * nx + jx
                        rows.append(i)
                        cols.append(j)
                        vals.append(w)

    H  = coo_matrix((vals, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    Hs = np.array(H.sum(axis=1)).ravel()   # row sums

    return H, Hs


def apply_filter(H, Hs, rho):
    """
    Apply density filter to cell densities.

    Args:
        H:   filter matrix from build_filter_matrix
        Hs:  row sums from build_filter_matrix
        rho: (n_cells,) raw cell densities

    Returns:
        rho_filtered: (n_cells,) smoothed densities
    """
    return (H @ rho) / Hs


def apply_filter_sensitivity(H, Hs, dc_drho):
    """
    Apply filter to sensitivities (chain rule for filtered density).

    When we filter rho before computing compliance, the sensitivity of
    compliance with respect to the *unfiltered* rho requires a chain rule:

        dc/drho_raw = H^T @ (dc/drho_filtered / Hs)

    This is the transpose of the density filter operation.

    Args:
        H:        filter matrix
        Hs:       row sums
        dc_drho:  (n_cells,) sensitivity w.r.t. filtered densities

    Returns:
        dc_drho_raw: (n_cells,) sensitivity w.r.t. raw (unfiltered) densities
    """
    return (H.T @ (dc_drho / Hs))
