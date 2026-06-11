"""
3D density filter for topology optimization.

Direct extension of the 2D filter to 3D.
Weight function: w_ij = max(0, r_min - dist(i,j))
where dist is Euclidean distance between cell centres in cell-index units.

The filter radius r_min is in cell units — so r_min=1.5 means averaging
over all cells within 1.5 cell-widths in each direction.
"""

import numpy as np
from scipy.sparse import coo_matrix


def build_filter_matrix_3d(nx, ny, nz, r_min):
    """
    Build sparse filter weight matrix H and row sums Hs for 3D.

    Cell flat index: iz*(nx*ny) + iy*nx + ix  (row-major)

    Args:
        nx, ny, nz: number of cells in each direction
        r_min:      filter radius in cell units

    Returns:
        H:  scipy sparse (n_cells, n_cells)
        Hs: (n_cells,) row sums for normalisation
    """
    n_cells = nx * ny * nz
    r_ceil  = int(np.ceil(r_min))

    rows, cols, vals = [], [], []

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                i = iz * (nx * ny) + iy * nx + ix

                for jz in range(max(0, iz-r_ceil), min(nz, iz+r_ceil+1)):
                    for jy in range(max(0, iy-r_ceil), min(ny, iy+r_ceil+1)):
                        for jx in range(max(0, ix-r_ceil), min(nx, ix+r_ceil+1)):
                            dist = np.sqrt((ix-jx)**2 + (iy-jy)**2 + (iz-jz)**2)
                            w = r_min - dist
                            if w > 0:
                                j = jz * (nx * ny) + jy * nx + jx
                                rows.append(i)
                                cols.append(j)
                                vals.append(w)

    H  = coo_matrix((vals, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    Hs = np.array(H.sum(axis=1)).ravel()
    return H, Hs


def apply_filter_3d(H, Hs, rho):
    """Apply density filter: rho_filtered = (H @ rho) / Hs"""
    return (H @ rho) / Hs


def apply_filter_sensitivity_3d(H, Hs, dc_drho):
    """
    Chain rule for filtered density.
    dc/drho_raw = H^T @ (dc/drho_filtered / Hs)
    """
    return (H.T @ (dc_drho / Hs))
