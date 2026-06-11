"""
3D mesh generation and boundary identification.

Uses scikit-fem's MeshTet.init_tensor() for structured tetrahedral meshes.
Each rectangular cell is split into 6 tetrahedra by scikit-fem.
"""

import numpy as np
import skfem


def create_box_mesh(width, height, depth, nx, ny, nz):
    """
    Create a structured 3D tetrahedral mesh over a box domain.

    Domain: x in [0, width], y in [0, height], z in [0, depth]
    Each rectangular cell is split into 6 tetrahedra by scikit-fem,
    giving 6*nx*ny*nz elements total.

    Args:
        width:  x-extent of domain
        height: y-extent of domain
        depth:  z-extent of domain
        nx, ny, nz: number of cells in each direction

    Returns:
        mesh: skfem.MeshTet object
    """
    xs = np.linspace(0.0, width,  nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    zs = np.linspace(0.0, depth,  nz + 1)
    return skfem.MeshTet.init_tensor(xs, ys, zs)


def identify_boundary_nodes_3d(mesh, tol=1e-10):
    """
    Identify nodes on each face of a box mesh.

    Args:
        mesh: skfem.MeshTet object
        tol:  coordinate tolerance

    Returns:
        dict with keys: 'left', 'right', 'bottom', 'top', 'front', 'back'
        Values are numpy arrays of node indices.

    Convention:
        left/right  = x = min/max
        bottom/top  = y = min/max
        front/back  = z = min/max
    """
    x, y, z = mesh.p[0], mesh.p[1], mesh.p[2]
    return {
        'left':   np.where(np.abs(x - x.min()) < tol)[0],
        'right':  np.where(np.abs(x - x.max()) < tol)[0],
        'bottom': np.where(np.abs(y - y.min()) < tol)[0],
        'top':    np.where(np.abs(y - y.max()) < tol)[0],
        'front':  np.where(np.abs(z - z.min()) < tol)[0],
        'back':   np.where(np.abs(z - z.max()) < tol)[0],
    }


def get_mesh_info_3d(mesh):
    """Return basic statistics about a 3D mesh."""
    return {
        'n_elements': mesh.t.shape[1],
        'n_nodes':    mesh.p.shape[1],
        'x_range':    (float(mesh.p[0].min()), float(mesh.p[0].max())),
        'y_range':    (float(mesh.p[1].min()), float(mesh.p[1].max())),
        'z_range':    (float(mesh.p[2].min()), float(mesh.p[2].max())),
    }


def build_elem_to_cell_3d(mesh, nx, ny, nz):
    """
    Map each tet element to its parent rectangular cell.

    Uses element centroids: floor(cx/dx) gives ix, etc.
    Each rectangular cell contains 6 tetrahedra.

    Args:
        mesh: skfem.MeshTet from init_tensor
        nx, ny, nz: number of cells in each direction

    Returns:
        elem_to_cell: int array (n_elements,)  flat cell index per element
    """
    # Centroid of each element
    cx = mesh.p[0, mesh.t].mean(axis=0)
    cy = mesh.p[1, mesh.t].mean(axis=0)
    cz = mesh.p[2, mesh.t].mean(axis=0)

    dx = mesh.p[0].max() / nx
    dy = mesh.p[1].max() / ny
    dz = mesh.p[2].max() / nz

    ix = np.floor(cx / dx).astype(int).clip(0, nx - 1)
    iy = np.floor(cy / dy).astype(int).clip(0, ny - 1)
    iz = np.floor(cz / dz).astype(int).clip(0, nz - 1)

    # Row-major flat index: iz * (nx*ny) + iy * nx + ix
    return iz * (nx * ny) + iy * nx + ix
