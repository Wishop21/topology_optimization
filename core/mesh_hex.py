"""
3D hexahedral mesh generation.

Uses scikit-fem's MeshHex.init_tensor() for structured hex meshes.
Each rectangular cell = exactly one hex element (1:1 mapping).
This is the key advantage over tet: elem_to_cell is trivial.

Quadrature order: always use intorder=2 (8 Gauss points per element).
For Q1 hex elements, this is the minimum exact order.
The default (intorder=None) uses 64 points — 8x wasteful.
"""

import numpy as np
import skfem


def create_hex_mesh(width, height, depth, nx, ny, nz):
    """
    Create a structured 3D hexahedral mesh.

    Domain: x in [0,width], y in [0,height], z in [0,depth]
    Exactly nx*ny*nz elements — one per cell.

    Args:
        width, height, depth: domain dimensions
        nx, ny, nz: number of cells (= elements) in each direction

    Returns:
        mesh: skfem.MeshHex object
    """
    xs = np.linspace(0.0, width,  nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    zs = np.linspace(0.0, depth,  nz + 1)
    return skfem.MeshHex.init_tensor(xs, ys, zs)


def create_hex_basis(mesh):
    """
    Create the vectorial basis for a hex mesh with correct quadrature.

    Always uses intorder=2 (8 Gauss points) — the minimum exact order
    for trilinear Q1 hex elements. The scikit-fem default uses 64 points,
    which is 8x more computation than needed.

    Args:
        mesh: skfem.MeshHex object

    Returns:
        basis: skfem.Basis with ElementVector(ElementHex1), intorder=2
    """
    return skfem.Basis(
        mesh,
        skfem.ElementVector(skfem.ElementHex1()),
        intorder=2   # 8 qp per element — exact for Q1, not 64 (the wasteful default)
    )


def identify_boundary_nodes_hex(mesh, tol=1e-10):
    """
    Identify nodes on each face of a hex box mesh.

    Returns:
        dict with keys: 'left', 'right', 'bottom', 'top', 'front', 'back'
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


def get_mesh_info_hex(mesh):
    """Return basic hex mesh statistics."""
    return {
        'n_elements': mesh.t.shape[1],
        'n_nodes':    mesh.p.shape[1],
        'x_range':    (float(mesh.p[0].min()), float(mesh.p[0].max())),
        'y_range':    (float(mesh.p[1].min()), float(mesh.p[1].max())),
        'z_range':    (float(mesh.p[2].min()), float(mesh.p[2].max())),
    }
