"""
Mesh generation and loading utilities.

Uses scikit-fem's MeshTri.init_tensor() for structured rectangular meshes.
For complex geometries, load external .msh files via meshio.
"""

import numpy as np
import skfem


def create_rectangle_mesh(width, height, nx, ny):
    """
    Create a structured 2D triangular mesh over a rectangle.

    Each rectangular cell is split into 2 triangles, giving
    2*nx*ny elements total.

    Args:
        width: Width of the rectangle (x-direction)
        height: Height of the rectangle (y-direction)
        nx: Number of element columns (x-direction)
        ny: Number of element rows (y-direction)

    Returns:
        mesh: skfem.MeshTri object

    Example:
        mesh = create_rectangle_mesh(10.0, 2.0, 50, 10)
        # Creates 1000 elements over a 10m x 2m domain
    """
    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    mesh = skfem.MeshTri.init_tensor(xs, ys)
    return mesh


def identify_boundary_nodes(mesh, tol=1e-10):
    """
    Identify nodes on each edge of a rectangular mesh.

    Args:
        mesh: skfem.MeshTri object
        tol: Coordinate tolerance for boundary detection

    Returns:
        dict with keys 'left', 'right', 'bottom', 'top'
        Values are numpy arrays of node indices.
    """
    x = mesh.p[0, :]
    y = mesh.p[1, :]

    boundaries = {
        'left':   np.where(np.abs(x - x.min()) < tol)[0],
        'right':  np.where(np.abs(x - x.max()) < tol)[0],
        'bottom': np.where(np.abs(y - y.min()) < tol)[0],
        'top':    np.where(np.abs(y - y.max()) < tol)[0],
    }

    return boundaries


def get_mesh_info(mesh):
    """
    Return basic mesh statistics as a dict.
    """
    return {
        'n_elements': mesh.t.shape[1],
        'n_nodes':    mesh.p.shape[1],
        'x_range':    (float(mesh.p[0].min()), float(mesh.p[0].max())),
        'y_range':    (float(mesh.p[1].min()), float(mesh.p[1].max())),
    }


if __name__ == "__main__":
    mesh = create_rectangle_mesh(10.0, 2.0, 50, 10)
    info = get_mesh_info(mesh)
    print("Mesh info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    bounds = identify_boundary_nodes(mesh)
    print("Boundary nodes:")
    for edge, nodes in bounds.items():
        print(f"  {edge}: {len(nodes)} nodes")
