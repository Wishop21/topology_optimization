"""
CAD export for topology optimization results.

Pipeline:
    density field (rho_cells)
        ↓  threshold + marching cubes
        triangle mesh (vertices + faces)
        ↓  smooth (Laplacian)
        ↓  cleanup (remove small fragments)
        ↓  save
    STL  ← binary, universal, works immediately
    OBJ  ← Inventor 2019+ opens as mesh body
    STEP ← requires pythonOCC (conda install -c conda-forge pythonocc-core)

Notes on STEP:
    The STEP output wraps the triangle mesh in a STEP shell.
    Inventor will open it as a mesh body, not an editable solid.
    True parametric STEP requires manual reconstruction in CAD software.
    This is a fundamental limitation of SIMP topology output, not this code.
"""

import numpy as np
import struct
import os


def density_to_mesh(rho_cells, nx, ny, nz, width, height, depth,
                    threshold=0.5, smooth_iter=50, smooth_relax=0.1):
    """
    Convert density field to triangle mesh using marching cubes.

    Args:
        rho_cells:    (n_cells,) density field from optimizer
        nx, ny, nz:   mesh cell counts
        width, height, depth: physical domain dimensions (metres)
        threshold:    density isosurface value (default 0.5)
        smooth_iter:  Laplacian smoothing iterations (0 = no smoothing)
        smooth_relax: smoothing relaxation factor (0.1 = gentle)

    Returns:
        verts: (n_verts, 3) vertex coordinates in physical units
        faces: (n_faces, 3) triangle vertex indices
        mesh_pv: PyVista PolyData mesh (for further processing)
    """
    from skimage.measure import marching_cubes
    import pyvista as pv

    # Reshape density to 3D grid: (nx, ny, nz)
    # marching_cubes operates on the first 3 dimensions
    grid = rho_cells.reshape(nz, ny, nx).transpose(2, 1, 0)

    # Cell spacing maps voxel indices to physical coordinates
    spacing = (width / nx, height / ny, depth / nz)

    # Extract isosurface
    verts, faces, _, _ = marching_cubes(grid, level=threshold, spacing=spacing)

    if len(verts) == 0:
        raise ValueError(
            f"No surface found at threshold={threshold}. "
            f"Density range: [{rho_cells.min():.3f}, {rho_cells.max():.3f}]. "
            f"Try a lower threshold."
        )

    # Convert to PyVista for smoothing and cleanup
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh_pv = pv.PolyData(verts, faces_pv)

    # Remove small disconnected fragments (isolated bits of grey material)
    # Keep only the largest connected component
    bodies = mesh_pv.split_bodies()
    if len(bodies) > 1:
        sizes = [b.n_points for b in bodies]
        largest = bodies[int(np.argmax(sizes))]
        # split_bodies returns UnstructuredGrid; convert back to PolyData for smoothing
        mesh_pv = largest.extract_surface(algorithm='dataset_surface')
        print(f"  Removed {len(bodies)-1} small fragment(s), "
              f"kept largest ({mesh_pv.n_points} vertices)")

    # Laplacian smoothing to remove voxel staircase artifacts
    if smooth_iter > 0:
        mesh_pv = mesh_pv.smooth(
            n_iter=smooth_iter,
            relaxation_factor=smooth_relax,
            boundary_smoothing=False   # preserve sharp boundary edges
        )

    # Extract numpy arrays from smoothed mesh
    verts_out = np.array(mesh_pv.points)
    faces_raw = mesh_pv.faces.reshape(-1, 4)[:, 1:]   # strip leading '3'
    faces_out = np.array(faces_raw)

    return verts_out, faces_out, mesh_pv


def export_stl(verts, faces, filename, binary=True):
    """
    Export triangle mesh to STL file.

    No dependencies beyond numpy. Binary STL is smaller and faster to load.

    Args:
        verts:    (n_verts, 3) vertex coordinates
        faces:    (n_faces, 3) triangle indices
        filename: output path (e.g. 'result.stl')
        binary:   True = binary STL (recommended), False = ASCII STL

    Returns:
        filename: path to saved file
    """
    if binary:
        _write_stl_binary(filename, verts, faces)
    else:
        _write_stl_ascii(filename, verts, faces)

    size_kb = os.path.getsize(filename) / 1024
    print(f"STL saved: {filename}  ({len(faces)} triangles, {size_kb:.0f} KB)")
    return filename


def export_obj(verts, faces, filename):
    """
    Export triangle mesh to Wavefront OBJ format.

    Inventor 2019+ can import OBJ files as mesh bodies.
    Simpler than STEP, no pythonOCC required.

    Args:
        verts:    (n_verts, 3) vertex coordinates
        faces:    (n_faces, 3) triangle indices
        filename: output path (e.g. 'result.obj')

    Returns:
        filename: path to saved file
    """
    try:
        import meshio
        mesh = meshio.Mesh(
            points=verts,
            cells=[("triangle", faces)]
        )
        meshio.write(filename, mesh)
    except ImportError:
        # Fallback: write OBJ manually
        _write_obj_manual(filename, verts, faces)

    size_kb = os.path.getsize(filename) / 1024
    print(f"OBJ saved: {filename}  ({len(faces)} triangles, {size_kb:.0f} KB)")
    return filename


def export_step(verts, faces, filename):
    """
    Export triangle mesh to STEP format using pythonOCC.

    The STEP file contains a triangulated shell — Inventor opens it as a
    mesh body, not an editable solid. This is the best achievable output
    from SIMP topology optimization without manual CAD reconstruction.

    Requires pythonOCC:
        conda install -c conda-forge pythonocc-core
        OR
        pip install pythonocc-core  (Windows: may need Visual Studio)

    Args:
        verts:    (n_verts, 3) vertex coordinates
        faces:    (n_faces, 3) triangle indices
        filename: output path (e.g. 'result.step')

    Returns:
        filename on success, None if pythonOCC not available
    """
    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon,
                                              BRepBuilderAPI_MakeFace,
                                              BRepBuilderAPI_Sewing)
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TopoDS import TopoDS_Shell
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IFSelect import IFSelect_RetDone
    except ImportError:
        print("STEP export requires pythonOCC.")
        print("Install: conda install -c conda-forge pythonocc-core")
        print("Falling back to STL export.")
        stl_path = filename.replace('.step', '.stl').replace('.stp', '.stl')
        export_stl(verts, faces, stl_path)
        return None

    print(f"Building STEP shell from {len(faces)} triangles...")

    # Build a sewing of triangular faces
    sewing = BRepBuilderAPI_Sewing(1e-6)

    for tri in faces:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        poly = BRepBuilderAPI_MakePolygon()
        poly.Add(gp_Pnt(*v0.tolist()))
        poly.Add(gp_Pnt(*v1.tolist()))
        poly.Add(gp_Pnt(*v2.tolist()))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        sewing.Add(face)

    sewing.Perform()
    shell = sewing.SewedShape()

    # Write to STEP
    writer = STEPControl_Writer()
    writer.Transfer(shell, STEPControl_AsIs)
    status = writer.Write(filename)

    if status == IFSelect_RetDone:
        size_kb = os.path.getsize(filename) / 1024
        print(f"STEP saved: {filename}  ({len(faces)} triangles, {size_kb:.0f} KB)")
        print("Note: Opens as mesh body in Inventor, not editable solid.")
        return filename
    else:
        print(f"STEP write failed (status={status})")
        return None


def export_all(rho_cells, nx, ny, nz, width, height, depth,
               basename='topology_result',
               threshold=0.5, smooth_iter=50,
               formats=('stl', 'obj')):
    """
    Convenience function: extract mesh and export to multiple formats.

    Args:
        rho_cells:   (n_cells,) density field
        nx, ny, nz:  cell counts
        width, height, depth: physical dimensions
        basename:    output filename without extension
        threshold:   density isosurface value
        smooth_iter: Laplacian smoothing iterations
        formats:     tuple of formats to export ('stl', 'obj', 'step')

    Returns:
        dict mapping format → output path
    """
    print(f"\nExtracting mesh (threshold={threshold}, smooth={smooth_iter} iters)...")

    verts, faces, mesh_pv = density_to_mesh(
        rho_cells, nx, ny, nz, width, height, depth,
        threshold=threshold, smooth_iter=smooth_iter
    )

    print(f"  Mesh: {len(verts)} vertices, {len(faces)} triangles")
    print(f"  Bounds: x=[{verts[:,0].min():.3f},{verts[:,0].max():.3f}] "
          f"y=[{verts[:,1].min():.3f},{verts[:,1].max():.3f}] "
          f"z=[{verts[:,2].min():.3f},{verts[:,2].max():.3f}]")

    outputs = {}
    for fmt in formats:
        path = f"{basename}.{fmt}"
        if fmt == 'stl':
            export_stl(verts, faces, path, binary=True)
            outputs['stl'] = path
        elif fmt == 'obj':
            export_obj(verts, faces, path)
            outputs['obj'] = path
        elif fmt == 'step':
            result = export_step(verts, faces, path)
            if result:
                outputs['step'] = path

    return outputs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_normals(verts, faces):
    """Compute face normals from vertex positions."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


def _write_stl_binary(filename, verts, faces):
    """Write binary STL. Compact and universally supported."""
    normals = _compute_normals(verts, faces)
    with open(filename, 'wb') as f:
        # 80-byte header
        header = b'Topology optimization result - SIMP'
        f.write(header + b'\x00' * (80 - len(header)))
        f.write(struct.pack('<I', len(faces)))
        for i, tri in enumerate(faces):
            f.write(struct.pack('<3f', *normals[i]))
            for idx in tri:
                f.write(struct.pack('<3f', *verts[idx]))
            f.write(struct.pack('<H', 0))


def _write_stl_ascii(filename, verts, faces):
    """Write ASCII STL. Human-readable but larger."""
    normals = _compute_normals(verts, faces)
    with open(filename, 'w') as f:
        f.write("solid topology_optimization\n")
        for i, tri in enumerate(faces):
            n = normals[i]
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for idx in tri:
                v = verts[idx]
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid topology_optimization\n")


def _write_obj_manual(filename, verts, faces):
    """Write OBJ manually without meshio."""
    with open(filename, 'w') as f:
        f.write("# Topology optimization result\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            # OBJ uses 1-indexed faces
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
