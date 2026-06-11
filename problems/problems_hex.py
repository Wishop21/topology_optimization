"""
3D topology optimization problems using hex elements.

Identical physics to problems_3d.py — only the mesh type and
DOF helper functions change (hex vs tet).
"""

import numpy as np
from core.mesh_hex import create_hex_mesh, identify_boundary_nodes_hex
from core.fem_solver_hex import node_dofs_hex, build_force_vector_hex


class CantileverHex:
    """3D cantilever with hex elements. Fixed left, load at right."""

    name = '3D Cantilever (Hex)'

    def __init__(self, nx=30, ny=15, nz=10,
                 width=2.0, height=1.0, depth=1.0,
                 E=1.0, nu=0.3, vol_frac=0.3,
                 load_direction='y'):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.load_direction = load_direction
        self.mesh = create_hex_mesh(width, height, depth, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_hex(self.mesh)
        return node_dofs_hex(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        bounds = identify_boundary_nodes_hex(self.mesh)
        right_nodes = bounds['right']
        y = self.mesh.p[1, right_nodes]
        z = self.mesh.p[2, right_nodes]
        y_mid = self.mesh.p[1].max() / 2.0
        z_mid = self.mesh.p[2].max() / 2.0
        dist  = np.sqrt((y - y_mid)**2 + (z - z_mid)**2)
        tip   = right_nodes[np.argmin(dist)]
        fy = -1.0 if 'y' in self.load_direction else 0.0
        fz = -1.0 if 'z' in self.load_direction else 0.0
        return [build_force_vector_hex(n_dofs, [tip], fy=fy, fz=fz)]


class CantileverHexEdgeLoad:
    """3D cantilever with load on bottom edge of right face."""

    name = '3D Cantilever Edge Load (Hex)'

    def __init__(self, nx=30, ny=15, nz=10,
                 width=2.0, height=1.0, depth=1.0,
                 E=1.0, nu=0.3, vol_frac=0.3):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_hex_mesh(width, height, depth, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_hex(self.mesh)
        return node_dofs_hex(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        bounds = identify_boundary_nodes_hex(self.mesh)
        right  = bounds['right']
        bottom = bounds['bottom']
        load_nodes = np.intersect1d(right, bottom)
        return [build_force_vector_hex(n_dofs, load_nodes, fy=-1.0)]


class MBBBeamHex:
    """3D MBB beam half-model with hex elements."""

    name = '3D MBB Beam (Hex)'

    def __init__(self, nx=30, ny=10, nz=10,
                 E=1.0, nu=0.3, vol_frac=0.5):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_hex_mesh(3.0, 1.0, 1.0, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_hex(self.mesh)
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10
        sym_dofs    = node_dofs_hex(bounds['left'], direction='x')
        right_bottom = np.where(
            (np.abs(x - x.max()) < tol) & (np.abs(y - y.min()) < tol))[0]
        roller_dofs = node_dofs_hex(right_bottom, direction='y')
        return np.concatenate([sym_dofs, roller_dofs])

    def get_load_cases(self, n_dofs):
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10
        top_left = np.where(
            (np.abs(x - x.min()) < tol) & (np.abs(y - y.max()) < tol))[0]
        return [build_force_vector_hex(n_dofs, top_left, fy=-1.0)]


class LBracketHex:
    """3D L-bracket with hex elements."""

    name = '3D L-Bracket (Hex)'

    def __init__(self, nx=30, ny=30, nz=10,
                 E=1.0, nu=0.3, vol_frac=0.4):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_hex_mesh(1.0, 1.0, 1.0, nx, ny, nz)

        # Void cells: top-right quadrant
        all_ix = np.arange(nx*ny*nz) % nx
        all_iy = (np.arange(nx*ny*nz) // nx) % ny
        self.void_cells   = np.where((all_ix >= nx//2) & (all_iy >= ny//2))[0]
        self.design_cells = np.setdiff1d(np.arange(nx*ny*nz), self.void_cells)

    def get_fixed_dofs(self, basis):
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10
        fixed_nodes = np.where(
            (np.abs(y - y.max()) < tol) & (x <= self.mesh.p[0].max()/2.0 + tol))[0]
        return node_dofs_hex(fixed_nodes, direction='all')

    def get_load_cases(self, n_dofs):
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10
        load_nodes = np.where(
            (np.abs(x - x.max()) < tol) & (y <= self.mesh.p[1].max()/2.0 + tol))[0]
        return [build_force_vector_hex(n_dofs, load_nodes, fy=-1.0)]


class TorsionBeamHex:
    """
    3D torsion beam with hex elements.

    Applies Saint-Venant torsion: distributed shear on right face
    creating pure torque about the beam (x) axis with zero net force.

    Each node on the right face receives:
        fy = -(z - z_mid)   fz = +(y - y_mid)

    This produces a moment about x with no net force or bending moment.
    Expected result: diagonal/helical bracing at ~45 degrees to beam axis.
    """

    name = '3D Torsion Beam (Hex)'

    def __init__(self, nx=30, ny=10, nz=10,
                 E=1.0, nu=0.3, vol_frac=0.3):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_hex_mesh(2.0, 1.0, 1.0, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_hex(self.mesh)
        return node_dofs_hex(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        mesh = self.mesh
        x, y, z = mesh.p[0], mesh.p[1], mesh.p[2]
        tol = 1e-10
        right = np.where(np.abs(x - x.max()) < tol)[0]

        y_mid = y.max() / 2.0
        z_mid = z.max() / 2.0

        # Saint-Venant torsion: distributed shear creating pure torque
        # about x-axis. Net force = 0, net moment about x = T > 0.
        f = np.zeros(n_dofs)
        scale = 1.0 / len(right)
        for node in right:
            dy = y[node] - y_mid
            dz = z[node] - z_mid
            f[3*node + 1] = -dz * scale   # fy
            f[3*node + 2] =  dy * scale   # fz
        return [f]


class CantileverHexMultiLoad:
    """3D cantilever with two load cases: downward + lateral."""

    name = '3D Cantilever Multi-Load (Hex)'

    def __init__(self, nx=30, ny=15, nz=15,
                 E=1.0, nu=0.3, vol_frac=0.3):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_hex_mesh(2.0, 1.0, 1.0, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_hex(self.mesh)
        return node_dofs_hex(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        bounds = identify_boundary_nodes_hex(self.mesh)
        right = bounds['right']
        y = self.mesh.p[1, right]
        z = self.mesh.p[2, right]
        dist = np.sqrt((y - self.mesh.p[1].max()/2)**2 +
                       (z - self.mesh.p[2].max()/2)**2)
        tip = right[np.argmin(dist)]
        f1 = build_force_vector_hex(n_dofs, [tip], fy=-1.0)
        f2 = build_force_vector_hex(n_dofs, [tip], fz=-1.0)
        return [f1, f2]
