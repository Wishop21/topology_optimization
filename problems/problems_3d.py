"""
3D topology optimization problem definitions.

Each class provides:
    mesh, nx, ny, nz, E, nu, vol_frac
    get_fixed_dofs(basis)    → fixed DOF array
    get_load_cases(n_dofs)   → list of force vectors (one per load case)

Single load: list of length 1
Multi-load:  list of length > 1, sensitivities are summed across cases

All problems use dimensionless E=1 by default so compliance values
are comparable across problems regardless of physical units.
"""

import numpy as np
from core.mesh_3d import create_box_mesh, identify_boundary_nodes_3d
from core.fem_solver_3d import node_dofs_3d, build_force_vector_3d


class Cantilever3D:
    """
    3D cantilever: fixed on left face (x=0), load on right face (x=max).

    Standard single-load: downward force at centre of right face.
    Multi-load variant: add second load case (horizontal or torsional).

    Expected topology: diagonal bracing extending from fixed face to
    load point, symmetric about beam centreline.
    """

    name = '3D Cantilever'

    def __init__(self, nx=30, ny=15, nz=10,
                 width=2.0, height=1.0, depth=1.0,
                 E=1.0, nu=0.3, vol_frac=0.3,
                 load_direction='y',
                 multi_load=False):
        """
        Args:
            nx, ny, nz:       cell counts (resolution)
            width/height/depth: domain dimensions
            E, nu:            material properties
            vol_frac:         target volume fraction
            load_direction:   'y' (downward), 'z' (out-of-plane), or 'yz' (both)
            multi_load:       if True, add second load case in z-direction
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.load_direction = load_direction
        self.multi_load = multi_load
        self.mesh = create_box_mesh(width, height, depth, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        """Fix all DOFs on left face (x=0)."""
        bounds = identify_boundary_nodes_3d(self.mesh)
        return node_dofs_3d(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        """
        Returns list of force vectors.

        Single load: [f_downward]
        Multi-load:  [f_downward, f_horizontal]
        """
        bounds = identify_boundary_nodes_3d(self.mesh)
        right_nodes = bounds['right']

        # Find node closest to centre of right face
        y = self.mesh.p[1, right_nodes]
        z = self.mesh.p[2, right_nodes]
        y_mid = self.mesh.p[1].max() / 2.0
        z_mid = self.mesh.p[2].max() / 2.0
        dist  = np.sqrt((y - y_mid)**2 + (z - z_mid)**2)
        tip   = right_nodes[np.argmin(dist)]

        load_cases = []

        # Primary load case
        fy = -1.0 if 'y' in self.load_direction else 0.0
        fz = -1.0 if 'z' in self.load_direction else 0.0
        load_cases.append(
            build_force_vector_3d(n_dofs, [tip], fy=fy, fz=fz))

        # Second load case: horizontal force (z-direction)
        if self.multi_load:
            load_cases.append(
                build_force_vector_3d(n_dofs, [tip], fz=-1.0))

        return load_cases


class MBBBeam3D:
    """
    3D MBB beam (half model, symmetric in x).

    Extends the 2D half-model by adding a depth dimension.
    The structure is no longer forced to be an extruded 2D profile —
    it can develop genuine 3D topology.

    BCs:
        Left face (x=0):          symmetry — fix u_x = 0
        Bottom-right edge (x=max, y=0): roller — fix u_y = 0
    Load:
        Top-left edge (x=0, y=max): downward force
    """

    name = '3D MBB Beam (half model)'

    def __init__(self, nx=30, ny=10, nz=10,
                 E=1.0, nu=0.3, vol_frac=0.5):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        # Standard 3:1 aspect ratio in x-y, square in z
        self.mesh = create_box_mesh(3.0, 1.0, 1.0, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_3d(self.mesh)
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10

        # Symmetry: fix u_x on left face
        sym_dofs = node_dofs_3d(bounds['left'], direction='x')

        # Roller: fix u_y on bottom-right edge (x=max AND y=min)
        right_bottom = np.where(
            (np.abs(x - x.max()) < tol) & (np.abs(y - y.min()) < tol))[0]
        roller_dofs = node_dofs_3d(right_bottom, direction='y')

        return np.concatenate([sym_dofs, roller_dofs])

    def get_load_cases(self, n_dofs):
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10
        # Load: top-left edge (x=0, y=max) — all z positions
        top_left = np.where(
            (np.abs(x - x.min()) < tol) & (np.abs(y - y.max()) < tol))[0]
        f = build_force_vector_3d(n_dofs, top_left, fy=-1.0)
        return [f]


class LBracket3D:
    """
    3D L-bracket: square domain with top-right quadrant void.

    Domain: full square box with the top-right quarter of cells treated
    as permanently void (density forced to E_min).

    BCs:
        Top face (y=max) of the LEFT half: fully fixed
    Load:
        Right face (x=max) of the BOTTOM half: downward force

    This is a standard stress-concentration test case.
    The expected result shows a diagonal brace at the re-entrant corner.

    Note:
        The void region is handled in the problem by zeroing out sensitivities
        and forcing densities to rho_min in the top-right quadrant.
        The optimizer is told about this via the 'void_cells' attribute.
    """

    name = '3D L-Bracket'

    def __init__(self, nx=30, ny=30, nz=10,
                 E=1.0, nu=0.3, vol_frac=0.4):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        # Square domain in x-y, depth in z
        self.mesh = create_box_mesh(1.0, 1.0, 1.0, nx, ny, nz)

        # Identify void cells: top-right quadrant (ix >= nx//2 AND iy >= ny//2)
        ix_cells = np.arange(nx * ny * nz) % nx
        iy_cells = (np.arange(nx * ny * nz) // nx) % ny
        # Cell flat index: iz*(nx*ny) + iy*nx + ix
        all_ix = np.arange(nx * ny * nz) % nx
        all_iy = (np.arange(nx * ny * nz) // nx) % ny
        self.void_cells = np.where(
            (all_ix >= nx // 2) & (all_iy >= ny // 2))[0]
        self.design_cells = np.setdiff1d(
            np.arange(nx * ny * nz), self.void_cells)

    def get_fixed_dofs(self, basis):
        """Fix top face of left half (x < 0.5, y = max)."""
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10
        fixed_nodes = np.where(
            (np.abs(y - y.max()) < tol) & (x <= self.mesh.p[0].max() / 2.0 + tol)
        )[0]
        return node_dofs_3d(fixed_nodes, direction='all')

    def get_load_cases(self, n_dofs):
        """Downward load on right face of bottom half."""
        x, y = self.mesh.p[0], self.mesh.p[1]
        tol = 1e-10
        load_nodes = np.where(
            (np.abs(x - x.max()) < tol) & (y <= self.mesh.p[1].max() / 2.0 + tol)
        )[0]
        f = build_force_vector_3d(n_dofs, load_nodes, fy=-1.0)
        return [f]


class Cantilever3DMultiLoad:
    """
    3D cantilever with two independent load cases:
        Case 1: downward force (y-direction) at tip centre
        Case 2: lateral force (z-direction) at tip centre

    The optimizer produces a structure stiff under both loads simultaneously.
    This results in a symmetric bracing pattern rather than the planar
    structure seen with a single load.
    """

    name = '3D Cantilever (multi-load)'

    def __init__(self, nx=30, ny=15, nz=15,
                 E=1.0, nu=0.3, vol_frac=0.3):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_box_mesh(2.0, 1.0, 1.0, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_3d(self.mesh)
        return node_dofs_3d(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        bounds = identify_boundary_nodes_3d(self.mesh)
        right_nodes = bounds['right']
        y = self.mesh.p[1, right_nodes]
        z = self.mesh.p[2, right_nodes]
        y_mid = self.mesh.p[1].max() / 2.0
        z_mid = self.mesh.p[2].max() / 2.0
        dist  = np.sqrt((y - y_mid)**2 + (z - z_mid)**2)
        tip   = right_nodes[np.argmin(dist)]

        f1 = build_force_vector_3d(n_dofs, [tip], fy=-1.0)        # downward
        f2 = build_force_vector_3d(n_dofs, [tip], fz=-1.0)        # lateral
        return [f1, f2]


class Cantilever3DEdgeLoad:
    """
    3D cantilever with load on the bottom edge of the right face.

    This produces a much more visually interesting topology than a central
    point load. The bottom-edge load forces the optimizer to create:
    - A clear tension chord along the bottom
    - A compression arch along the top
    - Diagonal members connecting them

    This is the 3D equivalent of the 2D cantilever and produces a
    recognisable truss structure rather than a blob.
    """

    name = '3D Cantilever (edge load)'

    def __init__(self, nx=30, ny=15, nz=10,
                 width=2.0, height=1.0, depth=1.0,
                 E=1.0, nu=0.3, vol_frac=0.3):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_box_mesh(width, height, depth, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_3d(self.mesh)
        return node_dofs_3d(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        bounds = identify_boundary_nodes_3d(self.mesh)
        # Load on bottom edge of right face (x=max, y=min) — all z positions
        right_nodes  = bounds['right']
        bottom_nodes = bounds['bottom']
        load_nodes   = np.intersect1d(right_nodes, bottom_nodes)
        return [build_force_vector_3d(n_dofs, load_nodes, fy=-1.0)]


class TorsionBeam3D:
    """
    3D beam under torsion: fixed on left face, opposing forces on right face
    (up on top edge, down on bottom edge) create a twisting moment.

    Expected result: helical/diagonal bracing pattern — clearly 3D topology
    that has no 2D equivalent.
    """

    name = '3D Torsion Beam'

    def __init__(self, nx=30, ny=10, nz=10,
                 E=1.0, nu=0.3, vol_frac=0.3):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vol_frac = vol_frac
        self.E, self.nu = E, nu
        self.mesh = create_box_mesh(2.0, 1.0, 1.0, nx, ny, nz)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes_3d(self.mesh)
        return node_dofs_3d(bounds['left'], direction='all')

    def get_load_cases(self, n_dofs):
        # Saint-Venant torsion: distributed shear on right face
        # creating pure torque about beam (x) axis. Net force = 0.
        mesh = self.mesh
        x, y, z = mesh.p[0], mesh.p[1], mesh.p[2]
        tol = 1e-10
        right = np.where(np.abs(x - x.max()) < tol)[0]
        y_mid = y.max() / 2.0
        z_mid = z.max() / 2.0
        f = np.zeros(n_dofs)
        scale = 1.0 / len(right)
        for node in right:
            dy = y[node] - y_mid
            dz = z[node] - z_mid
            f[3*node + 1] = -dz * scale
            f[3*node + 2] =  dy * scale
        return [f]
