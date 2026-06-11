"""
Standard topology optimization test problems.

Each problem class specifies:
    - mesh geometry (nx, ny cells)
    - material properties
    - volume fraction target
    - boundary conditions (fixed DOFs)
    - loading (force vector)

The optimizer is problem-agnostic — it only calls
    problem.get_fixed_dofs(basis)
    problem.get_force_vector(n_dofs)
so new problems only need those two methods.
"""

import numpy as np
import skfem

from core.mesh import create_rectangle_mesh, identify_boundary_nodes
from core.fem_solver import node_dofs, build_force_vector


class CantileverBeam2D:
    """
    Cantilever beam: fixed on left edge, point load at tip centre.

    Classic validation case. Expected result: diagonal bracing from
    fixed edge to load point.

    Domain: width x height
    BC:     all nodes on left edge fully fixed
    Load:   downward force at midpoint of right edge
    """

    name = 'Cantilever Beam 2D'

    def __init__(self, nx=60, ny=30, width=2.0, height=1.0,
                 E=1.0, nu=0.3, vol_frac=0.5, force=-1.0):
        """
        Args:
            nx, ny:    number of cells (design resolution)
            width:     domain width (aspect ratio matters, not absolute size)
            height:    domain height
            E:         Young's modulus (use 1.0 for dimensionless problem)
            nu:        Poisson's ratio
            vol_frac:  target volume fraction
            force:     total downward force (negative = downward)
        """
        self.nx, self.ny   = nx, ny
        self.vol_frac      = vol_frac
        self.E, self.nu    = E, nu
        self.force         = force
        self.mesh = create_rectangle_mesh(width, height, nx, ny)

    def get_fixed_dofs(self, basis):
        """Fix all DOFs on the left edge."""
        bounds = identify_boundary_nodes(self.mesh)
        return node_dofs(bounds['left'], direction='both')

    def get_force_vector(self, n_dofs):
        """
        Apply load at the midpoint of the right edge only.

        Using only the single closest node to the midpoint rather than
        the full edge gives a cleaner benchmark result.
        """
        bounds = identify_boundary_nodes(self.mesh)
        right_nodes = bounds['right']

        # Find node closest to vertical midpoint of right edge
        y_coords = self.mesh.p[1, right_nodes]
        y_mid    = self.mesh.p[1].max() / 2.0
        tip_node = right_nodes[np.argmin(np.abs(y_coords - y_mid))]

        return build_force_vector(n_dofs, [tip_node], fy=self.force)


class MBBBeam2D:
    """
    MBB (Messerschmitt-Bölkow-Blohm) beam benchmark.

    The MBB beam is the most widely used topology optimization benchmark.
    It exploits symmetry: only the right half is modelled.

    Full beam: simply supported at bottom corners, central downward load.
    Half model: left edge has symmetry BC (u_x = 0), bottom-right corner
    pinned, load applied at top-left corner.

    Expected result: classic arch/truss structure matching published results.

    Reference: Sigmund (2001), "A 99 line topology optimization code"
    """

    name = 'MBB Beam 2D (half model)'

    def __init__(self, nx=60, ny=20, E=1.0, nu=0.3,
                 vol_frac=0.5, force=-1.0):
        """
        Args:
            nx, ny:   cells in x and y (typical: nx = 3*ny for half-beam)
            E, nu:    material properties
            vol_frac: target volume fraction
            force:    applied force (negative = downward)
        """
        self.nx, self.ny = nx, ny
        self.vol_frac    = vol_frac
        self.E, self.nu  = E, nu
        self.force       = force

        # Standard MBB half-model: domain is 3 wide, 1 tall (fixed ratio).
        # nx/ny controls cell density, not domain shape.
        # A common mistake is setting width = 3*(nx/ny), which stretches
        # the domain rather than just refining it.
        width  = 3.0
        height = 1.0
        self.mesh = create_rectangle_mesh(width, height, nx, ny)

    def get_fixed_dofs(self, basis):
        """
        Half-model symmetry boundary conditions:
          - Left edge:  fix x-displacement only (symmetry plane)
          - Bottom-right corner: fix y-displacement only (roller support)
        """
        bounds = identify_boundary_nodes(self.mesh)

        # Symmetry: u_x = 0 on left edge
        sym_dofs = node_dofs(bounds['left'], direction='x')

        # Roller at bottom-right corner: u_y = 0
        right_nodes  = bounds['right']
        bottom_nodes = bounds['bottom']
        corner_node  = np.intersect1d(right_nodes, bottom_nodes)
        roller_dofs  = node_dofs(corner_node, direction='y')

        return np.concatenate([sym_dofs, roller_dofs])

    def get_force_vector(self, n_dofs):
        """
        Load at top-left corner (corresponds to beam centre in full model).
        """
        bounds = identify_boundary_nodes(self.mesh)
        left_nodes = bounds['left']
        top_nodes  = bounds['top']
        corner_node = np.intersect1d(left_nodes, top_nodes)

        return build_force_vector(n_dofs, corner_node, fy=self.force)


class BridgeStructure2D:
    """
    Bridge: simply supported at bottom corners, distributed load on top edge.

    Expected result: arch-and-cable truss structure.

    Domain: wide rectangle (width >> height)
    BC:     bottom-left pinned, bottom-right roller
    Load:   distributed downward force across top edge
    """

    name = 'Bridge Structure 2D'

    def __init__(self, nx=80, ny=40, E=1.0, nu=0.3,
                 vol_frac=0.4, force=-1.0):
        self.nx, self.ny = nx, ny
        self.vol_frac    = vol_frac
        self.E, self.nu  = E, nu
        self.force       = force

        # Bridge domain: 2 wide, 1 tall (fixed ratio).
        # nx/ny controls cell density only.
        width  = 2.0
        height = 1.0
        self.mesh = create_rectangle_mesh(width, height, nx, ny)

    def get_fixed_dofs(self, basis):
        bounds = identify_boundary_nodes(self.mesh)

        # Pin at bottom-left (fix x and y)
        left_nodes   = bounds['left']
        bottom_nodes = bounds['bottom']
        bl_corner    = np.intersect1d(left_nodes, bottom_nodes)
        pin_dofs     = node_dofs(bl_corner, direction='both')

        # Roller at bottom-right (fix y only)
        right_nodes  = bounds['right']
        br_corner    = np.intersect1d(right_nodes, bottom_nodes)
        roller_dofs  = node_dofs(br_corner, direction='y')

        return np.concatenate([pin_dofs, roller_dofs])

    def get_force_vector(self, n_dofs):
        """Distribute total force uniformly across top edge."""
        bounds = identify_boundary_nodes(self.mesh)
        return build_force_vector(n_dofs, bounds['top'], fy=self.force)
