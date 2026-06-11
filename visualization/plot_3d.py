"""
3D topology optimization visualization.

Two modes:
  1. Live monitoring (matplotlib): three orthogonal density slices updated
     each iteration. Fast, no 3D overhead during optimization.

  2. Final result (PyVista): interactive isosurface of converged density
     field. Rotate, zoom, export screenshots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Live matplotlib slice monitor
# ---------------------------------------------------------------------------

class SliceMonitor:
    """
    Shows three orthogonal slices through the 3D density field during
    optimization. Updates every N iterations without blocking the loop.
    """

    def __init__(self, nx, ny, nz, problem_name="3D Optimization"):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.problem_name = problem_name

        self.history_compliance = []
        self.history_volume     = []
        self.history_change     = []

        self.fig = None
        self.axes = None

    def init_figure(self):
        """Create the figure. Called once before optimization starts."""
        self.fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 4, figure=self.fig,
                               left=0.06, right=0.97,
                               top=0.92, bottom=0.08,
                               wspace=0.35, hspace=0.45)

        self.axes = {
            'xy': self.fig.add_subplot(gs[0, 0]),  # z mid-slice
            'xz': self.fig.add_subplot(gs[0, 1]),  # y mid-slice
            'yz': self.fig.add_subplot(gs[0, 2]),  # x mid-slice
            'conv': self.fig.add_subplot(gs[0, 3]),
            'vol':  self.fig.add_subplot(gs[1, 0]),
            'chg':  self.fig.add_subplot(gs[1, 1]),
            'info': self.fig.add_subplot(gs[1, 3]),
        }

        self.fig.suptitle(f'SIMP 3D Topology Optimization — {self.problem_name}',
                          fontsize=12, fontweight='bold')
        self.axes['info'].axis('off')
        plt.pause(0.01)

    def update(self, rho_cells, iteration, compliance, volume_frac, change,
               penal=3.0, vol_frac_target=0.5, final=False):
        """
        Redraw all panels.

        Args:
            rho_cells: (n_cells,) density field
            iteration, compliance, volume_frac, change: scalars
            penal, vol_frac_target: optimizer parameters for info panel
            final: if True, marks plot as CONVERGED
        """
        self.history_compliance.append(compliance)
        self.history_volume.append(volume_frac)
        self.history_change.append(change)

        # Reshape to 3D grid: (nz, ny, nx) — consistent with row-major indexing
        grid = rho_cells.reshape(self.nz, self.ny, self.nx)
        iters = np.arange(1, len(self.history_compliance) + 1)

        iz_mid = self.nz // 2
        iy_mid = self.ny // 2
        ix_mid = self.nx // 2

        # --- XY slice (z = mid) ---
        ax = self.axes['xy']
        ax.cla()
        ax.imshow(grid[iz_mid, :, :], cmap='gray_r', vmin=0, vmax=1,
                  origin='lower', aspect='auto')
        ax.set_title(f'XY slice (z={iz_mid})')
        ax.set_xlabel('x cells'); ax.set_ylabel('y cells')

        # --- XZ slice (y = mid) ---
        ax = self.axes['xz']
        ax.cla()
        ax.imshow(grid[:, iy_mid, :], cmap='gray_r', vmin=0, vmax=1,
                  origin='lower', aspect='auto')
        ax.set_title(f'XZ slice (y={iy_mid})')
        ax.set_xlabel('x cells'); ax.set_ylabel('z cells')

        # --- YZ slice (x = mid) ---
        ax = self.axes['yz']
        ax.cla()
        ax.imshow(grid[:, :, ix_mid], cmap='gray_r', vmin=0, vmax=1,
                  origin='lower', aspect='auto')
        ax.set_title(f'YZ slice (x={ix_mid})')
        ax.set_xlabel('y cells'); ax.set_ylabel('z cells')

        # --- Compliance ---
        ax = self.axes['conv']
        ax.cla()
        ax.semilogy(iters, self.history_compliance, 'b-', lw=1.5)
        ax.set_title('Compliance'); ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

        # --- Volume ---
        ax = self.axes['vol']
        ax.cla()
        ax.plot(iters, self.history_volume, 'g-', lw=1.5)
        ax.axhline(vol_frac_target, color='r', ls='--', lw=1, label='Target')
        ax.set_title('Volume Fraction'); ax.set_xlabel('Iteration')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # --- Change ---
        ax = self.axes['chg']
        ax.cla()
        ax.semilogy(iters, self.history_change, 'r-', lw=1.5)
        ax.axhline(0.01, color='k', ls=':', lw=1, label='Tol')
        ax.set_title('Max Density Change'); ax.set_xlabel('Iteration')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # --- Info panel ---
        ax = self.axes['info']
        ax.cla(); ax.axis('off')
        E_str = f"{1.0:.1f}" # dimensionless default; caller can override
        status = 'CONVERGED' if final else f'Iter {iteration}'
        info = (
            f"Problem : {self.problem_name}\n"
            f"Mesh    : {self.nx}×{self.ny}×{self.nz}\n"
            f"Cells   : {self.nx*self.ny*self.nz}\n\n"
            f"SIMP p  : {penal:.1f}\n"
            f"Vol frac: {vol_frac_target}\n\n"
            f"Status    : {status}\n"
            f"Compliance: {compliance:.4f}\n"
            f"Volume    : {volume_frac:.4f}\n"
            f"Change    : {change:.4f}\n"
        )
        ax.text(0.05, 0.95, info, transform=ax.transAxes,
                fontsize=9, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def close(self):
        """Print message and block until user closes window."""
        print("\nClose the plot window to continue (or press Ctrl+C).")
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            plt.close('all')


# ---------------------------------------------------------------------------
# PyVista isosurface — final result
# ---------------------------------------------------------------------------

def show_isosurface(rho_cells, nx, ny, nz, width, height, depth,
                    threshold=0.5, problem_name="Optimized Topology",
                    save_path=None):
    """
    Render an interactive 3D isosurface of the final density field.

    Displays the surface where rho = threshold (default 0.5),
    coloured by density value.

    Args:
        rho_cells:    (n_cells,) final density field
        nx, ny, nz:   mesh cell counts
        width, height, depth: physical domain dimensions
        threshold:    isosurface density value (0.5 = 50% density boundary)
        problem_name: title for the window
        save_path:    if given, save screenshot to this path (PNG)
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Run: pip install pyvista")
        return

    # Build a structured grid with density as cell data
    # PyVista ImageData: cell data on a regular voxel grid
    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)   # node dimensions
    grid.spacing    = (width / nx, height / ny, depth / nz)
    grid.origin     = (0.0, 0.0, 0.0)

    # Density in (nz, ny, nx) order, then flatten in Fortran order for VTK
    rho_3d = rho_cells.reshape(nz, ny, nx)
    # VTK cell ordering: x fastest, z slowest → transpose to (nx, ny, nz) then flatten
    grid.cell_data['density'] = rho_3d.transpose(2, 1, 0).flatten(order='F')

    # contour() requires point data, not cell data — convert first
    grid_pts = grid.cell_data_to_point_data()
    surface  = grid_pts.contour(isosurfaces=[threshold], scalars='density')

    # Smooth the surface slightly for nicer visuals
    if surface.n_points > 0:
        surface = surface.smooth(n_iter=20, relaxation_factor=0.1)

    # Always open an interactive window.
    # If save_path given, screenshot is taken when the window opens via show(screenshot=).
    pl = pv.Plotter(window_size=(1200, 800))
    pl.add_text(problem_name, font_size=12, position='upper_left')

    if surface.n_points > 0:
        pl.add_mesh(surface, color='steelblue', opacity=0.95,
                    smooth_shading=True, show_edges=False)
    else:
        print(f"Warning: no surface found at threshold={threshold}. "
              f"Try a lower value (density range: "
              f"{rho_cells.min():.2f}-{rho_cells.max():.2f})")

    pl.add_mesh(grid.outline(), color='black', line_width=1)
    pl.add_axes()
    pl.camera_position = 'iso'
    pl.set_background('white')

    if save_path:
        pl.show(screenshot=save_path, auto_close=False)
        print(f"Screenshot saved: {save_path}")
    else:
        pl.show()


def save_density_slices(rho_cells, nx, ny, nz, filename, iteration=None):
    """
    Save three mid-plane slice plots to a PNG file (no display needed).
    Useful for logging results without an interactive window.
    """
    grid = rho_cells.reshape(nz, ny, nx)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = [f'XY (z={nz//2})', f'XZ (y={ny//2})', f'YZ (x={nx//2})']
    slices = [grid[nz//2, :, :], grid[:, ny//2, :], grid[:, :, nx//2]]

    for ax, sl, title in zip(axes, slices, titles):
        ax.imshow(sl, cmap='gray_r', vmin=0, vmax=1,
                  origin='lower', aspect='equal')
        ax.set_title(title)

    suptitle = f'3D Density Field'
    if iteration is not None:
        suptitle += f' — Iteration {iteration}'
    fig.suptitle(suptitle, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Slices saved: {filename}")
