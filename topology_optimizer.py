"""
Main SIMP topology optimization loop.

Ties together: FEA solve → sensitivity → filter → OC update → repeat.

Live plotting shows the density field evolving each iteration (or every
N iterations for large meshes — update_interval is chosen automatically
based on mesh size so the display stays responsive).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import skfem
import time

from core.fem_solver import LinearElasticitySolver, node_dofs, build_force_vector, plane_stress_lame
from optimization.simp import build_elem_to_cell, assemble_simp_stiffness, compute_sensitivities
from optimization.filtering import build_filter_matrix, apply_filter, apply_filter_sensitivity
from optimization.optimality_criteria import oc_update, check_convergence


class TopologyOptimizer:
    """
    2D SIMP topology optimizer.

    Design variables: one density per rectangular cell (two triangles
    per cell share one rho value, avoiding triangle-asymmetry artifacts).
    """

    def __init__(self, problem):
        """
        Args:
            problem: a Problem object with attributes:
                mesh, nx, ny, E, nu, vol_frac
                get_fixed_dofs(basis) → array
                get_force_vector(n_dofs) → array
        """
        self.problem = problem
        self.mesh    = problem.mesh
        self.nx      = problem.nx
        self.ny      = problem.ny
        self.n_cells = problem.nx * problem.ny

        # FEA solver (used for stiffness assembly pattern only;
        # actual K is rebuilt each iteration with SIMP densities)
        self.solver = LinearElasticitySolver(self.mesh, E=problem.E, nu=problem.nu)
        self.basis  = self.solver.basis

        # Precompute cell mapping (doesn't change)
        self.elem_to_cell = build_elem_to_cell(self.mesh, self.nx, self.ny)

        # Cell volumes (area * unit thickness = area for 2D)
        # All cells have the same area for a structured mesh
        cell_width  = self.mesh.p[0].max() / self.nx
        cell_height = self.mesh.p[1].max() / self.ny
        self.cell_volumes = np.full(self.n_cells, cell_width * cell_height)

        # SIMP parameters
        self.E0    = problem.E
        self.E_min = problem.E * 1e-9   # void: effectively zero but non-singular
        self.penal = 3.0
        self.nu    = problem.nu

        # Boundary conditions (precomputed, fixed for all iterations)
        self.fixed_dofs = problem.get_fixed_dofs(self.basis)
        self.f_vec      = problem.get_force_vector(self.basis.N)

        # Decide how often to update the plot based on mesh size
        # Target: update takes < 0.5s on screen, so cap at ~1 update/s
        # For very large meshes, update every N iterations
        self.update_interval = max(1, self.n_cells // 2000)

        # History for convergence plot
        self.history_compliance = []
        self.history_volume     = []
        self.history_change     = []

    def run(self, max_iter=300, tol=0.01, r_min=1.5, vol_frac=None,
            penal=3.0, penal_continuation=True, verbose=True):
        """
        Run the SIMP optimization loop.

        Args:
            max_iter:            maximum number of iterations
            tol:                 convergence tolerance (max density change)
            r_min:               filter radius in cell units
            vol_frac:            target volume fraction (defaults to problem setting)
            penal:               final SIMP penalization exponent (typically 3)
            penal_continuation:  if True, ramp p: 1 -> 2 -> penal over first 40 iters.
                                 Prevents early cycling and reduces grey elements.
                                 Without this, OC can oscillate indefinitely on
                                 problems like MBB. Set False to use fixed penal.
            verbose:             print iteration log

        Returns:
            rho: (n_cells,) final density field
        """
        if vol_frac is None:
            vol_frac = self.problem.vol_frac
        self.penal    = penal
        self.vol_frac = vol_frac

        # Penalization schedule: (iteration_to_activate, p_value)
        # Starting with p=1 gives a smooth landscape; ramping to p=3 pushes
        # densities toward 0/1 once the gross topology is established.
        if penal_continuation:
            penal_schedule = [(1, 1.0), (20, 2.0), (40, penal)]
        else:
            penal_schedule = [(1, penal)]

        current_penal = penal_schedule[0][1]

        # Build filter (precomputed once — only depends on geometry)
        if verbose:
            print(f"Building filter matrix (r_min={r_min})...")
        H, Hs = build_filter_matrix(self.nx, self.ny, r_min)

        # Initialise: uniform density at vol_frac
        rho = np.full(self.n_cells, vol_frac)

        # Set up live plot
        fig, axes = self._init_plot()
        plt.pause(0.01)

        t_start = time.time()

        if verbose:
            print(f"\n{'Iter':>5}  {'p':>4}  {'Compliance':>14}  {'Volume':>8}  "
                  f"{'Change':>8}  {'Time':>6}")
            print("-" * 62)

        for iteration in range(1, max_iter + 1):
            rho_old = rho.copy()

            # Advance penalization schedule
            for start_iter, p_val in penal_schedule:
                if iteration >= start_iter:
                    current_penal = p_val
            self.penal = current_penal

            # 1. Filter densities before FEA (density filter)
            rho_f = apply_filter(H, Hs, rho)

            # 2. Assemble SIMP-modified stiffness and solve
            K, _ = assemble_simp_stiffness(
                self.basis, self.elem_to_cell, rho_f,
                self.E0, self.E_min, self.penal, self.nu
            )
            u = self.solver.solve(K, self.f_vec, self.fixed_dofs)

            # 3. Compliance and volume
            compliance = float(self.f_vec @ u)
            volume_frac = (rho_f * self.cell_volumes).sum() / self.cell_volumes.sum()

            # 4. Sensitivities w.r.t. filtered densities
            dc_drho_f = compute_sensitivities(
                self.basis, u, self.elem_to_cell, self.n_cells,
                rho_f, self.E0, self.E_min, self.penal, self.nu
            )

            # 5. Chain rule: sensitivity w.r.t. raw (unfiltered) rho
            dc_drho = apply_filter_sensitivity(H, Hs, dc_drho_f)

            # 6. OC density update
            rho, _ = oc_update(rho, dc_drho, vol_frac, self.cell_volumes)

            # 7. Convergence check
            converged, change = check_convergence(rho, rho_old, tol)

            # 8. Log
            elapsed = time.time() - t_start
            self.history_compliance.append(compliance)
            self.history_volume.append(volume_frac)
            self.history_change.append(change)

            if verbose:
                print(f"{iteration:>5}  {self.penal:>4.1f}  {compliance:>14.4f}  "
                      f"{volume_frac:>8.4f}  {change:>8.4f}  {elapsed:>5.1f}s")

            # 9. Update live plot (every update_interval iterations)
            if iteration % self.update_interval == 0 or converged or iteration == 1:
                self._update_plot(fig, axes, rho_f, iteration, compliance,
                                  volume_frac, change)
                plt.pause(0.01)

            if converged:
                if verbose:
                    print(f"\nConverged at iteration {iteration} "
                          f"(change={change:.4f} < tol={tol})")
                break
        else:
            if verbose:
                print(f"\nReached max_iter={max_iter} without convergence "
                      f"(final change={change:.4f})")

        # Final plot update to make sure last state is shown
        self._update_plot(fig, axes, rho_f, iteration, compliance,
                          volume_frac, change, final=True)
        plt.pause(0.1)

        # Keep the window open until the user closes it.
        # Catching KeyboardInterrupt means Ctrl+C closes cleanly
        # instead of producing a traceback.
        print("\nClose the plot window to continue (or press Ctrl+C).")
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            plt.close('all')

        self.rho_final = rho_f
        return rho_f

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _init_plot(self):
        """Set up the live-updating figure."""
        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                left=0.07, right=0.97,
                                top=0.92, bottom=0.08,
                                wspace=0.35, hspace=0.45)

        axes = {
            'density':     fig.add_subplot(gs[0, :2]),   # wide: density field
            'convergence': fig.add_subplot(gs[0, 2]),    # compliance history
            'volume':      fig.add_subplot(gs[1, 0]),    # volume history
            'change':      fig.add_subplot(gs[1, 1]),    # density change history
            'info':        fig.add_subplot(gs[1, 2]),    # text info panel
        }

        fig.suptitle('SIMP Topology Optimization', fontsize=13, fontweight='bold')

        axes['density'].set_title('Density Field (black=solid, white=void)')
        axes['convergence'].set_title('Compliance')
        axes['volume'].set_title('Volume Fraction')
        axes['change'].set_title('Max Density Change')
        axes['info'].axis('off')

        return fig, axes

    def _density_to_grid(self, rho_cells):
        """
        Reshape flat cell density array to 2D grid for imshow.

        Returns (ny, nx) array — imshow expects rows=y, cols=x.
        """
        return rho_cells.reshape(self.ny, self.nx)

    def _update_plot(self, fig, axes, rho_f, iteration, compliance,
                     volume_frac, change, final=False):
        """Redraw all subplots with current iteration data."""
        iters = np.arange(1, len(self.history_compliance) + 1)

        # --- Density field ---
        ax = axes['density']
        ax.cla()
        grid = self._density_to_grid(rho_f)
        # origin='lower' so y=0 is at the bottom (matches physical domain)
        im = ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1,
                       origin='lower', aspect='auto',
                       extent=[0, self.mesh.p[0].max(),
                               0, self.mesh.p[1].max()])
        ax.set_xlabel('x'); ax.set_ylabel('y')
        status = 'CONVERGED' if final else f'Iter {iteration}'
        ax.set_title(f'Density Field  [{status}]  '
                     f'c={compliance:.2f}  V={volume_frac:.3f}')

        # --- Compliance history ---
        ax = axes['convergence']
        ax.cla()
        ax.semilogy(iters, self.history_compliance, 'b-', lw=1.5)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Compliance')
        ax.set_title('Compliance')
        ax.grid(True, alpha=0.3)

        # --- Volume history ---
        ax = axes['volume']
        ax.cla()
        ax.plot(iters, self.history_volume, 'g-', lw=1.5)
        ax.axhline(self.vol_frac, color='r', ls='--', lw=1, label='Target')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Vol fraction')
        ax.set_title('Volume Fraction')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # --- Max density change ---
        ax = axes['change']
        ax.cla()
        ax.semilogy(iters, self.history_change, 'r-', lw=1.5)
        ax.axhline(0.01, color='k', ls=':', lw=1, label='Conv. tol')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Max |Δρ|')
        ax.set_title('Max Density Change')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # --- Info panel ---
        ax = axes['info']
        ax.cla(); ax.axis('off')
        # Format E sensibly: if >= 1e6 show in GPa, else show as-is (dimensionless)
        E_str = f"{self.E0/1e9:.0f} GPa" if self.E0 >= 1e6 else f"{self.E0:.1f}"
        info = (
            f"Problem : {self.problem.name}\n"
            f"Mesh    : {self.nx} × {self.ny} cells\n"
            f"Cells   : {self.n_cells}\n"
            f"DOFs    : {self.basis.N}\n\n"
            f"E       : {E_str}\n"
            f"nu      : {self.nu}\n"
            f"SIMP p  : {self.penal}\n"
            f"Vol frac: {self.vol_frac}\n\n"
            f"Iteration : {iteration}\n"
            f"Compliance: {compliance:.4f}\n"
            f"Change    : {change:.4f}\n"
        )
        ax.text(0.05, 0.95, info, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.canvas.draw_idle()

    def save_result(self, filename='topology_result.png'):
        """Save the final density field as an image."""
        fig, ax = plt.subplots(figsize=(10, 4))
        grid = self._density_to_grid(self.rho_final)
        ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1,
                  origin='lower', aspect='auto')
        ax.set_title(f'Optimized topology: {self.problem.name}')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Result saved: {filename}")
        plt.close(fig)
