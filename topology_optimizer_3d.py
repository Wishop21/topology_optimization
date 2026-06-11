"""
3D SIMP topology optimization loop.

Same structure as topology_optimizer.py but extended for 3D:
- 3D mesh, FEA, filtering
- Multi-load case support
- Matplotlib slice monitoring during optimization
- PyVista isosurface at convergence
"""

import numpy as np
import time
from collections import deque

from core.fem_solver_3d import LinearElasticitySolver3D, node_dofs_3d, build_force_vector_3d
from core.mesh_3d import build_elem_to_cell_3d
from optimization.simp_3d import (assemble_simp_stiffness_3d, compute_sensitivities_3d,
                              build_k0_cache)
from optimization.filtering_3d import (build_filter_matrix_3d,
                                        apply_filter_3d,
                                        apply_filter_sensitivity_3d)
from optimization.optimality_criteria import oc_update, check_convergence
from visualization.plot_3d import SliceMonitor, show_isosurface, save_density_slices


class TopologyOptimizer3D:
    """
    3D SIMP topology optimizer with multi-load case support.

    Multi-load case:
        If a problem defines multiple load cases, FEA is solved once per
        load case each iteration and sensitivities are summed:
            dc_total/drho = sum_k (dc_k/drho)
        The OC update then uses the combined sensitivity.
        This produces structures optimal under all loads simultaneously.
    """

    def __init__(self, problem):
        """
        Args:
            problem: a Problem3D object with attributes:
                mesh, nx, ny, nz, E, nu, vol_frac
                get_fixed_dofs(basis)       → array
                get_load_cases(n_dofs)      → list of force vectors
                                              (single load: list of length 1)
        """
        self.problem  = problem
        self.mesh     = problem.mesh
        self.nx, self.ny, self.nz = problem.nx, problem.ny, problem.nz
        self.n_cells  = self.nx * self.ny * self.nz

        self.solver = LinearElasticitySolver3D(
            self.mesh, E=problem.E, nu=problem.nu)
        self.basis  = self.solver.basis

        self.elem_to_cell = build_elem_to_cell_3d(
            self.mesh, self.nx, self.ny, self.nz)

        # Cell volumes (all equal for structured mesh)
        cell_w = self.mesh.p[0].max() / self.nx
        cell_h = self.mesh.p[1].max() / self.ny
        cell_d = self.mesh.p[2].max() / self.nz
        self.cell_volumes = np.full(self.n_cells, cell_w * cell_h * cell_d)

        self.E0    = problem.E
        self.E_min = problem.E * 1e-9
        self.nu    = problem.nu

        # Fixed DOFs and load cases (precomputed)
        self.fixed_dofs = problem.get_fixed_dofs(self.basis)
        self.load_cases = problem.get_load_cases(self.basis.N)

        # Precompute unit-material element stiffness (reused every iteration)
        self.K0_local = build_k0_cache(self.basis, problem.nu)
        n_loads = len(self.load_cases)

        # Plot update interval: cap at 1 update per ~2s
        self.update_interval = max(1, self.n_cells // 3000)

        print(f"Initialized 3D optimizer:")
        print(f"  Mesh    : {self.nx}×{self.ny}×{self.nz} cells")
        print(f"  Elements: {self.mesh.t.shape[1]}")
        print(f"  Nodes   : {self.mesh.p.shape[1]}")
        print(f"  DOFs    : {self.basis.N}")
        print(f"  Cells   : {self.n_cells}")
        print(f"  Loads   : {n_loads} load case(s)")
        print(f"  Plot interval: every {self.update_interval} iter(s)")
        print(f"  K0 cache built (saves ~1s/iter)")

    def run(self, max_iter=300, tol=0.01, r_min=1.5, vol_frac=None,
            penal=3.0, penal_continuation=True, sens_avg=3, verbose=True):
        """
        Run 3D SIMP optimization.

        Args:
            max_iter:           maximum iterations
            tol:                convergence tolerance (max density change)
            r_min:              filter radius in cell units
            vol_frac:           target volume fraction
            penal:              final penalization exponent
            penal_continuation: ramp p: 1→2→penal over first 40 iters
            verbose:            print iteration log

        Returns:
            rho: (n_cells,) final filtered density field
        """
        if vol_frac is None:
            vol_frac = self.problem.vol_frac
        self.vol_frac = vol_frac

        if penal_continuation:
            penal_schedule = [(1, 1.0), (20, 2.0), (40, penal)]
        else:
            penal_schedule = [(1, penal)]
        current_penal = penal_schedule[0][1]

        # Build 3D filter
        if verbose:
            print(f"\nBuilding 3D filter matrix "
                  f"({self.nx}×{self.ny}×{self.nz}, r_min={r_min})...")
        H, Hs = build_filter_matrix_3d(self.nx, self.ny, self.nz, r_min)
        if verbose:
            print(f"Filter built: {H.nnz} non-zeros")

        # Sensitivity history for averaging (reduces oscillation on asymmetric 3D loads)
        sens_history = deque(maxlen=max(1, sens_avg))

        # Initialise density
        rho = np.full(self.n_cells, vol_frac)

        # Set up slice monitor
        monitor = SliceMonitor(self.nx, self.ny, self.nz, self.problem.name)
        monitor.init_figure()

        t_start = time.time()

        if verbose:
            print(f"\n{'Iter':>5}  {'p':>4}  {'Compliance':>14}  "
                  f"{'Volume':>8}  {'Change':>8}  {'Time':>7}")
            print("-" * 65)

        for iteration in range(1, max_iter + 1):
            rho_old = rho.copy()

            # Advance penalization
            for start_iter, p_val in penal_schedule:
                if iteration >= start_iter:
                    current_penal = p_val
            self.penal = current_penal

            # 1. Filter densities
            rho_f = apply_filter_3d(H, Hs, rho)

            # 2. Assemble SIMP stiffness (once per iteration)
            K, _ = assemble_simp_stiffness_3d(
                self.basis, self.elem_to_cell, rho_f,
                self.E0, self.E_min, self.penal, self.nu)

            # 3. Solve all load cases and accumulate compliance + sensitivity
            total_compliance = 0.0
            dc_drho_total    = np.zeros(self.n_cells)

            for f_vec in self.load_cases:
                u = self.solver.solve(K, f_vec, self.fixed_dofs)

                # Compliance for this load case
                total_compliance += float(f_vec @ u)

                # Sensitivity for this load case
                dc_k = compute_sensitivities_3d(
                    self.basis, u, self.elem_to_cell, self.n_cells,
                    rho_f, self.E0, self.E_min, self.penal, self.nu,
                    K0_local=self.K0_local)
                dc_drho_total += dc_k

            volume_frac = (rho_f * self.cell_volumes).sum() / self.cell_volumes.sum()

            # 4. Chain rule for filtered sensitivity
            dc_drho = apply_filter_sensitivity_3d(H, Hs, dc_drho_total)

            # 4b. Average sensitivities over last N iterations
            # Smooths oscillation from degenerate optima in 3D problems
            sens_history.append(dc_drho.copy())
            dc_drho = np.mean(sens_history, axis=0)

            # 5. OC update (uses averaged sensitivity)
            rho, _ = oc_update(rho, dc_drho, vol_frac, self.cell_volumes)

            # 6. Convergence check
            converged, change = check_convergence(rho, rho_old, tol)

            elapsed = time.time() - t_start
            if verbose:
                print(f"{iteration:>5}  {self.penal:>4.1f}  "
                      f"{total_compliance:>14.4f}  "
                      f"{volume_frac:>8.4f}  {change:>8.4f}  "
                      f"{elapsed:>6.1f}s")

            # 7. Update slice plots
            if iteration % self.update_interval == 0 or converged or iteration == 1:
                monitor.update(
                    rho_f, iteration, total_compliance,
                    volume_frac, change,
                    penal=self.penal,
                    vol_frac_target=vol_frac,
                    final=converged
                )

            if converged:
                if verbose:
                    print(f"\nConverged at iteration {iteration} "
                          f"(change={change:.4f} < tol={tol})")
                break
        else:
            if verbose:
                print(f"\nReached max_iter={max_iter} "
                      f"(final change={change:.4f})")

        # Final slice plot update
        monitor.update(rho_f, iteration, total_compliance,
                       volume_frac, change,
                       penal=self.penal, vol_frac_target=vol_frac, final=True)
        monitor.close()

        self.rho_final = rho_f
        return rho_f

    def show_result(self, threshold=0.5, save_path=None):
        """
        Show PyVista isosurface of converged density field.

        Args:
            threshold: density isosurface value (default 0.5)
            save_path: optional path to save screenshot
        """
        if not hasattr(self, 'rho_final'):
            raise RuntimeError("Run optimization first before calling show_result()")

        show_isosurface(
            self.rho_final,
            self.nx, self.ny, self.nz,
            self.mesh.p[0].max(),
            self.mesh.p[1].max(),
            self.mesh.p[2].max(),
            threshold=threshold,
            problem_name=self.problem.name,
            save_path=save_path
        )

    def save_slices(self, filename):
        """Save mid-plane slices of final density to PNG."""
        if not hasattr(self, 'rho_final'):
            raise RuntimeError("Run optimization first")
        save_density_slices(
            self.rho_final, self.nx, self.ny, self.nz, filename)

    def export_cad(self, basename='topology_result', threshold=0.5,
                   smooth_iter=50, formats=('stl', 'obj')):
        """
        Export optimized topology to CAD files.

        Args:
            basename:    output filename without extension
            threshold:   density isosurface (0.5 = material/void boundary)
            smooth_iter: Laplacian smoothing passes
            formats:     ('stl',) for printing, ('stl','obj') for Inventor,
                         add 'step' if pythonOCC is installed

        Returns:
            dict mapping format → output path
        """
        if not hasattr(self, 'rho_final'):
            raise RuntimeError("Run optimization first")
        from export.cad_export import export_all
        return export_all(
            self.rho_final,
            self.nx, self.ny, self.nz,
            self.mesh.p[0].max(),
            self.mesh.p[1].max(),
            self.mesh.p[2].max(),
            basename=basename,
            threshold=threshold,
            smooth_iter=smooth_iter,
            formats=formats
        )
