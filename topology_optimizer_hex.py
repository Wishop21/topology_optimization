"""
3D SIMP topology optimizer for hexahedral elements.

Identical loop structure to topology_optimizer_3d.py.
Key differences from tet version:

  1. elem_to_cell is trivial (element i = cell i) — no mapping array
  2. K0_local has shape (n_cells, 24, 24) — 6x fewer matrices than tet
  3. Sensitivity accumulation is direct (no np.add.at needed)
  4. Assembly uses intorder=2 (8 qp per element vs 64 default)

These combine to make sensitivities ~6x faster than tet equivalent.
"""

import numpy as np
import time
from collections import deque

from core.fem_solver_hex import LinearElasticitySolverHex
from optimization.simp_hex import (assemble_simp_stiffness_hex,
                                    compute_sensitivities_hex,
                                    build_k0_cache_hex)
from optimization.filtering_3d import (build_filter_matrix_3d,
                                        apply_filter_3d,
                                        apply_filter_sensitivity_3d)
from optimization.optimality_criteria import oc_update, check_convergence
from optimization.mma import init_mma_state, mma_update
from visualization.plot_3d import SliceMonitor, show_isosurface, save_density_slices


class TopologyOptimizerHex:
    """
    3D SIMP topology optimizer using hexahedral elements.

    For the same mesh resolution (nx×ny×nz) as the tet optimizer:
    - 6x fewer elements → faster assembly and sensitivity
    - No volumetric locking → better accuracy at coarser mesh
    - Same DOF count → same linear solve cost
    """

    def __init__(self, problem):
        """
        Args:
            problem: a hex Problem object with:
                mesh (MeshHex), nx, ny, nz, E, nu, vol_frac
                get_fixed_dofs(basis) → array
                get_load_cases(n_dofs) → list of force vectors
        """
        self.problem  = problem
        self.mesh     = problem.mesh
        self.nx, self.ny, self.nz = problem.nx, problem.ny, problem.nz
        self.n_cells  = self.nx * self.ny * self.nz

        self.solver = LinearElasticitySolverHex(
            self.mesh, E=problem.E, nu=problem.nu)
        self.basis  = self.solver.basis

        # Verify 1:1 element-to-cell for hex
        assert self.mesh.t.shape[1] == self.n_cells, (
            f"Hex mesh elements ({self.mesh.t.shape[1]}) != cells ({self.n_cells})")

        cell_w = self.mesh.p[0].max() / self.nx
        cell_h = self.mesh.p[1].max() / self.ny
        cell_d = self.mesh.p[2].max() / self.nz
        self.cell_volumes = np.full(self.n_cells, cell_w * cell_h * cell_d)

        self.E0    = problem.E
        self.E_min = problem.E * 1e-9
        self.nu    = problem.nu

        self.fixed_dofs = problem.get_fixed_dofs(self.basis)
        self.load_cases = problem.get_load_cases(self.basis.N)

        # Precompute K0 (unit-material element stiffness, reused every iteration)
        self.K0_local = build_k0_cache_hex(self.basis, problem.nu)

        self.update_interval = max(1, self.n_cells // 3000)

        print(f"Initialized Hex optimizer:")
        print(f"  Mesh     : {self.nx}×{self.ny}×{self.nz} cells")
        print(f"  Elements : {self.mesh.t.shape[1]}  (1:1 with cells — hex advantage)")
        print(f"  Nodes    : {self.mesh.p.shape[1]}")
        print(f"  DOFs     : {self.basis.N}")
        print(f"  n_qp     : {self.basis.X.shape[-1]} per element  (8, not 64)")
        print(f"  Loads    : {len(self.load_cases)} load case(s)")
        print(f"  K0 shape : {self.K0_local.shape}")

    def run(self, max_iter=300, tol=0.01, r_min=1.5, vol_frac=None,
            penal=3.0, penal_continuation=True, sens_avg=3, verbose=True,
            optimizer='oc', mma_move=0.2, show_monitor=True):
        """
        Run hex SIMP optimization.

        Same API as TopologyOptimizer3D.run() — drop-in replacement,
        plus:

        Args:
            optimizer: 'oc' (default, unchanged behaviour) or 'mma'
                       (GCMMA-style with adaptive asymptotes, see
                       optimization/mma.py). For 'mma', also consider
                       passing sens_avg=1 -- see optimization/mma.py
                       module docstring / PATCH notes for why the
                       sensitivity moving average can mask the
                       oscillation signal MMA's asymptotes react to.
            mma_move:  move-limit fraction for MMA's alpha/beta box
                       (analogous role to OC's fixed move=0.2, but
                       MMA's effective step is also constrained by
                       the adaptive asymptotes). Ignored if
                       optimizer='oc'.
            show_monitor: if False, skip the live matplotlib SliceMonitor
                       entirely (no init_figure/update/close, no blocking
                       plt.show at the end). Use for batch/scripted runs
                       (e.g. comparing optimizers back-to-back) where the
                       blocking plot window would halt the script.
                       self.history (compliance/volume/change per
                       iteration) is populated regardless.
        """
        if optimizer not in ('oc', 'mma'):
            raise ValueError(f"optimizer must be 'oc' or 'mma', got '{optimizer!r}'")

        if vol_frac is None:
            vol_frac = self.problem.vol_frac
        self.vol_frac = vol_frac

        if penal_continuation:
            penal_schedule = [(1, 1.0), (20, 2.0), (40, penal)]
        else:
            penal_schedule = [(1, penal)]
        current_penal = penal_schedule[0][1]

        if verbose:
            print(f"\nBuilding 3D filter ({self.nx}×{self.ny}×{self.nz}, r={r_min})...")
        H, Hs = build_filter_matrix_3d(self.nx, self.ny, self.nz, r_min)
        if verbose:
            print(f"Filter: {H.nnz} non-zeros")

        sens_history = deque(maxlen=max(1, sens_avg))
        rho = np.full(self.n_cells, vol_frac)

        # Convergence history (compliance, volume, max-density-change per
        # iteration) -- stored on self for post-run comparison/plotting.
        # Independent of the live SliceMonitor's internal history lists.
        self.history = {'compliance': [], 'volume': [], 'change': []}

        # MMA state init. rho_min=1e-3 here MUST match oc_update's default
        # rho_min (1e-3) -- both define the lower density bound for the
        # design variables. If you change one, change the other.
        if optimizer == 'mma':
            mma_state = init_mma_state(rho, rho_min=1e-3, rho_max=1.0)

        monitor = None
        if show_monitor:
            monitor = SliceMonitor(self.nx, self.ny, self.nz, self.problem.name)
            monitor.init_figure()

        t_start = time.time()

        if verbose:
            print(f"\n{'Iter':>5}  {'p':>4}  {'Compliance':>14}  "
                  f"{'Volume':>8}  {'Change':>8}  {'Time':>7}")
            print("-" * 65)

        for iteration in range(1, max_iter + 1):
            rho_old = rho.copy()

            for s, p in penal_schedule:
                if iteration >= s:
                    current_penal = p
            self.penal = current_penal

            rho_f = apply_filter_3d(H, Hs, rho)

            # Assemble (no elem_to_cell needed — hex is 1:1)
            K, _ = assemble_simp_stiffness_hex(
                self.basis, rho_f, self.E0, self.E_min, self.penal, self.nu)

            total_compliance = 0.0
            dc_drho_total    = np.zeros(self.n_cells)

            for f_vec in self.load_cases:
                u = self.solver.solve(K, f_vec, self.fixed_dofs)
                total_compliance += float(f_vec @ u)
                dc_k = compute_sensitivities_hex(
                    self.basis, u, self.n_cells, rho_f,
                    self.E0, self.E_min, self.penal, self.nu,
                    K0_local=self.K0_local)
                dc_drho_total += dc_k

            volume_frac = (rho_f * self.cell_volumes).sum() / self.cell_volumes.sum()

            dc_drho = apply_filter_sensitivity_3d(H, Hs, dc_drho_total)

            sens_history.append(dc_drho.copy())
            dc_drho = np.mean(sens_history, axis=0)

            if optimizer == 'oc':
                rho, _ = oc_update(rho, dc_drho, vol_frac, self.cell_volumes)
            else:  # mma
                rho, mma_state = mma_update(
                    rho, dc_drho, vol_frac, self.cell_volumes,
                    mma_state, move=mma_move)

            converged, change = check_convergence(rho, rho_old, tol)
            elapsed = time.time() - t_start

            self.history['compliance'].append(total_compliance)
            self.history['volume'].append(volume_frac)
            self.history['change'].append(change)

            if verbose:
                print(f"{iteration:>5}  {self.penal:>4.1f}  "
                      f"{total_compliance:>14.4f}  "
                      f"{volume_frac:>8.4f}  {change:>8.4f}  "
                      f"{elapsed:>6.1f}s")

            if monitor is not None and (iteration % self.update_interval == 0 or converged or iteration == 1):
                monitor.update(
                    rho_f, iteration, total_compliance,
                    volume_frac, change,
                    penal=self.penal, vol_frac_target=vol_frac,
                    final=converged)

            if converged:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break
        else:
            if verbose:
                print(f"\nReached max_iter={max_iter} (change={change:.4f})")

        if monitor is not None:
            monitor.update(rho_f, iteration, total_compliance,
                           volume_frac, change,
                           penal=self.penal, vol_frac_target=vol_frac, final=True)
            monitor.close()

        self.rho_final = rho_f
        return rho_f

    def show_result(self, threshold=0.5, save_path=None):
        """Show PyVista isosurface of final result."""
        if not hasattr(self, 'rho_final'):
            raise RuntimeError("Run optimization first")
        show_isosurface(
            self.rho_final,
            self.nx, self.ny, self.nz,
            self.mesh.p[0].max(),
            self.mesh.p[1].max(),
            self.mesh.p[2].max(),
            threshold=threshold,
            problem_name=self.problem.name,
            save_path=save_path)

    def save_slices(self, filename):
        """Save mid-plane density slices to PNG."""
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
            smooth_iter: Laplacian smoothing passes to remove voxel staircase
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
