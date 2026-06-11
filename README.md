# Topology Optimization Engine

A from-scratch Python implementation of SIMP topology optimization, built to
develop a deep understanding of FEA and physics-engine internals rather than
to wrap an existing commercial tool. Long-term goal: a multiphysics-capable
inverse-design system in the spirit of Leap 71's Noyron architecture
(physics-driven, not generative-AI-driven).

## Status

- ✅ 2D FEA pipeline — validated against analytical benchmarks
- ✅ 2D SIMP topology optimization — live matplotlib monitoring
- ✅ 3D tetrahedral pipeline — converged results (cantilever, MBB, bridge, etc.)
- ✅ 3D hexahedral pipeline — faster, no volumetric locking (see below)
- ✅ Multi-load case support
- ✅ Iterative (PyAMG-preconditioned CG) solver
- ✅ STL/OBJ export via marching cubes + PyVista
- 🔄 GCMMA-style optimizer — implemented and validated in isolation, but
  **does not clearly outperform OC on the problems it was meant to fix**
  (see [MMA section](#oc-vs-mma) below — this is reported honestly, not
  glossed over)
- ⏳ STEP export (pythonOCC) — planned
- ⏳ FEniCS multiphysics backend — optional/future

## Repository structure

```
core/           mesh generation (tet + hex), FEM solvers, Lamé parameters
optimization/   SIMP material model, sensitivity analysis, filters,
                density update rules (OC, MMA)
problems/       boundary-value problem definitions (cantilever, MBB,
                L-bracket, torsion, edge-load, multi-load — tet and hex)
export/         CAD export (STL/OBJ/STEP) from density fields
visualization/  live optimization monitor (matplotlib) and final
                isosurface rendering (PyVista)
examples/       runnable scripts (ex01...ex05) demonstrating each stage
tests/          validation tests, including OC vs MMA comparison
```

## From Tet to Hex: why, and what changed

The pipeline was originally built on **linear tetrahedral (P1) elements**
because tet meshing is simple and `scikit-fem`'s `MeshTet` made early
validation straightforward. Once the 3D pipeline was working and validated
(cantilever, MBB, bridge — physically correct topologies confirmed visually
and against analytical deflection/stress benchmarks), it became clear tets
have two practical costs at this mesh resolution:

1. **Volumetric locking.** P1 tets are overly stiff in
   bending-dominated and near-incompressible problems, requiring finer
   meshes to get accurate displacements — directly inflating element count.
2. **elem_to_cell mapping overhead.** Structured topology optimization
   design grids are naturally voxel-based (one density variable per cubic
   cell), but a structured tet mesh splits each cube into multiple
   tetrahedra (commonly 5 or 6). Every sensitivity and stiffness
   computation then needs an `elem_to_cell` mapping array and an
   `np.add.at` scatter-accumulate to go from per-tet quantities back to
   per-cell densities — extra indexing, extra memory, extra opportunities
   for off-by-one bugs.

**Hexahedral (Q1, trilinear) elements** solve both:

- One hex element *is* one design cell — `elem_to_cell` is the identity
  mapping, accumulation is a direct array assignment, not a scatter.
- ~6× fewer elements than an equivalent tet mesh at the same cell
  resolution, with *better* accuracy (no volumetric locking) — net result
  is roughly 6× faster sensitivity computation for the same design
  resolution.
- Quadrature: Q1 hex elements are exact at `intorder=2` (8 Gauss points).
  scikit-fem's default is `intorder=None` → 64 points, an 8× waste that's
  easy to miss if you don't override it explicitly.

### What had to change going from tet to hex

This was **not** a drop-in element swap. Things that changed:

- **Mesh construction**: `MeshTet` → `skfem.MeshHex.init_tensor(xs, ys, zs)`.
  (Early on this was mistakenly attempted via a non-existent `init_sqmesh`
  — the correct constructor is `init_tensor`.)
- **Basis/quadrature**: explicit `intorder=2` everywhere a hex basis is
  created — the 64-point default is correct but wasteful for Q1.
- **K0 cache shape**: `(n_elem, 24, 24)` for hex (8 nodes × 3 DOF) vs the
  tet equivalent's per-tet matrices that then get mapped to cells.
- **Sensitivity accumulation**: direct 1:1 assignment
  (`dc_drho = -dE_drho * strain_energy`, no `np.add.at`) — this is the
  single biggest code-simplification from the tet version.
- **`scikit-fem` 12.x API quirk**: `ddot()` in the heterogeneous elasticity
  bilinear form takes 2 arguments, not 3, in this version — a breaking
  change from some older examples/tutorials.
- **Lamé parameters**: 3D problems use full 3D Lamé parameters
  (`lam = E*nu / ((1+nu)(1-2nu))`, `mu = E / (2(1+nu))`), **not** the 2D
  plane-stress form. Mixing these up gives plausible-looking but
  physically wrong stiffness.
- **`condense()` return signature**: returns 4 values
  (`K_cond, f_cond, _, I_free`) in this scikit-fem version, not 2 — used
  with `expand=True` to reconstruct the full DOF vector after solving the
  reduced system.
- **Domain aspect ratios**: for MBB and bridge problems, the aspect ratio
  must be a **fixed constant** (3.0), not computed as `3*(nx/ny)` — the
  latter silently produces the wrong physical domain shape if `nx/ny`
  isn't already an integer ratio matching the intended geometry.

### CAD export: a worked debugging example (axis ordering)

A concrete illustration of how easy it is to get 3D array layouts subtly
wrong: the STL export pipeline (`density_to_mesh` in `cad_export.py`)
initially produced geometry that, while superficially "blob-like" and
plausible, had **different topology** (different hole count, different
pillar connectivity) from the validated PyVista isosurface render of the
same density field.

Root cause: the flat density array `rho_cells` is indexed with
`i = iz*(nx*ny) + iy*nx + ix` (x fastest, z slowest — see
`filtering_3d.py`). The PyVista visualization (`show_isosurface`) handled
this correctly:
```python
rho_3d = rho_cells.reshape(nz, ny, nx)
grid.cell_data['density'] = rho_3d.transpose(2, 1, 0).flatten(order='F')
```
but the marching-cubes export path used `rho_cells.reshape(nx, ny, nz)`
directly — a C-order reshape that implicitly assumes `ix` is the
**slowest**-varying index, the opposite of the actual layout. The fix
mirrors the PyVista path:
```python
grid = rho_cells.reshape(nz, ny, nx).transpose(2, 1, 0)
```
After this fix, STL export topology matches the PyVista isosurface
exactly. This kind of bug is dangerous precisely because the *output looks
like a reasonable topology-optimized part* — only a side-by-side
comparison against an independently-validated reference (the PyVista
render) caught it. Lesson: any time the same field is consumed by two
different libraries with different axis conventions, write the
reshape/transpose with an explicit comment citing the indexing convention,
and validate against a reference, don't trust "it looks plausible."

## OC vs MMA

### Why MMA was considered

The Optimality Criteria (OC) update is a simple, fast heuristic
(bisection on a Lagrange multiplier + fixed move limit). It works well for
compliance minimization with a single volume constraint, but on certain
problems — observed here on **edge-load cantilever** and **torsion beam** —
the convergence metric (`max|rho_new - rho_old|`, "change") oscillates
rather than decaying smoothly toward the tolerance, even after the
compliance objective has visually flattened.

The Method of Moving Asymptotes (MMA, Svanberg 1987) and its GCMMA
extension (Svanberg 2002) are the standard alternatives in the topology
optimization literature, specifically because their moving asymptotes can
adapt to oscillatory vs monotonic design trajectories on a per-variable
basis.

### What was implemented

`optimization/mma.py` implements:

- The standard MMA separable convex subproblem (rational `p/(U-x) +
  q/(x-L)` approximation per variable, built from the objective and
  constraint gradients).
- A **closed-form primal recovery** `x*(λ)` for given dual variable λ.
- A **bisection-based dual solve for a single constraint** (m=1, volume),
  directly analogous to OC's own bisection — exact for m=1.
- An **adaptive asymptote update** (the "GC" part): asymptote distance for
  each design variable shrinks (×0.7) if its density history shows a sign
  change (oscillation) between consecutive steps, and expands (×1.2) if
  monotonic.

This was validated with three unit/integration tests
(`tests/test_mma.py`), all passing:

1. **Linear program bang-bang test**: for a problem with constant
   sensitivities (a linear objective), MMA's converged design exactly
   matches the analytically-known bang-bang LP optimum (rank cells by
   cost, fill to volume budget).
2. **Consistency with OC**: on a smooth synthetic sensitivity field, both
   OC and MMA satisfy the volume constraint and rank cell densities by
   sensitivity in the same way (correlation ≈ -0.88 for both).
3. **Asymptote adaptation mechanism**: a direct unit test of the
   shrink/expand logic confirms oscillating-history cells get tighter
   asymptotes and monotonic-history cells get looser ones, as designed.

These tests confirm the **dual solver, primal recovery, and asymptote
adaptation logic are each individually correct**.

### What was NOT implemented (scope limitations)

- This is **MMA with adaptive asymptotes, not full GCMMA**. Full GCMMA
  includes a second convexity parameter and an inner loop that re-solves
  the subproblem if the convex approximation underestimates the true
  constraint, providing a monotonic-convergence guarantee. That inner loop
  is absent here.
- The dual solver is **only correct for a single constraint (m=1)**. The
  interface accepts `(m, n)` constraint-gradient arrays for future
  extension (e.g. adding a stress constraint), but the bisection approach
  used does not generalize to `m > 1` — a proper coupled dual solve (e.g.
  primal-dual Newton) would be needed first.

### Result on the target problems: inconclusive

Running OC and MMA head-to-head on edge-load cantilever and torsion beam
(20×10×10 hex mesh, 150 iterations, `tol=0.005`):

| Case      | Final compliance (OC) | Final compliance (MMA) | Final `change` (OC) | Final `change` (MMA) |
|-----------|------------------------|--------------------------|------------------------|--------------------------|
| Edge-load | 208.02                 | 206.36                   | 0.0364                 | **0.1972**               |
| Torsion   | 8.015                  | 8.169                    | 0.1307                 | 0.0855                   |

Final compliance is essentially the same for both methods on both problems
(MMA marginally better on edge-load, marginally worse on torsion — within
noise). Neither method's `change` metric decays cleanly to the 0.005
tolerance within 150 iterations; both spend long stretches with `change`
pinned at the move-limit ceiling (0.2), particularly visible on torsion as
a persistent high-frequency sawtooth in OC's `change` curve from iteration
~60 onward.

**Interpretation, stated plainly**: MMA as implemented here did not resolve
the oscillation/non-convergence behavior on these two problems, and on
edge-load its `change` metric is *worse* than OC's. The most likely
explanations, in rough order of suspicion:

1. The observed `change=0.2` plateaus are the **move-limit box being
   actively hit**, not asymptote-driven oscillation — the adaptive
   asymptote mechanism (which reacts to sign changes in density history)
   doesn't address a design variable that's monotonically pushing against
   its move limit every iteration.
2. 150 iterations may simply be too few for either method to fully
   converge on these problems at this resolution — the compliance curves
   are nearly flat by iteration ~80 for both methods, suggesting the
   *design* may be close to converged even though the `change` metric
   (driven by a small number of cells, possibly near `rho_min` or `1.0`,
   or in a checkerboard pattern) is not.
3. It's possible the oscillation is a mesh/filter-resolution artifact
   (e.g. checkerboarding) that neither optimizer's update rule addresses —
   in which case the fix is in the filter or mesh, not the optimizer.

**Status**: MMA implementation is correct and validated in isolation
(tests 1-3 above), and is available as an option (`optimizer='mma'`,
default remains `'oc'` — no breaking change to existing scripts). Whether
to invest further in GCMMA's full inner-loop correction, or instead
investigate (1)-(3) above via longer runs and spatial visualization of
which cells are driving `change`, is an open decision — see
`tests/compare_oc_mma.py` for the comparison harness used to produce the
table above.

## Key implementation gotchas (running list)

These were each non-obvious bugs encountered during development —
recorded here so they aren't re-discovered the hard way:

- `scikit-fem` 12.x `ddot()` takes 2 args, not 3.
- 2D problems need plane-stress Lamé parameters; 3D needs full 3D Lamé
  parameters. These are **not interchangeable**.
- Hex mesh constructor is `MeshHex.init_tensor`, not `init_sqmesh`
  (doesn't exist).
- `condense()` returns 4 values (`K, f, _, I_free`) with `expand=True`,
  not 2.
- MBB/bridge domain aspect ratio must be the fixed constant `3.0`, not
  `3*(nx/ny)`.
- PyVista `contour()` requires **point data**; convert cell data first
  with `cell_data_to_point_data()`.
- PyVista screenshots require `off_screen=True` and an explicit
  `render()` call before `screenshot()`.
- `pyvista.MultiBlock.__getitem__` (`GetBlock`) requires a Python `int`
  index, not `np.intp`/`np.int64` — wrap `np.argmax(...)` results with
  `int(...)`.
- Marching-cubes input array axis order must match the flat density
  array's indexing convention (`i = iz*(nx*ny)+iy*nx+ix`) — see the CAD
  export worked example above. `reshape(nx,ny,nz)` is **wrong**;
  `reshape(nz,ny,nx).transpose(2,1,0)` is correct.

## Requirements

```
scikit-fem==12.0.1
numpy
scipy
matplotlib
pyvista
pyamg
trimesh / scikit-image  (boundary extraction, marching cubes)
gmsh                     (meshing, future use)
pythonocc-core           (planned, STEP export)
```

## Running examples

```
python examples/ex01_2d_cantilever.py
python examples/ex03_3d_cantilever.py
python examples/ex05_cad_export.py --problem cantilever_hex --threshold 0.4
```

See `examples/` for the full set and `tests/test_mma.py`,
`tests/compare_oc_mma.py` for validation and the OC/MMA comparison.
