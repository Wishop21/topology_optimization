# Topology Optimization Engine

A from-scratch Python implementation of SIMP topology optimization, built to
understand FEA and physics-engine internals through implementation rather
than using a commercial tool.

## Status

- ✅ 2D FEA + SIMP topology optimization, validated against analytical benchmarks
- ✅ 3D tetrahedral pipeline (cantilever, MBB, bridge — validated)
- ✅ 3D hexahedral pipeline (faster, no volumetric locking)
- ✅ Multi-load case support
- ✅ PyAMG iterative solver
- ✅ STL/OBJ export (marching cubes + PyVista)
- 🔄 GCMMA-style optimizer — implemented and validated in isolation, but
  **inconclusive vs OC on the target problems** (see below)
- ⏳ STEP export (pythonOCC) — planned
- ⏳ FEniCS multiphysics backend — future

## Repository structure

```
core/           mesh generation (tet + hex), FEM solvers
optimization/   SIMP material model, sensitivities, filters,
                density updates (OC, MMA)
problems/       boundary-value problems (cantilever, MBB, L-bracket,
                torsion, edge-load, multi-load — tet and hex)
export/         CAD export (STL/OBJ/STEP) from density fields
visualization/  live monitor + final isosurface rendering
examples/       runnable demo scripts
tests/          validation tests, OC vs MMA comparison
```

## Tet → Hex

Switched from tetrahedral (P1) to hexahedral (Q1) elements because:

- Tets have volumetric locking, requiring finer meshes for the same accuracy.
- A structured hex mesh has **one element per design cell** — no
  `elem_to_cell` mapping or scatter-accumulate needed for sensitivities.
- ~6x fewer elements than an equivalent tet mesh, with better accuracy.
- Hex is exact at `intorder=2` (8 quadrature points); scikit-fem's default
  (`intorder=None`) uses 64 — an 8x waste.

This wasn't a drop-in swap — mesh construction, basis setup, K0 cache shape,
sensitivity accumulation, and Lamé parameters (3D, not plane-stress) all
needed updating. See "Gotchas" below for specific bugs hit along the way.

## OC → MMA

OC (Optimality Criteria) works well generally, but on **edge-load
cantilever** and **torsion beam**, the convergence metric (`change`,
max density change per iteration) oscillates instead of decaying smoothly,
even after compliance has visually flattened.

MMA (Method of Moving Asymptotes) with adaptive asymptotes (the GCMMA
heuristic: shrink asymptotes for oscillating cells, expand for monotonic
ones) is the standard fix for this in the literature.

**What's implemented** (`optimization/mma.py`):
- Standard MMA separable convex subproblem + closed-form primal recovery
- Bisection-based dual solve (correct for a single constraint — volume)
- Adaptive asymptote shrink/expand based on density-history sign changes

**Validated** (`tests/test_mma.py`, all passing):
- Matches the analytical bang-bang optimum on a linear-objective toy problem
- Consistent with OC on a smooth synthetic sensitivity field
- Asymptote adaptation mechanism confirmed to shrink for oscillating
  cells and expand for monotonic ones

**Result on the actual target problems: inconclusive.**

| Case      | Final compliance (OC) | Final compliance (MMA) | Final `change` (OC) | Final `change` (MMA) |
|-----------|------------------------|--------------------------|------------------------|--------------------------|
| Edge-load | 208.02                 | 206.36                   | 0.0364                 | **0.1972**               |
| Torsion   | 8.015                  | 8.169                    | 0.1307                 | 0.0855                   |

Compliance is essentially the same for both methods. Neither converges
`change` below tolerance in 150 iterations — both spend long stretches
pinned at the move-limit (0.2), and on edge-load MMA's `change` is *worse*
than OC's.

**Most likely explanations** (untested):
1. The `change=0.2` plateaus may be the **move-limit box** being hit, not
   asymptote-driven oscillation — adaptive asymptotes don't help if a cell
   is monotonically pushing against its move limit every iteration.
2. 150 iterations may be too few — compliance looks converged by ~iteration
   80 even though `change` isn't.
3. Could be a mesh/filter checkerboarding artifact independent of the
   optimizer.

**Bottom line**: MMA is implemented correctly and available
(`optimizer='mma'`, default remains `'oc'`, no breaking changes), but
hasn't yet been shown to fix the original problem. `tests/compare_oc_mma.py`
is the comparison harness.

## Limitations / open issues

- MMA dual solve only handles a single constraint (volume). Adding a second
  constraint (e.g. stress) requires a proper coupled dual solve, not more
  bisection.
- MMA implements adaptive asymptotes only, not full GCMMA (no inner-loop
  convexity correction / monotonic convergence guarantee).
- Edge-load and torsion convergence behavior is unresolved (see above).
- STEP export not yet implemented (falls back to STL).
- `analysis.py` contains dead/broken code (2D-only stress recovery mixed
  into the 3D codebase, references an undefined `self` in a free function)
  — not on any active import path, but should be cleaned up or removed.

## Gotchas (bugs hit along the way)

- `scikit-fem` 12.x `ddot()` takes 2 args, not 3.
- 2D needs plane-stress Lamé parameters; 3D needs full 3D Lamé parameters —
  not interchangeable.
- Hex mesh constructor is `MeshHex.init_tensor`, not `init_sqmesh`.
- `condense(..., expand=True)` returns 4 values, not 2.
- MBB/bridge aspect ratio must be the fixed constant `3.0`, not `3*(nx/ny)`.
- PyVista `contour()` needs point data — use `cell_data_to_point_data()` first.
- PyVista screenshots need `off_screen=True` and `render()` before `screenshot()`.
- `pyvista.MultiBlock[idx]` needs a Python `int`, not `np.intp` —
  wrap `np.argmax(...)` with `int(...)`.
- Marching-cubes input must match the flat density array's axis order
  (`i = iz*(nx*ny)+iy*nx+ix`): use `rho.reshape(nz,ny,nx).transpose(2,1,0)`,
  not `reshape(nx,ny,nz)` — the latter silently produces wrong topology
  that still *looks* plausible.

## Requirements

```
scikit-fem==12.0.1
numpy, scipy, matplotlib
pyvista
pyamg
trimesh / scikit-image
gmsh
pythonocc-core   (planned)
```
