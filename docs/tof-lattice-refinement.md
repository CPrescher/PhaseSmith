# Shared TOF lattice refinement

`phasesmith-workflows` can refine one or more setting-aware unit cells against
two or more TOF detector banks while keeping empirical extraction state local
to each bank. This extends the atomic multi-bank Le Bail contract; it is not a
sequence of independent single-bank fits.

## Shared model and local state

`TofMultiBankLatticeInput` combines the validated `TofMultiBankInput` with an
ordered set of `TofSharedLatticePhase` records. Each selected phase supplies a
stable phase ID, a symmetry-aware `LatticeParameterization`, finite
`LatticeBounds`, and an initial cell. The same selected cell and HKL topology
are used in every bank, and the initial cell must reproduce the supplied bank
d-spacings exactly. Grids, instruments, masks, uncertainties, phase scales,
backgrounds, and extracted reflection intensities remain bank-local.

Every supported crystal system uses only its independent cell variables. For
reflection `h` and reciprocal metric `G*`,

```text
q = h^T G* h
d = q^(-1/2)
d(d)/dp = -(1/2) q^(-3/2) d(q)/dp

tof_b(d) = zero_b + difC_b d + difA_b d^2 + difB_b / d
d(tof_b)/dp = (difC_b + 2 difA_b d - difB_b / d^2) d(d)/dp
```

The reciprocal-metric derivative is shared. Each bank contributes its own
calibration chain and fused profile derivative.

## One accepted cycle

`refine_tof_multibank_lattice` alternates the existing joint Le Bail update
with one shared bounded cell step:

1. redistribute every bank's nonnegative reflection intensities and solve its
   optional Chebyshev background;
2. assemble one uncertainty-weighted Jacobian from all included samples using
   the fused local `dY/dd` rows and analytical `dd/dp` chains;
3. solve the diagonally damped normal equations in scaled cell coordinates,
   clip the step to both the declared bounds and maximum scaled step, and
   backtrack until the concatenated chi-square decreases;
4. publish the bank-local state and shared cells together.

Invalid intermediate cell/profile geometry is a rejected backtracking trial,
not a partially accepted result. If no improving cell trial exists, the valid
Le Bail intensity/background candidate is accepted with a zero cell step.
Cancellation during any evaluation discards the whole in-progress cycle.

Aggregate reduced chi-square counts all bank-local intensities and refinable
background coefficients plus every shared lattice parameter exactly once.
Bank-local reduced chi-square values remain descriptive and do not subtract
the shared variables a second time.

## Finite-support derivative convention

The fused TOF kernel evaluates values and all local/global derivatives in the
same supported sample pass. The outer active sample block is fixed for one
Jacobian evaluation. Inside that block, the analytical derivative includes
the moving truncated-convolution integration limits and the support radius
`R = support_fwhm H`, including the TCH total-width chain. At an exact internal
clamp boundary the bound derivative is defined as zero. Finite-difference tests
therefore perturb away from outer support membership changes and internal
clamp kinks.

This correction changes derivatives only; calculated profile values retain
the established deterministic operation order. The realistic benchmark uses
200 reflections on 5,001 samples and evaluates both local and all 15 global
TOF derivative rows for `tail_log` 8 and 20. On the review machine, the
optimized moving-bound implementation measured 134.80 ms at `tail_log=8`
versus 132.96 ms before the correction (about 1.4%).

## Runtime, restart, and validation

`refine_tof_multibank_lattice_with_runtime` provides structured events,
bounded evaluations, cancellation, checkpoint sinks, and exact continuation.
`TofMultiBankLatticeCheckpoint` stores every accepted bank-local state, shared
cell, and complete history. Continuation revalidates bank/phase/reflection
identity, fixed-phase d-spacings, selected-cell geometry, bounds, backgrounds,
and contiguous history before restoring anything.

Deterministic synthetic coverage uses two banks with different nonuniform
grids, calibrations, masks, scales, backgrounds, and intensity truth. It
recovers a displaced cubic cell, checks the full fused lattice column against
centered cell differences, rejects missing/duplicate phase selection, and
proves cancelled continuation identical to an uninterrupted run. The lattice
geometry and supported-profile equations also have independent NumPy
references. GSAS-II has no golden fixture for PhaseSmith's joint-cycle
sequencing; future real-data oracle work must compare plain arrays and physical
parameters rather than treating optimizer histories as interchangeable.

This API does not yet refine bank-local instrument coefficients, generate new
reflection topology during a run, expose the shared-cell workflow through the
Python application facade/project format, or refine structural intensities.
The separate fixed-cell
[`tof-instrument-refinement.md`](tof-instrument-refinement.md) workflow is the
first bounded analytical instrument layer; joint cell/instrument motion remains
separate because its correlation contract needs an explicit gate.
