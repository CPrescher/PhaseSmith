# Joint TOF lattice and instrument refinement

`phasesmith-workflows` combines shared symmetry-independent cell variables and
selected bank-local instrument coefficients in one analytical multi-bank
system. This is the correlation-aware geometry layer: it does not alternate a
lattice-only fit with independent instrument fits.

## Joint parameter contract

`TofMultiBankGeometryInput` owns a validated `TofMultiBankLatticeInput` plus
the same explicit `TofBankInstrumentModel` selections used by fixed-cell
instrument refinement. The cell is shared by stable phase ID. Instrument
coefficients remain local to stable bank IDs. Reflection IDs/HKL topology is
fixed for the run; extracted intensities, backgrounds, grids, masks,
uncertainties, and phase scales remain bank-local.

The packed physical order is deterministic:

1. every selected setting-aware lattice variable in phase/model order;
2. every selected instrument coefficient in bank-model/bound order.

For each reflection and bank, the fused profile supplies both `dY/dd` and the
15 direct instrument rows. The shared lattice block is

```text
J_cell,b = dY_b/dd * dd/dp_cell
```

and the bank-local block selects `dY_b/dp_instrument` directly. The workflow
concatenates these blocks horizontally for the same masked, uncertainty-
weighted residual vector:

```text
J_joint = [ J_cell | J_instrument ]
```

An internal invariant requires the two independently assembled residual vectors
to be bitwise identical before the blocks can be joined.

## Solve and atomic acceptance

One scaled damped normal-equation/SVD step moves both families together. Every
component is clipped to its declared physical bounds and the common maximum
scaled step. Backtracking installs the complete candidate cell and instruments,
regenerates d-spacings without changing topology, evaluates every bank, and
accepts only a strict decrease in concatenated chi-square. Invalid cell or
profile geometry is a rejected half-step.

The preceding Le Bail redistribution/background candidate and the joint
geometry trial are published together. Cancellation cannot expose new
intensities with old geometry, a new cell with old instruments, or one advanced
bank. Aggregate degrees of freedom count all shared cell variables once and all
selected local instrument variables once.

## Correlation diagnostics

`TofGeometryDiagnostics` reports numerical rank, maximum absolute weighted-
column correlation, and every unresolved pair by a typed
`TofGeometryParameterKey`. Keys distinguish

- `Lattice { phase_id, parameter_name }`; and
- `Instrument { bank_id, parameter }`.

This makes the common TOF ambiguity visible: a cell change shifts all banks
through `d`, while Zero/DIFC/DIFA/DIFB can imitate parts of that motion within
one bank and d-range. The synthetic acceptance test deliberately requires a
reported lattice–instrument correlation while still recovering a common cubic
cell and two local Zero terms at full rank. Rank-deficient selections are
returned as diagnostics; no covariance is fabricated.

## Runtime, validation, and performance

`refine_tof_multibank_geometry_with_runtime` stores every accepted cell,
instrument, phase/intensity, background, and history row. Restart validation
checks cell bounds and phase order, expected d-spacings, selected instrument
bounds, bitwise-fixed unselected coefficients, physical derived profiles,
background contracts, and contiguous bank-aligned history. A cancelled
four-cycle checkpoint resumes identically to an uninterrupted twelve-cycle run.

Python applications use the same solver through
`phasesmith.refinement.refine_tof_multibank_geometry`. The facade accepts typed
`TofLeBailBank`, `TofSharedLatticePhase`, and `TofBankInstrumentModel` records.
The shared lattice reuses the exact native `SpaceGroup` topology already owned
by `LatticeParameterization`, including nonstandard setting checks. The GIL is
released for the complete native solve. Final bank arrays are read-only NumPy
arrays, diagnostics preserve typed family/owner/parameter keys, and the opaque
`TofMultiBankGeometryCheckpoint` can be passed back for exact continuation.

The component lattice and all 15 instrument derivative families retain their
independent NumPy and centered finite-difference gates. The joint test adds
simultaneous synthetic recovery and typed cross-family correlation assertions.
The realistic benchmark performs one complete joint cycle for two banks, 80
reflections per bank, and 4,001 samples per bank; it measured 214.18 ms on the
review machine.

Native project format 4 persists the same complete state through
`TofMultiBankGeometryProjectState`, `save_tof_multibank_geometry_project`, and
`load_tof_multibank_geometry_project`. One histogram can belong to at most one
joint analysis. The codec stores every bank input, shared-cell bound,
instrument selection, solver control, and accepted checkpoint array; restoring
the bundle revalidates it against project histograms, phase definitions, and
the exact symmetry setting.

## Real multi-bank acceptance

The checksum-pinned LANL nickel tutorial now exercises the complete native
joint path on detector banks 2, 3, and 4. Each bank keeps its 4,431 measured
constant-step bins, profile-function-1 calibration, uncertainty, Smooth
Bruckner baseline, and 12-term Chebyshev residual. The banks share one complete
102-family Fm-3m nickel topology over 0.2--3.0 A, one cubic cell, and refine one
local Zero coefficient each. Reflections outside a particular bank's finite
microsecond grid contribute no samples; the shared topology itself is not
truncated to the intersection of bank coverage.

Starting from 3.523 A, 20 deterministic accepted cycles use 13,293 included
observations. The reviewed run returns joint Rwp 0.02273306, bank Rwp values
0.02335845/0.02205717/0.02260671, and a=3.52361196 A against the published
3.5234 A value. The four analytical columns (one cell plus three local Zero
terms) have numerical rank four and expose lattice/instrument correlations.
Acceptance requires joint and per-bank Rwp <= 0.03, |a-3.5234 A| <= 0.0005 A,
and rank 4/4.

The pinned GSAS-II oracle builds one independent three-histogram project and
exports only plain arrays. Across banks 2--4, PhaseSmith reproduces GSAS-II's
reflection positions exactly, variance to floating-point precision, and alpha/
beta chains within 4.5e-16. Reconstructing each GSAS-II peak-only pattern from
the same extracted intensities gives correlations 0.99999898--0.99999922 and
relative L2 differences 0.00174246--0.00181021. The independently refined cells
are 3.52361196 A and 3.52386668 A, differing by 0.00025472 A and both passing
their declared reference-cell gates. The oracle uses temporary one-bank RAW
views because the pinned scripting loader otherwise reuses the first dataset
on repeated legacy multi-bank imports.

GSAS-II selects 4,430 samples per bank at the nominal limits while the
PhaseSmith inclusive bin-center convention selects 4,431; the oracle asserts
both rather than trimming one implementation silently. Workflow Rwp values are
reported but are not a parity gate because the Le Bail redistribution and
background decompositions differ. This geometry gate remains extraction with
fixed reflection identities; the separate structural TOF workflow is described
in [`tof-structural-readiness.md`](tof-structural-readiness.md).
