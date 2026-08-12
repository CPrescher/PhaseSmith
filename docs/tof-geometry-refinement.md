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

The component lattice and all 15 instrument derivative families retain their
independent NumPy and centered finite-difference gates. The joint test adds
simultaneous synthetic recovery and typed cross-family correlation assertions.
The realistic benchmark performs one complete joint cycle for two banks, 80
reflections per bank, and 4,001 samples per bank; it measured 214.18 ms on the
review machine.

This native workflow is the numerical contract required before adding a Python
application facade, durable project records, or a multi-bank real-data oracle.
It remains Le Bail extraction with fixed reflection topology, not structural
TOF Rietveld.
