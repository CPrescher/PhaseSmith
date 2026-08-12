# Multi-bank TOF Le Bail

`phasesmith-workflows` provides an atomic fixed-cell multi-bank layer around the
validated single-bank TOF calculation. It is intended for detector banks that
observe one specimen/phase geometry through separate time grids and resolution
functions.

## Shared and local state

`TofMultiBankInput` contains two or more ordered `TofLeBailBank` records. Stable
bank IDs must be unique. The fixed-cell contract is shared exactly across all
banks:

- phase order, IDs, and labels;
- reflection IDs and Miller indices;
- d-spacings in ångströms.

The following remain bank-local:

- the bin-center `tof_us` grid and observations;
- uncertainty and inclusion arrays;
- all 15 calibration/profile coefficients;
- the fixed baseline and optional refinable Chebyshev background;
- phase scales and extracted integrated intensities.

This separation does not assume that detector efficiencies or bank-dependent
corrections make Le Bail intensities interchangeable. Shared lattice motion can
therefore be added later without falsely sharing empirical bank intensities.

## Atomic cycle

For every accepted redistribution cycle, the workflow performs three joint
model-evaluation stages:

1. calculate every bank at the current accepted state and form its
   observed/calculated redistribution ratios;
2. install every bank's nonnegative intensity candidate and solve each local
   analytical background;
3. calculate every complete candidate, evaluate local and aggregate residuals,
   then accept the entire bank set.

One runtime evaluation means one stage over all banks. Cancellation and budgets
are checked before each stage. If a stop arrives after a stage begins, all
candidate work from that cycle is discarded; no bank advances alone.

## Aggregate metrics

The joint residual convention is equivalent to concatenating all included bank
samples. With bank-local uncertainty weighting when requested,

```text
chi_square = sum_bank sum_included ((Ycalc - Yobs) / sigma)^2
Rwp = sqrt(chi_square / sum_bank sum_included (Yobs / sigma)^2)
Rp = sum_bank sum_included |Ycalc - Yobs|
     / sum_bank sum_included |Yobs|
```

For unit weights, `sigma` is one. Reduced chi-square subtracts every bank-local
reflection intensity and every refinable background coefficient from the total
included sample count. The result also retains each bank's complete
`ResidualEvaluation`, so hosts do not need to split flattened arrays.

## Runtime and continuation

Rust callers use `refine_tof_multibank` for the default bounded runtime or
`refine_tof_multibank_with_runtime` for cancellation, structured events, and
checkpoint sinks. `TofMultiBankCheckpoint` stores all accepted bank-local phase
intensities/backgrounds and the complete aggregate/local history. Continuation
requires the same bank order and IDs, shared reflection geometry, phase scales,
background identities/domains/orders, and a sufficient total cycle budget.

Focused tests cover different uniform/nonuniform grids, distinct instruments,
masks, uncertainties, scales, backgrounds, and intensity truth in two banks.
They also prove exact cancelled-state continuation and reject duplicate IDs or
one-bank d-spacing drift.

The current layer is fixed-cell Le Bail. It does not yet refine a common unit
cell, bank-local instrument coefficients, atomic structure, or structural
intensities. Those require separate analytical derivative and oracle gates.
