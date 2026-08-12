# Bank-local TOF instrument refinement

`phasesmith-workflows` refines explicitly selected calibration/profile
coefficients for one or more detector banks against the same atomic multi-bank
Le Bail objective. Instrument state remains bank-local: no coefficient is
silently shared merely because two banks observe the same specimen.

## Selection and bounds

`TofMultiBankInstrumentInput` combines a validated fixed-cell
`TofMultiBankInput` with one or more `TofBankInstrumentModel` records. Every
model names a stable bank ID and an ordered, non-empty set of
`TofInstrumentParameterBound` values. The selectable coefficients match the
15 fused global derivative rows exactly:

```text
zero, difc, difa, difb,
alpha,
beta0, beta1, betaq,
sigma0, sigma1, sigma2, sigmaq,
x, y, z
```

Bounds are finite, increasing, and must contain the initial coefficient. A
coefficient may occur only once per bank, and a bank model may occur only once.
Unselected coefficients remain bitwise fixed through refinement and restart.
The ordinary `TofLeBailInput` boundary now also evaluates every reflection's
derived alpha/beta rates and Gaussian/Lorentzian widths, so a nominally finite
instrument with a nonphysical profile combination fails before refinement.
The public `TofInstrumentParameter` enum, `TofInstrument::values`, and
`TofInstrument::from_values` bind selection order to the kernel Jacobian
without string or array-offset guesses.

## Analytical system

For each selected bank coefficient `p`, the fused kernel already evaluates

```text
dY/dp = sum_reflection I_h [
    dP/dtof * d(tof)/dp
  + dP/dalpha * d(alpha)/dp
  + dP/dbeta * d(beta)/dp
  + dP/dH_G * d(H_G)/dp
  + dP/dH_L * d(H_L)/dp
]
```

in the same supported sample pass as `Y`. The workflow selects the requested
dense rows, applies masks and uncertainty weights, and places each bank's
columns only on that bank's sample rows. It solves one diagonally damped system
in scaled physical coordinates, clips against declared bounds and the maximum
scaled step, and backtracks until the concatenated chi-square decreases.
Nonphysical trial profiles are rejected and halved; they do not abort or
partially advance the accepted bank set.

One accepted cycle publishes local extracted intensities, optional local
Chebyshev backgrounds, and every selected bank instrument together. If no
instrument trial improves the objective, the valid extraction candidate is
accepted with a zero instrument step. Aggregate reduced chi-square counts all
selected instrument coefficients once. Bank-local reduced chi-square remains
descriptive and does not subtract these joint-workflow parameters again.

## Identifiability

The result reports `TofInstrumentDiagnostics` for the final weighted selected
Jacobian:

- selected parameter count and numerical rank;
- the largest absolute normalized column correlation;
- every bank/parameter pair at or above the configured unresolved threshold.

Parameters from different banks have disjoint sample support and therefore
zero direct column correlation. Calibration terms within one bank can be
strongly correlated over a narrow d-spacing interval; the diagnostic is part
of the result rather than an optimizer warning that applications could lose.
Rank deficiency does not produce a fabricated covariance. The damped/SVD
solver may still return a bounded decrease, but callers must use the reported
rank and correlations when deciding whether the selected model is physically
identified.

## Runtime, restart, and validation

`refine_tof_multibank_instrument_with_runtime` checks cancellation and bounded
evaluation limits at atomic boundaries. Its checkpoint stores the accepted
instrument, phases/intensities, background, and complete history for every bank.
Continuation revalidates bank/phase/reflection identity, d-spacings, selected
bounds, bitwise-fixed unselected coefficients, physical derived profiles, and
background contracts before restoring state.

Synthetic tests recover an independent Zero shift in one bank and DIFC scale
in another, exercise same-bank correlation reporting, reject invalid
selections/bounds, return an exact one-reflection Zero/DIFC rank deficiency,
and prove cancelled continuation identical to an uninterrupted run. All 15
fused instrument rows have centered finite-difference
coverage and independent NumPy equation coverage, including moving finite-
support integration limits. The realistic workflow benchmark performs one
complete cycle for two banks, 80 reflections per bank, and 4,001 samples per
bank; it measured 426.04 ms on the review machine.

GSAS-II remains a pinned external oracle for coefficient translation and plain
profile arrays. Its optimizer history is not a golden reference for this joint
cycle ordering. A provenance-complete multi-bank real-data refinement with
reviewed parameter selection is required before promoting instrument motion
through the Python application facade or native project format.

This first instrument workflow keeps the supplied d-spacings fixed. The
separate [`tof-geometry-refinement.md`](tof-geometry-refinement.md) workflow
moves shared cells and local instruments in one correlation-diagnosed system.
Application persistence/facades and structural TOF Rietveld were separate
follow-on increments; both are now complete under the contracts documented in
[`tof-geometry-refinement.md`](tof-geometry-refinement.md) and
[`tof-structural-readiness.md`](tof-structural-readiness.md).
