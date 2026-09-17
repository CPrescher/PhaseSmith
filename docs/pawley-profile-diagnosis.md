# Pawley oracle profile discrepancy: diagnosis

The fixed-cell discrepancy is dominated by the **pinned GSAS-II FCJ axial-profile
calculation**, with a smaller contribution from different finite peak cutoffs.
It is not caused by the Pawley optimizer, F-squared-to-area conversion, or
Gaussian/Lorentzian width translation. The production equations are unchanged.

This investigation uses the original immutable 53-family Pawley fixture and
new black-box controls against the same GSAS-II revision and compiled binary.
The controls call the pinned profile API and its reported peak-support API;
no GSAS-II implementation code is copied or translated. The output is retained
under `oracle/diagnostics/pawley-profile-controls-20260917`, with source,
adapter, archive and binary hashes. The executable report is
`validation/results/pawley-20260917-profile-diagnosis.json`.

## Separating the effects

All full-pattern numbers below use the same denominator as the original
comparison, including its constant background of 1. Subtracting that background
from the denominator gives a peak-signal discrepancy of **0.3554%**, compared
with **0.03691%** for the full pattern. These are relative L2 errors, not a
maximum pointwise error or uncertainty estimate.

| Control | Relative L2, normalized to the original full pattern |
| --- | ---: |
| Original PhaseSmith versus GSAS-II histogram | 3.69131e-4 |
| PhaseSmith versus untruncated GSAS-II FCJ profiles | 3.68969e-4 |
| GSAS-II untruncated profiles versus its reported cutoffs | 1.11867e-5 |
| GSAS-II profiles with its reported cutoffs versus its histogram | 5.62354e-10 |
| Diagnostic per-peak translation/area adjustment plus matching cutoffs | 5.55682e-7 |

These are norms of separate residual vectors, so the numbers do not add as
scalars. The untruncated GSAS-II profiles, weighted by the existing converted
areas and restricted to its reported windows, reconstruct the original
histogram essentially exactly. This independently verifies the family-area
conversion and identifies the cutoff contribution. The native comparison used
10,000 FWHM support; GSAS-II uses much shorter asymmetric windows, approximately
50/75 FWHM plus its axial allowance. Neither convention is changed here.

## What differs in the axial profiles

At the fixture's formal SH/L=0 setting, the histogram uses the pinned floor
SH/L=0.002. PhaseSmith uses the published equal-height mapping S/L=H/L=0.001.
Zero-axial black-box profiles agree to **2.07e-6 relative L2 of peak signal**.
Turning on the same axial geometry produces the much larger discrepancy.

Most of that difference looks like a small additional low-angle translation
in the pinned GSAS-II profile. A two-column diagnostic fit of each native
profile and its analytical position derivative gives additional translations
between **-7.77e-5 and -1.56e-5 degrees**. Allowing those translations and small
area adjustments reduces the axial-profile mismatch about 650-fold; matching
cutoffs brings the full histogram discrepancy below 1e-6.

This fitted correction is only a diagnostic. It is not a physical recalibration,
not a production compatibility mode, and not a replacement for the unchanged
strict-equivalence acceptance check.

Controls exclude several alternative explanations:

- Stored Gaussian variances agree with translated instrument values within
  2.8e-20 degree²; Lorentzian widths agree within 1.1e-14 degrees. The latter
  comparison uses a local 2e-14-degree bound for stored-record roundoff.
- At four representative reflections, native FCJ agrees with independent
  128-point NumPy quadrature within 5.9e-13 relative L2. Increasing reference
  quadrature from 32 to 128 points changes the result by at most 3.1e-13.
- Increasing observation sampling from 401 to 40,001 points over the same
  isolated peak leaves the discrepancy at approximately 0.00500527. Sampling
  the observed grid more finely does not cure it.
- The retained axial/width sweep is not consistent with one constant geometry
  conversion factor: error changes non-monotonically with axial extent and
  depends on intrinsic width.

The evidence localizes the dominant numerical discrepancy to the pinned
compiled FCJ profile evaluator rather than PhaseSmith's quadrature resolution.
It does **not** separate the oracle's internal quadrature and floating-point
contributions or establish an exact defective line of code. “Different physical
models” was therefore too strong a description. The defensible classification
is a **pinned-oracle numerical profile discrepancy**, plus a known cutoff-policy
difference. The independent production integral follows
[Finger, Cox & Jephcoat (1994)](https://doi.org/10.1107/S0021889894004218) and the
[corrected derivative treatment](https://doi.org/10.1107/S0021889813016233);
its derivation is in [the FCJ guide](fcj-profile.md).

## Why overlapping areas differ

An independent NumPy linear least-squares solve using the GSAS-II profile basis
and its cutoffs recovers all 53 GSAS-II areas to **1.17e-8 relative error**.
The same solve using the native basis reproduces the roughly **1.16%** maximum
individual-area difference seen in the native Pawley optimizer. Thus the
optimizer is not responsible for that discrepancy. Near-overlapping basis
columns amplify small profile differences into larger changes in individual
areas, while their combined profile and area sum remain much closer.

A cell-refining fit can absorb part of the small apparent-position difference;
this is consistent with the previously recorded approximately 2 ppm cell-length
agreement. The detailed decomposition above uses the fixed-cell case so that
cell motion and optimizer behavior cannot obscure the source.

## Reproduction and outcome

Generate new external controls in the separate pinned oracle environment;
the generator refuses an existing destination:

```sh
python oracle/scripts/investigate_pawley_profiles.py \
  --gsas-root /path/to/pinned/GSAS-II --binary-dir /path/to/binaries \
  --output /new/controls
```

Then analyze them in the normal PhaseSmith environment, without GSAS-II:

```sh
python -m phasesmith.validation.pawley_profile_diagnostic \
  --fixture oracle/fixtures/pawley_optimizer_v1 \
  --controls oracle/diagnostics/pawley-profile-controls-20260917 \
  --output /new/report.json
```

Regression tests cover the decomposition, independent quadrature, grid control,
and revision guard. No numerical core, solver, production support convention,
original golden fixture, or existing acceptance tolerance was changed. Exact
GSAS-II equivalence remains unclaimed, but its failure now has an experimentally
isolated explanation. Altering production physics to imitate this particular
compiled oracle approximation is not justified by these controls.
