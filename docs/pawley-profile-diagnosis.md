# Pawley oracle profile discrepancy: diagnosis

The fixed-cell discrepancy is dominated by the **pinned GSAS-II FCJ axial-profile
calculation**, with a smaller contribution from different finite peak cutoffs.
It is not caused by the Pawley optimizer, F-squared-to-area conversion, or
Gaussian/Lorentzian width translation. A subsequent direct angular-integral
audit found and corrected a much smaller geometric-factor error in PhaseSmith;
see the independent equation audit below. It does not explain the main mismatch.

This investigation uses the original immutable 53-family Pawley fixture and
new black-box controls against the same GSAS-II revision and compiled binary.
The controls call the pinned profile API and its reported peak-support API;
no GSAS-II implementation code is copied or translated. The output is retained
under `oracle/diagnostics/pawley-profile-controls-20260917`, with source,
adapter, archive and binary hashes. The executable report is
`validation/results/pawley-20260917-profile-diagnosis.json`.

## Separating the effects

The decomposition and metrics in this section record the original September 17
investigation, before the September 18 geometric-factor correction.

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

## Independent equation audit and native correction

On September 18, the original angular density in FCJ equations 1 and 4–8 was
evaluated directly using mpmath adaptive tanh-sinh integration at 35 and 60
decimal digits. This reference does not use the height substitution or
Gauss-Legendre quadrature shared by the Rust and NumPy implementations. It
normalizes the angular convolution explicitly and evaluates the intrinsic TCH
profile separately in arbitrary precision. The published dimensions are
sample height `2S` and detector opening `2H`; the existing half-height mapping
is correct. See [FCJ (1994)](https://doi.org/10.1107/S0021889894004218).

This audit found a real PhaseSmith error. With `z` the axial separation divided
by radius, the angular density is proportional to `W(z)/(z cos(a))`.
Multiplication by the angular Jacobian gives
`W(z)/[(1+z²) sin(a)]`, not `W(z)/[sqrt(1+z²) sin(a)]`.
The missing square-root factor affected both Rust and the NumPy reference,
which explains why their mutual comparisons did not expose it. The kernel,
reference, analytical height derivatives and equation documentation are now
corrected together. Normalization and support conventions are preserved.

The retained before/after reports are
`validation/results/fcj-angular-20260918-before.json` and
`validation/results/fcj-angular-20260918-corrected.json`. They compare seven
observation points around each of four fixture reflections, rather than a
whole-grid error norm. Before correction, native errors were 2.4e-10–1.4e-9
relative L2 at those samples, while pinned-oracle errors were
9.23e-4–4.93e-3. After correction, native errors are below 8.2e-13;
the 35- and 60-digit references round to the same float64 sample values.
Thus there were **two issues**: a small native formula error
and a much larger discrepancy in this pinned oracle's evaluation of the same
published model. Fixing our error does not remove the latter.

Regression tests also compare equal and unequal larger axial dimensions with
a Gaussian-only intrinsic profile and verify the reflected high-angle case.
The calibration tool's separate point-source/point-sample FCJ branch receives
the same correction, without relaxing its equality test. Its general
finite-source ray model is outside this FCJ audit's scope.
The high-precision audit is an offline validation dependency, never a runtime
dependency. No GSAS-II implementation source was used to derive this correction.
This establishes agreement with the published integral for the tested cases;
the subsequent source audit below identifies the quadrature cause. Neither
audit makes a claim about every GSAS-II version or diffraction geometry.

Validation after correction: 362 Rust tests and 941 Python tests pass (11
external-data tests skipped and 33 opt-in tests deselected); strict Clippy,
formatting, targeted Ruff and the strict documentation build pass. The existing
200-peak, 5,001-sample benchmark with two warmups and nine repetitions shows no
median regression: CW FCJ 2.249→2.195 ms, neutron FCJ 2.225→2.191 ms, and
doublet FCJ 4.519→4.492 ms. These small differences are timing noise, not a
speedup claim. Raw output is retained in
`validation/results/fcj-angular-20260918-benchmark.json`.

Reproduce the direct equation audit with the test dependencies installed:

```sh
python -m phasesmith.validation.fcj_angular_reference \
  --fixture oracle/fixtures/pawley_optimizer_v1 \
  --controls oracle/diagnostics/pawley-profile-controls-20260917 \
  --output /new/angular-audit.json
```

## Source audit: the pinned path uses one convolution node

Reading the pinned GSAS-II source is compatible with the clean implementation
boundary: no source was copied or translated into PhaseSmith. The actual call
chain is `getFCJVoigt3` → `PYPSVFCJO` → `PSVFCJO`. The separate newer
`psvfcj.f90` routine is not the function called by this histogram path.
The wrapper divides `(S+H)/L` equally between `S/L` and `H/L`, confirming the
geometry mapping used in the independent audit.

In the pinned
[PSVFCJO source](https://github.com/AdvancedPhotonSource/GSAS-II/blob/c0bc79b259cdf0065480b5fbd57674ddf12c4a23/sources/powsubs/psvfcjo.for#L172),
the table selector truncates `300 × axial_span_degrees / gamma_centidegrees`
to an integer. This indicator ranges from 0.04583 to 0.41287 over all 53
fixture reflections, so every reflection selects the first, two-node table.
The convolution loop at lines 218–221 uses only its positive half: **one node**.
At lines 277–279 the normalization cancels that single node's weight, leaving
one intrinsic pseudo-Voigt profile displaced to the selected apparent angle.
This is an under-resolved approximation to the FCJ convolution, not an
alternative physical model. The selector's unit combination is recorded as
implemented; this audit does not claim an unverified unit-conversion fix.

An independent consequence of a two-point Legendre rule is that its positive
node is `1/sqrt(3)`. If the axial angular span is `d`, the single-node profile
is shifted by `-d(1-1/sqrt(3))`. In contrast, the small-height, equal-height
FCJ distribution has leading-order mean shift `-d/6`. For the first fixture
reflection these are approximately -1.2216e-4 and -4.8171e-5 degrees,
explaining the previously measured excess low-angle translation.

To separate this from floating-point effects, the **unmodified external
sources** were compiled normally and again with gfortran's real-kind promotion
flags. No GSAS-II files were edited. The double-precision build retains the
same quadrature selection. Calling its intrinsic `PSVOIGT` at the independently
predicted shifted angle reproduces its FCJ result within 4.9e-13 relative L2 at
all four sampled reflections. Nevertheless, that build differs from the direct
angular integral by 7.90e-4–4.69e-3: higher precision alone does not cure the
under-integration. The normal rebuild reproduces two sampled pinned profiles
exactly; another differs by 9.7e-8 and one by 2.4e-4 relative L2. Compiler/build
roundoff differences therefore remain secondary and are not claimed to be
bit-for-bit identical across the whole fixture.

The confirmed main cause is thus **single-node FCJ quadrature in this pinned
GSAS-II call path**, with secondary floating-point effects and the separately
documented finite-support difference. A fix on the oracle side would require
converged quadrature and stable arithmetic, validated against the angular
integral; merely increasing arithmetic precision is insufficient. PhaseSmith's
corrected integral should not acquire an empirical shift to imitate it.

The executable audit builds from external files in place and records their
hashes, compiler version, build flags and output comparisons. It refuses a
different revision, modified source files or an existing output directory:

```sh
python oracle/scripts/audit_fcj_source.py \
  --gsas-root /path/to/pinned/GSAS-II \
  --compiler gfortran --output /new/source-audit
```

The retained report is `validation/results/fcj-source-20260918-audit.json`.
No production code or golden fixture changed during this source audit.

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

Regression tests cover the decomposition, direct angular integral, independent
quadrature, grid control and revision guard. The native geometric factor and
its derivative were corrected after the equation audit; the solver, production
support convention, original golden fixtures and acceptance thresholds are
unchanged. Exact GSAS-II equivalence remains unclaimed. An empirical shift to
imitate this particular compiled oracle is not part of the correction.
