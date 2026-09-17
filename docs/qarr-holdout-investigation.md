# QARR holdout: convergence, width decomposition and composition accuracy

The September 17 investigation finds that the QARR 1h profile failure is not
evidence that the existing forward model cannot fit this pattern. More solver
work helps, but the existing explicit empirical Gaussian convention helps
further. The same frozen-background recipe already used on 1g reaches
19.44367% Rwp on 1h, with every stage converged and a maximum phase-fraction
error of 1.55045 percentage points. Full default profiles and the original
scientific thresholds are retained.

This is a diagnostic result, not a replacement validation gate or a claim of
globally optimal refinement. The frozen native holdout validator still reports
its existing failure at 28.65443% Rwp. No golden data, default refinement recipe,
physical equation, derivative or acceptance threshold changed in this work.

## Comparison contract

`benchmarks/investigate_qarr_holdout.py` verifies both checksum-pinned datasets
and applies the same cases to 1g and 1h. The measured compositions enter only
the final scoring, never the refinement objective, constraints or starting
scale estimate. All fits use the native production solver and full default
FCJ/support policy, with support multiple 30, fixed CIF anisotropic displacement,
the existing size/Gaussian-strain/preferred-orientation model, and the same
observed-data background estimate plus three residual-background coefficients.

The script assembles requests through Python and initializes scales with NumPy
least squares. It uses Python's default objective tolerance `1e-10` and scaled
parameter tolerance `1e-7`, whereas the frozen native validator uses objective
tolerance `1e-7`. Thus the bounded diagnostic is not a bitwise replay of that
validator: it gives 28.68085%, rather than 28.65443%, on 1h. Comparisons below
are within the common diagnostic driver, not across those two preparations.

Cases have a 3000-evaluation per-stage guard. The bounded case has iteration
budgets 8/35/10 and rejection allowance 20. Extended cases use 100/100/100 or
500/500/100 and rejection allowance 200. The empirical cases apply the existing
`EmpiricalGaussianConvention("ZnO", 2e-4)` before stage one and retain its
constraint in subsequent stages. Only the separately named joint-background
case frees background coefficients during stage two.

Every case records stage termination, accepted Rwp history, fitted parameters,
rank availability, residual localization, dataset/native/driver hashes, and
final quality checks. Failures are retained. Times are not used as performance
claims in this investigation.

## Results

Errors below are the maximum absolute difference from the independently weighed
phase fractions, expressed in percentage points. The profile gate is Rwp at
most 20%; the composition gate allows at most 2 percentage points.

| Sample | Case | Rwp | Phase-fraction error | Every stage converged? |
| --- | --- | ---: | ---: | --- |
| 1g | Bounded | 19.877602% | 0.711163 | No |
| 1g | Extended, 100 | 19.877586% | 0.709837 | Yes |
| 1g | Extended, 500 | 19.877586% | 0.709837 | Yes |
| 1g | Empirical, frozen background | 19.219936% | 0.873030 | Yes |
| 1g | Empirical, joint background | 18.911584% | 1.864564 | Yes |
| 1h | Bounded | 28.680850% | 1.521968 | No |
| 1h | Extended, 100 | 21.410620% | 1.600837 | No |
| 1h | Extended, 500 | 20.101510% | 1.445351 | Yes |
| 1h | Empirical, frozen background | 19.443669% | 1.550447 | Yes |
| 1h | Empirical, joint background | 19.131498% | **2.459780 — fails** | Yes |

The frozen empirical 1h fit also has unit-weight Rwp 12.99103% (limit 15%) and
background-subtracted profile correlation 0.99086247 (minimum 0.98). Its phase
fractions are Al2O3 33.62103%, ZnO 30.13852%, and CaF2 36.24045%, compared with
weighed values 35.12%, 30.19%, and 34.69% respectively.

Lower Rwp is not uniformly better composition accuracy. Even the successful
frozen empirical fit has slightly more composition error than the 500-iteration
original-width fit. Joint background lowers Rwp further while moving the
Al2O3 fraction to 32.66022%, outside the existing error allowance. It is not
selected as a better default simply because its profile residual is smaller.

Final scale covariance is available in these fits, but it is conditional on
the other fitted parameters being held fixed. For the empirical 1h case the
maximum propagated conditional standard uncertainty is about 0.086 percentage
points. That does not account for model bias or justify interpreting the
1.550-percentage-point composition error as a small uncertainty.

## What explains the failure?

1. **The short recipe is under-refined.** The bounded 1h diagnostic stops both
   substantive stages at their iteration limits. A final scale-only polish
   converges, but this does not establish convergence of the profile fit.
   Increasing budgets to 500 brings all stages to convergence, reducing Rwp
   substantially, yet still missing the 20% gate.
2. **Instrument/sample Gaussian decomposition affects the staged search.**
   The shared coefficient is `U + C epsilon_p^2`, with
   `C=(2*180/pi)^2`; see [the convention equations](empirical-gaussian.md).
   An equal transfer of variance preserves the starting total profile while
   changing which widths stage one can reach with sample strain fixed. In
   stage two the reference constraint removes the common U/strain ambiguity.
   This is a change to the staged feasible search, not additional profile
   physics. The tested initial full profiles differ by less than `6e-17` in
   relative L2 for the nominal convention on both samples.
3. **Background freedom can trade composition accuracy for residual reduction.**
   This is demonstrated by the paired frozen/joint experiments. Residual
   localization alone does not establish the physical cause of that tradeoff.

The experiments support improving width coordinates and convergence-aware
staging before replacing the solver or adding speculative profile terms.
They do not prove that every residual feature or phase-fraction bias has been
explained.

## Robustness and limits

- The nominal empirical cases and the 500-iteration original-width cases were
  repeated with one instead of eight workers. Every retained numerical result,
  parameter record, stage history and final profile hash is identical.
- Tightening objective tolerance from `1e-10` to `1e-12` and parameter tolerance
  from `1e-7` to `1e-9`, with a 500-iteration budget, changes nominal empirical
  1h Rwp from 0.19443669166783403 to 0.19443669166505947. All stages converge;
  composition error changes by less than `5e-7` percentage points.
- The explicit reference is still an assumption. A 1e-4 reference converges
  on both samples, with 1h Rwp 19.44337% and error 1.55048 percentage points.
  A 3e-4 reference gives 1h Rwp 19.46154% and error 1.55301 percentage points,
  but stage two hits its iteration limit. It therefore fails the stronger
  all-stages-converged condition despite passing the profile/composition limits.
  No automatic choice of reference is inferred from these runs.
- The nominal convention and frozen-background recipe predate this holdout
  investigation. Follow-up sensitivity experiments are exploratory. Future
  model or recipe tuning needs an additional untouched sample or calibration
  standard; repeatedly optimizing against 1h cannot supply that evidence.

The next implementation target is an opt-in, convergence-aware staged workflow
with explicit width assumptions and separate profile/composition evidence.
Keep independent instrument calibration available whenever absolute sample
strain or crystallite size is the scientific objective.

## Reproduction

Use the release package and pinned data, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/investigate_qarr_holdout.py --threads 8 \
  --json-output validation/results/qarr-holdout-20260917-diagnostic.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/investigate_qarr_holdout.py --threads 1 \
  --case extended_500 --case empirical --case empirical_tight \
  --json-output validation/results/qarr-holdout-20260917-controls.json
```

The full diagnostic contains 16 fits; the worker control repeats six. Neither
command regenerates oracle fixtures or changes the frozen validation suite.
