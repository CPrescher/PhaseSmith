# Refinement consolidation and development decision

Assessment date: 2026-09-17. Further QARR solver tuning is paused. The goal is a
trustworthy bounded workflow across real problems, with profile agreement,
scientific parameter accuracy, physical assumptions, termination and runtime
reported separately. The experimental feasible-width solver remains parked
and off by default.

## Fixed assessment contract

The protocol in `validation/refinement-assessment-v1.json` was written before
running the panel. It retains existing recipes, starts, profile policies,
iteration/evaluation/rejection limits and acceptance thresholds. There is no
post-failure retry with larger budgets, parameter tuning or gate relaxation.

Eight cases cover native QARR 1g/1h, PbSO4 X-ray/neutron structural workflows,
Rowles 1a/1e QPA, the Echidna neutron smoke test, and the nickel TOF validation.
Rowles, Echidna and nickel were not used to tune the recent QARR changes. They
are **transfer checks within the existing corpus**, not untouched data across
the entire project. A genuinely unused external/laboratory holdout is still
required before claiming generalization of a newly selected recipe.

Baseline is the retained release binary from before the convergence-verification
and coordinate-recovery changes, SHA-256
`286ad235a891aeb8752e23d46f51681245de4f43bb4add78743940c3f506ad4a`.
Its Python solver and validation sources match the checked-in `8d9d86e` tree.
Candidate is the current release binary, SHA-256
`a16dc6b02053f281956bfd44c419c3b4bc233f0e1bada6bf95cef64762d7d2f5`.
It includes verified stopping and coordinate recovery, with feasible-width
steps disabled. This comparison isolates neither stopping verification nor
recovery individually; those are the combined changes under assessment.

Both builds run in persistent Python 3.12 subprocesses. Each case has one
discarded warmup and two measurements, with alternating build order and no
concurrent campaign fits. BLAS/OpenMP/Accelerate thread variables are one.
Native validators retain their worker defaults; Rowles explicitly uses one
worker. Import, checksum verification and Rowles conversion are outside the
timed region; the complete fixed workflow is inside. The host remains a shared
desktop, so timing ranges are supporting evidence rather than universal ratios.

Data checksums, native and installed Python source hashes, protocol/driver
hashes, complete reports, stage stops, failed checks and raw timings are in
`validation/results/refinement-assessment-20260917.json`.

## Findings

All 48 executions completed: eight cases, two builds, one warmup and two
measurements. Each build/case repeats its scientific report exactly, including
stage notes; Rowles additionally repeats the calculated-profile hash. There
were no runner exceptions. Six candidate cases pass their existing gates,
one retains a known failure, and one introduces a termination-gate failure.
Passing the benchmark driver is therefore **not** an overall scientific pass.

| Fixed case | Candidate Rwp (%) | Candidate scientific accuracy / scope | Baseline → candidate gates | Median seconds, baseline → candidate |
| --- | ---: | --- | --- | ---: |
| Native QARR 1g | 19.896901 | Max QPA error 1.117931 pp | pass → **fail: termination** | 0.599 → 0.739 |
| Native QARR 1h | 28.654430 | Max QPA error 1.728960 pp | fail → fail: profile checks | 0.592 → 0.598 |
| PbSO4 X-ray | 10.344461 | Structural/profile checks pass | pass → pass | 3.931 → 4.339 |
| PbSO4 neutron | 4.217168 | Structural/profile checks pass | pass → pass | 0.577 → 0.587 |
| Rowles 1a | 8.782343 | Max QPA error 0.757867 pp | pass → pass; stage rejections remain | 3.159 → 3.727 |
| Rowles 1e | 8.283140 | Max QPA error 2.223483 pp | pass → pass; stage rejections remain | 2.398 → 3.428 |
| Echidna LaB6 | 40.857954 | Le Bail smoke/improvement contract only | pass → pass | 0.008 → 0.008 |
| Nickel TOF | 3.278208 | Structural multibank; cell error 0.000331 A | pass → pass | 51.973 → 52.308 |

These are different datasets and different frozen recipes; Rwp values across
rows are not a ranking of accuracy. Nickel time includes the complete profile,
multibank and structural validation workflow, not just the displayed structural
fit. It also reaches 2.485869% single-bank and 2.273306% joint-profile Rwp.
Its measured cell is 3.523731 A against the 3.5234 A reference; fitted Ni
Uiso is 0.003965 A² and passes the existing physical interval check. That
interval is not an independent measurement of Uiso accuracy. Its wrapper does
not export every structural termination reason; the report retains its actual
checks and accepted-step/evaluation counts without inventing missing labels.

Machine-readable decision summary:
`validation/results/refinement-assessment-20260917-summary.json`.

### A missed native-entry-point regression

The native `run_qarr_1g_validation(directory)` case changes **passed → failed**.
Final Poisson Rwp remains exactly 19.896901%, with the same 1.117931
percentage-point maximum composition error. Stage two changes from
`converged` to `repeated_rejections`; evaluations increase from 43 to 64.
The unchanged termination gate explicitly excludes repeated rejections.

This is not evidence of worse fitted intensities or composition. It is a
workflow acceptance/termination regression, and the candidate must not be
described as release-ready. Honest stopping must be retained; restoring the
old unverified convergence label would hide the issue.

Earlier four-case checks supplied an explicit execution policy to QARR 1g,
which selects the Python-assembled recipe with native refinement. The canonical
no-execution-argument entry point instead runs the native validator, with
different stage budgets/tolerances/preparation. Both can use a native solver
while still being different workflows. Their passing records are not
interchangeable. The same entry-point distinction changes the native PbSO4
X-ray profile setup, which explicitly retains the fixed doublet.

### Transfer checks show recovery cost with little scientific gain

Rowles 1a and 1e pass their original numerical/QPA gates in both builds, but
neither has all stages converged. Existing gates are deliberately left as-is;
passing them must not be presented as a convergence certificate.

| Rowles case | Rwp %, baseline → candidate | Max composition error, pp | Nonlinear evaluations | Seconds, baseline → candidate |
| --- | ---: | ---: | ---: | ---: |
| 1a | 8.782365 → 8.782343 | 0.756049 → 0.757867 | 201 → 245 | 3.156–3.161 → 3.720–3.734 |
| 1e | 8.283129 → 8.283140 | 2.223435 → 2.223483 | 143 → 220 | 2.397–2.399 → 3.422–3.433 |

Stage stops become dominated by repeated rejections. These results support
reviewing the scope and cost of coordinate recovery, rather than assuming that
more solver work is inherently better. Rowles 1e uses its pre-existing
3-percentage-point QPA gate; it would not satisfy QARR's 2-point gate. The
acceptance rules remain case-specific and are not homogenized after fitting.

For residual localization, the largest equal-sample-width interval accounts
for 22.24% of weighted error at 21.01–33.90 degrees on 1a, and 23.55% at
33.91–46.80 degrees on 1e. These distributions barely change across builds.
They identify remaining misfit locations, not its physical cause. Other native
reports expose aggregate metrics and checks; missing detailed residual arrays
are not replaced by fabricated diagnoses.

### Existing limits remain visible

The bounded native QARR 1h workflow still fails its Rwp, unit-weight Rwp and
profile-correlation checks: Rwp is 28.654430% in both builds. Composition error
is 1.728960 percentage points and passes its separate QPA gate. Its first two
stages hit iteration limits. The much better extended/empirical QARR results
in earlier reports used different recipes/budgets and do not repair this
frozen default workflow automatically.

Echidna is a deliberately limited Le Bail smoke test. Its approximately 40.86%
Rwp and 0.9062 correlation pass that test's original improvement/coverage
contract; this does not establish a high-quality structural refinement. TOF
and Le Bail checks exercise unaffected solver families and provide breadth
and regression control, not evidence that CW coordinate recovery improved.

## Production decisions

1. **Keep established improvements:** native execution, selected derivatives,
   accepted-state reuse, deterministic accumulation, physics corrections and
   explicit approximation contracts. This panel provides no reason to undo
   those earlier changes.
2. **Keep truthful termination as a requirement.** The combined uncommitted
   stopping/recovery implementation is not cleared for promotion while the
   native QARR 1g acceptance regression remains.
3. **Review recovery separately from correctness.** Fix the bounded workflow
   interaction using a small reproducer and the same acceptance gates. Measure
   evaluations as well as elapsed time; do not solve it by silently multiplying
   budgets or treating every finite endpoint as converged.
4. **Park the feasible-width experiment.** Keep its results and off-by-default
   implementation available for research. No further QARR-specific active-set
   or tolerance tuning is justified by this assessment.
5. **Retain explicit scientific assumptions.** Calibrated instrument profiles
   are needed for independent microstructure interpretation. An empirical
   strain anchor is an optional assumption, not a measured strain. Choose it
   for a scientific workflow before fitting, not from the best QPA score.
6. **Freeze the assessment panel as a regression boundary.** A subsequent fix
   must resolve the new canonical QARR stop failure without restoring false
   convergence, preserve existing scientific gates, and show its cost on
   Rowles. Recipe improvements for the pre-existing 1h problem require a
   separate proposal and genuinely unused validation data.

The immediate engineering priority is the reproducible bounded-workflow
termination issue. A general optimizer replacement, additional profile terms,
or renewed competition over a single timing/Rwp row is not supported.

## Reproduction

Install the candidate release and unpack the retained baseline wheel into a
separate package directory, then run:

```sh
python benchmarks/refinement_assessment.py \
  --baseline-package /path/to/baseline-package \
  --json-output /path/to/assessment.json
```

The driver verifies the registered local datasets, invokes no GSAS-II process,
does not download data, and leaves known/novel failures in the report. Review
each case's `status`, checks, errors and `exact_repeatability`; successful
report generation alone is not a scientific pass.

The new driver passes Ruff, formatting, Python compilation and whitespace
checks. No production numerical code changed during this consolidation.
Previous ordinary unit-suite pass counts do not override this panel's failed
real-data gate: real-data cases are opt-in/ignored in those default suites,
and the existing Python QARR regression explicitly selects the other entry
point. The release check must exercise the canonical native call too.
