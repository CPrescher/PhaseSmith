# Original QARR recipe: time to reported convergence and stopping audit

The original-width, full-profile recipe was run with larger numerical budgets,
without the empirical Gaussian convention or optional profile approximations.
Its standard-tolerance results repeat exactly, but an additional feasible-descent
audit demonstrates that the solver's `converged` label can represent damping-
induced stagnation. These measurements are therefore **time to reported
convergence**, not certified convergence or evidence of a global optimum.

No production solver, default recipe, profile equation, derivative, scientific
acceptance threshold or oracle fixture was changed by this experiment.

The subsequent implementation and its more conservative termination results
are documented in [Rietveld stopping verification and feasible recovery](refinement-convergence-recovery.md).

## Repeated measurements

Full workflows include data loading, background and scale preparation, all
three refinement stages, final scale covariance and result assembly. Imports
and checksum verification are outside timing. Each worker/sample combination
has one warmup and three measured repetitions, with alternating order. Other
known numerical validation jobs finished before timing; ordinary macOS services
remained active. Numerical-library thread limits were one throughout.

| Sample | Final Rwp | Maximum composition error | Median, one worker | Median, eight workers |
| --- | ---: | ---: | ---: | ---: |
| QARR 1g | 19.877586% | 0.709837 percentage points | 5.742 s | 1.720 s |
| QARR 1h | 20.101510% | 1.445351 percentage points | 7.433 s | 2.413 s |

Each stage reports `converged`; no numerical budget is exhausted. Every
retained numerical record, stage parameter record and final profile/covariance
hash matches exactly across repetitions and one/eight workers. The 1g result
passes the existing profile/composition checks. The 1h result still misses the
20% Rwp gate while passing the composition, unit-weight residual and correlation
limits. Those acceptance checks do not certify optimizer stationarity.

Standard settings use objective tolerance `1e-10`, scaled parameter tolerance
`1e-7`, and per-stage limits of 1000 iterations, 20000 model evaluations and
500 consecutive rejections. Stage step caps and parameter selections remain
the established original recipe: background changes only in stage one; stage
two adds sample/displacement parameters; the final stage adjusts scales only.
Covariance is calculated only at the final scale stage, as in the original
timing recipe.

Accepted iterations and evaluation counts are respectively 17/41/1 and
310/269/2 for 1g, and 21/131/1 and 366/312/2 for 1h. The earlier roughly
0.35-second eight-worker bounded result must not be described as a fully
converged fit. It used much shorter budgets and different stage stopping points.

The preparation driver is shared with the preceding holdout investigation.
It uses Python's observation-only initial-scale least squares and objective
tolerance, rather than reproducing the separately frozen native validator
bit for bit. This is why its numbers differ slightly from that validator.

## Tolerance and restart stress tests

A separate diagnostic tightens objective tolerance to `1e-12` and parameter
tolerance to `1e-9`, with the same large guards and full profiles. Fresh restarts
begin from accepted physical states, reset damping/history, and rebuild the
native canonical numerical scales. Scale changes are recorded; this is not
checkpoint continuation with guaranteed unchanged numerical scaling.

| Sample | Standard | Tight from original start | Tight with fresh restarts at each stage |
| --- | ---: | ---: | ---: |
| 1g Rwp | 19.877586% | 19.877584% | 19.877584% |
| 1h Rwp | 20.101510% | 21.006423% | 21.606800% |

All these calls report convergence. Tighter tolerances do not guarantee a
better staged solution: small changes near the stage-one width boundary alter
the subsequent search, and its background becomes fixed in stage two.
The worse 1h outcomes are retained rather than selected away.

Repeated full-profile/scale polishing from the **better standard endpoint**,
while retaining that endpoint's stage-one background, barely changes the fit:

- 1g: Rwp 0.19877585975285172 → 0.19877585975122405.
- 1h: Rwp 0.20101509761276720 → 0.20101509761198352.

Those restarts are operationally stable, but stability under the same optimizer
is not sufficient evidence that no useful feasible direction remains.

## Feasible-descent evidence

The diagnostic evaluates PhaseSmith's residual and analytical Jacobian at each
returned state through the existing independent orchestration adapter. Weighted
squared-error parity with the native result is asserted at relative tolerance
`5e-12`. It then tries small coordinate moves in the negative gradient direction,
scaled by the corresponding diagonal Gauss--Newton curvature, capped at 0.01
scaled units, with factors 1, 0.1 and 0.01. Box bounds are respected and trials
outside the normal profile domain are rejected. These probes do not modify the
accepted fits and do not constitute a full constrained-optimality test.

A valid, improving trial is sufficient to demonstrate that a returned point
is not stationary to that accuracy. Examples:

- The standard 1g stage-one state permits a small zero-shift change that reduces
  its weighted squared error by about 0.1475%, despite `converged` status.
- The tight 1h stage-two state permits a small ZnO strain change that reduces
  its weighted squared error by about 0.2422% (Rwp 21.04542% → 21.01992%).
- The restarted tight 1h stage-two state permits a small ZnO size change that
  reduces weighted squared error by about 0.3048%.
- The standard stage-two states are much more stable numerically, but the
  coordinate test still detects smaller improvements: about `1.90e-6` relative
  squared error for 1g and `7.09e-8` for 1h. Final scale-only polish has no
  improving witness above the diagnostic's `1e-10` relative reporting threshold.

The native general solver currently checks the norm of the **damped proposed
step** against `parameter_tolerance` before line search. Failed trials increase
damping, which can make that step arbitrarily small even with a non-negligible
feasible descent direction. The tight 1h stage-two stop has damping around
`3.5e13`. The Python fallback has the same small-step test. This explains why
increasing iteration budgets or merely tightening tolerances is not a robust
convergence remedy.

This finding does not invalidate the recorded Rwp or known-composition errors.
It limits what can be inferred from the termination flag. The next solver work
should distinguish convergence from damping-induced stagnation and provide
feasible-direction recovery when coupled width constraints block a joint step.
It must preserve the profile model and physical bounds, and be validated before
changing production behavior. No single smaller Rwp should bypass composition
or convergence checks.

## Reproduction and evidence

Use the release benchmark environment, then run sequentially:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/qarr_convergence.py --case standard \
  --json-output validation/results/qarr-convergence-20260917-timing.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/qarr_convergence.py --workers 8 --probe \
  --warmups 0 --repetitions 1 \
  --json-output validation/results/qarr-convergence-20260917-audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/qarr_convergence_polish.py \
  --json-output validation/results/qarr-convergence-20260917-polish.json
```

The probe path additionally uses the optional SciPy environment already used
by the solver-isolation adapter; it does not invoke SciPy optimization. Raw
records contain script/native/data hashes, limits, timings, numerical-scale
changes, final quality checks and feasible-descent witnesses. The timing sweep
contains sixteen warmup/measured workflows, followed by six probe workflows
and two continuation controls. Python lint/format checks and finite-JSON,
provenance and numerical-repeatability assertions pass. No production source
changed, so the earlier full Rust/Python test results are not a test of a new
solver implementation.
