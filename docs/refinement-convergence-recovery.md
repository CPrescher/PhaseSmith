# Rietveld stopping verification and feasible recovery

Historical investigation of the preserved experimental implementation. The
integrated solver now uses [one shared recovery trial allowance](refinement-bounded-recovery.md).
The long-budget measurements below have not been regenerated or relabelled.

The [consolidated assessment](refinement-assessment.md) now reviews this change
on fixed practical recipes. It separates the correctness of honest stopping
from the cost and compatibility of recovery. The long QARR investigations
below are historical diagnostics, not the recommended general runtime budget.

The complete native CW Rietveld solver and independent Python orchestration now
verify candidate stopping conditions. A small **damped** step alone is not a
convergence certificate. Near the coupled instrument-width domain boundary,
the former implementation could increase damping until it reported convergence
while a valid change to a different parameter still decreased weighted error.
The evidence and baseline are in `qarr-convergence-audit.md`.

## Equations and behavior

In scaled free coordinates, with the existing mask and uncertainty weights,
let `r = W(ycalc - yobs)`, `f = r.T r / 2`, `g = J.T r`, and `H = J.T J`.
The ordinary step still solves `(H + lambda I) d = -g`, with the existing
step cap, backtracking and strict descent acceptance. No profile, support,
reflection, weighting, physical bound or parameter scale is changed.

A candidate stop occurs when the box-projected proposed step is smaller than
`parameter_tolerance`, or the previous accepted objective improvement is below
`objective_tolerance * max(f, 1)` after `min_iterations`. Instead of immediately
stopping, the solver checks:

1. A residual-checked, effectively undamped normal solve (regularization
   `1e-18`) may establish a step below `parameter_tolerance`. The native dense
   path uses its existing small direct solve; the Python path independently
   uses CG and verifies its residual. If the native direct solve is unavailable
   or fails, the diagonal check below remains available.
2. Otherwise let `h_i = H_ii`, and `s_i` be `-g_i/h_i` projected onto the free
   parameter's physical box bounds. Sum the nonnegative one-coordinate model
   improvements `-g_i*s_i - h_i*s_i*s_i/2`. Only if this sum is below the
   objective tolerance may the solver report convergence by this check.
   The user step cap does not shrink this stopping measure. Zero Jacobian
   columns contribute zero. This is a local numerical stopping test, not a
   global optimum or full constrained KKT certificate.
3. If material predicted improvement remains, try one-coordinate directions,
   capped by `max_scaled_parameter_step`, in decreasing predicted-gain order.
   Equal scores retain parameter order. Each direction uses the usual
   backtracking, full constraint expansion and physical/calculation-domain
   validation. Accept only an actual objective decrease greater than the
   objective tolerance. These trials use the same full numerical profile
   policy as the rest of the fit.
4. If none succeeds, report `stagnated`, unless a runtime guard already stopped
   the solve. Do not relabel a budget, cancellation or rejection limit.

An accepted recovery step reuses the existing accepted-state/checkpoint path
and ordinary damping decrease. Resetting damping after each recovery step was
experimentally much slower, repeatedly rediscovering the same domain boundary.
No additional checkpoint state is needed: a pending objective-change check is
derived from the last accepted history record. History records the accepted
coordinate direction's norm and backtrack count; damping and CG iterations
still describe the normal solve attempted before that accepted trial. Dense diagonal curvature comes
from the existing fused Jacobian, with no additional profile evaluation.
Matrix-free diagonal products retain runtime accounting and cancellation.
Recovery stores coordinate descriptors and reuses a single direction vector,
so its direction storage is linear in the parameter count.

This change is limited to the complete CW solver used by the public Python
Rietveld API and its Python fallback. Separate legacy structural/joint solver
implementations and TOF workflows are not changed. No GSAS-II implementation
or new oracle data is involved; profile/derivative equations are unchanged.

## Validation

The regression suite includes a single-site, nonlinear occupancy fit initialized
with damping `1e20`: both native and independent Python solvers must recover
occupancy 0.5 and Rwp below `1e-8`, rather than accepting the initial state.
Additional tests cover a misleading small accepted objective change in a
linear scale fit, the physical occupancy upper bound, an artificially tiny
step cap, existing rejected-state reuse, trace isolation, constraints and
exact checkpoint continuation after recovery in both implementations. Existing profile derivative/normalization/oracle
fixtures remain unchanged.

Real-data measurements and remaining limitations from the release build are
recorded below. Scientific acceptance gates are not loosened.

## Real-data results

The original-width recipe retains full profiles, support 30, the same staged
parameter selections, frozen stage-two background, and per-stage limits of
1000 iterations / 20000 evaluations / 500 consecutive rejections.

| Sample | Previous Rwp | Verified-stop/recovery Rwp | Maximum composition error, before → after |
| --- | ---: | ---: | ---: |
| 1g | 19.877586% | 19.690019% | 0.709837 → 0.597903 percentage points |
| 1h | 20.101510% | 19.689517% | 1.445351 → 1.411222 percentage points |

Both standard-tolerance fits pass all four unchanged profile/composition
checks. In both, stage one and stage two now report **stagnated**, and the final
scale-only stage converges. The independent feasible-coordinate probe finds no
remaining witness above its `1e-10` relative squared-error reporting threshold
at any stage. This improves the fit and removes the earlier false-success
claim; it does **not** establish full constrained convergence.

Accepted iteration/evaluation counts by stage:

- 1g: 89/2761, 97/4577, 1/2.
- 1h: 78/2808, 205/1731, 1/2.

Tighter tolerances still do not ensure a better staged fit: 1g reaches
19.690104% Rwp (0.596559 percentage-point composition error), and 1h reaches
20.273201% (1.982626 percentage points). Tight 1h therefore fails the 20% Rwp
check. All main stages report stagnation, and none of these calls exhausts
its enlarged budget. Every stage again has no coordinate-descent witness
above the probe threshold. The stage-one background is frozen thereafter;
different stage-one endpoints can change the subsequent fit. A more capable
search within the coupled width domain remains necessary.

The established empirical frozen-background recipe remains a distinct option:
1g reaches Rwp 19.219774% with 0.873305 percentage-point composition error;
1h reaches 19.443693% with 1.550494 percentage points. Their physical/profile
checks pass. Their initial stages now report stagnation; subsequent joint
and final scale stages converge. Do not continue describing these workflows
as having every stage converged merely because their final stage converged.

The earlier explicit-execution QARR 1g and PbSO4 X-ray/neutron checks retain
their gate statuses; the known native QARR 1h failure remains. These calls
do not cover every canonical native validator. The subsequent consolidated
assessment finds that `run_qarr_1g_validation(directory)` with no execution
argument changes from passed to failed on the termination gate, with identical
final Rwp and QPA. The earlier statement that all frozen validators retained
their gate statuses was too broad; see `refinement-assessment.md`.
PbSO4 X-ray Rwp improves from 10.344569% to 10.344490%; its other checks still
pass. `check_recovery_real_data.py` deliberately exits nonzero because it
compares complete records, including measured numbers, and was originally
written for the prior bitwise-preserving optimization. Its retained output
shows changed measurements, **not** a new failed scientific gate.

Results are retained in:

- `validation/results/convergence-recovery-qarr-audit-20260917.json`
- `validation/results/convergence-recovery-empirical-20260917.json`
- `validation/results/convergence-recovery-real-data-20260917.json`

The audit records the release binary, solver source, driver and data hashes.
Probe runs include extra calculations and concurrent validation work; their
wall times are not benchmark measurements. The empirical and frozen-validator
records precede the final linear-memory direction-storage refactor, which
changes no direction arithmetic; the final audit confirms identical original
recipe results after that refactor.

## Checks run

- `cargo fmt --check` and `cargo clippy --workspace --all-targets --all-features` pass.
- `cargo test --workspace --all-features`: 344 passed, 34 ignored.
- `pytest`: 871 passed, 11 skipped, 33 deselected under the existing marker policy.
- The separately enabled native/Python affine-history comparison passes.
- Ruff and `git diff --check` pass.

No profile equations, derivatives, independent profile references, finite-support
rules, normalization conventions or golden oracle fixtures changed. Existing
kernel/derivative tests therefore remain the relevant mathematical regression
coverage. The new tests exercise solver progress and termination directly.

## Runtime cost and worker determinism

The final release ran one warmup and two measured full workflows per sample
and worker count, with alternating order. Imports and checksum verification
were outside timing; data preparation, all stages, covariance and reporting
were inside. Numerical-library thread limits were one. All stage parameters,
history summaries, final profile/covariance hashes and scientific checks agree
**exactly** across all twelve runs and one/eight workers. Timed endpoints also
match the feasible-descent audit exactly.

| Sample | One worker, measured range | Eight workers, measured range |
| --- | ---: | ---: |
| 1g | 73.13–74.29 s | 19.00–41.46 s |
| 1h | 40.25–46.31 s | 11.75–20.73 s |

These are **contended desktop timings**. Our build/validation jobs had finished,
but a separate VirtualMachine process subsequently used approximately four
cores during the second measured eight-worker 1g run; system services also
consumed substantial CPU. The variation is retained in the JSON, not discarded.
Do not infer an isolated scaling ratio or compare these medians directly with
the older machine-state timings. No unrelated process was stopped.

The algorithmic cost increase is unambiguous: the original standard recipe
uses 7340 total evaluations on 1g (previously 581), and 4541 on 1h (previously
680). This is an accuracy/termination repair, **not a speed improvement**.
Efficient feasible joint directions are the next optimization priority;
restoring the old unverified early stop would obscure the numerical problem.

Timing records, repeatability assertion and contention note:
`validation/results/convergence-recovery-qarr-timing-20260917.json`.

Reproduction (use a quiet machine for interpretable elapsed times):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/qarr_convergence.py --case standard --workers 1 8 \
  --warmups 1 --repetitions 2 --json-output timing.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/qarr_convergence.py --case standard --case tight --workers 8 \
  --warmups 0 --repetitions 1 --probe --json-output audit.json
```
