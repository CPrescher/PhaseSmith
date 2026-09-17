# Quality-first refinement and rejected-step recovery

The target is a stable, scientifically valid fit. Runtime is compared at fixed
profile physics, parameter roles, convergence tolerances and attained quality.
A low Rwp alone does not establish accurate phase fractions or a correct model.

## Solver change

The native complete Rietveld solver previously discarded the accepted-state
prepared objective when a whole backtracking sequence failed. On its next
iteration it evaluated that unchanged model and Jacobian again. It now retains
them until a different state is accepted. Parameter identities, constraints,
weights and profile policy remain fixed during this reuse. Rejected trial
objects never become the accepted-state cache.

Damping also used to decay to `1e-18` during successful sequences, then recover
by one factor of ten per failed backtracking sequence. In the QARR reference
case this required about twenty failed outer iterations to reach useful
regularization near 100. The failure recovery rule is now

\[
\lambda_{next}=\max(\gamma_{increase}\lambda,\lambda_{initial}).
\]

This is a recovery heuristic for the existing scaled Gauss--Newton system
`(J_w^T J_w + lambda I) delta = -J_w^T r_w`. It does not change the residual,
Jacobian, step scaling, physical bounds, half-step backtracking, strict
objective-improvement acceptance, convergence tolerances or rejection budget.
Successful steps retain the existing damping-decrease rule and lower bound.
The caller's existing `initial_damping` supplies the recovery floor; there is
no new global numerical tolerance or profile approximation.

Both native Rietveld solvers and the independent Python orchestration use the
same damping rule. The Python implementation already retained its current
linearization across rejection. Native complete refinement now does so too.
A changed solver trajectory is possible on other problems, so scientific
regression checks are required; identical answers are not promised universally.
No claim of global optimality follows from an ordinary convergence flag.

Model-evaluation counts decrease when redundant calculations are removed.
Evaluation-budget stops can therefore occur at different accepted states.
Hard iteration, evaluation, runtime and consecutive-rejection guards remain
in force. Checkpoints contain the accepted state and next damping; optional
wall-clock diagnostics are kept out of deterministic checkpoint/history data.

## Benchmark contract

`benchmarks/qarr_quality.py` compares isolated installed baseline/current
release packages in persistent subprocesses, with alternating run order,
one warmup and repeated complete workflows. Shared preparation honors the
requested worker budget and numerical-library threads are pinned to one.

All runs use full default profile evaluation, the same explicit ZnO reference
strain of 0.0002, and the same 100-iteration / 1500-evaluation / 200-rejection
limits in each stage. Two separately named recipes are compared:

- **Frozen background:** background adjusts in stage one and stays fixed in
  subsequent stages, matching the existing empirical benchmark.
- **Joint background:** background remains adjustable during the second
  profile/sample stage. This changes the model's adjustable parameters and is
  compared only with its own baseline.

The primary reported outcomes are final Rwp, existing QPA/profile checks,
convergence of every stage, and final profile/parameter/covariance fingerprints.
The benchmark does not silently discard failing configurations. Intermediate
Rwp targets are 20%, 19.5%, 19.3%, 19.25%, 19% and 18.9%. A missed target is
stored as null, never replaced with elapsed time at an unsuccessful stop.

An optional private native binding trace records runtime events inside Rust.
No Python callback changes native dispatch. Approximate workflow time to a
target combines the native accepted-event timestamp with the adapter-entry
workflow offset, including preceding preparation/stages. A first accepted
crossing is not evidence of intermediate QPA accuracy or convergence; final
checks are always separate. Full-workflow timing includes final diagnostics
and QPA. The trace is off by default and does not enter saved projects.

## Numerical verification

### QARR 1g results, 2026-09-17

Both recipes converge at every stage before and after the change. Final
calculated profiles and covariance matrices are bit-for-bit identical, final
parameter records are identical, and every scientific check is unchanged.
These are checks of the complete result, not just rounded Rwp agreement.

| Recipe, eight workers | Final Rwp, both builds | Maximum phase-fraction error, both builds | Evaluations before → after | Median seconds before → after |
| --- | ---: | ---: | ---: | ---: |
| Frozen background | 19.2199358% | 0.873030 percentage points | 376 → 251 | 1.168 → 0.898 |
| Joint background | 18.9115837% | 1.864564 percentage points | 736 → 484 | 2.356 → 1.671 |

Counts sum all three stages. Frozen-background stage one drops from 330 to
205 evaluations; joint-background stage two also drops from 404 to 277.
This removes wasted work while preserving each recipe's attained solution.
The lower-Rwp joint recipe has a larger composition error, so it is not
promoted to the default on the strength of its residual alone.

Approximate median time to an accepted Rwp of 19.25% is 1.091 → 0.817 s
for the frozen recipe. The joint recipe reaches 19% in 1.176 → 0.850 s.
Neither reaches 18.9%; the benchmark records that target as missing.

**Timing limitation:** these medians use three measured repetitions after a
warmup with alternating build order, but another Pawley validation job and
macOS background services were active. They are provisional shared-machine
timings, not an uncontended speed claim or a new comparison against rietx.
The exact evaluation counts and numerical-equality results do not depend on
that timing limitation. Raw runs, hashes, traces, checks and the machine-load
note are retained in
`validation/results/quality-recovery-20260917-frozen-8workers.json` and
`validation/results/quality-recovery-20260917-joint-8workers.json`.

The one-worker frozen-background control likewise has identical baseline/
candidate results and the same evaluation reduction. Its provisional median
is 3.698 → 2.611 s, recorded in
`validation/results/quality-recovery-20260917-frozen-1worker.json`.
Candidate final profiles, parameter records and covariance are also identical
between one and eight workers.

### Regression coverage

The synthetic rejection regression uses intensity proportional to occupancy
squared. Its deliberately overshooting trial is rejected; the recovered fit
must reach the known occupancy and low residual with unchanged tolerances.
It checks absence of redundant accepted-state evaluations, restoration of
useful damping, strict accepted-objective descent, exact continuation through
an interrupted rejection sequence, and the hard rejection limit.

Python tests check recovery against the native solution, exact continuation,
and equality of traced/untraced native fits, histories and checkpoints.
Existing independent profile references, finite differences, support and
normalization tests remain applicable: no forward equation or derivative
changed, so no oracle fixture was regenerated.

`benchmarks/check_recovery_real_data.py` compares the unchanged QARR 1g recipe,
PbSO4 X-ray and neutron fits, and the independent QARR 1h holdout across builds.
The first three retain identical passing scientific checks. The holdout
retains its existing profile failure at Rwp 28.65443%; this improvement does
not solve that transferability problem, and its failure threshold is unchanged.
Raw checks are in `validation/results/quality-recovery-20260917-real-data.json`.

The complete Rust test run passes 343 tests (34 ignored), including the new
rejection/restart regression. Python passes 867 tests (11 skipped and 33
deselected by the existing suite configuration). Workspace formatting, Clippy
with all targets/features, and the focused Python lint checks pass. Skipped
external-data tests are not counted as scientific validation passes.

## Reproduction

Install the baseline wheel into an isolated package directory, and install the
candidate release wheel into the active benchmark environment. The retained
baseline includes the optional trace adapter but precedes the solver change.
Then run:

```sh
python benchmarks/qarr_quality.py --baseline-package /path/to/baseline \
  --threads 8 --repetitions 3 --json-output frozen.json
python benchmarks/qarr_quality.py --baseline-package /path/to/baseline \
  --threads 8 --repetitions 3 --joint-background --json-output joint.json
python benchmarks/check_recovery_real_data.py --baseline-package /path/to/baseline \
  --json-output real-data.json
```

The drivers set child BLAS/OpenMP/Accelerate limits to one. Do not run builds,
tests or other benchmarks concurrently with timing measurements.
