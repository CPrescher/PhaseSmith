# Exact reuse in categorical repair evaluation

The opt-in `cached_repair_evaluation` option requires
`joint_assignment_refinement`. It changes execution only. Original proposal
order, scalar shift seeds, beam width/depth, count diversity, physical fits,
acceptance gates, source observations, atom caps and deadlines remain unchanged.
The original evaluator remains available for same-binary comparisons.

## Measured bottleneck

A five-second native sample repeats the complete COD 2242589 development solve,
using only its existing input packet and HKL observations. The original repair
stage costs about 1.8 seconds on the recorded complete iteration. Among collapsed
main-thread leaves, `shift_profile` has 304 samples and `joint_neighbors` has 111;
exponential and norm routines also appear in their call stacks. Matrix assembly
and residual guards remain major whole-solver costs. These are sampling counts,
not isolated wall-time fractions. Raw stacks, native hashes, image identifiers
and exact drivers are retained under `runs/repair-evaluation-profile/` and in the
committed profile artifact. No holdout data are used.

The scalar line search repeatedly builds the same shifted reflection field,
allocates amplitude/derivative vectors and evaluates unused final derivatives.
Successful trial fields are then recomputed at the next Gauss-Newton iteration.
Categorical scoring also recalculates the same complex magnitudes.

## Equations and cache lifetime

No equations change from `solution-joint-assignment.md` or its independent NumPy
reference. The call-local column cache stores the exact original multiplication
`P_i(h)*f_e(s_h)` in contiguous site/element/reflection order. No exponential
reassociation or approximate math is introduced.

Two reusable field slots retain complex totals, magnitudes and scalar-shift
magnitude derivatives. Their keys are the exact shift float bits. Both slots
are invalidated for every categorical move, so background, site and element
cannot leak between moves. Full scalar profiles retain their original sum and
normalization order. Backtracking requests only the objective; magnitude values
and derivatives are still evaluated together, and the final derivative reduction
is performed if that trial becomes a subsequent Gauss-Newton point.

The categorical scorer preserves its distinct original denominator
`sum(norm_sqr(F))`, its original final residual normalization, and original
reduction order. It reuses magnitudes from the same complex field rather than
substituting the scalar-profile objective. This distinction matters for exact
floating-point equivalence. The original two variants and optimized scalar seed
still count as the same logical evaluations, including all rejected proposals.

The background subtraction is prepared once per site and reused across possible
elements. The immutable orbit images and neighbor U-equivalent medians are also
reused between the two coherence strengths, preserving the same expanded-image
and median ordering. Scalar/Gauss-Newton work remains bounded exactly as before.

Column payload is capped at 4 MiB. Above that size, column multiplication falls
back to direct evaluation while reusable fields remain active. Reflection
buffers add at most 104*M bytes (M<=2048), so the evaluator payload is at most
4 MiB plus 208 KiB. This excludes ordinary vector headers, allocator overhead
and the pre-existing model/cache. The small orbit-image/median cache is separately
bounded by the original 128-image model. Everything is owned by one synchronous
repair call and released before the full physical fits; no global cache, new
threads or retained numerical state is installed.

Audit fields distinguish logical proposal counts from actual field builds,
reuses, objective-only requests and maximum evaluator payload bytes. Independent
physical equations, finite differences, forced-cap fallback, bitwise scalar
values/derivatives/categorical scores, deterministic beam candidates and synthetic
whole searches validate the change. Existing Python references remain independent
and unchanged. No new differentiable physical term, finite support or oracle
fixture is needed; GSAS-II's powder oracle does not cover this stage.

## Comparison protocol

A balanced serial three-mode comparison uses all 35 development cases:
`off` measures the preceding no-repair overhead baseline, `control` enables the
original repair and `cache` enables repair plus exact reuse. All other best
controls, the seed, original caps, one numerical worker, final anisotropic mode
and 13/15-second safeguards are identical. All 210 predictions precede all
three reference barriers. The required quality comparison is original repair
versus cached repair: the exact previous 19 structures and 27 correct total-count
cases must survive. The off reference intentionally has 26 correct total counts.
The 65-case holdout stays closed. Source and binary snapshots are verified;
no task-owned tests, builds or kernel runs overlap timed corpus collection.

## Results on all 35 development cases

The matched same-binary collection retained the exact preceding 19 labelled
structures, 27 correct total non-H atom counts, 19 correct per-element count
vectors and 15 structures within the strict 0.02-Angstrom coordinate tolerance.
All 35 predicted count vectors match the preceding repair report, including
incorrect outcomes. Four no-model cases and seven references above the unchanged
native atom cap remain in the primary denominator. No PhaseSmith process timed out.

| Measurement across all 35 cases | Original repair | Cached repair |
| --- | ---: | ---: |
| Repair stage, summed seconds | 4.477 | 3.466 |
| Library solve, summed seconds | 39.605 | 38.566 |
| Process wall time, summed seconds | 46.880 | 45.639 |
| Logical repair scores | 184,318 | 184,318 |
| Completed beam depths | 188 | 188 |
| Full physical fits / accepted rounds | 18 / 5 | 18 / 5 |

Repair stage time falls 22.59%, and summed complete solver time falls 2.62%.
Summed process time falls 2.65%. Median paired solver time falls only 0.26%
(paired process time 0.98%): improvements concentrate in expensive repairs rather
than uniformly affecting typical cases. These are one balanced collection on
this host, not confidence intervals on complete solve speed.

The no-repair reference takes 35.118 summed solver seconds and retains 26 correct
total counts. Relative to that same-collection reference, repair overhead falls
from 12.78% to 9.82%; about 23.15% of the added solve time is removed. The remaining
repair work still costs time. Displacement-coherence preparation and full physical
fits are outside the categorical field kernel.

The eight-orbit, 1054-reflection Criterion workload takes 35.125 ms originally
versus 17.479 ms with reuse (2.01 times faster; ten raw samples retained per mode).
That isolated kernel gain does not imply a twofold complete-solver improvement.
The corpus records 115,510 actual field builds, 66,523 field reuses and 56,666
score-only requests, with a maximum evaluator payload of 4,145,152 bytes.

All 429 Rust tests and 2,743 Python tests pass, with 34 ignored Rust tests,
11 skipped Python tests and 34 deselected Python tests. Formatting and strict
workspace Clippy checks pass. The holdout remains unopened, and the option stays
experimental/off by default.

## Reproduction and artifacts

Use `cached_repair_evaluation=True` together with
`joint_assignment_refinement=True` in `solve_structure_elements`; the other
accepted solver controls are set explicitly in the collector below.

```sh
.venv/bin/python benchmarks/solution_repair_cache_interleaved.py --output runs/repair-evaluation-final --seconds 15
for mode in off control cache; do
  .venv/bin/python benchmarks/solution_corpus_reaudit.py --manifest validation/solution-corpus-v5.json --run "runs/repair-evaluation-final/$mode" --output "runs/repair-evaluation-final/$mode/comparison-verified.json"
done
.venv/bin/python benchmarks/solution_repair_cache_verify.py --base runs/repair-evaluation-final
cargo bench -p phasesmith-single-crystal --features solution --bench solution -- repair_evaluation_ --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Committed reports under `benchmarks/structure-solution-repair-evaluation-` include
the three full outcome reports, paired comparison, exact success-set/count-vector
retention, native profiles, raw Criterion samples and validation provenance.
The adapter-only protocol lock is `validation/solution-corpus-v5.benchmark36.lock.json`.
Raw predictions, schedules and source snapshots remain under
`runs/repair-evaluation-final/`.
