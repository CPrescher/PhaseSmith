# Bounded structure-solution execution

`solve_structure_elements(..., workers=N)` accepts a Python integer in 1..64
and caps it to available logical CPUs. The default is one. Each synchronous
solve lazily creates one reusable pool through `phasesmith-execution`; no global Rayon
pool or numerical state is installed. Python releases the GIL for native work.
The pool is released when that call finishes. Worker counts, parallel maps,
submitted tasks and peak active pool threads appear in `SharedSearchAudit`.
A serial scope avoids pool creation and task buffers. Nested parallel roots are
rejected; internal maps reuse the existing pool. Scope restoration handles errors
and unwinding. Concurrent independent callers own independent budgets, so an
application must allocate its total budget across calls.

## Equations and ordering

There is no change to the independent Python reference equations. For each
reflection h, the existing symmetry orbit sum remains

    F(h) = sum_j f_j(h) exp[-2 pi^2 q_h^T U_j q_h]
                         exp[2 pi i h.x_j].

Values and all coordinate and displacement derivatives are evaluated together.
Each reflection retains the original atom/orbit and parameter operation order.
A map partitions contiguous reflection ranges; results return in range order
and are copied into the original parameter-major Jacobian. Amplitude scale,
residual, objective, R1, scaled-Jacobian and normal-equation reductions retain
original serial order. This leaves independent reference and finite-difference
tests applicable without new tolerances. Direct tests compare every float bit
at 1, 2, 4 and 8 requested workers for both displacement models, uncached and
cached geometry, supplied symmetry, family weighting and complete short fits.

Phase starts, seeded randomness, Fourier trajectories, candidate ordering,
assignment decisions and stopping remain sequential. Wall-time safeguards can
still change completed work if the machine is slower or busy; deterministic
kernels do not make wall deadlines deterministic.

Evaluations below 256 reflection rows or 16384 full-cell atom-row products remain
serial. Otherwise there are at most N tiles, at least 128 rows per tile and a task-count
limit of floor(full-cell atom-row products/8192). Small calls never construct
the pool.
The tile Jacobians together have exactly the original Jacobian's element count;
extra storage is one additional bounded Jacobian plus amplitudes (at most
16 MiB for the existing two-million-entry cap), not N complete model matrices.
There is one owned pool and no nested numerical-library threading.

## Reproduction and comparison

`benchmarks/solution_multicore.py` measures both:

* Single-solve latency: one program at a time, PhaseSmith workers=N against
  SHELXT `-tN`, with N=1,2,4,8. Worker settings rotate in the serial case order.
* Cold batch throughput: at most N independent single-worker processes, for
  each program separately, across the same 35 development cases. This includes
  process startup, Python imports, parsing and serialization. It does not
  claim persistent-service or import-amortized throughput.

Input/resource preparation and reference grading are outside timings. The
native time budget remains 13 seconds; subprocess safeguards remain 15 seconds.
Every latency and batch prediction (560 total) is saved and sealed before the
first development-reference audit. Adapter revision 34 inherits the unchanged
registry from revision 33 without opening raw or holdout data. No holdout
outcomes or raw holdout structures are opened. Source and executable hashes,
input caps, actual worker work and all failures remain in the reports.

The supplied SHELXT 2018/2 executable is x86_64 under the host compatibility
runtime; PhaseSmith is native ARM. Both have matching total requested worker
budgets, with other numerical libraries limited to one thread. Their internal
refinement endpoints and algorithms differ. Batch throughput and solver-only
latency must not be interchanged.

## Measured development results

The 14-core Mac has ten performance and four efficiency cores. All eight panels
retain the exact previous 19 successful structure cases, 26 correct total non-H
counts, 19 correct per-element counts and 15 strict 0.02-Angstrom structures.
Every inferred count vector across all 35 cases is unchanged, including failures.
The seven references above the unchanged input-only resource cap stay in the
denominator. PhaseSmith has no hard timeout in any panel. All 560 predictions
were sealed before any development reference was opened.

Single solves execute one case at a time. The table reports summed wall seconds
for all 35 outcomes, including startup, and library-only PhaseSmith solve seconds.
SHELXT's selected output is graded by the same adapter at every thread count.

| Requested workers | PhaseSmith library sum, s | PhaseSmith wall sum, s | SHELXT wall sum, s | SHELXT structures | SHELXT correct total counts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.269 | 46.176 | 50.877 | 13 | 29 |
| 2 | 35.909 | 43.983 | 42.450 | 14 | 32 |
| 4 | 34.412 | 42.245 | 34.688 | 13 | 32 |
| 8 | 34.997 | 42.893 | 28.480 | 10 | 28 |

Four workers give the smallest PhaseSmith summed solve time: 10.08% less than
one worker, or a 1.112x aggregate speedup. The median paired solve speedup is
1.088x (8.11% less time). Eight workers do not improve on four for these cases.
Twenty-nine cases use parallel maps; peak actual active pool threads reach
2, 4 and 8 at the corresponding settings. Small models remain serial.
The realistic 48-atom, 1054-reflection, eight-step kernel benchmark improves
12.776-to-10.470 ms at eight workers (1.220x), including lazy pool creation.
The 8- and 12-atom kernels stay serial; their timing scatter is not a parallel gain.

SHELXT scales a single solve more strongly. On the same 34 completed cases at
all settings, its median paired reported internal-time speedup is 1.425x at two,
2.479x at four and 2.194x at eight threads. PhaseSmith's median paired library
solve/reported-SHELXT time ratios on those common cases are 0.844, 1.073, 1.709
and 1.463. Reported internal clocks have different boundaries from library and
process timers. The faster SHELXT settings also change its selected models;
these are observed algorithm outcomes rather than an identical-work kernel
comparison. The one-, two- and four-thread SHELXT panels each have one 15-second
hard timeout. Eight threads return before that safeguard, but recover fewer
correct structures. Its low all-35 wall sum therefore needs both qualifications.

Cold batches use N independent single-worker processes. Elapsed time includes
all 35 outcomes and process/import/output overhead; per-job slowdown under load
must not be confused with whole-batch speedup.

| Outer workers | PhaseSmith elapsed, s | PhaseSmith speedup | SHELXT elapsed, s | SHELXT speedup |
| --- | ---: | ---: | ---: | ---: |
| 1 | 46.546 | 1.000x | 50.038 | 1.000x |
| 2 | 24.675 | 1.886x | 25.704 | 1.947x |
| 4 | 14.098 | 3.302x | 19.442 | 2.574x |
| 8 | 9.967 | 4.670x | 17.396 | 2.876x |

At eight outer workers, PhaseSmith processes 3.512 cases/second versus 2.012 for
SHELXT (1.745x throughput). SHELXT's single-worker batches retain 13 correct
structures and 29 correct total counts, with one timeout in every batch. That
15-second tail limits its batch speedup. PhaseSmith retains 19 structures and
26 correct total counts at every batch size. Batch worker intervals and inner
worker statistics independently confirm the total budget bounds.

PhaseSmith's phase starts, seeded trajectories and fit reductions remain serial;
only reflection work is parallelized inside a solve. This explains the modest
single-solve gain. Wall-based phase safeguards complete slightly different cycle
counts under changing load (22947/23029/22941/22957 for latency; 22960 down to
21743 in batches), while all 35 inferred counts and every previous successful
case set remain. These are single-collection, host-sensitive timings.
For this corpus, use four internal workers for one solve and eight outer jobs
with one internal worker each for throughput. The public default remains one.

Raw audited panels, common-completed internal comparisons, worker work, exact
retention sets, source/executable hashes and all per-case count vectors are in
`benchmarks/structure-solution-multicore-*-macos.json`; the main report is
`benchmarks/structure-solution-multicore-comparison-macos.json`.
Kernel samples are in `benchmarks/structure-solution-multicore-kernels-macos.json`.

Reproduce on the same pinned inputs with:

```sh
.venv/bin/python benchmarks/solution_multicore.py --output runs/bounded-multicore-new
.venv/bin/python benchmarks/solution_multicore_verify.py --base runs/bounded-multicore-new
```

Independent Python physical equations and finite differences, exact worker-count
comparisons, scopes/lifetimes and nested-budget tests pass. The complete workspace
validation is recorded separately in
`benchmarks/structure-solution-multicore-validation-macos.json`.
