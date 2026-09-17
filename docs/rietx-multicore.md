# rietx multicore follow-up

The optimization target is the lowest stable, scientifically valid Rwp with
verified convergence. Faster fits stopped at higher Rwp are useful diagnostics,
but are not the target result. Compare time to comparable fit quality while
retaining physical-model and QPA checks; the lowest observed Rwp is not proof
of a global optimum.

## Existing parallelism

Inspection of installed rietx 1.4.0 and its upstream
[`compiled.py`](https://github.com/yue-here/rietx/blob/main/src/rietx/model/compiled.py)
confirms that it has compiled Numba kernels with `nogil=True`, distributed over
a persistent `ThreadPoolExecutor`. `RIETX_COMPILED_THREADS` chooses the worker
budget before pool creation. The default caps the budget at eight; an explicit
environment override can choose a larger value.

`_spread` runs batches smaller than 512 rows inline. The QARR case dispatches
batches of 36, 58 and 128 reflection/line rows, so no pool is created under the
normal threshold. Setting eight workers alone therefore does not make this
particular refinement run on eight cores.

Profile values and derivative bases can run concurrently. Accumulating
reflections into overlapping output windows remains serial to avoid races and
preserve addition order. Successive optimizer iterations also depend on prior
results. Source comments about historical `prange` timing are the author's
measurements, not a general limitation of Numba or Python.

## Forced-thread experiment, 2026-09-17

`benchmarks/probe_rietx_thread_threshold.py` temporarily lowers the threshold
inside a fresh process. Installed rietx files, its model, optimizer settings and
accuracy choices are unchanged. Each configuration has one full-workflow warmup
and five timed repetitions, with BLAS/OpenMP restricted to one and shared
PhaseSmith preparation explicitly restricted to one worker. Configurations ran
sequentially, not in alternating order, so small timing differences should not
be interpreted as precise universal overhead estimates.

| rietx setting | Pool created | Full workflow median (s) | After preparation (s) | After-preparation CPU/wall |
|---|---|---:|---:|---:|
| 1 worker, normal threshold | no | 0.4184 | 0.3502 | 1.000 |
| 2 workers, threshold 1 | yes | 0.4315 | 0.3625 | 1.034 |
| 8 workers, threshold 1 | yes | 0.4475 | 0.3780 | 1.107 |

Every scientific result record matches exactly across all configurations and
repetitions: Rwp 19.260110440383335%, maximum phase-fraction error
0.9125382758793987 percentage points. All existing rietx workflow quality gates
pass. CPU/wall sums process thread CPU times; it estimates average CPU usage,
not simultaneous peak activity or CPU affinity. Forced eight-worker execution
averages only about 1.11 busy cores over the fit/reporting portion.

These results support keeping the small-batch threshold on this dataset. They
do not measure rietx's scaling on larger reflection sets. Plausible further
parallelization designs include batching more work per dispatch, phase-level
work with private buffers and a deterministic reduction, or partitioning output
ranges to avoid scatter races. Each needs an actual implementation and scaling
measurement before claiming a speedup.

Independent patterns or independent starting points can also be scheduled in
separate processes. That can improve throughput or exploration of local minima;
it does not make one optimizer trajectory parallel, and a sequential series
that initializes each fit from its predecessor has a dependency to preserve.

Raw audit records: `validation/results/rietx-20260917-serial.json`,
`rietx-20260917-forced-2workers.json`, and `rietx-20260917-forced-8workers.json`.
