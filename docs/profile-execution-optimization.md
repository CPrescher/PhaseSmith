# Bounded profile execution optimizations

These changes preserve the finite voxel model, mask/reference selection,
iteration counts, derivatives, accumulation order and scientific controls.
They change execution and local work reuse only. Large data/products remain
outside Git. Every stage has its own archived benchmark and fresh MgSiO3 run.

## Stage 1: immutable learning geometry cache

For a fixed grid, teacher centre, detector map, local scale and sample footprint,
the exact overlap `A_ij` is constant throughout learning. The forward operation
is `expected_i = sum_j A_ij p_j`; the original gradient accumulates
`reference_weight_i * A_ij * (target_i - expected_i)` into each global/local
bank. Cache the original nonzero `(voxel, overlap)` visits in source sample and
voxel order. Initialization, both forward bank sums and backward accumulation
retain the original floating-point operation sequence. No rounding, interpolation,
renormalization or new finite-support convention is introduced.

The native `profiles::Execution` / optional Python `ProfileExecution` controls
additional geometry-cache bytes (default 64 MiB; zero disables caching; maximum
1 GiB). Count before allocating; use exact vector reservations and reject an
allocation exceeding the declared cap. If admission or reservation fails, use
the original streaming equations. Cache rows include only accepted teachers,
and the cache is local to the immutable call: it is never reused after geometry,
reference selection, grid or source data changes. Actual admitted capacity,
row count and intersection count are reported separately from the profile model.

The independent NumPy reference remains unchanged because the equations are
unchanged. Native and Python tests compare cached/streaming paths bitwise for
axis-aligned and affine/signed-rocking inputs, reference weighting and tiny
budgets. Existing independent value/partial/support/normalization tests remain
authoritative. GSAS-II does not provide this empirical single-crystal oracle.

`benchmarks/profile_execution.py` measures the complete real learned bank and
baseline amplitude integration on the preserved 7,277-reflection shoebox case.
Three deterministic repeats, explicit execution controls, admitted cache bytes
and process peak RSS are reported. Fresh full runs use the existing observational
wall-time wrapper and compare every numerical array and saved candidate with
retained v11. Timings and correctness are reported separately at each stage.

The first real-case measurement at the 64 MiB library default correctly falls
back to streaming (38.81 s versus retained 38.78 s). Its result is preserved.
An admission diagnostic measures 177,855,620 cache-capacity bytes (169.6 MiB),
506,475 teacher samples and 13,808,351 intersections, with a bitwise-identical
learned bank. The full v13 example explicitly requests a 256 MiB additional
cache cap; v12's unadmitted 64 MiB protocol/measurement are retained. This
resource choice uses geometry capacity, never R1 or an atomic reference.

Stage 1 completes with a three-repeat learning median of 2.543 s versus
38.780 s (15.25x). The fresh v13 numerical worker takes 54.89 s versus
87.77 s for retained v11; full CLI wall time is 63.39 s versus 88.80 s.
The parent setup/assessment portion varies, so these are single full-run
measurements rather than a claimed stable distribution. Every numerical array,
full candidate record and phasing history is bitwise identical. Source files
are unchanged. Native workspace tests pass 510 cases and Python 3,239, plus
strict Clippy/formatting. Reports:
`benchmarks/enstatite-profile-execution-stage1-macos.json` and the cumulative
`benchmarks/enstatite-profile-execution-macos.json`.

## Stage 2: deterministic parallel banks

Global and local profile banks have independent objectives: a bank's update
reads only its own previous voxels and fixed teacher data. There is no joint
normalization between iterations. Run the complete requested iteration sequence
for each bank as a separate native task, keeping every within-bank sample/voxel
visit and arithmetic operation in source order. Final support selection and
normalization are unchanged. This avoids nondeterministic floating reductions
and requires no duplicated geometry matrix.

A stable local-bank row index is admitted within the same cache cap. A private
Rayon pool uses up to the requested 1–4 workers (no more than the bank count).
If geometry/index admission, allocation or pool creation fails, use the serial
cached/streaming path and report actual versus requested workers. Index bytes
are included in admitted cache capacity. Per-worker gradient buffers replace
the serial bank-wide gradient allocation; at most one gradient per active bank
is live. The global bank contains all teachers, so bank parallelism does not
imply fourfold speedup. Measurements determine the actual additional gain.

Stage 2's measured learning median is 2.493 s, versus 2.543 s after caching:
only a 1.02x incremental gain, 15.56x versus the original. All numerical arrays
and candidate/search histories remain bitwise identical. The four-worker
cache including its row index uses 181,907,500 capacity bytes under the same
256 MiB cap. The full CLI run takes 61.82 s; single-run parent/machine overhead
varies, so no large additional end-to-end gain is claimed. The global bank and
serial preparation limit this coarse bank-level parallelism. Report:
`benchmarks/enstatite-profile-execution-stage2-macos.json`.

## Stage 3: fixed-profile integration

Keep the legacy streaming `profiles::fit` as a control/convenience entry point.
The resource-controlled `fit_with_execution` first evaluates each sample's
profile mass in its original voxel order. Independent chunks may fill disjoint
probability rows in parallel; mass sums retain source order. Only integer
overlap flags couple reflections. Group these by exact exposure-bound bit
patterns and keep one exposure's pixel map at a time, preserving the original
flag criterion without an all-scan occupancy hash.

Stable per-reflection group indices retain the original group/sample order.
Each reflection independently executes the unchanged eight expected-count
updates (or one supplied-variance pass). The first pass saves each original
per-voxel derivative contribution `profile[j]*dp[j]`, including zeros and signs,
in callback order. Subsequent noise passes replay these contributions with the
current weights and residuals, retaining every multiplication/addition order.
Do not sum them into one derivative first: that would reorder floating point.
Mass/partial evaluation still uses the original fused finite-intersection kernel.

Divide the declared extra derivative-cache cap among active workers. Each
working reflection admits its row pointers first and grows its contiguous
contributions only within that cap; admission/allocation overflow discards
that cache and continues streaming without changing the fit. Cache capacities,
counts of cached/uncached rows, actual/requested workers, and a conservative
concurrent-cache upper bound are reported. The cap excludes caller input,
output, ordinary algorithm workspace and OS thread stacks; process peak RSS is
measured separately. The overlap map now scales with a single exposure.

The final integration is benchmarked both with one worker and four, before
fresh full runs for each resource configuration. All science controls, signed
intensities, covariance propagation, support and quality flags must remain exact.

Stage 3a (v15) measures the serial cache plus per-exposure overlap workspace.
The three-repeat integration median falls from 17.792 s to 7.971 s (2.23x).
Fresh raw-to-search worker time is 44.75 s and full CLI time 52.89 s, versus
53.97/61.82 s after stage 2. Every numerical array, candidate record and phasing
history is bitwise identical to v11. The largest live reflection cache is
1,583,672 bytes under the explicit 256 MiB cap; process memory is reported
separately. Report: `benchmarks/enstatite-profile-execution-stage3a-macos.json`.

Stage 3b (v16) measures four native integration workers with the same cache cap.
The integration median is 2.185 s: 3.65x versus the serial cache and 8.14x versus
the original. The fresh numerical worker takes 37.86 s versus the original
87.77 s (2.32x); full CLI takes 38.95 s versus 88.80 s (2.28x). All numerical
arrays, full candidate records and phasing histories are bitwise identical;
R1 remains .04085182070733226. The derivative-cache concurrent capacity bound
is 6,334,688 bytes, versus a 256 MiB declared cap. The learning cache is
181,907,500 bytes, admitted separately during learning. Full worker peak RSS
is 3,246,407,680 bytes; the cache cap is not a cap on the entire process.
Native workspace tests pass 510 cases; Python passes 3,245, with 11 external-data
skips and 34 slow-test deselections, plus strict Clippy and formatting.
Report: `benchmarks/enstatite-profile-execution-stage3b-macos.json`.

## Measured progression

| Configuration | Learning median (s) | Baseline integration median (s) | Full worker (s) | Full CLI (s) |
|---|---:|---:|---:|---:|
| Original v11 | 38.780 | 17.792 | 87.77 | 88.80 |
| 1: learning geometry cache, v13 | 2.543 | unchanged path | 54.89 | 63.39 |
| 2: four-worker learning, v14 | 2.493 | unchanged path | 53.97 | 61.82 |
| 3a: serial integration cache, v15 | unchanged path | 7.971 | 44.75 | 52.89 |
| 3b: four-worker integration, v16 | unchanged path | 2.185 | 37.86 | 38.95 |

Component benchmarks are medians of three isolated sequential repeats using the
same saved real shoeboxes; full runs independently reread raw frames and have
one measurement each. Full runs include strong shape optimization in addition
to baseline integration. Parent cold-start overhead is larger in v13–v15; the
CLI change from v15 to v16 should not be attributed entirely to parallelism.
The modest 2% learning gain in stage 2 is retained honestly. The initial 64 MiB
cache rejection and measurement are preserved, as are all earlier trials.
The cumulative machine-readable report is
`benchmarks/enstatite-profile-execution-macos.json`.

## Follow-up 1: continuous fitting queue and worker scaling

A fixed-size native queue replaces per-batch barriers in conditional profile
fitting. Each independent reflection retains exactly the same finite-support
value/partial passes, optimizer controls and scalar operation order. Results
and progress callbacks are consumed in source order on the calling thread.
Cancellation stops new work, drops the bounded result channel and joins active
workers; no partial result escapes. Explicit profile worker limits are now
1–32, with default one unchanged. Cache bytes remain a shared per-pool cap,
not a per-worker multiplication. The 4/6/8/10 worker choice will use observed
runtime, never intensity agreement or atomic-reference scores. Independent
references and GSAS-II coverage are unchanged because equations are unchanged.

Follow-up 1 measures the complete real-case amplitude/146-strong-fit stage:
old four-worker batches 8.731 s; continuous scheduling with 4/6/8/10 workers
6.128/4.178/3.372/2.921 s (three-repeat medians). All output hashes match.
Fresh v17 passes every bitwise gate; worker time is 33.91 s versus 37.86 s.
Its CLI is 42.49 s versus 38.95 s: cold parent/setup overhead accounts for
the reversal, so this single run does not establish a CLI gain. Both are
retained. Report: `benchmarks/enstatite-next-execution-stage1-macos.json`.

## Follow-up 2: bounded independent frame preparation

`FrameExecution` adds optional 1–32 frame workers and an explicit owned-payload
budget (default 512 MiB, maximum 1 GiB), independently of scientific integration
options. The optional Python adapter owns ordered loading and immutable snapshots;
background fitting, sample collection and final cross-frame accumulation stay in
Rust. Each frame has the same kernel inputs and scalar operations; results,
sample sinks, errors and progress are consumed in source order. Cancellation
joins active work and returns no partial result. The default serial path retains
its callback timing. No independent equation/reference or oracle fixture changes.

Admission charges raw snapshots plus normalized/native image/mask/variance
copies; queued charge is bounded and recorded. Ordinary result/scratch arrays,
thread stacks, and at most one transient loader snapshot are separate from this
payload cap. Requested concurrency can be reduced by memory admission.
Tests exercise reused loader buffers, reverse exposures, masks, supplied and
expected-count variances, mean/plane backgrounds, exact sample/covariance bytes,
ordered errors/cancellation and budget-limited concurrency.

The raw benchmark measures 142 source frames and the 7,277 frozen target
predictions, without competing-domain exclusions; each execution control uses
exactly that same cohort. The separate full run retains all competing domains.
Serial baseline is 8.964 s; new serial/4-worker/10-worker medians are
8.849/5.039/5.072 s, all hashes identical. Four workers are selected using time
and lower queued memory, not scientific agreement. Their charged payload is
411,333,472 bytes, so the full example requests a 512 MiB cap.

The full all-domain v18 run also passes every bitwise identity gate. Frame
preparation takes 5.132 s versus 9.350 s in v17. The numerical worker falls
from 33.91 s to 29.41 s; CLI from 42.49 s to 37.80 s, with the same cold-parent
qualification. Report: `benchmarks/enstatite-next-execution-stage2-macos.json`.

## Follow-up 3: native lossless numeric archives

`phasesmith_persistence::numeric_archive::write_numeric_npz` writes NPY members
inside a standard NPZ archive using the existing pure-Rust ZIP/DEFLATE stack.
Level zero stores members; levels 1–9 select lossless compression effort. Shape,
dtype/endianness, signed zeros and NaN payload bytes are preserved exactly.
Metadata remains ordinary uint8 arrays; no pickle or executable objects are
accepted. This is a separate I/O API; existing project-bundle encoding is unchanged.

Validate flat names, duplicate keys, supported numeric dtypes, rank (up to 32),
shape/byte-count agreement, at most 4,096 arrays and at most 1 GiB total raw
payload before creating a file. Creation is exclusive, timestamps deterministic,
and a newly created partial file is removed on write failure. The optional
NumPy adapter copies payloads before releasing the interpreter; that copy and
compressor workspace are distinct from frame/profile cache caps.

The example can explicitly choose the native writer for mask/case snapshots
and every numeric stage archive; legacy controls remain reproducible. Benchmark
three complete real-case writes per backend/level, with output sizes and exact
NumPy read-back, then run the complete raw workflow again. Native tests also
write/read ZIP/NPY bytes without Python. No numerical equations, support or
reference/oracle fixtures change.

The first archive trial completed three exact level-1 writes (median 1.589 s)
but ran out of disk space during the third uncompressed write. Its error log
and measurements are retained. Only verified duplicate files made by this
benchmark were removed (1,299,741,524 bytes); source data and all full workflow
products remain untouched. Repeats now journal each verified result and can
discard only their own successful archive after exact read-back.

The clean bounded-storage rerun gives native level-1/uncompressed medians
1.167/.579 s. Level 1 is 152,576,907 bytes versus NumPy level 6's 143,890,029
bytes (about 6% larger); uncompressed is 353,403,826 bytes. Level 1 is selected
for the full v19 run to retain compression. The interrupted trial remains in
the report; no numerical thresholds or scientific settings changed.

## Completed follow-up measurements

| Configuration | Numerical worker (s) | Full CLI (s) |
|---|---:|---:|
| Previous v16 | 37.86 | 38.95 |
| Continuous fitting queue, ten fit workers (v17) | 33.91 | 42.49 |
| Four bounded frame workers (v18) | 29.41 | 37.80 |
| Native compressed archives (v19, first run) | 25.22 | 33.39 |
| Same v19 protocol, second fresh raw run | 24.20 | 25.31 |

Every row passes all numerical-array, full-candidate and search-history bitwise
gates against retained v11; R1 remains .04085182070733226. No masks, choices,
optimizer budgets, equations or scientific thresholds were changed. The first-use
CLI rows include about eight seconds outside the worker; the final repeat has
about one second. Indexing also varies (1.42 s first run, .23 s repeat). Do not
attribute that variation to archive compression or claim a stable timing
distribution from these individual full runs.

| Component, three-repeat median | Before (s) | After (s) | Speedup |
|---|---:|---:|---:|
| Complete amplitude/strong-profile fitting | 8.731 | 2.921 | 2.99x |
| Raw frame preparation, fixed target cohort | 8.964 | 5.039 | 1.78x |
| Lossless archive writing, matched bounded-storage policy | 4.817 | 1.167 | 4.13x |

The new archive is 6.04% larger. The initial disk-limited and baseline trials
remain available separately; the bounded-storage baseline was remeasured after
removing only verified duplicate benchmark output. Both backend comparisons
retain exact byte read-back, all repeat times and sizes. No original source or
full workflow archive was removed.

Final example resource controls are four learning workers, ten baseline/shape-fit
workers, four frame-preparation workers, and native archive level 1. The profile
cache cap remains 256 MiB per phase; frame queued-payload cap is 512 MiB. Peak
worker RSS in the repeat is 3,212,050,432 bytes (2.99 GiB), separate from those
cache/payload caps. Production numerical kernels and archive compression are
Rust; optional Python orchestration remains.

Validation passes 511 Rust tests (34 ignored) and 3,258 Python tests (11 external
data skips, 34 slow deselections), strict all-feature Clippy and formatting.
Stage reports are `benchmarks/enstatite-next-execution-stage1-macos.json` through
`stage3-macos.json`; the cumulative summary is
`benchmarks/enstatite-next-execution-macos.json`, also linked by the earlier
`benchmarks/enstatite-profile-execution-macos.json`. All large products are
external.

## Detection follow-up

The subsequent v20 detection worker study measures 4/6/8/10 workers and retains
eight as nominally fastest, with a serial-loading plateau above six. Full fresh
runs are 23.46/23.76 s and remain bitwise identical. The complete current timing
partition and next bottlenecks are in
[detection execution optimization](detection-execution-optimization.md).
The record is `benchmarks/enstatite-detection-execution-macos.json`.
