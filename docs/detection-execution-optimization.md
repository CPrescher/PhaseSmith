# MgSiO3 detection scaling and complete workflow timing

The example now uses eight detection workers, independently of the unchanged
four-worker shadow/acquisition policy. All native numerical kernels, thresholds,
masking, linking radii and scientific choices are unchanged. The current frozen
protocol is `validation/enstatite-raw-workflow-v20.json`. This is the same exposed,
prior-assisted regression dataset; earlier provenance and convergence qualifications
remain in force.

## Detection and linking scaling

Each configuration rereads all 142 original HDF5 frames, with three sequential
repeats. The isolated study uses the frozen original-count mask; saved positions
are compared only after detection and never supplied to the detector/linker.
Every position array and frame fingerprint is bitwise identical in all 12 trials.
There are 4,098 detections linked into 3,047 position tracks before later quality
selection.

| Workers | Median wall time (s) | Speedup over 4 |
|---|---:|---:|
| 4 | 6.568 | 1.000x |
| 6 | 5.041 | 1.303x |
| 8 | 5.003 | 1.313x |
| 10 | 5.048 | 1.301x |

Eight is nominally fastest: detection/linking is about 24% shorter (1.31x).
The 6–10 differences are under 1%; these counts effectively reach a loading
plateau. Ten workers provide no extra speed benefit. The entire isolated sweep
peaks at 1,459,568,640 resident bytes; this is a process high-water mark, not a
per-worker-count RSS comparison.

Two fresh complete raw-frame runs take **23.46 s** and **23.76 s**, versus the
previous v19 repeat's **25.31 s**: about 6–7% shorter. Their numerical workers
take **22.40 s** and **22.66 s**. Both recompute masks, observations and profiles
and pass every numerical-array, complete-candidate and search-history bitwise
gate against v11. R1 is unchanged at **.04085182070733226**; Mg16 Si16 O48
recovery and every original-source hash are unchanged. Reduction/search workers
load no SciPy, h5py or hdf5plugin. Individual full runs are retained rather than
presented as a stable timing distribution.

## Full wall-time breakdown

This table partitions the first 23.46 s full run. Frame reading/decoding is
included in the consuming stage. Summed parallel thread times are excluded.

| Area | Seconds | Share of full CLI |
|---|---:|---:|
| Pixel collection, integration and background | 5.052 | 21.5% |
| Detection and linking, including frame I/O | 4.946 | 21.1% |
| Shadow masks, including frame I/O | 3.263 | 13.9% |
| Profile amplitudes and strong shape fits | 2.781 | 11.9% |
| Profile learning | 2.531 | 10.8% |
| Other worker preparation and bookkeeping | 1.389 | 5.9% |
| Native lossless output archives | 1.145 | 4.9% |
| Process launch, source verification and reference assessment | 1.058 | 4.5% |
| Structure search, including child launch | 0.785 | 3.3% |
| Physical corrections, empirical scaling and error model | 0.271 | 1.2% |
| Grain indexing and geometry fitting | 0.238 | 1.0% |
| **Total** | **23.460** | **100%** |

The fitting row includes fixed-profile amplitudes for all 7,277 target
observations and the same 146 strong shape fits. Within the small combined
rows, geometry fitting takes .0036 s, empirical scaling .216 s and error-model
estimation .0225 s. The other-worker row includes prediction, domain/background
bookkeeping, array preparation and stage setup. The launch/verification/assessment
row includes parent source checks and reference assessment after the numerical
worker stops. Worker peak RSS is 3,221,454,848 bytes (about 3.00 GiB).

## Why additional detection workers stop helping

The full observer records these detection subparts without changing arguments:

| Detection subpart | Seconds | Relationship |
|---|---:|---|
| Ordered loading/detection pipeline | 4.901 | Wall time |
| Adjacent-frame linking | .00035 | After detection |
| Track summaries and table assembly | .0445 | After detection |
| Image loading | 4.111 | Inside pipeline; overlaps worker tasks |
| HDF5 reading/decoding and array boundary | 2.194 | Included in image loading |
| Original-count source verification | .895 | Included in image loading |
| Orientation, reserved mask, second hash and loader bookkeeping | 1.022 | Remaining image-load work |
| Mask loading/construction | .683 | Inside pipeline; overlaps worker tasks |

Serial image plus mask loading occupies about 4.79 s of the 4.90 s pipeline.
Detector tasks overlap that work; their cumulative elapsed thread time is
21.20 thread-seconds, **not additional wall time**. Linking itself is under a
millisecond and track assembly only about .04 s. The measured limit is the
loading pipeline, not the linking algorithm.

## Where to optimize next

1. **Pipeline frame validation and masks.** HDF5 reading/decoding takes about
   2.19 s, followed on the caller thread by verification, additional hashing,
   orientation and mask construction. Benchmark moving independent per-frame
   work into bounded workers while preserving source-order callbacks/errors.
   The loader hashes the original counts for both validity verification and its
   own record; reuse a digest only after successful verification, retaining all
   source-identity checks. Detection and pixel preparation share this loader,
   so both could benefit. The gain requires measurement; more detector threads
   alone have reached their plateau.
2. **Reuse original decoded frames within an explicit budget.** Masks, detection
   and pixel preparation each read the selected scan. A shared immutable cache
   could avoid repeated HDF5 passes, with source identity/invalidation retained.
   The complete selected uint32 payload costs **2,539,537,088 bytes (2.37 GiB)**
   before masks/workspace; compare time against memory explicitly.
3. **Profile fitting and learning.** They still account for about 5.31 s.
   Measure repeated allocation/fixed-coordinate work in strong-fit objectives
   and the serial/global-bank learning path. Previous bank-level learning
   parallelism gained only about 2%, so simply raising that worker count is
   unlikely to be the next useful change. Preserve operation order and support.

Archive output, scaling, geometry and structure search are now smaller shares.
Linking has negligible elapsed cost at this dataset size. No projected gain
above is claimed as measured.

## Reproduction and validation

The committed isolated protocol is `validation/enstatite-detection-scaling-v1.json`.
Run `benchmarks/detection_execution.py` using an external completed v19 run, then
`benchmarks/enstatite_profiled_workflow.py` with the original external data root
and a new external output directory. The latter also writes the separate
`timings-detection.json` alongside the full exclusive timing records.

Fifty focused rotation/workflow tests pass, including exact position output
checks at 6/8/10 workers, source-order callbacks, reused loader buffers, masks
and cancellation. Production Rust and Python numerical kernels are unchanged;
this change selects an existing execution capability and adds observational
timing. Raw files, full arrays, outputs and every trial remain outside Git.
The complete repeat data, exact-equivalence reports and full timing events are
in `benchmarks/enstatite-detection-execution-macos.json`.

## Completed original-frame follow-up

The loading/validation/mask and decoded-original-frame proposals above are now
implemented and measured separately. Digest reuse reduces the isolated median
5.003→4.138 s; bounded native preparation and direct owned HDF5 buffers reach
3.562 s without a decoded cache. The final v23 adds an optional 1 GiB source-bound
decoded-count cache: complete runs take 18.65/18.56 s and match v11 bitwise.
The cache contributes only .52–.61 s over warm uncached execution and raises
peak RSS from 3.02 to about 4.03 GiB. It stays opt-in in the library.

Native preparation now performs verification and mask composition inside image
load. A zero Python verification timer therefore means the work moved into Rust;
all source checks still run. Detection takes 3.321 s in the final repeat, with
21.602 summed measurement thread seconds overlapping across eight workers,
.000351 s linking and .0433 s track assembly. The next detector candidates are
workspace/buffer reuse and array/native copy reduction, measured within the
two-preparation/eight-detection worker balance. Profile learning/fitting still
consume 5.23 s and are also useful targets. The current complete wall partition,
all slower trials and memory tradeoffs are in
[frame input optimization](frame-input-optimization.md) and
`benchmarks/enstatite-frame-input-macos.json`.
