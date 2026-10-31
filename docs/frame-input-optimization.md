# Measured original-frame input optimization

Every stage preserves original integer counts, masks, source identities and
all scientific controls. Native detector/profile/geometry equations and finite
support remain unchanged; independent numerical references stay authoritative.
Each full protocol is committed before frames, and each full run must reproduce
all retained v11 arrays/candidates/search histories bit for bit. Large input and
output products remain outside Git.

## 1. Reuse a freshly verified digest

Validity providers now return their freshly computed digest after checking the
current original frame. Combined providers still execute every check. Geometric
providers return None; ordinary hashing remains the fallback. A digest is reused
only for the same unchanged owned frame, never supplied unchecked by the caller.

At eight detection workers, three-repeat median time falls from 5.003 s to
4.138 s (1.21x), with all positions and frame fingerprints byte-identical.
Fresh full v21 runs take 32.34/23.48 s CLI and both pass all identity gates.
First-use indexing/setup and archive-write variation mean this is not evidence
of an overall improvement over the v20 23.46/23.76 s pair. These full measurements
are retained, not replaced by the isolated gain. Seventy-five focused tests pass.
The complete record is `benchmarks/enstatite-frame-input-stage1-macos.json`.

## 2. Native count preparation

`phasesmith_spots::frame_input` accepts owned original u32 snapshots and an
immutable scan plan. It applies the declared orientation in place, computes a
native-endian SHA256 once, verifies every inferred-provider digest, and composes
the reserved-count/block/fixed/DAC masks. The DAC kernel, whole-block expansion,
closed exposure and finite support are unchanged. SHA256 and complete masks are
checked independently against hashlib and the ordinary array frontend, including
partial edge blocks, fixed masks, reversed rows, zeros and high/reserved counts.
Expected hashes come from mask provenance; no unchecked current digest is accepted.

A private Rust pool prepares independent frames; results/errors remain ordered.
The byte cap admits input snapshots, native owned counts, masks, temporary DAC
mask and digest scratch before native copies. Plan storage, downstream detector
workspace and thread stacks are separate. Native row errors are delivered when
that row is requested, preserving earlier valid results/callbacks; no partial
scientific workflow result is returned on failure. The optional frontend batches
ordered HDF5 reads; existing numerical workers can overlap these native batches.

The initial native trial regresses to 6.042 s and is retained. Its software-only
ARM SHA backend and per-pixel mask divisions were corrected with upstream sha2's
AArch64 hardware feature (runtime CPU fallback) and row-run mask expansion. No
handwritten cryptography or numerical equation change is introduced. Cargo pins
sha2/sha2-asm; other architectures retain the ordinary upstream path. Corrected
four-frame batches take 4.325 s; one/two-frame batches 4.038/3.674 s. Every trial
keeps exact original positions/fingerprints. Two preparation workers plus eight
detector workers stay within the ten-worker execution choice. The full v22
example explicitly selects two-frame batches and a 256 MiB payload cap.

## 3. Source-bound decoded counts

`phasesmith_hdf5::cache::CachedReader` adds an optional 0–4 GiB decoded-payload
cap. It admits the first unique original frames that fit, without eviction or
scan-thrashing, and returns independent owned copies. Cache payloads never
contain masks, normalized counts or refined information. One fresh open source
and dataset own each cache, with canonical path, size, timestamps and platform
file identity checked before reuse. Changed/missing sources clear the cache
and fail until reopening; indirect/external storage is rejected for cached reads
until separate source validation exists. The uncached reader remains available.

Fallible cache reservation can decline admission and retain streaming. Payload
capacity excludes metadata, caller copies and ordinary workspace; process RSS
is measured separately. Caller mutation cannot alter cached counts. Tests cover
source changes, isolation between handles, tiny-budget streaming, bounds and
external links. Full runs create an empty cache and never read archived arrays.
It is shared across raw masking/detection/collection and closed before search.

The v23 example requests 1 GiB, which admits about 60 of 142 original frames.
The choice is a memory/payload control, never an atomic-reference or R1 choice.
Exact cache hit/decode/admission counters and observed performance are recorded
separately from native preparation. All physics, support, geometry and independent
reference/oracle fixtures remain unchanged.

The matching final binary exposed an uncached-reader regression (4.241/4.270 s
detection medians) even though native preparation itself stayed near .87 s
summed batch wall time. The read adapter now transfers HDF5's owned C-order
buffer directly, with a logical-order fallback for nonstandard layouts, instead
of traversing/copying it through a generic array iterator. The measured final
uncached native median is **3.562 s**, retaining every position/fingerprint byte.
All older trials, including the initial slowdown, remain in the stage report.


## Full measured progression and cache tradeoff

| Stage | Isolated detection median | Full CLI first / repeat | Worker first / repeat |
| --- | ---: | ---: | ---: |
| Prior eight-worker v20 | 5.003 s | 23.46 / 23.76 s | 22.40 / 22.66 s |
| Digest reuse, v21 | 4.138 s | 32.34 / 23.48 s | 23.82 / 22.20 s |
| Native preparation + owned HDF5 buffer, uncached v22 | 3.562 s | 27.52 / 19.17 s | 19.38 / 18.13 s |
| Same native binary + decoded cache, v23 | measured separately below | 18.65 / 18.56 s | 17.62 / 17.57 s |

All full runs pass strict v11 equivalence: every retained array's dtype, shape
and bytes, complete candidate records and phasing histories. Source manifests
remain unchanged. The structure contains Mg16 Si16 O48, R1
**.04085182070733226**, with the same .007524 Å maximum / .004539 Å RMS
reference displacement. Original HKLs, archived numerical arrays and atomic
reference coordinates never enter the numerical worker; reference assessment
runs in the parent after it exits. Workers reject scipy/h5py/hdf5plugin imports.
There are still 4,125 signed observations (220 negative), 246 teachers, 7,277
profile targets and 13,267,065 pixel samples. No scientific equation, support,
convergence rule or uncertainty threshold changes. Python orchestration remains
in this example; a standalone Rust command is not added here.

The full pass reduces mean CLI elapsed time by **21.2%** versus the v20 pair.
These are fresh workers and empty decoded caches, with ordinary filesystem and
code caches not artificially flushed. First-use setup/indexing varies: v22's
first CLI is 27.52 s versus its 19.17 s repeat, with numerical workers
19.38/18.13 s. Preserve both; do not interpret parent/setup variation as a native
speedup. Likewise, digest reuse establishes its isolated benefit but no clear
net full-run benefit alone. Both v22 controls and v23 runs use the same immutable
final-v4 release wheel, with explicit zero/1 GiB decoded budgets.

Three fresh-reader repeats of three original-frame passes take median
**5.461 s uncached / 4.004 s cached (1.36x)** for native reads and owned array
returns. Independent byte digests outside the timers match every original count.
The full cached gain is only **.52–.61 s (2.7–3.2%)** over the 19.17 s warm
uncached run, because reads overlap compute. A 1 GiB budget actually admits
**1,073,043,840 bytes / 60 frames**, with **426 requests, 120 hits, 306 decodes,
zero invalidations** per full run. Peak worker RSS is **3,245,719,552 bytes
(3.02 GiB)** uncached and **4,327,161,856 / 4,325,588,992 bytes (about 4.03 GiB)**
cached. Payload caps exclude normal caller/workspace allocations and are not
whole-process memory limits. The library default remains zero; the example
selects caching explicitly and users can choose the uncached v22 protocol.

## Final full wall partition

This partitions the final **18.56 s** CLI repeat. Raw reads, hashing, mask
composition and native parallel work are included in their owning stages;
nested and overlapping worker times must not be added again. The final row
accounts for parent verification/assessment, launch and uninstrumented process
initialization. The structure-search row includes its subprocess startup.

| Area | Seconds | Share |
| --- | ---: | ---: |
| Shadow estimation | 2.730 | 14.7% |
| Detection and linking | 3.321 | 17.9% |
| Pixel collection/background | 2.669 | 14.4% |
| Profile learning | 2.516 | 13.6% |
| Profile fitting | 2.719 | 14.7% |
| Lossless archive output | 0.973 | 5.2% |
| Corrections/scaling/error model | 0.266 | 1.4% |
| Indexing/geometry | 0.243 | 1.3% |
| Structure-search process | 0.758 | 4.1% |
| Other worker assembly/bookkeeping | 1.381 | 7.4% |
| Launch/parent checks/remaining initialization | 0.986 | 5.3% |
| **Total** | **18.560** | **100%** |


Detection now takes 3.321 s, with 21.602 summed measurement thread seconds across
eight workers. Image load is 2.044 s including 1.167 s HDF5 reads; preparation
and loading overlap detector work. Python verification and mask-access timers
are approximately zero because native batch preparation now owns those checks
and composition inside image load. They have not been removed. Linking is
.000351 s and track assembly .0433 s, so linking remains a negligible target.

## Remaining useful measurements

1. **Profile objective and learning work (5.23 s combined).** Measure repeated
   allocations, fixed-coordinate work and the global learning bank. Previous
   bank-level parallelism gained only about 2%, so more learning threads alone
   are unlikely to help much. Preserve exact scalar order and finite support.
2. **Detector workspace and copies (3.32 s).** Loading is no longer almost the
   whole pipeline. Profile reusable detector buffers and array/native copies,
   then remeasure the two-preparation/eight-detector balance within ten workers.
   Any additional worker choice needs a new measured deterministic sweep.
3. **Shadow estimation and pixel preparation (2.73/2.67 s).** Profile filter and
   normalization allocations and sample/background preparation. A larger decoded
   cache would require a separate time/memory comparison; the 1 GiB gain is small.
4. **Assembly and memory lifetime.** Other worker assembly/bookkeeping is 1.38 s,
   including 1.09 s integration orchestration. Profile that before a larger
   Rust-driver port. Releasing decoded counts immediately after the last raw
   read is a future memory experiment, not part of the current implementation.

## Reproduction, tests and retained trials

The full protocols `validation/enstatite-raw-workflow-v21.json` through `v23.json`
were committed before frames. Run `benchmarks/enstatite_profiled_workflow.py`
with the original external root, an explicit protocol and a new external output;
then compare using `tools/compare_enstatite_execution.py` against retained v11.
`benchmarks/detection_execution.py --input-execution <execution.json>` measures
native count preparation; `benchmarks/decoded_frame_cache.py --run <completed-run>`
measures repeated original-count reads independently of detector work.

The machine-readable cumulative record is
`benchmarks/enstatite-frame-input-macos.json`; individual records are
`benchmarks/enstatite-frame-input-stage1-macos.json` through `stage3-macos.json`.
They retain the initial 6.042 s regression, corrected four/one/two-frame trials,
4.241/4.270 s cache-wrapper regressions, paired reader diagnostics, final 3.562 s
uncached median and every full run. Hardware-SHA/row expansion and buffer-transfer
corrections are measured as their combined configurations; no isolated gain is
invented for each internal edit. Independent hashlib/array comparisons cover
native byte hashing, orientation, partial blocks, composed DAC/fixed/reserved
masks, ordered failure and mutation isolation. Source-bound cache tests cover
file changes, tiny-budget streaming and external-link rejection. All numerical
reference equations and oracle fixtures remain unchanged: hashing/lossless cache
and execution changes are outside those equations. All raw files, full products,
wheels and logs remain external under `/Users/clemens/ChatGPT/PhaseSmith`.

Final validation passes **515 Rust tests** (34 opt-in tests ignored) and
**3,267 Python tests** (11 external-data skips; 34 slow tests deselected by the
repository's default configuration), plus strict all-target/all-feature Clippy,
Cargo formatting and diff whitespace checks. Full fresh MgSiO3 runs separately
exercise the real external-data workflow at both cache budgets. Final validation
logs and wheel provenance remain external and are linked from the stage records.
