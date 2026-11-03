# Rotation position processing: serial optimization and bounded parallelism

**Historical research record.** Measurements and conclusions below apply to their
recorded builds. The 3 November 2026 [reconciliation](integration/unfinished-work-reconciliation.md)
retains this evidence and adapts reproduction helpers to the supported API; it
does not rerun or replace the historical scientific comparisons.

A subsequent [threshold-loop optimization](threshold-loop-optimization.md)
reduces native time further while retaining these exact scientific records.
Its report contains the current one/four-worker comparison with DIALS.

Measured 2026-10-25 on an Apple M4 Pro (14 CPU cores, 48 GiB RAM), macOS arm64,
Python 3.13.7 and NumPy 2.4.1. These are desktop engineering measurements, not
cross-platform guarantees or physical peak precision/recall validation. All work
remains under the opt-in experimental-spots feature and development branches.

## How peaks are found

For every detector pixel, the existing finder estimates background and its
variance in a clipped square annulus (default outer radius 8, inner radius 2).
A pixel must exceed both an absolute threshold and the noise-scaled significance
threshold. A second pass excludes first-pass bright pixels from background
estimation. Summed-area tables make these background passes linear in pixel
count. Eight-connected bright components become padded finite apertures;
positive background-subtracted counts determine spot centroids. Detector sums
are internal weights, not integrated rocking intensities.

After frame detection, the existing deterministic linker joins nearby detections
in adjacent angularly contiguous frames and computes compensated weighted
x/y/omega moments. Thresholds, component rules, supports, linking and centroid
equations have not changed. Improved spot discrimination or deblending would
change results and needs separate labelled validation.

## Changes

The single-thread native threshold loop now precomputes clipped column supports
and row offsets, rather than recomputing window clipping/strides at every pixel.
It reads variance-table entries only after passing the absolute threshold.
Rectangle subtraction, division and summation orders are preserved. Source
fingerprints consume contiguous array buffers without allocating temporary
Python byte strings. The independent reference remains unchanged because these
are execution changes to the same equations.

`locate_series(..., workers=W)` adds a private bounded thread pool. Each detector
still runs on one native thread outside Python's interpreter lock. Loaders run
serially on the caller's thread; at most W frames are in flight. Owned snapshots
protect reused loader buffers. Results, budgets, callbacks and linking retain
source order; cancellation stops further loading and joins running work.

The library default remains one worker. Dioptas's Find 3D positions action uses
four frame workers, with preview images retained by frame index. This avoids
mixing a completed detection with a later prefetched image. The application
continues to queue one background job at a time. No process/global worker pool,
new runtime dependency or normal-release API is introduced.

## Controlled timing and equality results

Each entry is the median of three complete runs after one excluded full warm-up.
The benchmark includes loading/masking, detection, hashing complete frame records,
source/mask fingerprinting, linking and position summaries. Configurations run
sequentially, with a warm filesystem cache. Qt rendering is excluded.

| Workload | Original serial | Optimized serial | 2 workers | 4 workers | 8 workers | Speedup at 4 |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic, 32 × 512² | 0.343 s | 0.233 s | 0.120 s | 0.062 s | 0.034 s | 5.52× |
| Synthetic, 16 × 2048² | 2.890 s | 2.040 s | 1.052 s | 0.589 s | 0.325 s | 4.91× |
| Fe7C3 P1, 76 × 2048² | 10.544 s | 7.494 s | 3.836 s | 1.968 s | 1.509 s | 5.36× |
| Fe7C3 P2, 76 × 2048² | 10.496 s | 7.676 s | 3.793 s | 1.971 s | 1.147 s | 5.32× |
| Fe7C3 P7, 76 × 2048² | 10.743 s | 7.427 s | 3.883 s | 2.355 s | 1.185 s | 4.56× |

The single-thread improvement is 1.37–1.45× on measured scans (27–31% less
elapsed time). Four workers add 3.15–3.89× relative to optimized serial, for
4.56–5.36× overall. Eight workers are faster here, but multiply detector-buffer
memory and show more timing variability; four is the application default.

Every timed repeat and warm-up matches the unchanged baseline: exact dtype,
shape and bytes of boxes, frame spans, values, flags, observations and exposure
bounds; identical source/mask hashes; and matching SHA-256 of complete per-frame
spot values/boxes/flags, rejected-component counts and noise conventions. This
includes NaN representations. P1/P2/P7 retain respectively 2,125/2,421/6,436
tracks from 4,703/4,882/11,787 observations. No tolerance was loosened.

An additional installed-Dioptas run with ordered image-preview callbacks enabled
and all source files/sidecars in provenance takes 2.101/2.107/2.254 s for P1/P2/P7
(three-run medians after warm-up), again with identical records and fingerprints.
The process high-water resident memory is about 1.0 GiB for four workers,
including imported application dependencies. This is process memory, not a
per-worker allocation guarantee, and excludes Qt rendering. Code fingerprints
change as expected; exported provenance additionally records the worker count.

## Reproduction and evidence

The unchanged baseline is revision `f166525af743bcfa63c6e132101730d907b5927d`.
Both baseline and candidate use release-mode builds with experimental-spots.
The baseline package/native extension were copied before editing; native and
source SHA-256 manifests are retained with the local run. Ordinary golden data
were not regenerated. Optional measured files are checksum-recorded and use
the reviewed explicit sidecar start/width convention, zero/nonfinite masks and
the conservative uint16 ceiling flag. No UB, indexed text, known cell or material
label is supplied to detection.

Run `benchmarks/rotation_performance.py` in separate processes for the preserved
baseline and candidate package paths:

```sh
PYTHONPATH=/path/to/baseline-site python benchmarks/rotation_performance.py \
  --legacy-api --workers 1 --output baseline.json --records baseline-records \
  --measured-source /path/to/Fe7C3_SC
PYTHONPATH=python python benchmarks/rotation_performance.py \
  --workers 1 2 4 8 --output candidate.json --records candidate-records \
  --compare baseline-records --measured-source /path/to/Fe7C3_SC
```

Pillow is required only for optional measured TIFF benchmarking. The default
synthetic benchmark requires only the package and NumPy. The compact portable
timing/equality report and raw-report hashes are in
`benchmarks/experimental-rotation-performance-macos.json`. Detailed local reports,
preserved baseline binaries/sources and arrays are in the ignored
`runs/positions-performance-20261025/` directory.

Validation: cargo fmt; workspace/all-target/all-feature clippy; 380 workspace
Rust tests/doctests; 1,195 Python tests (11 unavailable external-input skips and
34 external/real-data deselections); and 56 Dioptas single-crystal tests pass.
Coverage includes actual concurrent detectors, ordered observers, reusable
loader buffers, bounded read-ahead and error/cancellation cleanup, immutable
matching previews, stale GUI results, nondefault clipped supports/noise and the
independent detector reference. GSAS-II's powder oracle does not cover this
detector operation; the independent reference and explicit baseline comparison
are the relevant checks.
