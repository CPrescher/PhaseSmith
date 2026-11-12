# GPU row-search experiment — November 12, 2026

**The tested GPU row-preparation paths did not speed up complete grain finding
on the M4 Pro. The production row implementation stays on CPU.** This is an
analysis-only experiment, not a new runtime backend or a general claim that all
GPU row-search designs must be slower.

## What was tested

The original independent Rust implementation of Aslandukov et al. (2022),
[DOI 10.1107/S1600576722008081](https://doi.org/10.1107/S1600576722008081), remains
the CPU baseline. It is copied into an isolated experimental cdylib with three
checked source anchors for one initial-preparation hook. The production source,
crate manifests and Python APIs are untouched. `bridge-provenance.json` records
the source hash; the unchanged bridge CPU mode is compared with the installed
native CPU kernel for every input.

Two variants retain the same requested options, 50,000-direction cap, eight
row-search workers, cache policy, and exact candidate selection:

1. **GPU sorting:** Rust computes the original float64 projection keys. Metal
   sorts batches of integer keys, detects qualifying segments, and compacts
   retained point records.
2. **GPU projections and sorting:** Metal computes projections and keys and
   sorts/compacts their rows. Pairs near any projection-cell boundary are
   recomputed in CPU float64 before sorting. All subsequent phase/spacing
   calculations retain the original float64 equations.

Normalization, naive/smart direction nomination, gap/phase fitting, candidate
competition, and residual iterations remain on CPU. Subsequent cell inference
also keeps the unchanged CPU FFT, following the previous FFT experiment. This
is a hybrid row-preparation test, not an implementation of every stage on GPU.

## Equations, boundaries and ordering

For normalized point `p`, direction `v`, dominant axis `k`, and the two other
axes `a,b`, the original equations are

```text
t = p[k]/v[k]
u = p[a] - v[a]*t
w = p[b] - v[b]*t
cell_a = floor((u + offset_a*epsilon)/(2*epsilon))
cell_b = floor((w + offset_b*epsilon)/(2*epsilon))
```

The four offset pairs are `(0,0), (1,1), (1,0), (0,1)`. Cells have the original
half-open floor boundaries. Integer cell keys and qualifying row sizes are
unchanged. A fixed-grid row initially needs at least three points under the
selected settings; the original rule discards a whole grid without a row of
four points that can supply gap evidence. Accepted row-spacing/phase predicates
are inherited without alteration from `ROW_SEARCH_ALGORITHM.md` and the Rust
kernel. No new profile, smooth derivative or intensity equation is introduced;
a powder-profile/GSAS-II oracle does not cover this positional grouping task.

The screened projection prototype supports the selected `epsilon=0.002` only.
It marks a point/direction pair when any of the four scaled projected coordinates
is within `0.002` of an integer (a guard of `8e-6` in normalized projected
coordinates), then replaces **all** its keys using the original float64
expressions. This guard controls recomputation, not row-membership tolerance:
no scientific threshold is widened. Normalized points lie in `[-1,1]` and the
chosen direction component has magnitude at least `1/sqrt(3)`, bounding the
float32 projection error. Exhaustive sampled key audits and adjacent-float
boundary controls check this prototype; they are not a universal numerical
certificate for arbitrary compiler/hardware changes or other epsilons.

GPU integer sorting has no floating-point reduction. Segment compaction uses an
integer atomic counter; arrival order cannot select candidates. Rust restores
key order and the original float64 longitudinal/point-ID tie ordering on retained
records, then constructs the original packed-row objects. Coincident points,
zero-spread axes, duplicates and cache settings keep their original handling.
There is no silent fallback if a GPU callback fails. Unsupported preparation
options raise an error. Complete default-budget results are compared; wall-time
partial searches are not treated as quality-equivalent. GPU preparations check
the existing deadline between bounded batches; an in-flight GPU command is joined.

The default batch holds at most 8,388,608 keys (32 MiB), at most 4,096 directions.
Native output scratch has capacity for every point record (up to 96 MiB), so
there is no lossy truncation. The initial prepared-row pool is temporary and
additional to the unchanged 512 MiB retained-cache budget. GPU scratch, sorting
buffers and host allocations are also additional memory, not a claim of equal
RSS to the CPU implementation.

## Panel and clocks

The panel contains three frozen measured training-coordinate fixtures (644,
1,026 and 696 points), the existing 2,674-point two-grain/noise synthetic input,
and new fixed-seed 6,000- and 12,000-point two-grain/noise inputs. Large clouds
contain two lattice signals with positional noise plus 20% diffuse contamination.
The full searches use the same deterministic 80/20 split, so the three synthetic
training populations contain 2,139, 4,800 and 9,600 points and use **smart**
direction nomination. The measured fixtures use **naive** nomination. The original
2,048-unique-point crossover is unchanged.

Fixtures contain old training populations; their new internal splits are exposed
regression inputs, not original held-out scientific validation. No fixture row
label or generating cell/HKL is supplied to the search. Measured fixtures 1-C
and 2-A yield no accepted grain under these unchanged full-search settings in
both backends; 2-C yields one. Identical failures remain failures, not claimed
physical grain recovery. All three synthetic full-search cases accept two
models. Separate known-orientation controls verify both large synthetic lattices
within the existing 1% volume/integer-basis checks.

Complete timings include row normalization/nomination, preparation, transfers,
GPU completion, compaction, CPU residual grouping, per-group FFT cell inference,
consolidation, ownership and held-out acceptance/null controls. Raw detector
loading and peak extraction are outside this position-table benchmark. Imports,
input loading, reference audits and result-file writes are excluded equally.
Three warmed alternating CPU/GPU pairs follow a separately recorded case-first
pair. The first GPU call in a process may compile kernels; later case-first calls
share compiled kernels. No point-cloud result is reused across complete calls.
The three numerical library thread limits are 1 before Python startup.

## Measured results

M4 Pro: 14 CPU cores, 20 GPU cores, 48 GB. Median **complete** row-based grain
finding, with GPU projections plus sorting:

| Input points | Method | CPU (s) | GPU hybrid (s) |
| ---: | --- | ---: | ---: |
| 644, measured 1-C | Naive | 0.0851 | 0.2438 |
| 1,026, measured 2-A | Naive | 0.1228 | 0.3552 |
| 696, measured 2-C | Naive | 0.0746 | 0.1732 |
| 2,674, synthetic | Smart | 0.1262 | 0.1499 |
| 6,000, synthetic | Smart | 0.2487 | 0.3151 |
| 12,000, synthetic | Smart | 0.6368 | 1.0481 |

The CPU-key/GPU-sorting variant also loses on every case: complete GPU/CPU ratios
range from 1.17 to 2.86. For projection-plus-sorting the ratios are 1.19–2.89.
Standalone row grouping, separate from later cell finding, is about 1.86–2.31
times slower with GPU projections/sorting on these inputs.

The CPU's saturating cell counts discard sparse cells before sorting, and its
work/cache handling is already optimized. This GPU prototype sorts batched dense
keys, synchronizes for boundary corrections and compaction, and returns retained
records to CPU phase fitting. Those costs outweigh the benefit here. Smart
nomination remains a CPU cost on larger inputs. These are alternating paired
measurements on a shared host, not idle-host confidence intervals or a bound on
a fully resident GPU redesign.

Across both variants, **36 warmed complete grain-search pairs and 36 warmed
standalone grouping pairs** preserve every scientific result field bit for bit:
ordered groups, labels, normalization, cache/work counters, options, fitted
bases/cells, HKLs, residuals, translation vectors/scores and acceptance. Only
clock fields are excluded; the requested `seconds` deadline itself is preserved.
Each backend also reproduces its own scientific fingerprints on repeated runs.
An independent float64 audit checks **879,333,240 projected keys with zero
mismatches** after correction. This audit is excluded from speed measurements.

## Reproduce and inspect

Use the isolated environment from the FFT experiment (`mlx==0.32.4`,
`numpy==2.5.4`, `scipy==1.18.1`), the same native PhaseSmith extension and a Metal
Mac. The bridge needs Rust plus cached serde/serde_json build dependencies.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=python:benchmarks/grain_gpu
python benchmarks/grain_gpu/build_rows_bridge.py /absolute/row-artifacts
export PHASESMITH_ROW_GPU_LIBRARY=/absolute/row-artifacts/bridge/target/release/libphasesmith_row_gpu_experiment.dylib
python -m pytest -q benchmarks/grain_gpu/test_metal_rows.py
python benchmarks/grain_gpu/rows_compare.py --output /absolute/row-artifacts/full --repeats 3
python benchmarks/grain_gpu/rows_compare.py --output /absolute/row-artifacts/grouping --mode grouping --repeats 3
python benchmarks/grain_gpu/rows_compare.py --output /absolute/row-artifacts/full-projection --gpu-projection --repeats 3
python benchmarks/grain_gpu/rows_compare.py --output /absolute/row-artifacts/grouping-projection --mode grouping --gpu-projection --repeats 3
python benchmarks/grain_gpu/audit_row_projections.py --output /absolute/row-artifacts
```

`MetalRows(library=...)` also accepts an explicit generated dylib path.
Full timings, input snapshots, all result fingerprints,
projection audit, known-orientation controls, bridge source/provenance and logs:

`/Users/clemens/Codex/2026-11-12/phasesmith-row-gpu/`

Validation passed: 23 dedicated GPU checks, 24 isolated bridge tests, 577 workspace
Rust tests, and 3,444 Python tests (28 skipped, 34 deselected), plus workspace
formatting, Clippy and script linting. Workspace Rust tests have 34 ignored tests.

The compact machine-readable receipt is `rows-summary.json` beside this report.
Production numerical source and the extension hashes are checked unchanged.
