# Metal FFT and peak-detection experiment (2026-11-12)

The separate [GPU row-search experiment](ROWS.md) covers row grouping and larger clouds.

This is an analysis-only GPU prototype. It does not register a production backend,
change a search default, modify frozen fixtures, or add a package dependency.
`mlx` is needed only in the separately created experiment environment.

## Numerical contract

For ordered reciprocal positions `q`, cloud-in-cell deposition uses
`p=q/step`, `low=floor(p)`, and eight trilinear weights. Coordinate division,
weights, and radial bounds are prepared in float64 on the CPU. Each weight is
rounded to float32 before addition, as in the existing Rust implementation.
Contributions are stably grouped by voxel. One Metal thread per touched voxel
adds its contributions in the original `dx,dy,dz,point` order. No floating-point
atomic addition or unordered reduction is used. The test checks every deposited
float32 voxel bit against independent NumPy deposition.

The GPU executes an unnormalized float32 forward transform in Z, Y, X order.
For voxel density `rho[k]`, the equation is
`F[m] = sum_k rho[k] exp(-2*pi*i*(kx*mx+ky*my+kz*mz)/n)`, with score `Re(F[m])`.
It computes Z transforms only for occupied XY columns and retains only the
frequency windows needed for the outer radius between transforms. It does not
skip every spherical-outside X transform as the Rust implementation does.
MLX's GPU FFT has its own butterfly arithmetic; CPU/GPU transform bits are **not**
expected to match RustFFT. No GSAS-II source or external indexing code is used.
The independent reference remains NumPy's FFT on independently deposited data.

The score is the real transform, not its magnitude or square. The score domain
is inclusive `min_length² <= (fx²+fy²)+fz² <= max_length²`, using the original
float64 boundary decisions. CPU-prepared Z intervals preserve those decisions
without changing them to float32. Maxima use strict neighbor `>` rejection,
wrapped X/Y neighbors, clamped half-spectrum Z neighbors, and retain plateaus.
A GPU integer counter compacts maxima; deterministic CPU ordering uses descending
float32 score, then original voxel index. All maxima are retained before selecting
the requested prefix; there is no capacity truncation or hidden CPU fallback.

Only the FFT/deposition/maxima method is replaced in the benchmark process.
The production Rust engine still performs the value/gradient/Hessian pass,
vector refinement, fitting and assignment. Python's search, budgets, training
split, validation and acceptance gates are unchanged. No profile function,
intensity normalization, or derivative equation is added. The powder-profile
oracle does not cover these positional Fourier calculations.

## Reproduce

Run from the repository root with a separate Python 3.13 environment containing
the same PhaseSmith native extension plus `mlx==0.32.4`, `numpy==2.5.4`,
`scipy==1.18.1`, and pytest. Metal is required; no CPU fallback is enabled.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=python:benchmarks:benchmarks/grain_gpu
python -m pytest -q benchmarks/grain_gpu/test_metal_fft.py
python benchmarks/grain_gpu/kernels.py --output /absolute/output/kernels.json
python benchmarks/grain_gpu/compare.py --output /absolute/output/single --repeats 5
python benchmarks/grain_gpu/compare.py --output /absolute/output/ensemble --mode ensemble --repeats 3
python benchmarks/grain_gpu/detector_control.py --output /absolute/output/detector --repeats 3
```

Each timing waits for GPU completion and includes preparation, allocation/input
handoff, GPU work and compaction. Complete calls also include native refinement,
ownership and validation. Input loading, package imports and writing audit files
are excluded equally. The first call for each case is recorded separately, then
backend order alternates. Only the first GPU call in a process includes entirely
uncompiled custom kernels; later case-first calls share compiled kernels. No
prior point-cloud fields or candidate results are reused between complete calls.
All cases run serially with the same four-worker search budget, except the
unchanged strict detector control's own two-worker budget. An eight-process CPU
ensemble and a shared GPU multiprocessing service are outside this experiment.

The position-only panel is the established 2,674-point two-grain/noise synthetic
case and three existing measured-coordinate fixtures (644, 1,026, 696 points).
The fixtures contain historically reserved *training* populations; a new 80/20
split is made within each for this regression experiment. These are exposed
regression inputs, not new independent evidence of grain truth. Fixtures' row
labels are never passed to FFT searches. Raw detector loading and peak extraction
are outside the position-only timings. Old example-4/5 detector-run archives are
not replayed here. The separate forward-generated detector control retains its
known-two-component assertions and original fitting/recovery settings.

## Results

**This prototype did not improve complete grain-finding speed. Keep the CPU
production path.** Medians of three alternating paired runs:

| Complete search | CPU (s) | GPU FFT/maxima (s) | GPU time increase |
| --- | ---: | ---: | ---: |
| 28-search ensemble, synthetic two-grain cloud | 2.520 | 2.614 | 3.7% |
| 28-search ensemble, measured fixture 1-C | 1.121 | 1.223 | 9.1% |
| 28-search ensemble, measured fixture 2-A | 1.187 | 1.198 | 0.9% |
| 28-search ensemble, measured fixture 2-C | 1.188 | 1.391 | 17.1% |
| Strict detector control, ordinary Fourier refinement | 36.871 | 39.564 | 7.3% |
| Strict detector control, native Fourier refinement | 35.419 | 37.484 | 5.8% |

Five warmed single-search repeats also favored CPU: synthetic 0.1556 versus
0.1660 s; measured fixtures were 8.8–15.5% slower with GPU. Nine warmed kernel
repeats on each of three clouds at 128³, 192³ and 256³ found the combined GPU
staging/FFT/maxima path approximately 1.65–3.60 times as slow as optimized CPU.
The 192³ two-grain kernel took 4.36 ms CPU versus 7.42 ms GPU. Separate submitted
GPU/wait/compaction timings can be smaller than CPU for some larger workloads,
but omitting the required staging would misrepresent a replacement's cost.

The targeted CPU FFT/maxima calls account for about 2.2–5.6% of the complete
position-only ensemble times. Even eliminating those calls entirely would save
only that fraction on these cases. This is a measured workload-specific bound
for replacing this stage, not a bound on accelerating the rest of the search.

Across **38 warmed paired complete searches**, every label, candidate count,
and acceptance sequence matches. All 93 accepted model comparisons preserve
indexed membership and residuals exactly. Eighty-two compared fitted bases are
bitwise equal; the other eleven comparisons are exact unimodular integer basis
changes, with HKLs transformed exactly and zero lattice reconstruction error.
These counts include repeats. Both backends reproduce their own complete
result fingerprints on repeat. In all six detector-control pairs, fitted bases,
HKLs, residuals, labels and acceptance are bitwise equal; only diagnostics differ.
Both known components pass the existing control's truth checks.

**Raw results are not fully bitwise equal across backends.** Small FFT rounding
differences change some tied peak order, translation-vector representations,
and discovery diagnostics. All nine kernel panels preserve the top-400 peak
sets and total maxima counts in this experiment, but peak ordering differs.
The GPU is therefore not promoted as an exact production replacement.

Validation passed: 21 dedicated GPU checks; 577 Rust tests; 3,444 Python tests
(28 skipped, 34 deselected); workspace formatting and all-feature Clippy; and
Ruff for the experimental scripts. The Rust count includes integration/doc tests
and has 34 ignored tests. Production FFT/grain source and the native extension
hashes are checked unchanged when regenerating the summary.

Results and the final validation receipt are recorded in `summary.json`.
Full per-call timing, exact result fingerprints, compact input snapshots, logs,
and the isolated environment are under:

`/Users/clemens/Codex/2026-11-12/phasesmith-grain-gpu/`

The host is an Apple M4 Pro with 14 CPU cores, 20 GPU cores and 48 GB memory.
Measurements use alternating pairs on a shared host; unrelated activity was
observed. Small complete-call differences should not be treated as statistically
established speed differences. GPU-stage timings are consistently slower once
its preparation and synchronization are included. This particular prototype
cannot establish a universal limit on more optimized Metal implementations.
