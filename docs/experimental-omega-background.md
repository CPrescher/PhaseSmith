# Experimental omega background fitting

`phasesmith.experimental.omega_background` owns the reusable CPU arPLS engine
for detector-pixel intensity traces along uniformly spaced omega exposures.
It is opt-in with the existing `experimental-spots` native feature. It has no
Dioptas, detector-file, calibration, GUI, mask-estimator or GPU dependency.
Dioptas owns those application concerns and signed frame export.

```python
from phasesmith.experimental.omega_background import (
    ArPLSParameters, arpls_masked_rows,
)

baseline, iterations, converged = arpls_masked_rows(
    rows, excluded,  # equal (pixels, frames) arrays; true means unavailable
    ArPLSParameters(smoothness=10000, tolerance=0.001, max_iterations=30),
    workers=4,
    minimum_frames=12,
)
corrected = rows - baseline
```

The library defaults to one worker. Callers choose a bounded worker allowance
and coordinate concurrent jobs. Up to four private worker pools are cached by
worker count for reuse across tiles; no process-global Rayon pool is changed.
Native inputs are owned snapshots while the Python interpreter lock is released.
Parallelism distributes pixels, retaining scalar reduction order within a fit.
The same worker count handles every interval in a tile, avoiding nested pools.

## Equations and scientific contract

For each contiguous valid interval, minimize the penalized least-squares system

\[
(W + \lambda D^T D)z = Wy,
\]

where each row of the unpadded second-difference matrix is `[1, -2, 1]`.
Boundaries are natural/truncated, with no padding or extrapolation. All starting
weights are one. After a solve, let `d=y-z`, and let `m` and `s` be the population
mean and standard deviation of negative residuals. Update with

\[
w_i = \max\{10^{-8},\ [1+\exp(2(d_i-(2s-m))/s)]^{-1}\}.
\]

Traces are shifted by their first value and scaled by their largest absolute
difference. Constant traces return themselves with zero solves. Negative-residual
variance is accumulated using Welford's algorithm. Standard deviation has a
floor of `1e-8`; the logistic argument is clipped to `[-40,40]`. Stop when fewer
than two residuals are negative, the maximum normalized residual is below
`1e-10`, or the relative L2 weight change is below the requested tolerance.
Return the last solved baseline, including when the iteration cap is reached.
The baseline uses float64. Subtraction retains negative signal.

Masked intervals retain the original frame spacing: there is no time
compression or interpolation. Excluded samples and intervals shorter than the
requested minimum remain NaN. The minimum defaults to 12; the generic unmasked
`arpls_rows` permits at least three samples. Per-pixel iteration counts are the
maximum over fitted intervals. Convergence requires at least one fitted
interval and convergence of every fitted interval. Nonfinite samples are
permitted only outside fitted intervals. The caller must validate uniform
angular spacing and keep angle/source provenance.

The Rust kernel uses O(frames) banded work per pixel and iteration. Native calls
accept `solver="cholesky"` (the library default, preserving the original
floating-point convention) or `solver="ldlt"`. Both solve the same penalized
system with the same weights, safeguards and stopping rule. The SciPy reference
continues to construct and solve the equations independently; the solver choice
applies to the native backend only.

For the faster LDLᵀ solve, write `A = L diag(d) L.T`, with unit diagonal in `L`.
Let `a_i = w_i + lambda * (D.T @ D)[i,i]`, let `b_i` be the first lower band,
and let the second lower band be `c=lambda`. With absent terms treated as zero:

\[
q_i = c/d_{i-2},\qquad
e_i = b_i-c p_{i-1},\qquad p_i=e_i/d_{i-1},\qquad
d_i=a_i-e_i p_i-c q_i.
\]

The first two rows omit the unavailable prior bands; in particular the `c*p`
term is absent at row one. The forward/diagonal solve is
`v_i = (w_i*y_i - e_i*v_{i-1} - c*v_{i-2}) / d_i`, followed by
`z_i = v_i - p_{i+1}*z_{i+1} - q_{i+2}*z_{i+2}` in reverse order. The native
implementation stores `1/d_i`, using one reciprocal per sample and no square
roots in factorization, rather than the original four divisions and square
root. Positive finite pivots remain mandatory. Normalization, interval masks,
natural boundaries and nonlinear reweighting are unchanged. Reciprocal
multiplication and LDLᵀ round differently from Cholesky; cross-solver equality
is numerical, while worker-count equality within either solver is exact.

This is preprocessing, without a refinable peak-model derivative API. It does
not estimate baseline uncertainty or remove original counting noise. It does
not establish quantitative reflection-integral accuracy on measured scans.

## Provenance and migration

The application along omega was inspired by Konstantin Glazyrin's
[background-sc](https://gitlab.desy.de/konstantin.glazyrin/background-sc).
The equations are from Baek et al., *Analyst* 140 (2015), 250–257,
[doi:10.1039/C4AN01061B](https://doi.org/10.1039/C4AN01061B).
The independent implementation was initially developed in Dioptas; its CPU
numerical engine is now in `crates/phasesmith-spots/src/omega_background.rs`,
behind an opt-in PyO3 boundary. No background-sc or GSAS-II code is reused.
GSAS-II is not an oracle for this detector/omega preprocessing convention.

`omega_background_reference.py` independently constructs the banded penalty
and uses SciPy/LAPACK and NumPy statistics. It is selected with `backend="scipy"`
and requires SciPy, available through the existing `refinement` extra. The Rust
engine requires only the package's normal NumPy Python dependency. `auto` uses
Rust when available and otherwise the SciPy reference. An explicit `native`
request fails clearly if the experimental kernel is absent.

Migration preserves lambda, tolerances, iteration limits, normalization,
interval boundaries and convergence decisions on the tested scans. It is not
bit-identical to the former C compiler's floating-point evaluation: the full
Fe7C3 P1 comparison over 208,939,231 fitted samples has maximum float64 baseline
difference `1.37585e-6` counts and RMS `1.62946e-9` counts. Per-pixel solve counts
and convergence flags are exactly identical. The largest signed float32 frame
difference is `6.10352e-5` counts. These are local migration differences, not
relaxed global tolerances. Exact worker-count equality is tested for the Rust
engine; independent-reference tolerances are local to the corresponding tests.

## Validation and benchmarks

`tests/test_experimental_omega_background.py` covers randomized reference
comparisons across lengths/penalties, fragmented and short intervals, excluded
NaNs, invalid fitted values, constant/negative traces, signed peak areas and
centroids, explicit caps, empty batches, unchanged inputs and native boundary
validation. The existing detector and rotation tests retain original-count
variance-loader behavior for background-subtracted images.

```sh
maturin build --release --features experimental-spots
python benchmarks/omega_background.py --pixels 262144 --frames 76 \
  --workers 1,4,8 --repeats 3 --output omega-benchmark.json
```

Add `--baseline-native /path/to/preserved/_core.abi3.so` to alternate matched
direct native calls between the preserved and current binaries. This comparison
includes owned input snapshots and output allocation. With the Cholesky solver,
it checks every returned baseline/iteration/convergence value exactly. With
`--solver ldlt`, it checks explicit local baseline tolerances and exact
diagnostics, and reports maximum/RMS differences. Hashing, comparison and
summary reductions are excluded from timing. `--extra-peaks` adds three narrow
peaks to every sixteenth trace to exercise multi-peak inputs.

The benchmark includes noisy traces and narrow peaks on a gently varying
baseline, with a second spatially correlated interval-mask workload. It excludes
files and GUI. Dioptas provides the full pipeline benchmark, original C baseline
snapshots and measured scan reports in its experimental single-crystal checkout;
see its `docs/omega-background-performance-20261025.md`.

Local three-repeat P1 fitting medians improve from 8.783 to 4.222 seconds with
eight workers (including interval handling), using the preserved Dioptas C
baseline and exactly the same v2 masks. Four-worker fitting improves from
10.834 to 8.430 seconds. The complete Dioptas Qt workflow improves from 16.105
to 10.977 seconds; that total includes file preparation, fitting, export,
cleanup and completion. These are warm-input-cache timings on an M4 Pro,
with eight workers as Dioptas's feature ceiling. The independently measured
mask-preparation improvement belongs to Dioptas, outside this engine.
Raw repeats, compiler/binary provenance and synthetic scaling are in
`benchmarks/omega-background-performance-macos.json`. The frozen wheel preserves
the corrected-series variance-loader adapter. Subsequent concurrent aperture
integration development is outside this measurement snapshot.

### Worker-budget follow-up

The arPLS Rust implementation remains unchanged in a subsequent eight/ten-worker
comparison. With 262,144 noisy 76-frame traces and 4096-pixel calls, three-repeat
medians fall from 0.397 to 0.327 s unmasked and 0.339 to 0.280 s masked. All
returned baseline/iteration/convergence bytes match. Dioptas now uses its shared
CPU/memory/task allowance instead of adding an eight-worker feature cap; that
permits ten on the development machine while leaving CPU capacity for other work.
Its fresh complete-pipeline P1/P2 medians improve by approximately 14–15%.

An experiment reusing initial unit-weight Cholesky factors was numerically exact
but did not improve the measured workloads consistently, and was discarded.
The accepted scaling measurements and rejected experiment are retained in
`benchmarks/omega-background-worker-budget-macos.json`; the library's equations,
reference implementation and serial-by-default API are unchanged.

### Reciprocal LDLᵀ solver

The numerical solver itself now has an explicit `solver="ldlt"` option. The
standalone library still defaults to `solver="cholesky"`; Dioptas selects LDLᵀ
in its experimental background task and records that selection. Both share the
same normalization, reweighting and stopping code; their linear routines are
specialized outside the nonlinear iteration. No approximation, iteration-cap
reduction or tolerance relaxation is used.

Three-repeat native comparisons on a realistic multi-peak 131,072 × 76 grid
reduce time by approximately 18–20% at one/ten workers, with/without masks.
Complete measured-scan processing (including tile copies and summaries) improves
from 3.689 to 2.978 s for P1 and 3.502 to 2.847 s for P2. File export remains
variable, so these are processing-stage gains rather than guaranteed whole-app
speedups. The retained Cholesky compatibility benchmark stays within roughly
−0.8% to +1.1% of its preserved binary's median times.

On 208,939,231 / 208,816,215 fitted P1/P2 samples, maximum float64 baseline
differences are 7.2024e-7 / 1.5672e-6 counts, RMS differences about 1.4e-9 counts,
and iteration/convergence maps are unchanged. Signed float32 frame differences
reach 6.1035e-5 counts; cross-solver byte identity is not claimed. The retained
Cholesky path is bit-identical to the original binary on both full scans and
126 deterministic parameter cases. At the deliberately strong lambda=1e9,
cross-solver differences reach 0.0005293 counts and some short-run iteration
counts differ. The library's original default remains available for reproducing
that convention.

The independent SciPy reference and dense first-solve tests validate both
factorizations without sharing native code. Existing local fit tolerances are
unchanged; first-solve normalized backward error remains below 5e-16 through
lambda=1e9. Synthetic peak area/centroid tests, interval boundaries, invalid
inputs and exact worker-count equality cover both methods. Detailed repeats,
source/native fingerprints and the full consumer comparison are retained in
`benchmarks/omega-background-ldlt-macos.json`. This operation introduces no new
peak-profile term or derivative API; GSAS-II does not cover this omega convention.
