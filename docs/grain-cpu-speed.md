# Exact CPU grain-search speed improvements — November 12, 2026

The CPU implementation now avoids repeated work during cell refinement and
coherent recovery. Search configurations, hypotheses, sample order, refinement
cycles, stopping conditions, thresholds, precision and final validation remain
unchanged. The GPU experiments remain analysis-only.

## Changes and numerical contract

### Reuse an identical least-squares update within one refinement

The original update is the same rank check and least-squares problem
`B = argmin_B ||H B - Q||²`, followed by `A = inverse(B).T`. The refinement still
performs every original nearest-HKL selection and each original cycle's delta
and stopping comparison. If both ordered `H` and selected `Q` have the exact
same shape, dtype and float bytes as the preceding successful solve, its update
is reused. Keys distinguish signed zeros. A changed correspondence, selected
point value, dtype, shape or row order cannot reuse it.

Only the immediately preceding solve is retained, and only inside one `_refine`
call. No fit, point-cloud or cross-search cache is introduced. At the public
100,000-point bound, the retained two float64 triple-array keys total at most
4.8 MB, plus small metadata and the 3×3 result; transient key construction and
existing fitting arrays are additional. No rank failure or invalid solution is
cached as a successful update. Regression tests check signed-zero changes,
unchanged minimum cycle counts/tolerances, and absence of cross-call reuse.

### Stop the private integer-index check once Z³ is established

The unchanged Euclidean row operations build an upper-triangular integer basis
with positive diagonal entries. Once all three diagonal pivots are one, its
index is exactly one and the rows generate Z³. Adding later integer rows cannot
enlarge Z³. The cell-alias consumer only needs to know whether the index exceeds
one, so its private fast path can return the identity at that point and continue
with the original `index <= 1` rejection. It does not terminate grain search or
change any proposal budget.

The complete input is validated before any early return. Nonprimitive and
rank-deficient cases retain the original complete row operations. The default
`integer_lattice_basis` path retains its original returned basis representation;
only the internal `_stop_if_primitive=True` consumer uses the shortcut. This
is not a scientific proof that an experimentally discovered cell is primitive.
An independent readable reference computes the gcd of exact 3×3 minors to check
the index, rather than repeating the row-reduction implementation.

### Reuse the frozen observation-chart measure during coherent recovery

For chart `K_i`, the model-independent normalization is still

```text
measure_i = (2*pi)^(3/2) * xy_scale^2 * omega_scale / abs(det(K_i))
```

with the original validity mask and zero values for invalid entries. Charts and
scales are fixed throughout a coherent recovery invocation. The serial path
computes this exact array once; each independent parallel parent worker computes
it once for its own unchanged chart input. All original determinant/division
arithmetic is retained. Proposal-specific evidence, physical eligibility,
spacing/phase fits, ownership competition and validation are recomputed normally.
No measure is reused between rotated validation charts or separate inverses.

These changes do not introduce a profile term, alter finite support, or change
analytical derivative accumulation. Independent Fourier/ownership equations and
existing derivative/boundary tests remain applicable. A powder-profile GSAS-II
oracle does not cover integer positional lattice fitting or this execution reuse;
exact comparison with the first-party baseline is the relevant regression check.

## Profiling and benchmark design

The task-start source was frozen before editing, with native extension and source
hashes recorded. Both arms load the same native extension and numerical library
versions; the baseline swaps only the saved `grains`, `ownership` and `recovery`
Python modules. It supplies its own original option classes, including for lazy
imports. Replacing a function in a live new module is not used as the baseline.

The profile of the full detector-space control found substantial time in repeated
integer-basis checks, least-squares/rank calculations and repeated chart
determinants during coherent recovery. Profiling is diagnostic and is excluded
from speed measurements.

The six-case position-only panel contains the same exposed measured training
fixtures (644, 1,026 and 696 positions), the established 2,674-position synthetic
two-grain/noise cloud and fixed-seed 6,000/12,000-position controls. Full row
searches keep the normal 80/20 split and original naive/smart crossover. Measured
row fixtures 1-C and 2-A still have no accepted grain at these fixed settings;
matching their failures is not claimed as grain recovery or fresh validation.

Complete position-only searches include discovery, all cell fitting,
consolidation, ownership and held-out/null validation. The detector-space control
additionally includes conditioning, stability, proposal polishing, alias-cell
work and coherent recovery, in both ordinary and native Fourier-refinement
modes. It must recover exactly the two original forward-generated components.
Raw image loading and peak extraction are outside these position-table clocks.
The detector control uses its original explicit search portfolio; the ordinary
ensemble panel retains all 28 default geometric configurations.

Five alternating warmed pairs are recorded for each single FFT/row case and
three for each 28-search ensemble case. A separate case-first pair is retained.
The detector control has three alternating pairs, each including both complete
refinement modes. Inputs/imports and audit-file writes are outside the clocks
in both arms. Numerical thread limits are one before Python startup; native
search worker budgets remain unchanged. No timed numerical result is reused
across complete calls. The host is an M4 Pro with 14 CPU cores and 48 GB RAM.
Measurements were made on a shared host and are not idle-host confidence limits.

## Results

Median complete-search wall times:

| Search | Before (s) | After (s) | Wall-time reduction |
| --- | ---: | ---: | ---: |
| Detector control, ordinary Fourier refinement | 39.605 | 28.468 | 28.1% |
| Detector control, native Fourier refinement | 34.879 | 25.152 | 27.9% |
| Row search, 2,674 positions | 0.1253 | 0.0926 | 26.1% |
| Row search, 6,000 positions | 0.2452 | 0.2036 | 17.0% |
| Row search, 12,000 positions | 0.5764 | 0.5471 | 5.1% |
| FFT ensemble, 2,674 positions | 2.2384 | 2.1320 | 4.8% |
| FFT ensemble, 6,000 positions | 3.8993 | 3.7939 | 2.7% |
| FFT ensemble, 12,000 positions | 6.1073 | 6.0861 | 0.3% |

The measured-fixture FFT ensembles save 3.7–5.7% in these observations. Single
FFT searches save 2.8–7.7% on the small/fixture inputs, while the 6,000/12,000-point
single FFT medians are 0.2/0.3% slower. Measured-fixture row medians vary from
1.0% slower to 2.3% faster. These near-zero differences should be treated as
essentially unchanged on a shared host. Large-cloud raw row grouping is unchanged;
its native direction/grouping work can dominate the complete row-search clock.
There is no claim of a uniform speedup or a one-second detector-space search.

Every scientific result field is bitwise identical in **84 warmed full-search
comparisons** (30 single FFT, 30 row, 18 ensembles and six detector-space searches
across the two modes), plus the separately retained case-first comparisons.
This covers raw row groups/labels, normalization and work counters, all accepted
and rejected candidate arrays, fitted bases/cells, HKLs, residuals, translation
vectors/scores, inference histories, null counts and acceptance flags. Only clock
fields are excluded; the requested row-search deadline itself remains included.
Each arm also reproduces its own complete scientific fingerprint on repeat.

Validation passed: 77 focused tests, then the full suite of 3,450 Python tests
(28 skipped, 34 deselected) and 577 Rust tests (34 ignored), plus workspace
formatting, all-feature Clippy, script linting and `git diff --check`. Existing
serial/process parity, derivative and support-boundary checks are included.

## Reproduce and inspect

Saved baseline, source/native hashes, profiles, per-pair timings, exact result
fingerprints and validation logs:

`/Users/clemens/Codex/2026-11-12/phasesmith-grain-cpu-speed/`

The baseline directory contains the task-start `single_crystal` Python sources.
The owning files were clean at task start on commit
`4c1def58bd23a0d78783b0031530256790183145`; unrelated automatic structure-recovery
work in the checkout was preserved. The benchmark does not require MLX or GPU.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=python
python benchmarks/grain_cpu_speed.py --baseline /absolute/baseline --output /absolute/fft --mode fft --repeats 5
python benchmarks/grain_cpu_speed.py --baseline /absolute/baseline --output /absolute/rows --mode rows --repeats 5
python benchmarks/grain_cpu_speed.py --baseline /absolute/baseline --output /absolute/ensemble --mode ensemble --repeats 3
python benchmarks/grain_cpu_speed.py --baseline /absolute/baseline --output /absolute/detector --mode detector --repeats 3
```

The benchmark's detector mode imports the repository's original strict control.
The final compact receipt is stored with the local artifacts. For this Python
coordinator experiment, production Rust and the native extension were unchanged.
The native follow-up is reported below. No change was committed or published.

## Native Rust coordinator

The separate native coordinator in
`crates/phasesmith-workflows/src/single_crystal/grains.rs` now also removes
repeated work. It retains the same FFT discovery recipe, 32 attempted candidate
budget, five discovery rounds, eight refinement cycles, selection cutoffs,
null rotations, setting enumeration, and geometry-fit gates. This coordinator
currently selects candidates through FFT; the Python row-search timings above
are a different workflow and should not be compared directly with these clocks.

* Consecutive identical ordered `H,Q` arrays reuse the successful least-squares
  update `A = inverse(argmin_B ||H B - Q||²).T`. Comparison uses float64 bits,
  including signed zero, and lengths. Every original scoring/selection cycle and
  delta/stopping test still executes. The cache belongs to one refinement call
  and retains at most two arrays of triples (4.8 MB at the 100,000-point bound).
* The private integer-index consumer can stop at three positive unit pivots,
  after checking every input row. The full integer-basis path remains available
  in tests. Nonprimitive results keep the original row operations and basis.
  Independent exact gcd-of-minors tests check both paths on 300 deterministic
  integer matrices, including nonprimitive indices and rank deficiency.
* Refinement returns its already-computed final scores with its chosen basis.
  If an alias wins, its own scores are retained; otherwise the original reduced
  basis scores are retained. Alias tests still use the original sorted rows.
  Discovery consumes this exact array instead of calling the same kernel again.
* Fixed model inverses are computed once for observed/null ownership assignment,
  rather than reconstructed for each of the 64 rotations. Every assignment,
  count, statistic and decision still executes in its original order.

These are execution changes to existing positional equations. No profile term,
finite support, derivative, parameter convention or public API changes. The
independent Python reference remains independent. GSAS-II does not cover this
coordinator; the frozen first-party baseline is the numerical comparison.

### Native benchmark scope and reproduction

The ignored test in `grains_benchmark.rs` calls the actual private native
coordinator, with no alternate implementation or shortened search. The nine
frozen inputs include three measured training fixtures, three two-component
reciprocal-space controls (2,674 / 6,000 / 12,000 positions), and three synthetic
detector-position controls (600 / 1,800 / 5,000 tracks). Detector controls contain
40% positions from each of two independent orientations and 20% diffuse points,
with 0.08-pixel and 0.015-degree positional noise. They use an explicit first-cell
prior, as required by this native recipe. These exposed performance controls are
not a fresh scientific validation set.

Discovery clocks include selection, Fourier refinement, lattice fitting,
consolidation, ownership and all 64 rotated-null validations. Detector-position
clocks additionally include position-to-reciprocal conversion, native cell-setting
selection, family-partitioned geometry fitting and its diagnostics/gates, and
competing-domain calibration. Raw frame loading, peak extraction, integrated
intensity/profile fitting and downstream structure search are outside these
position-table clocks. Fixture loading and bit serialization are also excluded.

The 600-track detector control fails the existing training-supported cell-prior
gate in both versions. Its matching failure is retained as a regression control;
it is not reported as a successfully completed geometry fit. The 1,800- and
5,000-track controls complete the full native indexing stage.

The test serializer records every returned field: grain order/bases, all HKLs,
ownership/training/eligibility, accepted status, training/withheld/null counts,
selected track ids, observations, complete geometry state and Jacobians,
initial/final predictions and flags, covariance, objective history, convergence,
evaluation counts, RMS diagnostics and all competitor records. Float64 values
are serialized with `to_bits`, retaining NaN payloads and signed zeros. Errors
are compared as exact strings. Timing is the only excluded diagnostic.

To reproduce, first save the unoptimized source tree and add the same test-only
benchmark harness to both trees. Build each release executable in a **separate
Cargo target directory** (sharing the directory can reuse a local-package
artifact across relocated workspaces). Resolve the executable from Cargo's JSON
`compiler-artifact` record, rather than guessing its hashed filename:

```sh
cargo test --release -p phasesmith-workflows --features single-crystal \
  --lib --no-run --message-format=json --target-dir /tmp/native-grain-after
# Run the same command in the saved baseline, using /tmp/native-grain-before.
.venv/bin/python benchmarks/native_grain_cpu.py --prepare --output /tmp/native-grain-results
.venv/bin/python benchmarks/native_grain_cpu.py --output /tmp/native-grain-results \
  --baseline-exe /absolute/path/to/baseline-test-executable \
  --optimized-exe /absolute/path/to/optimized-test-executable --repeats 12 --batch-size 5
```

The driver verifies input hashes, rejects identical executables, warms each case
in each arm, alternates execution order, rotates case order, compares every
binary result directly, and saves raw clocks, hashes, medians and std. Each
process performs five complete searches and checks every repeated output against
its first output before returning. The reported per-process time is the median
of those five clocks, and the summary takes the median over 12 paired processes.
All individual clocks are retained. Both arms
use release settings, the same Rust/dependency versions and the same four-worker
kernel configuration. Sampling profiles are diagnostic and excluded from timings.

### Native results on November 12, 2026

Apple M4 Pro, macOS 26.6.2, Rust 1.99.1; four kernel workers in both arms.
The following are medians of the 12 paired process batches, in milliseconds;
std describes variation between batch medians. Positive reductions mean faster.

| Case | Before (ms) | After (ms) | Reduction | Before / after std (ms) |
| --- | ---: | ---: | ---: | ---: |
| discovery-12000 | 231.30 | 230.46 | +0.36% | 7.48 / 5.06 |
| discovery-2674 | 91.24 | 91.48 | -0.27% | 4.22 / 4.85 |
| discovery-6000 | 176.93 | 174.78 | +1.22% | 6.83 / 6.72 |
| discovery-dataset-1-C | 63.57 | 63.55 | +0.03% | 1.41 / 1.62 |
| discovery-dataset-2-A | 46.84 | 46.06 | +1.66% | 1.56 / 1.38 |
| discovery-dataset-2-C | 50.18 | 50.15 | +0.06% | 1.49 / 1.50 |
| index-1800 | 159.77 | 156.53 | +2.03% | 4.18 / 4.82 |
| index-5000 | 222.27 | 222.54 | -0.12% | 7.31 / 5.91 |
| index-600 | 123.27 | 124.90 | -1.32% | 2.82 / 2.33 |

Every returned result in all 1,170 complete executions (585 before/after pairs,
including warmup) matched its case's frozen baseline bit for bit. The successful
full-indexing cases retained the same five returned hypotheses, selected domain,
geometry counts (462/258 and 1,285/715 training/validation), and one/two competitor
records, respectively. The 600-track failure also matched exactly.

The native speed benefit is small: about 2% on the 1,800-track detector control,
and effectively unchanged on the 5,000-track control and most discovery cases.
These differences are smaller than the observed batch-to-batch variation; this
panel does **not** establish a convincing general end-to-end native speedup.
Sampling showed the shared FFT path dominates native runtime. The earlier
20-single-search-process comparison also showed only small, noisy differences;
its raw results are retained rather than selecting the more favorable timings.
The safe reuse changes reduce coordinator work, while a material overall speed
improvement would need further work on the dominant shared kernels.

An additional trial used the existing `FittingCloud.selected` kernel to avoid
full score arrays inside refinement. It preserved every result but was 2–3%
slower across the five larger controls in six paired batches of three searches.
It was discarded; it is not part of the final implementation or table above.

Final checks: `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets --all-features` (no warnings),
`cargo test --workspace --all-features` (581 passed), `pytest` (3,450 passed,
28 skipped, 34 deselected), and benchmark-driver Ruff checks. Targeted release
refinement tests also passed. The benchmark executable rebuilt from the final
source was checked against the recorded final benchmark SHA256.

The frozen production coordinator was verified against its task-start Git
source. Both final executables use the identical test-only harness and independent
Cargo target directories. Final source hashes, executable hashes, input manifest,
binary outputs, individual clocks, profiles and check logs are retained under
`/Users/clemens/Codex/2026-11-12/phasesmith-native-grain-cpu/`.
The table comes from `batched/summary.json`; the earlier single-process results
are in the root `summary.json`, and the rejected trial is in `selection-trial/`.

The production FFT and row paths remain on CPU. GPU implementations are retained
only under `benchmarks/grain_gpu/` as optional analysis tools; they add no runtime
backend or package dependency. Compact results are versioned in
[`grain_cpu_speed.summary.json`](../benchmarks/grain_cpu_speed.summary.json) and
[`native_grain_cpu.summary.json`](../benchmarks/native_grain_cpu.summary.json).
