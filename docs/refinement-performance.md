# Refinement performance changes

The subsequent [profiling investigation](refinement-performance-followup.md)
identifies remaining work and records experiments without changing defaults.
Its [implementation follow-up](refinement-performance-round2.md) records the
next production changes and their validation.

The optimization changes work scheduling, derivative selection and small-system
linear algebra. It adds no physical terms. Existing independent NumPy profile,
structure-factor and derivative references remain the numerical authorities;
GSAS-II fixtures are unchanged. Finite support is still closed at the configured
FWHM radius, including FCJ's node-relative support convention.

## Selected derivatives

For physical parameters p and free variables u, `J_u = J_p (dp/du)`.
Structural selection takes the union of native rows reached by the physical
layout, including tied parameter targets. It never infers activity by summing
coefficients, which could cancel. Omitted native rows are represented by zeros
for stable internal indexing. The reflection pass evaluates values and selected
analytical structure-factor derivatives together; the peak/sample pass evaluates
values and the required local profile derivatives together. Fixed axial geometry
omits its two derivative chains while retaining zero rows in the internal layout.
Public full-derivative calculation functions retain their original behavior.

## Preparation reuse

A per-solve cache holds up to 32 exact keys for reciprocal geometry, scattering
arrays and intensity corrections. Keys include the full cell, HKLs, ordered
species, fixed dispersion offsets, scattering model and correction model (which
includes wavelength when relevant). Coordinates, occupancy, displacement,
scales, instrument widths and position corrections are evaluated at the current
state; none is substituted from a stale cache. The cache is shared safely by
native workers, bounded, and discarded after the solve. No global cache is used.

## Trial evaluation

The first trial retains a fused full selected linearization so an ordinary
accepted step reuses its calculated values and derivatives. Subsequent
backtracks calculate values and local profile derivatives without constructing
a dense structural Jacobian. If one is accepted, its structural linearization
is prepared at the next iteration. This hybrid avoids duplicating the expensive
profile pass on every successful first trial while reducing rejected-step work.
Evaluation budgets count the extra accepted-state preparation explicitly.

## Small damped solves

For up to 64 free parameters, form the symmetric Gram matrix and solve
`(J_w^T J_w + damping I) step = -J_w^T r_w` by Cholesky factorization.
The result is accepted only when its residual, evaluated using the original
Jacobian operator, meets `cg_tolerance * max(norm(rhs), 1)`. Nonfinite, failed,
large or insufficiently accurate solves use the existing conjugate-gradient
fallback. Step clipping, damping updates and objective acceptance are unchanged.
History reports zero CG iterations for a direct solve. Different floating-point
summation can change an optimization trajectory; scientific acceptance and
repeatability are checked explicitly rather than claiming old trajectories are
bit-for-bit identical.

## Python fixed-spectrum routing

Compatible fixed-spectrum CW requests now attach the validated wavelengths and
weights to the native request. Existing restrictions on lattice/wavelength
refinement remain. Python callbacks, unsupported custom physics, and incompatible
checkpoints retain the established Python path. Native results preserve the
spectrum, masks, constraints, component diagnostics and checkpoint state.
Custom dense-linearization budgets also retain the Python path because the
native solver currently uses its fixed default budget.

## Measured real-data result

Measured on 2026-09-16 using the same Apple M4 Pro, release build and isolated
Python 3.12 environment as the [baseline comparison](rietx-comparison.md).
The existing `benchmarks/real_data.py` QARR 1g recipe is unchanged: 7,251
measured samples, 110 reflection families, three phases, fixed Cu Kα doublet
and FCJ asymmetry. Baseline medians use three warm repetitions; optimized
medians use five. No tests or builds ran concurrently with these timings.

| Requested workers | Baseline | Optimized | Time reduction | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 1.283 s | 1.029 s | 19.8% | 1.25× |
| 8 | 0.509 s | 0.357 s | 29.8% | 1.42× |

These are complete bounded workflows, including input preparation, all three
refinement stages and final QPA covariance. They are not kernel-only timings.
Scientific records repeat exactly across cold/warm runs and both worker
settings. No support radius, iteration budget or acceptance threshold changed.
The optimizer follows a slightly different accepted trajectory:

| Scientific metric | Baseline | Optimized | Existing gate |
|---|---:|---:|---:|
| Poisson-weighted Rwp | 19.80945% | 19.81646% | ≤20% |
| Unit-weight Rwp | 13.17062% | 13.17936% | ≤15% |
| Maximum composition error | 0.6223 percentage points | 0.6180 percentage points | ≤2 percentage points |
| Stage 1 / 2 / 3 model evaluations | 28 / 49 / 2 | 31 / 29 / 3 | Explicit existing budgets |

All quality gates pass, including finite propagated covariance. Stage 1 and
stage 2 still stop at their budgets; the final scale polish converges. This is
faster time to an accepted result, not a claim of identical convergence or an
improved profile residual. The combined changes are measured together; these
results do not assign an isolated speedup to each individual change.

Five new interleaved repetitions against rietx 1.4.0 gave:

| Requested workers per library | PhaseSmith | rietx |
|---:|---:|---:|
| 1 | 1.009 s | 0.399 s |
| 8 | 0.357 s | 0.400 s |

rietx remains faster in the single-worker workflow. PhaseSmith finishes sooner
in the eight-worker workflow. Both pass the real-data quality gates, but
cross-library equivalence still fails: starting-profile relative L2 is
0.009092 versus the 0.001 limit, and final Rwp difference is 0.005563 versus
the 0.005 limit. This cannot establish an identical-calculation speed ratio.
The baseline comparison documents the physical and optimizer differences.

The supporting synthetic recovery benchmark also passes its unchanged forward,
parameter and Rwp gates. PhaseSmith's median fell from 206.630 to 92.292 ms
with one worker and from 60.862 to 39.511 ms with eight workers. The new paired
rietx medians were 60.408 and 59.630 ms, respectively. This is the documented
423-reflection, four-parameter Gaussian case; it does not replace the real-data
result above.

Raw timings, scientific records, dataset hashes and environment details are
retained under `validation/results/rietx-20260916-optimized-*.json`; the
baseline files remain unchanged. Reproduce the primary measurements with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/real_data.py --dataset iucr-qarr-1g --threads 1 --threads 8 \
  --require-release --repetitions 5 --json-output optimized-real-data.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx_qarr.py --threads 1 --repetitions 5 \
  --json-output optimized-qarr-1.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx_qarr.py --threads 8 --repetitions 5 \
  --json-output optimized-qarr-8.json
```

## Regression coverage

Selected and cached calculations are checked against the original full
linearization, including exact selected derivative rows, omitted zeros and
cache-key changes. Existing finite-difference, independent NumPy reference,
support-boundary and normalization tests remain in place. End-to-end native
doublet tests cover FCJ, masks, affine phase ties, finite covariance, recovery
against the Python solver, exact checkpoint continuation, callbacks and custom
linearization-budget fallback. No oracle golden data was regenerated because
this change introduces no new physical equation or profile convention.

Additional checksum-pinned PbSO4 workflows pass for both X-ray and neutron
data, with exact repeated scientific records. Their Poisson Rwp values are
0.103446 and 0.042172 against unchanged limits of 0.11 and 0.05. Raw results
are in `validation/results/rietx-20260916-optimized-pbso4.json`.

Final checks: `cargo fmt --check`, all-target/all-feature Clippy, and
`cargo test --workspace --all-features` pass (336 passed, 34 ignored).
The Python suite passes with 793 passed, 11 skipped and 33 deselected by its
default external-data/oracle marker policy. The real QARR and PbSO4 benchmark
runs above execute their scientific gates separately; live oracle regeneration
was not part of this performance change.
