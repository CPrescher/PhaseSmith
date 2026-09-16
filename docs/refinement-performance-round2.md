# Second refinement performance pass

This implements the main candidates from the
[profiling investigation](refinement-performance-followup.md). Physical
equations, support, quadrature orders, solver tolerances and iteration budgets
are unchanged.

## Batched FCJ values and derivatives

Production CW contribution accumulation evaluates four adjacent samples at a
time. Each sample still visits quadrature nodes in the original order and
accumulates values and analytical derivatives together. Node-relative support
remains closed: `abs(x - apparent_position) <= radius`. The last incomplete
batch repeats its last coordinate internally and discards the extra lanes.
The implementation uses portable Rust; it makes no architecture-specific SIMD
or fast-math assumption. The scalar evaluator remains available as a check.

Structural Jacobian assembly now visits active rows outside the reflection and
sample loops. Each row retains the original reflection summation order, while
writes are contiguous and fixed-row mask checks occur once per row.

## Reusable basis for scale-only fits

When every physical parameter is a phase scale and the existing dense-memory
ceiling admits the Jacobian, one full calculation at unit phase scales builds
an immutable profile basis. For phase p,

`y_p(s_p) = s_p y_p(1)` and `dy_p/ds_p = y_p(1)`.

The complete pattern is `sum_p y_p(s_p) + background`. Integrated intensities,
position derivatives and global profile derivatives scale by `s_p`; local
derivatives with respect to integrated intensity remain unchanged. Structure
factors and reflection geometry remain independent of phase scale. Because the
basis is calculated at one, zero initial scales are supported without division
by zero or extrapolation from an empty profile.

The basis is reused for accepted states, trials and covariance products. The
existing damped solver still applies parameter bounds, affine constraints,
step limits, acceptance checks, cancellation and checkpoint rules. There is no
unchecked unconstrained least-squares replacement. A non-scale parameter or a
layout above the memory ceiling retains the general path. The basis is local
to one solve and reconstructed deterministically on resume.

Basis construction counts as a model evaluation; cached objective preparation
and covariance products do not repeat that expensive evaluation. Trial
calculations remain counted. Floating-point multiplication is regrouped from
peak accumulation to a unit-profile scaling, so comparisons with the previous
implementation use explicit local tolerances. Repeated runs and checkpoint
continuation remain deterministic.

## Native phase diagnostics

The Python adapter now decodes phase arrays exported by the native result,
instead of calling Python `calculate` after refinement. Arrays preserve the
public phase profiles, reflection identities, wavelength-component indices,
and complete local/global derivatives.

There is an important limit to reuse: ordinary selected FCJ linearizations
omit fixed axial derivative rows, which the public diagnostics still promise.
Those rows require one complete final native calculation for such fits.
Scale-only bases already contain complete derivatives and are reused directly;
non-axial results also need no replacement profile pass. This change does not
claim to eliminate every final FCJ calculation or the Python metadata assembly.

## Worker budgets

The bounded two-worker library default is retained. For an interactive fit on
the benchmark host, the already-supported
`RietveldOptions(execution=ExecutionPolicy(threads=8))` uses the useful part of
the measured scaling curve. Callers running many fits concurrently should
choose a smaller per-fit budget. The implementation does not silently increase
CPU use based on a benchmark from one machine.

## Validation contract

- Batched/scalar FCJ comparisons cover zero, equal and unequal axial geometry,
  both angular asymmetry directions, and representable coordinates immediately
  inside/on/outside node support boundaries.
- Existing independent NumPy value/derivative references, finite differences,
  normalization and support tests remain authoritative; no oracle golden data
  was regenerated because the profile equations and conventions are unchanged.
- Native Python tests compare complete diagnostic derivative arrays against a
  fresh calculation and forbid a Python recalculation inside native result
  assembly.
- Scale-basis tests start at zero and nonzero scales, with masked observations,
  uncertainties and tied phases. Recovery and covariance are compared with the
  independent Python solver, and checkpoint continuation is exact.

## Release results

All checks pass: 337 Rust tests (34 ignored), 796 Python tests (11 skipped,
33 deselected by the default external-data/oracle marker policy), formatting,
Ruff and all-target/all-feature Clippy. The unchanged real-data QARR workflow
and PbSO4 X-ray/neutron workflows also pass their separate scientific gates.

The primary before/after comparison alternates calls to the retained previous
release wheel and the current release wheel in separate persistent Python
processes. Each configuration has one warmup and five measured repetitions.
Imports are outside timing; the complete existing QARR recipe, preparation,
diagnostics and final covariance are inside. No builds or tests ran concurrently
with these timings. Hardware and numerical dependencies match the earlier
Apple M4 Pro / Python 3.12 measurements.

| Workers | Previous optimized build | This build | Time reduction |
|---:|---:|---:|---:|
| 1 | 1.020 s | 0.833 s | 18.3% |
| 2 (default) | 0.607 s | 0.582 s | 4.0% |
| 8 | 0.355 s | 0.348 s | 1.9% |

The main gain is single-worker performance. The eight-worker result is nearly
unchanged; this is not evidence of a broad large parallel speedup. At one
worker, the final scale-only stage falls from 64.43 to 21.85 ms (66% less time).
At eight workers, that stage falls from 20.36 to 13.22 ms. The production path
preserves solver controls and complete results, so its timing should not be
equated with the investigation's smaller unconstrained prototype.

An independent run of unchanged `benchmarks/real_data.py` gave 833.29 ms at
one worker and 349.10 ms at eight. Every repeated scientific record matches
exactly within each build, including across worker counts. Relative to the
previous build, QARR Poisson Rwp changes from 0.19816460010715795 to
0.19816460010715792, and maximum phase-fraction error from
0.006179751338109052 to 0.006179751338108996. These are rounding-level
differences. The final stage still converges in one accepted step; its model
evaluation count falls from three to two because covariance reuses the basis.
PbSO4 Poisson Rwp is unchanged: 0.10344569200723859 for X-rays and
0.04217168086166863 for neutrons.

The isolated preliminary FCJ-batch measurement returned exactly the prior
scientific record. Combined changes are reported using the alternating-build
comparison above; individual stage/profile savings must not be added as
independent whole-workflow gains.

Raw data are retained in `validation/results/rietx-20260916-round2-*.json`.
The new `benchmarks/compare_refinement_builds.py` verifies quality and exact
repeatability on every run and times each of the three stages. Install the
previous wheel into an isolated package directory and run with the current
release build installed in the benchmark environment:

```sh
python -m pip install --no-deps --target /tmp/phasesmith-previous PREVIOUS_WHEEL
python benchmarks/compare_refinement_builds.py \
  --baseline-package /tmp/phasesmith-previous --repetitions 5 \
  --json-output comparison.json
python benchmarks/compare_refinement_builds.py \
  --baseline-package /tmp/phasesmith-previous --threads 2 --repetitions 5 \
  --json-output comparison-default.json
```

The script pins BLAS/OpenMP/Accelerate worker settings to one in both child
processes, independently of PhaseSmith's requested worker count.

## Supporting comparisons

Five newly interleaved QARR runs against rietx 1.4.0 give 802.59 versus
394.02 ms at one worker, and 341.58 versus 395.22 ms at eight workers.
Both packages pass the existing real-data quality gates. The stricter
cross-library equivalence gates still fail (starting-profile relative L2
0.009092 > 0.001, final Rwp difference 0.005563 > 0.005), so these are bounded
workflow timings and do not establish an identical-calculation speed ratio.
No acceptance threshold was changed.

The supporting 423-reflection synthetic recovery benchmark passes its unchanged
forward/parameter/Rwp gates. Single-worker PhaseSmith time falls from the prior
92.29 ms to 73.11 ms. Eight-worker time is effectively flat (39.51 ms previously,
40.25 ms now, a 1.9% increase in these separate runs). Its new paired rietx
medians are 61.72 and 62.51 ms. This diagnostic reinforces the single-worker
benefit without substituting synthetic data for the real-data acceptance case.
