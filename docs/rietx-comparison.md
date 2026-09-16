# rietx comparison and capability development

Measured on 2026-09-16 with the PhaseSmith 0.5.0 development kernel (base
commit `9b18acb`) and PyPI rietx 1.4.0, Apple M4 Pro (14 logical CPUs),
macOS arm64, Python 3.12.13,
NumPy 2.5.3, SciPy 1.18.1 and Numba 0.67.0. PhaseSmith used a fresh release
wheel. rietx compiled kernels were explicitly verified enabled and warmed.
BLAS/OpenMP/Accelerate thread settings were one; kernel workers are specified
separately below. Packages were installed in an isolated temporary environment.

The baseline measurements below precede the optimization follow-up. See
[implemented refinement improvements](refinement-performance.md) for the
current results and numerical contracts.

## Baseline primary benchmark: existing real-data QARR 1g workflow

The primary benchmark is PhaseSmith's existing, checksum-pinned IUCr QARR 1g
validation: **7,251 observations, 110 reflection families, three phases**
(Al2O3, ZnO, CaF2), 5–150 degrees. Observations are measured counts, with
`sigma = sqrt(max(counts, 1))`. The independently weighed composition is
31.37%, 34.21%, 34.42%. This is the existing acceptance case, not a synthetic
replacement or a reduced scale-only fit.

`benchmarks/real_data.py` ran unchanged. Three warm measured repetitions gave
**1.283 s with one worker and 0.509 s with eight workers**. The scientific
records are identical between worker settings. Every existing validation gate
passes, including finite propagated QPA covariance. The recipe intentionally
budgets its first two stages; only its final scale polish converges. These are
timings to the existing accepted result, not proof of an unconstrained optimum.

The new `benchmarks/compare_rietx_qarr.py` also runs that unchanged PhaseSmith
workflow, interleaved with a mapped rietx workflow (one warmup, three measured
runs). It checks real-data quality and scientific repeatability on **every**
run. These separate interleaved measurements were:

| Requested workers per library | PhaseSmith | rietx |
|---:|---:|---:|
| 1 | 1.241 s | 0.384 s |
| 8 | 0.496 s | 0.396 s |

| Scientific result (same at both worker settings) | PhaseSmith | rietx |
|---|---:|---:|
| Poisson-weighted Rwp | 19.8095% | 19.2601% |
| Unit-weight Rwp | 13.1706% | 13.2684% |
| Maximum error against weighed phase fractions | 0.622 percentage points | 0.913 percentage points |
| Al2O3 / ZnO / CaF2 weight percentages | 30.748 / 34.232 / 35.021 | 30.457 / 34.314 / 35.228 |

Both meet the existing real-data quality thresholds: Poisson Rwp ≤20%,
unit-weight Rwp ≤15%, profile correlation ≥0.98, and maximum phase-fraction
error ≤2 percentage points, with finite QPA uncertainties. Lower Poisson Rwp
alone does not mean more accurate phase fractions.

**The stricter equivalence gates fail.** At mapped starting parameters, the
crystalline profiles differ by relative L2 **0.009092** (0.9092%), exceeding
the diagnostic limit 0.001. Final Poisson Rwp differs by **0.005493**,
exceeding the existing cross-implementation limit 0.005. The JSON explicitly
records `equivalence.passed = false`. Neither limit was loosened. The table
therefore describes practical bounded workflows at accepted real-data quality;
it does **not** establish a speed ratio for identical numerical work.

The adapter preserves the fixed Smooth Bruckner background plus three
Chebyshev coefficients, Cu Kα doublet and polarization, fixed dispersion,
FCJ geometry, CIF anisotropic tensors, size/strain and March–Dollase texture.
It explicitly maps Uiso to Biso, Gaussian variance to FWHM squared, tensor
component ordering, and normalized doublet weights. Its stages refine the
same parameter roles, with nominal budgets 8/28/10 and fresh input state each
run. Nominal budgets do not imply equal optimizer iteration counts.

Remaining differences are observable and retained: PhaseSmith uses 30-FWHM
support; rietx uses area-tail windows frozen per stage. rietx's phase size
coefficient uses the primary wavelength, whereas PhaseSmith evaluates size
per wavelength component. Optimizer stopping, bounds, site snapping and
covariance work also differ. rietx estimates covariance at every public fit;
PhaseSmith does so only at the final scale polish. The initial discrepancy is
not attributed solely to support without further controlled tests.

Both timed operations include rebuilding their inputs. rietx time additionally
includes the shared PhaseSmith background/scale initialization and explicit
model mapping, giving it the same starting recipe conservatively. Imports,
JIT warmup and checksum verification are outside timing. These are warm
in-process measurements, not cold command-line startup times. The single-worker
gap motivated the profiling and subsequent optimization follow-up.

### Profiling follow-up: why rietx finishes sooner

**Backend correction:** the explicit `execution` argument selects the Python
validation recipe. Its Cu Kα doublet is `ComponentRadiation`, which the baseline
Python `rietveld.refine()` dispatch excluded from the fully native solver.
The measurements above therefore exercise **Python refinement orchestration
with Rust numerical kernels**, not an entirely Rust workflow. Earlier wording
calling this a native workflow was too broad.

Three warmed single-worker runs were profiled with Python `cProfile`; the
Rust extension was also sampled with macOS `sample`. Profiling overhead means
these attribution times must not replace the uninstrumented benchmark times.

- PhaseSmith spent **2.921 of 4.219 profiled seconds (69%)** inside its Rust
  dense `linearize` calls: **237 calls, or 79 per workflow**. Dense calculation
  is also requested while evaluating candidate steps, before acceptance.
- The engine constructs all physical structural columns (cell and site
  derivatives included), then projects them onto the selected fit parameters.
  Fixed cell/coordinate/occupancy parameters therefore still cause derivative
  work. The sampled native stack concentrates in FCJ accumulation and dense
  derivative assembly, consistent with the Python profile.
- Repeated preparation consumes another **0.677 seconds (16%)**, including
  phase/component objects and sample-physics evaluation. This is distinct
  from the dense calculation above.
- rietx makes **99 Jacobian calls, or 33 per workflow**, including diagnostic
  calls. Its SciPy solver requests residuals and Jacobians separately, and its
  derivative dispatch skips axial derivatives when aperture parameters are
  fixed and profile-shape derivatives in intensity-only stages. The two call
  counts describe different APIs, but demonstrate the difference in work
  scheduling; they are not a per-call speed comparison.
- rietx's different finite-support policy is another workload difference.
  Its individual contribution to the speed gap has **not** been isolated.

The separate fully native `run_qarr_1g_validation(root)` path was also checked:
**0.706 s median over five warmed runs**, with its fixed two-worker default.
All scientific gates pass (Poisson Rwp 19.8284%, maximum phase error 1.007
percentage points). That recipe uses a 35-iteration second-stage budget versus
the Python recipe's 28 and returns a different accepted state, so it is an
additional diagnostic, not a replacement row in the matched-worker table.
Python overhead alone does not account for the performance gap.

The resulting optimization targets were selected-parameter derivative
evaluation, avoiding full dense derivative work on rejected trials while
retaining fused value/derivative evaluation where required, reuse of invariant
preparation, native doublet dispatch, and small-system solver tuning. These are
now implemented and validated in the linked follow-up. Changing
tail support requires a separate tolerance analysis; it is not a free speedup.

## Symmetric profile and derivative accumulation

Both implementations calculate identical 20-FWHM closed support intervals,
the summed profile, and local intensity/position/FWHM/mixing derivatives.
Randomized inputs are deterministic. Every value and local derivative is
checked before timing. Eleven interleaved repetitions follow three warmups.

| Peaks | Samples | PhaseSmith, 1 worker | rietx, 1 worker | rietx, 8 workers |
|---:|---:|---:|---:|---:|
| 400 | 20,001 | 0.736 ms | 1.117 ms | 1.162 ms |
| 1,600 | 20,001 | 2.842 ms | 4.405 ms | 1.997 ms |
| 1,600 | 100,001 | 13.873 ms | 21.562 ms | 8.207 ms |

These are medians, not complete-fit timings. The paired PhaseSmith medians in
the eight-worker rietx run were 0.766, 2.769 and 14.072 ms. The simple
`accumulate` primitive is serial: eight workers applies only to rietx in that
comparison. rietx elects not to thread its smallest case.

The adapter calls rietx's version-pinned internal compiled APIs. Its frozen
padded window layout is prepared outside timing; PhaseSmith performs public
input validation and support discovery each call. Both allocate derivative
outputs during timing. rietx basis derivatives are scaled to physical
intensity units during timing. No implementation code is copied. Values pass
`rtol=2e-12, atol=2e-10`; derivatives pass `rtol=2e-11, atol=2e-8`.

## Complete controlled refinement

One P1 crystal with eight Si sites and 423 reflection families is sampled at
20,001 points. Four parameters refine: phase scale, Gaussian W, and two
Chebyshev background coefficients. The cell and sites remain fixed. Both
start from scale 0.9, W 15% high and background (400, 80), and recover
(1, 0.003 degrees squared, 500, 100). Both estimate covariance; optional
rietx history and stage reports are off. Seven interleaved repetitions follow
two warmups, and every fit must converge and satisfy the same recovery gate.

| Configured workers per library | PhaseSmith | rietx | Interpretation |
|---:|---:|---:|---|
| 1 | 206.630 ms | 63.987 ms | rietx about 3.23 times faster |
| 8 | 60.862 ms | 62.671 ms | approximately tied |

The common observations are noiseless synthetic data calculated by PhaseSmith,
with the same explicit square-root uncertainties in both fits. Before timing,
the independent forward curves must agree to relative L2 below `1e-7`, without
rescaling. All four recovered physical values pass `rtol=2e-6, atol=2e-6`,
and Rwp must be below `1e-6`. The Gaussian width conversion is explicit:
rietx's U/V/W FWHM-squared coefficients are `8 ln(2)` times PhaseSmith's
Gaussian variance coefficients.

The Gaussian limit is intentional: the packages' default Lorentzian-tail
support policies differ observably, so timing their unconstrained default
profiles would compare different calculations. The Gaussian case retains
both support defaults and passes the forward gate. It is not evidence about
FCJ, general atomic refinement, multiphase fits, difficult starts, or real-data
accuracy. Input construction and imports are outside timing; optimization,
internal preparation, covariance and result assembly are inside. PhaseSmith
owns pre-generated fixed HKLs; rietx compiles reflection state within `fit`.
This compares the public workflows, not identical internal operations or
stopping algorithms. Worker counts are requested maxima, not proof every
operation uses that many threads.

## Reproduce

Install a release PhaseSmith wheel and `rietx==1.4.0` into a separate Python
3.12 environment; rietx is a benchmark dependency only. Retain JSON outputs.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/real_data.py --dataset iucr-qarr-1g --threads 1 --threads 8 \
  --require-release --repetitions 3 --json-output real-baseline.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx_qarr.py --threads 1 --json-output qarr-1.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx_qarr.py --threads 8 --json-output qarr-8.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx.py --rietx-threads 1 --json-output kernels-1.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx.py --rietx-threads 8 --json-output kernels-8.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx_refinement.py --threads 1 --json-output fit-1.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/compare_rietx_refinement.py --threads 8 --json-output fit-8.json
```

The retained records under `validation/results/rietx-20260916-*.json` contain
all raw timings, environments and numerical gates. No blanket speed claim is
supported. The single-worker complete-fit gap is a useful target for profiling
preparation, derivative construction and result assembly before optimizing.

## Capability development

The first implemented slice is [read-only fit evidence](fit-report.md): native
mask-aware residual localization, typed reports, rank/covariance disclosure,
and conservative review actions. This is a foundation, not full rietx parity.

The proposed sequence after this slice is:

1. **Constraint-aware candidate diagnostics:** project candidate derivative
   columns out of the active free-parameter span, report predicted local
   chi-square gains, and abstain for unresolved directions, active-bound
   restrictions and unsupported models. Validate against explicit augmented
   least-squares solves and synthetic misfit cases. Advice never grants
   authorization or changes a refinement recipe automatically.
2. **Broader anisotropic models:** remaining Stephens symmetries and refinable
   site-symmetry-constrained displacement tensors, with independent references,
   analytical chains, positivity behavior and real-data validation.
3. **Interoperability:** prioritized vendor readers and declarative import
   reports that disclose every omitted physical term. Select formats from
   actual user files rather than claiming universal project conversion.
4. **Additional workflows:** Pawley extraction, geometric restraints and
   indexing as separate evidence-driven implementation units.

GUI work remains an external consumer of the native application boundary.
Adding an integrated Python GUI is not required to match scientific workflow
capabilities. No new runtime dependency or copied third-party physics is
introduced by this comparison.
