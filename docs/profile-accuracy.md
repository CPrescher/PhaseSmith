# Choosing CW profile accuracy

`ProfileAccuracy` makes the two numerical approximations independently opt-in.
Existing calculations retain their current support and quadrature by default.

```python
from phasesmith import ProfileAccuracy, RietveldProject
from phasesmith.refinement.rietveld import RietveldOptions

accuracy = ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)
options = RietveldOptions(profile_accuracy=accuracy)
project = RietveldProject(request, options)
result = project.refine()
project.save("accelerated-fit")
```

- `ProfileAccuracy()` preserves the existing 8/48-node FCJ quadrature and
  `support_fwhm` setting. It is the default, not an infinite-support exact
  integral.
- `fast_fcj=True` retains axial asymmetry but uses four Gauss--Legendre nodes
  per nonempty overlap interval when axial span / TCH FWHM is at most 0.02.
  Otherwise the existing rule applies: eight nodes at ratios up to 0.2 and
  48 above it. Equal-height geometry has only one nonempty interval. The
  zero-geometry limit remains the existing symmetric profile.
- `tail_area_tolerance=0.01` replaces the FWHM-multiple support with a radius
  that discards at most 1% of each normalized continuous node profile's area.
  Supported budgets are `1e-8` through `0.1`. `None` retains `support_fwhm`.
  Profiles are not renormalized after truncation. This budget does not bound
  fitted parameter error, final Rwp, discrete-grid quadrature error, or area
  lost outside the measured angular range.

Both controls can be used independently. The same `profile_accuracy` keyword
is accepted by `rietveld.calculate`, `PreparedStructuralPattern`,
`calculate_structural_pattern`, and `accumulate_cw_contributions`.
`profile_fcj` accepts the quadrature control only because it is untruncated.
For example, compare a fitted state with the established calculation using:

```python
from phasesmith.refinement import rietveld

reference = rietveld.calculate(
    request.pattern,
    result.experiment,
    result.phases,
    background=result.background,
    support_fwhm=options.support_fwhm,
    profile_accuracy=ProfileAccuracy(),
)
```

Native built-in CW structural models support the policy, including fixed
wavelength spectra, sample physics, matrix-free products and dense Jacobians.
The Python callback/refinement path uses the same policy through the native
structural calculator. Unsupported external structural models reject nondefault
policies explicitly. TOF and Le Bail controls are unchanged.

## Numerical contract

The FCJ height integral and derivative equations remain those in
[FCJ profiles](fcj-profile.md). Values and analytical derivatives share one
node/sample pass. No axial term or derivative is set to zero by `fast_fcj`.
Four-node quadrature is an approximation to that same integral, not a new
physical model. The adaptive rule has discrete switches at ratios 0.02 and
0.2: derivatives hold the selected rule fixed, as they already do at the
existing 0.2 switch. Finite-difference tests avoid crossing these switches.
Randomized checks bracket both switches and compare all direct derivatives
against an independent 256-node NumPy integral. There is no promise of exact
global smoothness at the switches or of a universal fit-accuracy bound.

For a normalized pseudo-Voigt with Lorentzian fraction eta and support
half-width `k * FWHM`, the exact discarded area is

```text
D(k) = eta * (2/pi) * atan(1/(2k))
       + (1-eta) * erfc(2*sqrt(ln(2))*k).
```

The Rust kernel avoids an additional special-function dependency by solving
the conservative upper bound

```text
B(k) = eta * (2/pi) * atan(1/(2k))
       + (1-eta) * exp(-4*ln(2)*k*k) >= D(k).
```

Forty-eight bisection steps return the upper bracket with `B(k) <= budget`.
The Gaussian inequality is `erfc(z) <= exp(-z²)` for `z >= 0`.
Because FCJ is a positive normalized mixture of node profiles, using the same
area budget at every node bounds the mixture's discarded continuous area too.
This is an independently derived conservative rule; its windows do not match
rietx's frozen windows or their movement slack.

Support is closed at each node: `abs(x - node_position) <= radius`. The outer
stored support covers the union over the full geometric axial span. No
support-derivative impulse is added at changing boundaries; analytical
derivatives hold the active samples fixed, following the existing convention.

## Solver, results and persistence

Rejected backtracking trials automatically omit fixed axial derivative rows.
Their values, required derivatives and closed support are unchanged. Full
final diagnostics still include the axial rows. This scheduling optimization
does not change solver bounds, step selection, tolerances or budgets.

Projects, refinement checkpoints and JSON result reports record the policy.
Native and Python project readers give old files the existing default.
Checkpoint continuation rejects a different profile policy; begin a new fit
from the accepted parameters when changing accuracy. Native wire fields are
optional additions to the shared Rietveld options/checkpoint schema and are
omitted when at their defaults. Older software rejects the new nondefault
fields rather than silently using another calculation.

No external-oracle golden fixture is regenerated: the default remains covered
by existing fixtures; the optional quadrature is checked against an independent
high-order integral and the optional support against the analytical tail area.

## Validation and measurement

`tests/test_profile_accuracy.py` covers independent profiles and fused
derivatives, high-order integration, centered finite differences, area and
moment checks, masks/ties, native and callback refinement, exact resume, policy
mismatch, and both native and Python persistence. Rust tests cover invalid
budgets, closed support at adjacent representable coordinates and exact
value preservation when fixed axial rows are omitted.

Run the existing measured-data workflows with each policy:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/profile_accuracy.py --repetitions 5 \
  --json-output accuracy-results.json
```

This uses the unchanged QARR recipe at one, two and eight workers, alternates
measurement order, requires exact repeatability within each policy and across
worker counts, and retains the existing scientific gates. PbSO4 X-ray and
neutron fits are checked separately. Results from different policies must not
be presented as identical-calculation speed ratios.

### Measured real-data results (2026-09-16)

These historical measurements precede the [powder Friedel correction](powder-friedel.md).
Use that document's corrected measurements for the current implementation.

Release build, Apple M4 Pro, Python 3.12.13; one warmup and five measured runs
per policy/worker count in alternating order, with BLAS/OpenMP workers fixed
to one. These are complete existing QARR workflows, including preparation,
three refinement stages, covariance and result assembly.

| Policy | 1 worker | 2 workers | 8 workers | Poisson Rwp |
|---|---:|---:|---:|---:|
| Default | 0.780 s | 0.572 s | 0.341 s | 19.8165% |
| Fast FCJ only | 0.569 s | 0.457 s | 0.309 s | 19.8884% |
| 1% tail budget only | 0.546 s | 0.413 s | 0.282 s | 19.9302% |
| Both | 0.424 s | 0.352 s | 0.264 s | 19.9237% |

Both controls reduce measured time by 45.6% at one worker and 22.7% at eight.
The combined fit's largest phase-fraction error is 0.549 percentage points,
versus 0.618 for the default. Every run passes the unchanged quality gates;
scientific result records repeat exactly within a policy and across worker
counts. The different fitted Rwp values demonstrate why the policy remains
explicit even when pointwise quadrature errors are small: support and numerical
changes can alter the bounded optimizer trajectory.

PbSO4 X-ray and neutron fits also pass their existing gates for all four
policies. With both controls, Poisson Rwp is 10.3466% for X-ray (default
10.3446%) and 4.2889% for neutron (default 4.2172%). The tail budget is a
continuous profile-area guarantee, not a fit-quality guarantee on other data.

A separate interleaved comparison against the retained pre-change release
wheel isolates the default-path scheduling change. Median single-worker time
is 0.811 s before and 0.785 s after (3.3% less time); two workers improve 1.6%,
and the 0.2% difference at eight workers is negligible. Scientific records are
identical across builds at all three worker counts. This automatic optimization
does not require an accuracy flag.

Validation: 339 Rust tests passed (34 ignored); 825 Python tests passed
(11 skipped, 33 deselected). Formatting, Ruff and workspace/all-feature Clippy
checks pass. New numerical checks include angular limits, both sides of 90
degrees, zero geometry, and all five FCJ derivatives.

Raw measurements:

- [Policies and PbSO4 checks](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/validation/results/rietx-20260916-profile-accuracy.json)
- [Default before/after comparison](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/validation/results/rietx-20260916-accuracy-default-comparison.json)

### Fresh paired rietx comparison

The same host also ran interleaved five-repeat comparisons against rietx 1.4.0
with both PhaseSmith controls enabled. These measurements are separate from
the policy table above.

| Requested workers | PhaseSmith, both controls | rietx defaults |
|---|---:|---:|
| 1 | 0.422 s | 0.385 s |
| 8 | 0.261 s | 0.389 s |

Both workflows pass the existing real-data quality gates. Strict equivalence
still fails: the initial crystalline-profile relative L2 difference is
0.008755 (limit 0.001), and fitted Poisson Rwp differs by 0.006635 (limit 0.005).
These times therefore compare accepted bounded workflows with different
numerical/physical details, not interchangeable implementations of identical
calculations. PhaseSmith retains the axial correction under its fast policy;
rietx's default skips it on this dataset. The
[workload audit](rietx-workload-audit.md) explains the remaining differences.

Reproduce with `benchmarks/compare_rietx_qarr.py --fast-fcj
--tail-area-tolerance 0.01 --threads 1 --repetitions 5 --json-output result.json`
under the same single-thread BLAS/OpenMP environment, then repeat at eight
workers. Records include the selected accuracy policy and unchanged gates:

- [One worker](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/validation/results/rietx-20260916-profile-accuracy-vs-rietx-1thread.json)
- [Eight workers](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/validation/results/rietx-20260916-profile-accuracy-vs-rietx-8threads.json)
