# Background estimation and subtraction

Background subtraction is a preprocessing operation, distinct from the
additive differentiable background models used during refinement. The public
module boundary is `phasesmith.background`: it accepts and returns plain NumPy
arrays and has no dependency on Dioptas, xypattern, or a GUI.

## Smooth Bruckner compatibility model

The named procedure traces to Sergio Brückner, [*Estimation of the background
in powder diffraction patterns through a robust smoothing
procedure*](https://doi.org/10.1107/S0021889800003617), *Journal of Applied
Crystallography* **33**, 977–979 (2000). Brückner introduced it for estimating
the background beneath Bragg peaks, with particular attention to diffraction
patterns from semicrystalline polymers. The paper describes the method as an
extension of traditional smoothing that treats Bragg peaks as removable
fluctuations alongside random profile noise.

The documented origin of the named algorithm is therefore powder diffraction,
not astronomy. Astronomy uses related moving-window smoothing and iterative
clipping techniques for sky and spectral-background estimation, but that
similarity does not establish an astronomical provenance for the Brückner
procedure. `SmoothBrucknerBackground` is the ASCII software class name used by
xypattern; it is not the title used in the original publication.

The first estimator reproduces the observable algorithm in xypattern's pinned
[`smooth_bruckner.pyx`](https://github.com/CPrescher/xypattern/blob/6e4574d75d2d6fcefc633f9fbecc27b8f1bcd817/xypattern/util/smooth_bruckner.pyx),
used by Dioptas-style workflows. Compatibility is pinned to xypattern revision
`6e4574d75d2d6fcefc633f9fbecc27b8f1bcd817` (release 1.2.3). xypattern is MIT
licensed; its notice is retained in `THIRD_PARTY_NOTICES.md`.

For an input intensity vector `y` of length `n` and a non-negative half-window
`N`, the native kernel:

1. extends `y` by `N` copies of each endpoint;
2. initializes a mean over the first `2 N + 1` extended samples;
3. scans extended indices `i = N, ..., n - N - 3` on every iteration;
4. replaces `y_i` by the current moving mean only when `y_i` is above it; and
5. updates that mean incrementally before advancing to the next sample.

The last `2 N + 2` returned samples are therefore not clipped by the scan. This
looks unusual but is retained deliberately because the linked Cython range is
observable compatibility behavior. The separate upstream Python fallback also
contains a global pre-clipping step that the linked Cython implementation does
not perform; this implementation follows the linked Cython path exactly.

`smooth_bruckner(y, smooth_points=N, iterations=k)` exposes the point-window
kernel directly. `SmoothBrucknerBackground` is the script-facing physical-width
model. It converts `smooth_width` to `N` with

```text
N = int(smooth_width / (x[1] - x[0]))
```

on a validated uniform increasing grid. By default it then fits and evaluates a
degree-50 Chebyshev polynomial on `x` normalized to `[-1, 1]`, matching the
xypattern `SmoothBrucknerBackground` pipeline. Setting `chebyshev_order=None`
returns the raw smoothed envelope.

The subtraction result retains the estimated background, the corrected signal,
the raw smoothed intermediate, and the effective point half-window. Arrays are
read-only and contiguous. The estimated array can be supplied directly as
`PowderPattern(..., background=result.background)`.

## Validation boundary

- Rust is compared with an independent NumPy loop on deterministic random and
  peak-rich signals.
- A frozen small fixture is generated from the pinned upstream Cython
  implementation and compared with a tight floating-point tolerance.
- Edge cases cover zero iterations, zero-width windows, large windows, invalid
  counts, non-finite input, nonuniform grids, and polynomial order limits.
- A realistic multi-peak pattern benchmark reports native smoothing and the
  complete smoothing-plus-Chebyshev pipeline separately.

An isolated install of xypattern 1.2.3 at the pinned revision gives a maximum
absolute difference of `8.89e-16` for the raw 32-sample compatibility fixture
and `8.31e-14` for a 1,001-sample default smoothing-plus-Chebyshev pipeline.
xypattern remains absent from the project environment and dependency metadata.

On the development host's release build, a realistic 20,001-sample, 200-peak,
40-point-half-window, 50-iteration case takes 3.61 ms median through the public
native smoother. The complete degree-50 Chebyshev subtraction takes 13.15 ms
median. The corresponding Rust Criterion kernel measurement is 3.76 ms, or
approximately 5.31 million input samples per second for each complete
50-iteration call. Run `benchmarks/background.py --require-release` and the
`background_smoother` Criterion case to reproduce the measurements.

This estimator is non-linear and non-differentiable at clipping decisions. It
is intentionally not a refinable parameter family and does not participate in
Rietveld JVP/VJP calculations.

## Refinable analytical backgrounds

Refinable backgrounds remain distinct from preprocessing and are owned by
`phasesmith.refinement.background` in Python and the Python-free
`phasesmith-workflows` crate in Rust. Both interfaces provide the same ordered
models:

- a power series `sum(c_k t^k)` on the input grid normalized to `t in [-1,1]`;
- a Chebyshev series `sum(c_k T_k(t))` on an explicit closed degree domain;
- linear interpolation through fixed ordered knots, with constant values
  outside the first and last knots;
- area-normalized broad Gaussian amorphous components; and
- ordered additive composition with unique immediate component IDs.

For one amorphous component with area `A`, center `mu`, FWHM `H`,
`q = 4 ln(2)`, and `d = x - mu`,

```text
g(x) = sqrt(q / pi) / H * exp(-q (d/H)^2)
b(x) = A g(x)

db/dA  = g(x)
db/dmu = b(x) * 2 q d / H^2
db/dH  = b(x) * (-1/H + 2 q d^2/H^3)
```

Power, Chebyshev, and point bases are coefficient-invariant and may be cached
across optimizer trials. Amorphous derivatives depend on the current center
and width; a composite is invariant only when all its components are. Native
matrices are checked row-major sample-by-parameter records, and constructors
prevent invalid domains, knots, coefficients, widths, IDs, or duplicate
component IDs.

Each native derivative column is checked against centered coefficient
differences for all model families. A configured differential gate compares a
mixed composite's values and complete basis with the existing NumPy
implementation. On the development host, the quick release Criterion run for
a realistic mixed 20,001-sample background measured a 295.98 microsecond
central estimate for values and 472.81 microseconds for the complete analytical
basis. Run `cargo bench -p phasesmith-workflows --bench backgrounds -- --quick`
to repeat that smoke measurement; omit `--quick` for a full measurement.
