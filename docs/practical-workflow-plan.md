# Practical monochromatic workflow plan

This document freezes the equations, parameter ownership, public interfaces,
and validation gates for implementation unit 18. The scope is monochromatic
constant-wavelength powder diffraction. GSAS-II may be used later as a pinned
external numerical oracle, but it is not a source dependency or implementation
source.

## Instrument position corrections

The first supported geometry is a Bragg--Brentano flat-plate diffractometer.
For a reflection with spacing `d`, wavelength `lambda`, Bragg angle `theta`,
goniometer radius `R`, specimen displacement `s`, and zero shift `z`, the
reported position in degrees is

```text
beta = 2 asin(lambda / (2 d))
p = beta (180 / pi) + z - (2 s / R) cos(beta / 2) (180 / pi).
```

`beta` is in radians; `p` and `z` are degrees.
`lambda` and `d` use the same length unit, while `s` and `R` use millimetres.
Positive `s` means displacement toward the source and shifts peaks to lower
angle. The convention follows McCusker et al., *J. Appl. Cryst.* 32, 36--50
(1999), DOI `10.1107/S0021889898009856`, section 6.2.

The analytical derivatives are

```text
dp/d(beta) = (180 / pi) [1 + (s / R) sin(beta / 2)]
dp/d(base position in degrees) = 1 + (s / R) sin(beta / 2)
dp/dz = 1
dp/ds = -(2 / R) cos(beta / 2) (180 / pi)
d(beta)/dlambda = 1 / (d cos(beta / 2)).
```

The wavelength derivative also includes any wavelength dependence of the
selected integrated-intensity correction. Geometry configuration is typed and
immutable during a calculation. `lambda`, `z`, and `s` may be selected as
refinement parameters; `R` is fixed configuration in this unit.

## Refinable sample physics

The existing typed providers remain the production models:

- isotropic crystallite size contributes Lorentzian broadening and owns
  `isotropic_size.crystallite_size_nm`;
- isotropic microstrain contributes Gaussian broadening and owns
  `isotropic_microstrain.rms`;
- March--Dollase preferred orientation contributes integrated intensity and
  owns `march_dollase.ratio`.

Providers expose stable parameter names and analytical derivative arrays.
Rietveld phase selections map those names to phase-owned refinable parameters;
no provider-specific Python reflection loop is introduced. Composite provider
order is insertion order and therefore observable. A lattice update must also
refresh the reciprocal metric used by March--Dollase before the next native
calculation.

## Differentiable background contract

Refinable backgrounds implement a common script-facing contract:

```text
evaluate(x) -> y
parameter_names -> stable tuple[str, ...]
parameters -> one-dimensional finite float64 array
with_parameters(parameters) -> new background
derivatives(x) -> shape (n_parameters, n_samples)
```

The built-in models are:

1. the existing power-series polynomial;
2. a Chebyshev series on an explicit finite domain;
3. a point background with fixed strictly increasing knots and refinable knot
   values, evaluated by linear interpolation with constant end values;
4. broad Gaussian amorphous components with positive area and width; and
5. a composite background with deterministic component and parameter order.

Every model validates shape, finiteness, domain, and positive parameters at the
Python boundary. Analytical derivatives are checked with centered finite
differences. Smooth Bruckner remains a separate preprocessing estimator and is
never silently inserted into this differentiable contract.

## Script-first facade and reports

`RietveldProject` is an application-neutral convenience facade over the typed
request objects. It owns pattern, experiment, phases, background, parameter
selection, options, and optional restart state. It provides explicit
`calculate()`, `refine()`, `stop()`, persistence, and result-report methods.
The underlying request types remain public for advanced scripts and plugins.

Progress uses the existing structured events and cooperative cancellation.
Stopping returns the last accepted resumable state; numerical code does not
read standard input. JSON reports contain provenance, termination, metrics,
parameters, diagnostics, and phase/reflection summaries. CSV reports contain
plain columns for observed, calculated, background, residual, weights, and
included-mask values. Reporting has no plotting or GUI dependency.

## Persistence and validation

Persistence format 5 records instrument geometry/corrections, all background
variants, phase sample models, and the facade state while continuing to load
formats 1--4. Unknown model kinds and malformed arrays fail explicitly.

The completion gate includes:

- deterministic Rust/Python and finite-difference checks for every new
  derivative, including wavelength intensity dependence;
- integrated intensity and profile-moment checks away from grid boundaries;
- synthetic laboratory X-ray and monochromatic-neutron recovery cases;
- cancellation and checkpoint continuation through the facade;
- realistic calculation and refinement benchmarks, with an optional pinned
  GSAS-II comparison kept outside normal installation and tests;
- source and wheel runs of formatting, strict Clippy, Rust tests, Ruff, and
  Python tests.
