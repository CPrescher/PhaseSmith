# PhaseSmith Implementation Plan

## Purpose and planning rules

This document converts `PROJECT_BRIEF.md` into dependency-ordered implementation
units. It is intentionally more specific than the project brief: it defines the
expected data flow, public API direction, validation artifacts, and exit gates
for each milestone.

The plan follows four rules:

1. Each numerical feature starts with equations, units, and an independent
   Python reference before production Rust code.
2. Each feature includes analytical derivatives in the same production pass as
   profile values.
3. No milestone is complete until it has finite-difference tests, integral and
   moment checks, pinned-oracle coverage, and a realistic benchmark.
4. GSAS-II remains an external process/environment that produces plain fixture
   data. Its objects and internal structures never enter the core API.
5. Public Python development follows `docs/public-api.md`: instrument, phase,
   pattern, calculation, and refinement modules remain separated; Le Bail is a
   first-class method; integration adapters remain optional and NumPy-based.

## Current baseline

The repository currently has a working first primitive:

- normalized symmetric pseudo-Voigt parameterized by total FWHM `H` and mixing
  fraction `eta`;
- exact inclusive support `abs(x - position) <= support_fwhm * H`;
- fused Rust accumulation with derivatives for intensity, position, FWHM, and
  eta;
- PyO3 and NumPy APIs plus an independent Python reference;
- local Rust/Python value, support, normalization, moment, and finite-difference
  tests;
- an external GSAS-II adapter and an opt-in direct profile comparison;
- Rust and Python benchmarks.

Implementation unit 0 was completed on 2026-08-05. The repository now also has:

- a versioned fixture schema and hash-validating normal-environment reader;
- a pinned private-profile fixture covering three width regimes and overlap;
- a deterministic public-scripting fixture containing `X`, `Ycalc`, background,
  and a documented powder reflection list;
- separate external generators that never import `phasesmith`;
- baseline Rust/Python CI and optimized benchmark build-mode reporting.

The baseline is not yet a release milestone for two numerical reasons:

1. The production primitive uses `(H, eta)`, while the GSAS-compatible TCH
   profile is naturally driven by Gaussian and Lorentzian component widths.
2. The current Jacobian is dense with size `peaks * 4 * samples`; allocating and
   zeroing it defeats the memory-scaling benefit of finite-support evaluation.

These are the next problems to solve before adding instrument broadening.

## Target numerical architecture

### Parameter layers

Profile calculation will be separated into three explicit layers:

```text
reflection and instrument parameters
                |
                v
component widths and intensity modifiers
                |
                v
normalized peak-shape primitive
                |
                v
finite-support fused accumulation
```

- The primitive layer evaluates a normalized shape from physical widths and
  shape parameters.
- The broadening layer maps reflection angle, instrument parameters, size, and
  strain into those widths and supplies chain-rule derivatives.
- The intensity layer applies phase scale, multiplicity, structure-factor
  magnitude, preferred orientation, and later correction factors.
- The accumulator owns grid lookup, support selection, summation, and derivative
  storage.

No layer accepts an untyped nested mapping.

### Units

The public/core convention will use:

- degrees for constant-wavelength `2theta` and angular widths;
- radians only in local trigonometric calculations, with conversion explicit;
- standard deviation for Gaussian component input and FWHM for Lorentzian
  component input unless a type name explicitly states otherwise;
- microseconds for TOF coordinates and widths;
- angstroms for wavelength, lattice dimensions, d-spacing, and crystallite size;
- dimensionless strain and preferred-orientation parameters.

GSAS-II centidegree and variance conventions are confined to the oracle adapter.
Every public parameter type and fixture field includes a unit suffix or unit
statement.

### Batch layout

Production input will use structure-of-arrays batches so Python can pass
contiguous NumPy memory without constructing one Rust object per reflection:

```text
PeakBatchView
  positions:   &[f64]
  intensities: &[f64]
  width terms: &[f64] or shared instrument model
  shape terms: &[f64] or shared instrument model
```

An owned Rust convenience type may wrap `Vec<f64>` fields. The hot kernel takes
borrowed validated slices. Validation occurs once per call or once when a future
`PreparedPattern` object is constructed.

### Derivative layout

Use a hybrid layout based on parameter ownership:

- Per-reflection derivatives are support-block sparse. For each reflection,
  store `start`, `length`, and contiguous derivative values only for active
  samples.
- Shared instrument/phase derivatives are dense arrays shaped
  `(global_parameter, sample)` because contributions from many peaks overlap and
  are accumulated directly.
- The calculated profile `y` is always dense.

The Python representation will use only NumPy arrays:

```text
SupportJacobian
  starts:  int64[peak]
  offsets: int64[peak + 1]
  values:  float64[active_sample, local_parameter]

PatternDerivatives
  local:  SupportJacobian
  global: float64[global_parameter, sample]
  local_parameter_names: tuple[str, ...]
  global_parameter_names: tuple[str, ...]
```

During migration, `jacobian_layout="dense"` will preserve the existing API for
small problems. The support layout should become the default before the first
published release. Converting support blocks to a dense array is a Python
convenience operation, not work performed unconditionally by the Rust kernel.

### Support policy

Support must remain deterministic and observable. Introduce:

```text
SupportPolicy::FwhmMultiple(f64)
SupportRange { left: f64, right: f64 }
```

Symmetric profiles use equal left/right radii. FCJ and TOF profiles can use
different radii. Samples exactly on either boundary are included. Derivatives
hold the active sample set fixed. A later tail-tolerance policy may be added only
after its reproducibility and oracle implications are documented.

## Implementation unit 0: make the baseline reproducible

Status: complete (2026-08-05).

### Goal

Turn the current local demonstration into a repeatable validation baseline.

### Work

1. Add `oracle/fixtures/schema.json` covering:
   - fixture format version;
   - exact GSAS-II commit and integer tag;
   - GSAS-II/Python/NumPy/platform versions;
   - input GPX SHA-256 and generation timestamp;
   - histogram name/type and coordinate units;
   - NPZ member names, shapes, dtypes, and parameter conventions;
   - reflection-table column metadata.
2. Add an explicit oracle generator executable under `oracle/scripts/` that runs
   only inside the pinned GSAS-II environment and imports no `phasesmith` code.
3. Add a normal-environment fixture reader under `python/phasesmith/oracle/` that
   validates schema version, hashes, shapes, dtypes, finite values, and pin.
4. Commit the smallest redistributable fixture set:
   - one isolated symmetric peak;
   - two overlapping symmetric peaks;
   - one minimal powder histogram with background and reflection lists.
5. Execute and review the direct `getPsVoigt` comparison. Confirm the
   centidegree density conversion and the sign/scaling of GSAS derivatives.
6. Make oracle tests fail, rather than skip, in a dedicated opt-in CI job.
   Ordinary CI remains independent of GSAS-II.
7. Add CI for formatting, strict Clippy, Rust tests, Python 3.11 and current
   Python tests, wheel build, and artifact installation smoke tests.
8. Make the Python benchmark build an optimized extension and print build mode,
   platform, peak count, active sample count, output bytes, and elapsed time.
9. Add benchmark cases for values only, values plus support Jacobian, and values
   plus dense Jacobian so allocation costs are visible.

### Exit gate

- A clean checkout can build and test without GSAS-II.
- The pinned external job can regenerate byte-structured fixtures explicitly.
- Golden fixture changes produce reviewable metadata and numerical diffs.
- At least one real GSAS-II value comparison has run and passed.
- Benchmark reports distinguish release Rust/Python execution from GSAS-II.

Final audit result: ordinary CI excludes the `external_oracle` marker and stays
GSAS-II-free. A dedicated manually dispatched, self-hosted job requires the
exact configured checkout/interpreter/binaries and fails on missing setup,
revision mismatch, or numerical mismatch. Its optional regeneration input
writes a new symmetric fixture into runner-temporary storage and validates its
schema and hashes without overwriting committed golden data. The live direct
`getPsVoigt`/`getdPsVoigt` comparison passed locally at the pinned revision,
including centidegree density, width scaling, and position-sign conversion.

## Implementation unit 1: scalable accumulation and derivative storage

Status: complete (2026-08-05).

### Goal

Remove `O(peaks * samples)` derivative allocation while preserving the current
numerical results exactly.

### Rust work

1. Add validated `GridView` and `PeakBatchView` types.
2. Replace the temporary `Vec<Peak>` constructed in the binding with borrowed
   parameter slices.
3. Split accumulation output into dense `y`, sparse local support blocks, and an
   optional dense global Jacobian.
4. Compute support indices once per peak and use the same range for values and
   all derivatives.
5. Preserve deterministic peak-order summation initially. Do not parallelize
   overlapping writes in this unit.
6. Add checked allocation arithmetic for support-value counts and global rows.
7. Keep dense Jacobian materialization as an explicit compatibility method.

### Python work

1. Add frozen `SupportJacobian` and `PatternDerivatives` result classes.
2. Add `SupportJacobian.to_dense(sample_count)` for tests and small consumers.
3. Add `jacobian_layout="support" | "dense"` during transition.
4. Validate that all peak arrays have equal one-dimensional shapes, supported
   dtypes, finite values, and contiguous converted storage.

### Tests

- Bitwise-equal `y` and numerically equal dense Jacobian versus the current
  implementation on deterministic cases.
- Empty grid, empty peak batch, peak entirely outside the grid, single active
  sample, boundary inclusion, overlapping supports, and allocation overflow.
- Randomized support-to-dense reconstruction against the NumPy reference.
- Memory-scaling assertion based on allocated derivative elements, not RSS.

### Exit gate

- Support mode stores `O(total active peak samples)` local derivatives.
- Dense compatibility output remains covered.
- Release benchmarks demonstrate that runtime and memory scale with active
  support, not full peak-grid products.

Review result: the 200-peak, 5,001-sample release benchmark stores 0.504 MB in
support mode versus 32.046 MB in dense compatibility mode. Median end-to-end
Python times were 0.079 ms and 0.292 ms respectively on the recorded development
machine. Rust and Python tests cover borrowed inputs, sparse reconstruction,
fixed-support derivatives, boundary inclusion, empty/outside supports, checked
allocation arithmetic, and deterministic peak-order summation.

## Implementation unit 2: GSAS-compatible TCH symmetric pseudo-Voigt

Status: complete (2026-08-05).

### Goal

Retain the tested `(H, eta)` primitive while adding an independently implemented
Thompson-Cox-Hastings mapping from Gaussian and Lorentzian component widths.

### Equations and API

Document and implement:

```text
TchWidths
  gaussian_fwhm_deg
  lorentzian_fwhm_deg

TchShape
  total_fwhm_deg
  eta
  d_total_fwhm / d_component_widths
  d_eta / d_component_widths
```

The mapping uses the published fifth-order total-width approximation and cubic
mixing approximation. Derivatives are derived symbolically in project
documentation and implemented directly; production code must not finite
difference the transform.

Public helpers should accept either Gaussian standard deviation or Gaussian
FWHM through unambiguous function/type names. Internally, one canonical unit is
used.

### Work order

1. Add the TCH equations and derivative derivation to `docs/equations.md` or a
   dedicated `docs/tch-profile.md`.
2. Implement the independent NumPy mapping and composed profile.
3. Add a dependency-free Rust `TchShape::from_component_fwhm`.
4. Chain primitive derivatives to Gaussian and Lorentzian component widths in
   the same sample pass.
5. Expose `profile_tch` and `accumulate_tch` in Python without removing the
   simpler `profile` teaching/primitive API.
6. Add oracle fixtures for Gaussian-dominant, mixed, and Lorentzian-dominant
   profiles at narrow and broad widths.

### Validation

- Pure-Gaussian and pure-Lorentzian limits.
- Positivity, symmetry, half maximum, integral, and centroid.
- TCH transform values against published examples or independently evaluated
  equations.
- Rust versus NumPy randomized comparison.
- Finite differences for component widths, intensity, and position.
- Pinned GSAS-II values and derivatives after explicit unit/sign conversion.

### Exit gate

The component-width API matches the pinned oracle within documented tolerances,
and all chain-rule derivatives pass finite differences away from support edges.

Review result: the Rust and independent NumPy transforms use the published
fifth-order/cubic equations with analytical component-width derivatives and a
scale-normalized evaluation that avoids intermediate overflow and underflow.
The pinned fixture covers narrow and broad Gaussian-dominant, mixed, and
Lorentzian-dominant profiles; the maximum derivative error is below `6e-5` of
the corresponding oracle derivative peak magnitude after documented unit/sign
conversion. The 200-peak release benchmark evaluates the support Jacobian in
about 0.103 ms end to end through Python on the recorded development machine.

## Implementation unit 3: constant-wavelength U/V/W/X/Y broadening

Status: complete (2026-08-05).

### Goal

Calculate reflection-dependent TCH widths from a typed constant-wavelength
instrument model and accumulate profile plus local/global derivatives.

### Data model

```text
ConstantWavelengthInstrument
  wavelength_angstrom
  u_deg2
  v_deg2
  w_deg2
  x_deg
  y_deg

CwReflectionBatchView
  two_theta_deg
  integrated_intensity
  optional reflection_id
```

The public Python object lives in `phasesmith.instrument`; array-oriented CW
operations live in `phasesmith.cw`. Shared result containers live in
`phasesmith.results`, independent of native-extension and refinement state. This
is the first enforced slice of the public module contract.

The exact parameter scaling and formula variants are frozen only after the
published convention and pinned GSAS-II adapter are reconciled. Any conversion
from GSAS-II's stored units occurs in the adapter, not in the kernel.

### Numerical work

1. Compute `theta = two_theta / 2` and the Gaussian variance contribution from
   U/V/W.
2. Compute the Lorentzian width contribution from X/Y.
3. Map component widths through TCH to total FWHM and eta.
4. Derive and implement chain-rule derivatives for U, V, W, X, and Y.
5. Include the position dependence of width in the derivative with respect to
   reflection position. The position derivative is not only translation of the
   primitive profile.
6. Accumulate the five shared instrument derivatives directly into dense global
   rows while retaining sparse per-reflection intensity and position rows.
7. Define behavior for invalid negative variance/width. Prefer a clear domain
   error for public calculation; do not silently clamp unless an explicitly
   named compatibility policy is selected.

### Validation cases

- Reflections at low, middle, and high `2theta`.
- Each of U, V, W, X, and Y isolated, then representative mixtures.
- Near-zero but valid Gaussian and Lorentzian limits.
- Finite differences for all five instrument parameters and reflection
  position.
- Width, eta, support bounds, profile values, integrated intensity, and
  reflection-list parameters against pinned GSAS-II fixtures.
- A full synthetic pattern with overlapping reflections and background removed
  before profile comparison.

### Exit gate

One array-oriented Python call evaluates an entire CW reflection batch with no
per-reflection Python orchestration, and all seven derivative classes (five
global, two local) meet documented oracle and finite-difference tolerances.

Review result: the physical-unit U/V/W/X/Y model, TCH width chain, inclusive
support, sparse reflection intensity/position columns, and dense shared
instrument rows execute in one deterministic Rust reflection/sample pass. The
position derivative includes both profile translation and angular width
variation. Eighty Python tests and fifteen Rust tests pass, including randomized
NumPy comparisons, centered finite differences for all seven derivative
classes, area/centroid checks, invalid domains, and low/middle/high plus overlap
cases from the pinned GSAS-II #5838 fixture. Fixture-normalized peak derivative
errors remain below `6e-6`. On the recorded development machine, the 200-peak,
5,001-sample release benchmark takes about 0.137 ms through Python, produces
0.442 MB of value/local/global derivative arrays, and the Rust kernel benchmark
is about 131 microseconds.

## Implementation unit 4: FCJ asymmetry and K-alpha doublets

Status: complete (2026-08-05).

### Goal

Add axial-divergence asymmetry and wavelength doublets without expanding peaks
in Python.

### FCJ plan

1. Document the Finger-Cox-Jephcoat geometry, integration variable, singular
   limits, normalization, and parameter convention from the publication.
2. Prototype quadrature and derivative equations independently in Python.
3. Choose a deterministic quadrature rule and convergence order using an error
   study against high-accuracy reference integration and the pinned oracle.
4. Implement an asymmetric Rust support range with independently calculated
   low- and high-angle radii.
5. Fuse FCJ convolution, normalization, and derivatives with accumulation;
   avoid allocating a full temporary profile per reflection.
6. Include derivatives for axial parameters and all inherited symmetric-profile
   parameters. Use published improved FCJ derivative expressions where
   appropriate, deriving the implementation independently.

FCJ review result: the public `FcjGeometry` owns only dimensionless axial
half-heights, while the `cw_fcj` composition module combines it with CW widths
in one Rust reflection/sample pass. Production uses two fixed 48-point smooth
Gauss-Legendre pieces, exact asymmetric union support, component-local support,
and simultaneous derivatives for two local plus seven shared parameters. The
independent 256-point study controls quadrature error; direct Rust/NumPy and
finite-difference tests cover low/middle/high angle, equal/unequal geometry,
zero asymmetry, normalization, centroid, and skew reversal. The pinned GSAS-II
#5838 fixture records its discretized `SH/L` behavior and the documented
fixture-local difference from the converged published integral. On the recorded
development machine, a 200-reflection, 5,001-sample release batch takes about
7.06 ms and returns 0.525 MB of values and derivatives.

### K-alpha plan

1. Add a typed `WavelengthComponents` model containing wavelength and relative
   intensity arrays with a documented normalization convention.
2. Expand components inside the Rust reflection loop. A crystallographic
   reflection remains one logical local-derivative owner.
3. Derive component positions through Bragg's law and chain derivatives to the
   base d-spacing/position, wavelength ratio, and intensity ratio.
4. Compute the union of component supports once and accumulate both components
   deterministically.

### Validation

- FCJ normalization, centroid shift, skewness, low-angle tail, high-angle limit,
  and zero-asymmetry limit.
- Quadrature convergence tests with fixed reference values.
- Doublet area conservation, component separation, intensity ratio, centroid,
  and unresolved/resolved limits.
- Finite differences for every new parameter.
- Pinned GSAS-II cases across low/mid/high angle, with and without K-alpha2.
- Benchmarks report cost by quadrature order and wavelength component count.

### Exit gate

FCJ plus doublets execute in one Rust batch call, use asymmetric finite support,
and reproduce oracle values and peak moments across the angular test matrix.

K-alpha review result: `WavelengthComponents` represents an optional discrete
spectrum whose first component is the CW reference; a one-component model is
bit-for-bit identical to the original monochromatic CW and CW+FCJ paths. The
Rust kernel expands component positions through Bragg's law inside each
reflection loop, accumulates over the exact union of component supports, and
retains one logical intensity/position derivative owner per reflection.
Analytical rows cover all inherited instrument and FCJ parameters plus every
secondary wavelength and intensity ratio. Seventeen focused Python tests,
including centered finite differences for all nine shared rows and both local
columns, supplement the full 138-test Python and 21-test Rust suites. A pinned
GSAS-II #5838 fixture covers resolved low/middle/high-angle FCJ doublets and
checks component positions, values, area, centroid, and moments. On the
recorded development machine, the 200-reflection, 5,001-sample release
benchmark takes about 6.93 ms for one FCJ wavelength and 14.07 ms for two in
Rust (about 6.97 ms and 14.09 ms through Python), returning approximately
0.525 MB and 0.633 MB respectively.

## Implementation unit 5: size, microstrain, and preferred orientation

Status: complete (2026-08-05).

### Goal

Introduce sample-dependent width and intensity modifiers while preserving the
separation between profile shape and reflection physics.

### Data model

```text
ReflectionGeometryBatchView
  h, k, l
  d_spacing_angstrom
  two_theta_deg
  reciprocal_direction or metric inputs
  base_integrated_intensity

SampleBroadening
  isotropic_size parameters
  isotropic_microstrain parameters
  later anisotropic tensors/models

PreferredOrientation
  model enum
  axis_hkl
  model parameters
```

This unit also establishes the first versioned physics-provider boundary.
Built-in size, microstrain, and preferred-orientation implementations satisfy
the same explicit batch protocols available to external packages. Broadening
providers return contiguous reflection arrays and derivative chains for one
native accumulation call; arbitrary Python code is not invoked from the Rust
peak/sample loop. Tests include one external-style vectorized provider to prove
that custom broadening can participate in calculation and finite-difference
validation without changes to refinement orchestration.

### Work order

1. Implement isotropic crystallite-size broadening with explicit shape-factor
   convention and units.
2. Implement isotropic microstrain broadening.
3. Define how each contribution combines with instrument Gaussian and
   Lorentzian components before TCH mapping.
4. Add March-Dollase preferred orientation as an intensity multiplier, including
   reciprocal-metric geometry and derivatives.
5. Add anisotropic size/strain only as separate, documented sub-milestones after
   isotropic cases are oracle-validated.
6. Keep structure-factor calculation outside this unit; accept base integrated
   intensities as inputs.

Provider and isotropic-broadening review result: provider API version 1 uses
immutable contiguous arrays for additive Gaussian variance, additive
Lorentzian FWHM, multiplicative integrated-intensity correction, position
chains, and parameter-major analytical chains. `ReflectionGeometryBatch` and
the stateless `calculate_cw_pattern` entry point call any compatible provider
once, then dispatch one fused native batch. Built-in Scherrer size and explicit
RMS Gaussian microstrain providers compose through the same public protocol as
the tested external quadratic provider. Neutral contributions reproduce the
instrument-only Rust result bit-for-bit. Eleven focused tests cover equations,
units, disabled limits, randomized native/NumPy equality, finite retained
area, centroid, all local/provider derivatives, deterministic repetition,
invalid inputs, and the external-provider contract. At this checkpoint, 149
Python and 22 Rust tests pass. For 200 reflections and 5,001 samples, size and
strain expand active work from 12,444 to 91,741 reflection/sample pairs; the
recorded Rust benchmark is about 1.008 ms versus 0.126 ms instrument-only, and
the end-to-end Python/provider/native median is about 1.08 ms with 1.791 MB of
result arrays.

Preferred-orientation review result: `MarchDollasePreferredOrientation` uses an
explicit reciprocal metric and preferred reciprocal-lattice axis, returns an
integrated-intensity modifier through the same provider protocol, and exposes
the analytical March-ratio row. Reciprocal-angle geometry and axis-coordinate
chains are independently finite-difference checked. Twelve focused tests cover
the equation, random-orientation limit, sign-equivalent reflections,
composition with size and strain, profile derivatives, and invalid geometry.
The pinned GSAS-II #5838 fixture contains the public scripting `X`, `Ycalc`,
background, and complete 125-reflection table plus three private normalized
profile probes. After the explicit GSAS-unit translation, all reflection
Gaussian variances, Lorentzian widths, and March factors agree to floating-point
precision; selected peak-relative profile errors are below `2.0e-5`, with area
and moments checked locally. The completed repository passes 166 Python and 22
Rust tests. For 200 reflections and 5,001 samples, adding orientation to the
size/strain provider increases the recorded end-to-end median from about 1.09
ms and 1.791 MB to 1.24 ms and 1.831 MB.

### Validation

- Correct dimensional scaling with wavelength, angle, d-spacing, size, and
  strain.
- Infinite-size and zero-strain limits recover the instrument-only result.
- Preferred-orientation parameter `1` recovers an unmodified intensity.
- Symmetry-equivalent reflections receive equivalent modifiers.
- Finite differences for all modifier parameters and orientation geometry.
- Oracle reflection parameters, widths, orientation factors, values, areas, and
  moments.

### Exit gate

Sample effects compose through typed width/intensity layers, not special cases
inside the primitive peak function.

## Implementation unit 6: multiple phases and scale composition

Status: complete (2026-08-05).

### Goal

Accumulate several phases and their derivatives deterministically in one pattern
calculation.

### Data model

```text
ReflectionBatch (phasesmith.phase)
  reflection_id
  h, k, l
  d_spacing_angstrom
  two_theta_deg
  integrated_intensity

Phase (phasesmith.phase)
  phase_id
  name
  reflection batch
  scale
  optional phase-level correction parameters

Pattern (phasesmith.pattern)
  grid
  observed intensity/uncertainty/mask when available
  optional supplied background array or background model

CalculationInput (phasesmith.calculation)
  pattern
  phases
  instrument
  sample models by phase
  calculation options
```

### Work

1. Flatten phase reflection batches into contiguous arrays with explicit phase
   offsets.
2. Validate durable phase/reflection IDs and preserve them in derivative and
   diagnostic labels.
3. Accumulate phase-scale derivatives as dense global rows.
4. Preserve stable phase and reflection ordering for deterministic summation and
   result labeling.
5. Allow a supplied background array to be added once, but keep background
   model evaluation outside the profile core.
6. Return optional phase-separated `y` only in a diagnostic mode; the production
   default returns the total pattern.
7. Add cancellation/negative-intensity tests even if normal physical inputs are
   non-negative, since derivative and difference calculations may contain signs.
8. Provide both stateless `calculate_pattern(...)` and reusable
   `PreparedPattern.calculate(...)` interfaces; both dispatch one flattened
   native batch call.

Review result: `ReflectionBatch`, `Phase`, and `PowderPattern` provide durable
IDs and validated immutable NumPy inputs without exposing core or oracle
objects. `calculate_pattern` evaluates each optional provider once, prefixes
its rows by phase ID, flattens phases in stable input order, and dispatches one
generic Rust accumulation. `PreparedPattern` caches that immutable translation.
Phase-scale rows are analytical; background is added once outside the profile
kernel; negative base intensities and exact duplicate-position cancellation are
supported. Optional phase curves are reconstructed from the same local support
blocks without additional native calls. Eleven focused tests cover single-phase
equivalence, fused/separate sums, order and labels, scale and provider finite
differences, zero scales, cancellation, diagnostics, prepared reuse, boundary
validation, and one-call dispatch. A pinned GSAS-II #5838 two-phase fixture
stores public `X`, total `Ycalc`, background, two complete reflection lists,
and a controlled private overlap. Widths, fused values, phase components, and
scale rows agree within fixture-local `2.8e-6` peak-relative error. The full
repository passes 179 Python and 22 Rust tests. For 200 reflections split over
four size/strain phases and 5,001 samples, the recorded release medians are
about 2.12 ms for the stateless API and 1.84 ms for the prepared API, returning
2.191 MB of result arrays.

### Validation

- Single-phase equivalence to the preceding API.
- Sum of separately calculated phases equals fused multi-phase output within a
  tight, documented summation tolerance.
- Scale derivatives, zero-scale phases, overlapping reflections, duplicate
  positions, and phase ordering.
- Oracle fixtures with at least two phases, reflection lists, phase scales,
  background, and total `Ycalc`.

### Exit gate

A multi-phase profile is computed in one Rust call with labeled local/global
derivatives and no GSAS-style project structure in the API.

## Implementation unit 7: neutron constant-wavelength profiles

Status: complete (2026-08-05).

### Goal

Reuse the validated CW shape pipeline while adding neutron-specific parameter
and oracle coverage.

### Work

1. Introduce an explicit radiation/instrument enum rather than inferring
   behavior from string codes.
2. Reuse U/V/W/X/Y and FCJ components only where the documented neutron profile
   convention matches.
3. Exclude K-alpha doublets by construction for monochromatic neutron inputs.
4. Keep neutron scattering-length and magnetic structure-factor calculation in
   a separate future physics layer; accept integrated intensities initially.
5. Add neutron-specific GSAS-II fixtures and reflection metadata.

Review result: `RadiationProbe`, `MonochromaticRadiation`, and
`ConstantWavelengthExperiment` separate probe identity from instrument response
while enforcing exact wavelength agreement. The neutron entry points accept
only the monochromatic experiment type; no wavelength-component or K-alpha
argument exists. Symmetric neutron CW, multi-phase composition, batch sample
providers, and FCJ asymmetry reuse the existing fused kernels exactly. Eight
focused tests cover typed persistence-safe data, exact X-ray/neutron profile
equivalence for equal physical widths, high-level reuse, analytical finite
differences, area and centroid, provider reuse, FCJ reuse, and invalid probe or
wavelength combinations. A pinned GSAS-II #5838 `PNC` fixture stores public
`X`, total `Ycalc`, background, and a complete 128-reflection table, with
private symmetric/FCJ probes at low, middle, and high angle. Widths agree to
floating-point precision; symmetric values are within `3.3e-5`, and FCJ values
within `4.3e-4`, with area, centroid, and resolved asymmetric moments checked.
The full repository passes 191 Python and 22 Rust tests. The 200-reflection,
5,001-sample typed neutron benchmark is about 0.134 ms for symmetric CW and
6.74 ms for FCJ, matching the shared untyped-kernel costs and allocations.

### Validation

- Cross-check that identical physical width inputs produce identical primitive
  shapes across radiation types.
- Neutron CW oracle patterns at multiple angles, including asymmetric and
  symmetric cases.
- Derivative, integral, and moment tests identical in rigor to X-ray CW.

### Exit gate

Neutron CW is a typed configuration of shared numerical layers, not a copied
parallel implementation.

## Implementation unit 8: time-of-flight profiles

Status: complete (2026-08-05).

### Goal

Add TOF peak positions, d-dependent broadening, and asymmetric exponential
profile terms using the same batch and derivative architecture.

### Data model

```text
TofInstrument
  calibration/position coefficients
  alpha coefficients
  beta coefficients
  Gaussian-width coefficients
  Lorentzian-width coefficients

TofReflectionBatchView
  d_spacing_angstrom
  integrated_intensity
```

Exact coefficient sets and units are fixed from publications plus the pinned
oracle before the public type is stabilized.

### Work order

1. Implement and validate d-spacing to TOF position calibration separately.
2. Implement d-dependent alpha, beta, Gaussian, and Lorentzian terms with
   analytical derivatives.
3. Build the normalized asymmetric TOF primitive in Python, then Rust.
4. Add asymmetric support estimation with separate left/right tail rules.
5. Chain derivatives from primitive parameters to all instrument coefficients
   and d-spacing.
6. Accumulate shared instrument derivatives densely and reflection intensity/
   d-spacing derivatives in support blocks.
7. Treat TOF bin coordinates explicitly: fixtures record whether values are bin
   minima or centers, and the public API accepts one documented convention.

Review result: `TofInstrument` exposes a physical-unit, flat 15-coefficient
model for d-to-TOF calibration, alpha/beta rates, Gaussian variance, and
Lorentzian FWHM. The independently implemented asymmetric primitive convolves
the validated TCH profile with normalized truncated back-to-back exponentials;
values and five direct derivatives share every quadrature evaluation. The
fused Rust path maps 192-point composite quadrature onto the exact active
finite-support intervals, stores intensity/d-spacing rows in support blocks,
and accumulates all 15 instrument rows densely in the same pass. Public NumPy
coordinates are explicitly calculation-bin centers in microseconds. Thirty-eight
focused Python tests cover independent high-order quadrature, coefficient
equations, direct/local/global finite differences, overlap, exact support,
normalization, centroid, skew, equal-rate symmetry, both one-sided exponential
limits, and invalid inputs. A pinned
GSAS-II #5838 `PNT` fixture stores public bin-center `X`, `Ycalc`, background,
and a complete 2,066-row, 18-column reflection table; three private profile
probes validate values, all five direct derivatives, areas, centroids, and third
moments with explicit approximation-local tolerances. The compatibility-only
`sig-q * d` convention is named and documented separately from published
reciprocal-d variants. The full repository passes 231 Python and 24 Rust tests.
For 200 reflections and 5,001 samples, recorded release medians are about
131 ms for `tail_log=8`/63,837 active reflection-samples and 156 ms for
`tail_log=20`/85,142 active reflection-samples, returning 1.665 MB and 2.006 MB
respectively.

### Validation

- Calibration values and derivatives across the d-spacing range.
- Symmetric/one-sided/strongly asymmetric limits.
- Normalization, centroid, skewness, tail mass, and support truncation.
- Randomized Rust/reference comparisons and finite differences for every
  coefficient family.
- Pinned GSAS-II TOF `X`, profile, reflection parameters, background, and
  `Ycalc` fixtures.
- Benchmarks divided by active support size and tail severity.

### Exit gate

TOF profiles use the common accumulation/derivative interfaces and meet the same
validation standard as CW profiles.

## Implementation unit 9: refinement infrastructure and first-class Le Bail

Status: complete (2026-08-05).

### Goal

Provide a clean scripted refinement layer, then implement Le Bail extraction as
the first complete workflow. This unit begins only after CW X-ray, CW neutron,
multi-phase, and TOF kernels are stable. It does not include a GUI or GSAS-II
project compatibility layer.

### Module and data model

```text
phasesmith.refinement
  ParameterKey / ParameterSpec / ParameterSet
  Bounds and typed constraint transforms
  ResidualOptions / IterationRecord / TerminationReason
  Jacobian-vector and transpose-Jacobian-vector operations

phasesmith.refinement.lebail
  LeBailInput / LeBailOptions / LeBailResult
  extract_intensities(...)
  refine(...)

phasesmith.refinement.rietveld
  Reserved separate orchestration module; no structure-factor implementation is
  invented before its physics layer exists.
```

### Shared refinement work

1. Pack selected parameters with stable typed keys, units, bounds, scales, and
   deterministic ordering.
2. Express equality/fixed/dependent constraints as typed transforms, not string
   expressions or mutable global dictionaries.
3. Evaluate masked weighted residuals and standard powder residual metrics.
4. Provide dense-small-problem and matrix-free Jacobian-vector/
   transpose-Jacobian-vector paths over the hybrid derivative representation.
5. Add an optimizer protocol plus a documented adapter to a maintained Python
   least-squares implementation; keep the objective independently callable.
6. Record every iteration as immutable plain data with convergence metrics,
   warnings, and parameter changes.
7. Serialize inputs/checkpoints/results through a versioned plain-data schema;
   arrays use NPZ or another explicit non-pickle representation.

### Le Bail work

1. Accept `Pattern`, one or more `Phase` reflection batches, instrument/sample
   models, background, masks, weights, and selected refinable parameters.
2. Initialize reflection intensities deterministically or accept supplied
   values. Keep reflection IDs stable throughout the workflow.
3. Implement non-negative Le Bail intensity redistribution from observed minus
   background intensity using calculated reflection contributions.
4. Partition exactly coincident and unresolved multiplets deterministically and
   report rank-deficient groups rather than allowing order-dependent collapse.
5. Alternate intensity extraction with bounded profile/position/scale parameter
   least-squares updates using analytical derivatives.
6. Expose a one-call `lebail.refine(...)` convenience API and lower-level
   iteration primitives for notebooks and custom automation.
7. Return refined intensities by phase/reflection ID, final `Ycalc`, background,
   phase components, residual metrics, covariance information where valid,
   iteration history, and an explicit termination reason.

### Validation

- Synthetic exact-recovery patterns with isolated, overlapping, coincident,
  absent, and zero-intensity reflections.
- Multiple phases with shared peaks, variable background, masks, nonuniform
  uncertainties, bounds, constraints, and deliberately poor starting values.
- Finite-difference checks of the packed refinement Jacobian and adjoint
  consistency for JVP/VJP operations.
- Deterministic repeated runs and checkpoint/resume equivalence.
- Pinned GSAS-II Le Bail cases compare extracted intensities, `Ycalc`, residual
  metrics, profile parameters, and convergence trends without sharing workflow
  objects or code.
- Benchmarks separate profile-kernel time, intensity extraction, optimizer
  overhead, and total iteration time.

### Exit gate

A user can perform and inspect a robust Le Bail refinement from a short Python
script without constructing parameter vectors, optimizer callbacks, project
files, or per-reflection loops. The refinement layer consumes the calculation
API without introducing refinement state into `phasesmith-core`.

Review result: immutable typed parameters, exact fixed/affine transforms,
masked residuals, hybrid JVP/VJP products, a dependency-free default solver,
and an injectable lazy SciPy-compatible optimizer protocol are implemented.
Every accepted profile step records typed before/after/scaled parameter
changes. `iterate_once` exposes the same state transition as the one-call
workflow for notebooks and GUI scheduling. Tests cover isolated, overlapping,
absent, zero, coincident, cross-phase shared reflections, constraints,
instrument-width and position updates, phase-scale identifiability, custom
optimizer injection, rank diagnostics, deterministic checkpoint resume, finite
differences, and adjoint consistency. The pinned GSAS-II case compares all CW
profile parameters, unresolved-group integrated intensities, sampled `Ycalc`,
residuals, and convergence trends. On the recorded 200-reflection/5,001-sample
release benchmark, medians were 0.217 ms for the profile kernel, 0.424 ms for
intensity extraction, 0.829 ms estimated profile-optimizer overhead, 5.580 ms
for one complete intensity iteration, and 14.480 ms for ten iterations while
retaining final phase-component curves.

## Implementation unit 10: persistence and application integration boundary

Status: complete (2026-08-05).

### Goal

Make the mature calculation and Le Bail interfaces straightforward to embed in
external applications through an application-neutral NumPy contract.

### Work

1. Finalize a versioned plain-data schema for instruments, phases/reflections,
   patterns, calculation options, and refinement checkpoints/results.
2. Add `phasesmith.integrations.dioptas` with conversion functions operating on
   NumPy arrays and plain metadata. It must import without Dioptas installed.
3. Define a minimal adapter protocol for grid/observed/background/mask input and
   calculated/component/diagnostic output; keep GUI events and widgets out of
   the library.
4. Add an optional installed-Dioptas smoke adapter only if its public API is
   stable enough; otherwise ship a tested protocol example and document the
   small glue layer expected inside Dioptas.
5. Add cancellation/progress callback hooks at calculation/refinement batch or
   iteration boundaries, never inside the native reflection hot loop.
6. Document thread ownership and GIL behavior for responsive GUI execution.

### Validation and exit gate

- Round-trip every public plain-data model without pickle or native objects.
- Contract tests use a fake Dioptas consumer to verify dtype, shape, units,
  labels, masks, background separation, and errors.
- A documented example passes a Dioptas-style pattern through calculation and
  Le Bail and receives display-ready NumPy arrays and diagnostics.
- Normal installation, import, calculation, and refinement never require
  Dioptas.

Review result: format version 1 stores finite JSON records plus hash-validated,
non-pickle NPZ arrays for CW/TOF instruments, radiation/FCJ configuration,
phases and built-in or explicitly coded third-party providers, patterns,
calculation inputs/results, typed refinement parameters/options, constraints,
complete Le Bail checkpoints, and complete results. Undefined positive-infinite
ratio metrics use an explicit JSON-null representation; NaN remains invalid.
`PersistenceBundle.to_lebail_input()` reconstructs a runnable typed request. A
machine-readable schema and corruption/version tests cover the boundary. The
Dioptas module imports without Dioptas, normalizes included/excluded masks,
exposes explicit coordinate/intensity units, returns background-separated
display arrays and labeled phase curves, and is tested through a fake two-method
consumer. Progress and cancellation occur only at calculation/iteration
boundaries; cancelled Le Bail runs return resumable checkpoints. Documentation
states current caller-thread and GIL behavior and recommends a worker process
for responsive GUI integration. Final validation passes strict Ruff, Rust
formatting, strict Clippy, 24 Rust tests, 269 normal Python tests, and one live
pinned-oracle test; normal installation has no GSAS-II, Dioptas, or SciPy
requirement.

Roadmap clarification (2026-08-06): the completed Dioptas adapter is retained
only as an isolated compatibility convenience. Dioptas-specific expansion is
not a forward milestone; new scriptability work targets the typed public API,
plain persistence, and generic NumPy interchange.

## Implementation units 11 through 19: crystallography and Rietveld

Status: units 11 through 18 implemented and independently validated (2026-08-07),
with their live pinned GSAS-II behavior fixtures pending an available external
checkout. The complete equations, data contracts,
ordering, derivative strategy, validation gates, benchmarks, and discipline are
specified in
[`docs/crystallography-plan.md`](docs/crystallography-plan.md).

The dependency order is:

11. Rust crystallography crates, general cells, and a P1 structure-factor slice.
12. Exact symmetry operations, special positions, systematic absences,
    reflection generation, orbits, and multiplicity.
13. Optional CIF import into typed plain structures plus fixed-cell CIF-to-Le
    Bail through the native reflection generator. The parser performs no
    diffraction calculation.
14. Provenance-reviewed Rust X-ray and neutron scattering models plus a
    versioned batch-provider boundary.
15. Structural intensities, analytical derivatives, and one-call native
    structure-factor/profile composition.
16. CIF-backed lattice refinement and guarded reflection-domain management.
17. First full CIF-backed Rietveld refinement using native JVP/VJP operations.
18. Practical monochromatic workflows: background preprocessing, instrument
    corrections, refinable sample physics, richer backgrounds, and reports.
19. Anisotropic, anomalous, absorption, magnetic, electron, and specialist
    crystallographic physics as separate later increments.

The completed unit-18 implementation contract and results are recorded
in [`docs/practical-workflow-plan.md`](docs/practical-workflow-plan.md).

The cross-cutting real-data checkpoint and the dependency-ordered Cu K-alpha
structural-component/QARR plan are recorded in
[`docs/real-data-validation-plan.md`](docs/real-data-validation-plan.md).

The design deliberately separates file interpretation from numerical work.
The optional CIF backend may resolve names/settings and emit exact symmetry
operations, but Rust owns reciprocal mathematics, reflection generation,
structure factors, integrated intensities, and their derivative chains. The
production Rietveld path may not materialize an unconditional dense
`(pattern_sample, structural_parameter)` Jacobian or call Python per atom or
reflection.

Before implementing each new structural parameter family, the pinned GSAS-II
environment is used as a black-box behavioral study. Public scripting output
and narrowly revision-gated probes record parameter units/ownership,
reflection intermediates, constraints, and one-at-a-time perturbation results.
Published equations and reviewed data remain the implementation sources; the
study determines conversions and validation cases, not architecture. The full
protocol is in `docs/crystallography-plan.md`.

Unit-11 review result: the workspace separates profile, crystallography,
native composition, and PyO3 crates without adding crystallography to
`phasesmith-core`. General triclinic metrics, volume, d-spacing, and six cell
derivatives are analytical. The P1 kernel calculates complex `F`,
`scale |F|^2`, all coordinate/occupancy/`Uiso`/cell/scale derivatives, dense
diagnostics, and native JVP/VJP products. Eight Python and eight Rust tests
cover independent NumPy comparisons, every centered finite difference,
invariances, invalid inputs, and adjoint consistency; all 277 normal Python and
32 Rust tests pass. On a 5,000-reflection, 64-site release benchmark, medians
are approximately 2.502 ms for values, 2.843 ms for one JVP, and 6.895 ms for
one intensity VJP. The pinned documentation survey and live-study schema are
recorded in `oracle/STRUCTURAL_PARAMETER_STUDY.md`; no external-oracle
equivalence is claimed until that fixture runs.

Unit-12 review result: exact rational affine operations are canonicalized and
checked for identity, uniqueness, finite rotation order, and closure. Rust owns
special-position expansion, exact cyclotomic systematic-absence tests,
reciprocal orbits, stable IDs/multiplicities, crystal-system inference, exact
metric equations and nullspace parameterizations, and conservative bounded
reflection generation. Typed inclusive ranges cover d, `Q = 2 pi/d`,
monochromatic CW `2theta`, and TOF calibration. Twenty-three Python and ten new Rust
tests cover P1/P-1, I/F centring, screw/glide extinction, all crystal systems,
a non-standard setting, randomized triclinic enumeration, invalid inputs, and
finite-difference cell derivatives. A prepared eight-operation C-centred
orthorhombic release benchmark generates 3,174 families in approximately 14.95
ms before the conservative-bound review and 14.64 ms after it. The external
study schema is in
`oracle/SYMMETRY_REFLECTION_STUDY.md`; no oracle equivalence is claimed yet.

Unit-13 review result: `phasesmith.structure` owns immutable parser-independent
cell/symmetry/site/provenance records and versioned JSON-compatible round trips.
`phasesmith.io.cif` defines injectable backend and resource-limit contracts; its
lazy Gemmi 0.7.5 adapter handles blocks, loops, quoted values, standard
uncertainties, missing/unknown states, current and legacy tags, explicit/Hall/
HM/number symmetry precedence, Cartesian coordinates, B/U conversion,
anisotropic preservation, and structured strict/permissive diagnostics.
`LeBailPhase.from_structure()` and `.from_cif()` generate exact allowed
monochromatic families through Rust while accepting cell-and-symmetry-only
files. Ten focused CIF tests plus the existing Le Bail suite pass. Gemmi remains
an optional parser/setting dependency under its upstream license; it supplies
no diffraction calculation and no parser object crosses the adapter.
No numerical-kernel benchmark is added for file parsing; the downstream native
reflection generation used by CIF-to-Le Bail is covered by the unit-12 batch
benchmark.

Unit-14 source gate: the non-resonant X-ray model uses every neutral/ionic row
from the CC0/public-domain XrayDB Waasmaier--Kirfel source pinned at
`663d2171bd301dc51dbe048cae459934e60347c2`; the constant coherent neutron
model uses natural/isotope `b_c` rows from public-domain `periodictable` pinned
at `182ef63a9ec118ef725aae5bb81860f4ba0fb573`. Checksums, equations, units,
species resolution, provider versioning, and explicit exclusions are frozen in
`docs/scattering-models.md` before numerical implementation. No scattering data
comes from GSAS-II.

Unit-14 review result: deterministic generation produces 211 exact
Waasmaier--Kirfel X-ray states and 367 natural/isotope coherent-neutron
records. Prepared Rust kernels deduplicate exact species before their hot loop
and return values plus `df/ds` in one pass. Thin PyO3 bindings feed immutable
typed Python results; a versioned one-call provider contract supports custom
vectorized research models without process-global registration. Independent
compact NumPy equations, finite differences, table fingerprints, invalid-state
tests, and provider-contract tests cover the slice. On a 20,000-reflection,
eight-site release benchmark, X-ray and constant-neutron medians are about
1.09 ms and 89 microseconds respectively; the optimized public Python boundary
measures about 1.61 ms and 0.65 ms including result-array construction and
validation. Strict Ruff, Rust formatting, strict Clippy, 46 Rust tests, and 317
normal Python tests pass; one external-oracle test is deselected normally.

Unit-15 source gate: general-symmetry structure-factor, isotropic displacement,
integrated-intensity, cell/scattering/position derivative, and monochromatic
unpolarized Bragg--Brentano integrated LP conventions are frozen in
`docs/structural-intensities.md`. The document also fixes multiplicity,
preferred-orientation and scale ownership, special-position derivative scope,
parameter ordering, fused execution, and independent/oracle validation
boundaries before the kernel is changed.

## Cross-cutting validation matrix

Every numerical implementation unit must cover this matrix where applicable:

| Category | Required checks |
| --- | --- |
| Values | Closed form or high-accuracy reference, Rust versus NumPy, oracle |
| Derivatives | Analytical versus centered finite difference for every parameter |
| Area | Infinite-domain normalization and finite-support retained mass |
| Moments | Centroid, width/FWHM, variance-like finite-window moment, skewness for asymmetric profiles |
| Limits | Zero/pure-component limits, narrow/broad profiles, low/high angle |
| Support | Exact bounds, inclusive edges, outside-grid peaks, moving-boundary exclusion in derivative tests |
| Composition | Overlap, doublets, sample effects, phases, background separation |
| Invalid input | Shapes, sorting, finiteness, parameter domains, overflow |
| Determinism | Repeated runs, stable ordering, documented floating-point tolerance |
| Performance | Scalar primitive, realistic batch, active support, derivative memory |

Oracle tolerances are fixture-specific. Each tolerance must identify whether it
accounts for floating-point rounding, published approximation differences,
quadrature error, unit conversion, or sampled integration. No single global
oracle tolerance is permitted.

## Source ledger

Before implementing an equation, add it to a source ledger under `docs/` with
the precise equation number or section, parameter/units translation, and an
independent derivation note. Initial primary sources are:

- P. Thompson, D. E. Cox, and J. B. Hastings, “Rietveld refinement of
  Debye-Scherrer synchrotron X-ray data from Al2O3,” *J. Appl. Cryst.* 20,
  79-83 (1987), DOI `10.1107/S0021889887087090`.
- L. W. Finger, D. E. Cox, and A. P. Jephcoat, “A correction for powder
  diffraction peak asymmetry due to axial divergence,” *J. Appl. Cryst.* 27,
  892-900 (1994), DOI `10.1107/S0021889894004218`.
- J. R. Hester, “Improved asymmetric peak parameter refinement,” *J. Appl.
  Cryst.* 46, 1219-1220 (2013), DOI `10.1107/S0021889813016233`.
- W. A. Dollase, “Correction of intensities for preferred orientation in powder
  diffractometry: application of the March model,” *J. Appl. Cryst.* 19,
  267-272 (1986), DOI `10.1107/S0021889886089458`.

GSAS-II source code is not an equation source. Its pinned executable behavior is
used to validate parameter translations and observable numerical results.

## Proposed pull-request sequence

Keep implementation units reviewable through these ordered changes:

1. Oracle fixture schema, generator, first real fixtures, and optimized
   benchmark harness.
2. Borrowed batch input and support-block Jacobian with dense compatibility.
3. TCH transform, derivatives, Python API, and symmetric oracle matrix.
4. CW U/V/W/X/Y broadening and global derivatives.
5. FCJ reference/quadrature study, followed by the production kernel.
6. K-alpha component model and fused doublet accumulation.
7. Isotropic size and microstrain.
8. March-Dollase preferred orientation.
9. Multi-phase flattening, phase scales, and background composition.
10. Neutron CW fixtures and typed configuration.
11. TOF calibration, then TOF profile and derivatives.
12. Shared refinement parameters, constraints, residuals, and matrix products.
13. Le Bail extraction, orchestration, oracle cases, and benchmarks.
14. Plain-data persistence and the application-neutral integration boundary.
15. Native cell/P1 structure-factor foundation and derivative products.
16. Symmetry, reflection generation, and systematic absences.
17. CIF import and CIF-to-Le Bail.
18. Provenance-reviewed X-ray and neutron scattering tables.
19. Fused structural pattern calculation and persistence migration.
20. Full Rietveld orchestration and benchmarks.
21. Native application boundary: owned structural preparation, execution
    policy, fixed-spectrum/multiphase composition, and a Rust-only calculation
    vertical slice.
22. Owned native domain/I/O/persistence records with Python/Rust compatibility,
    followed by native refinement workflows and application-neutral APIs.
23. Typed multi-histogram Rietveld objective with shared structural and local
    experiment parameters, followed by a joint PbSO4 X-ray/neutron benchmark.
24. Python-free real-data validation crate and CLI, native fixed-spectrum
    Rietveld workflows/persistence, Rust QPA, Rust QARR/PbSO4 runners, and thin
    Python report adapters.
25. Effective constant-wavelength starting-profile estimation from one
    predominantly single-phase pattern, with fixed wavelength, optional bounded
    nuisance lattice alignment, Le Bail independent intensities, and
    conservative W/UVW/UVWXY model selection.
26. Typed microsecond-domain fixed-instrument TOF Le Bail calculation and
    nonnegative intensity extraction, followed by checksum-pinned POWGEN LaB6
    real-pattern acceptance.
27. Offline first-slice fundamental-profile calibration: independent isolated
    physical targets from emission lines, ideal equatorial apertures, and FCJ;
    bounded compression to production U/V/W/X/Y plus equal-height SH/L with
    explicit rejection diagnostics.
28. Public POWGEN fixed-instrument TOF application slice: bounded SLOG FXYE/GSA
    and GSAS type-3 PRM import, a distinct Python microsecond-domain pattern,
    CIF reflection generation, and one-call native Le Bail extraction.
29. Broader independent experimental validation on in-house benchmark examples
    to establish profile transferability before selecting any specialized
    LPSD, tube-tail, continuum, or coupled-dispersion production increment.
30. Facility-neutral TOF interchange and independent non-POWGEN acceptance:
    plain center/density columns, logarithmic FXYE and packed constant-step STD
    adapters, legacy profile functions 1/3, and a checksum-pinned LANL nickel
    refinement gate.
31. TOF application runtime hardening: cooperative cancellation at safe
    boundaries, structured progress events, typed last-accepted checkpoints,
    and exact continuation through the Rust and Python application boundaries.
32. Native TOF project format: explicit microsecond histograms, validated TOF
    experiments, runnable Le Bail analyses, and exact checkpoint persistence.
33. Atomic multi-bank fixed-cell TOF Le Bail: shared phase/reflection geometry,
    bank-local experimental state, aggregate metrics, runtime events, and exact
    joint checkpoint continuation.
34. Shared analytical TOF lattice refinement: native setting-aware TOF geometry
    chains first, then one bounded cell optimizer over every detector bank.
35. Selected bank-local TOF instrument refinement: identifiable calibration or
    profile coefficients enter the same summed objective through the existing
    15 analytical global derivative rows, with explicit bounds and correlation
    diagnostics.
36. Joint shared-cell plus bank-local instrument refinement with one analytical
    system and explicit lattice/calibration identifiability diagnostics.
37. Shared-cell/instrument application facades, native persistence, and a
    provenance-complete multi-bank real-data/oracle acceptance case.
38. Structural TOF Rietveld only after structure-factor, correction, and
    analytical shared/local parameter contracts pass synthetic and oracle gates.

Unit 23 is complete. The Rust workflow crate now owns stable shared/local
packing, matrix-free joint products, a bounded constraint-aware summed solver,
restart/cancellation state, aggregate metrics, and an optimized Rust-only
benchmark over the pinned PbSO4 X-ray/neutron patterns and CIF.

Unit 24 is complete. The validation CLI calls Rust directly with no Python
sidecar; ordinary scripting validation calls
reconstruct the same public report types from native JSON. Release-mode Rust
tests cover deterministic QARR, both PbSO4 probes including the exact Cu K-alpha
doublet, mixed-radiation joint products, and independent Python differential
comparisons.
The PbSO4 X-ray scientific gate is accepted at 10.346% Poisson Rwp after the
final polish materially improves its accepted state and then terminates with
`repeated_rejections`. This remains a diagnostic non-convergence reason rather
than being relabeled by the solver. General recipe defaults still reject it;
the PbSO4 validation alone recognizes safe final-stage stagnation when all
stages were attempted, at least one step was accepted, and the stage improves
Rwp by an explicit minimum of 0.0001. Rejected trials never alter the returned
accepted state or checkpoint.

Unit 25's first slice is complete. `phasesmith-workflows` owns the staged
estimator and fixed-wavelength invariant; the Python facade delegates width
fitting to that native implementation. Synthetic tests cover W recovery,
automatic complexity rejection, masks, and wavelength preservation. Bundled
calibrant CIFs, CeO2/LaB6 geometry calibration, in-situ Si separation,
persistence, and GUI presentation are deliberately deferred until real
representative data are available. The estimator is documented as an effective
starting profile because it may absorb sample broadening. Real-pattern
validation remains the exit condition for promoting those deferred features.

Unit 26 is complete. `TofPatternRecord` remains non-interchangeable with the CW
angle record; the workflow carries d-spacing reflection coordinates through the
fused TOF value and derivative pass, applies masks and nonuniform-grid weights,
and extracts nonnegative intensities without a Python dependency. The official
POWGEN bank-2 type-3 coefficients are translated by the bounded I/O adapter and
the 6,824-sample, 330-family LaB6 run passes calibration, derivative, coverage,
profile-correlation, and intensity gates. Structural TOF Rietveld refinement is
outside this fixed-instrument Le Bail slice.

Unit 27's first slice is complete. Fundamental targets are generated in the
independent NumPy layer, while candidate values and analytical derivatives use
the production Rust CW-component/FCJ pass. The fitted model remains the normal
runtime U/V/W/X/Y plus single SH/L profile. Global and per-angle diagnostics
make failed compression observable. Full divergence, transparency, tube tails,
monochromator/analyser passbands, and PSD defocusing remain deferred to
equation- and data-backed increments.

Unit 27's second slice adds a deterministic independent axial-ray target with
finite source, illuminated-sample, and receiving-slit lengths plus triangular
incident and diffracted Soller transmissions. A transformed two-dimensional
quadrature integrates the exact finite-coordinate overlap, recovers the FCJ
point-incident limit, and has an explicit NIST-geometry convergence test. The
target remains offline; the fitted production profile and its analytical Rust
derivatives are unchanged. The NIST probe improves materially but fails the
normal compression gates, leaving the graphite analyser/spectral passband as a
separate next increment rather than loosening the empirical-profile contract.

Unit 27's third slice adds an explicit Gaussian wavelength transmission before
the full axial/equatorial target convolution. The center and FWHM are required
physical inputs; no bandwidth is inferred from NIST's graphite analyzer angle.
Analytical Gaussian and transformed Lorentzian integrals determine effective
transmitted component areas and centroids for the compressed fixed spectrum.
The full-order NIST probe is worse than the existing narrow effective doublet,
which is retained as a negative scientific result rather than hidden by
specimen-specific passband tuning. Coupled incident-monochromator dispersion
remains outside this post-analyzer slice.

Unit 27's Rowles transferability audit is complete. The neutral TOPAS converter
now records the deposited physical geometry, and a contained pinned-GSAS-II FPA
worker compresses it into plain `U/V/W/X/Y/SH/L` coefficients. Although the
synthetic target is represented at 6.952% Rwp, fixed use on measured 1a/1e
patterns is 4.843 and 3.515 percentage points worse than empirical GSAS-II
calibration. This is the stop gate for additional Rowles-specific optics:
continue with independent in-house benchmark examples before considering LPSD,
tube-tail, continuum, or coupled-dispersion production terms.

Unit 28's first public slice is complete. The bank-aware GSAS type-3 calibration
translation now lives in the bounded I/O crate and is shared by validation and
applications. Python exposes immutable `TofPowderData`/`TofPowderPattern`
records, the calibration reader, generated fixed-cell TOF phases, a
`TofLeBailInput.from_files` composition boundary, and a thin native refinement
facade returning stable reflection keys, intensities, backgrounds, metrics, and
cycle history. Synthetic extraction and FXYE/PRM/CIF composition tests cover
the boundary. Multi-bank requests, lattice/instrument refinement, and structural
TOF Rietveld were follow-on slices at this milestone; Units 33--38 now complete
them through separate APIs and validation gates.

Unit 31 is complete. `refine_tof_lebail_with_runtime` adopts the shared native
runtime without changing the fixed-instrument numerical algorithm. Hosts can
cancel before cycle/evaluation boundaries, consume stable events, durably sink
each accepted checkpoint, and resume that checkpoint under the same immutable
phase/background topology. Rust tests prove that a cancelled four-cycle state
continued to twelve cycles is bitwise identical to an uninterrupted run.
Python exposes the same cancellation token, progress dictionaries, opaque
checkpoint, termination reason, and continuation path while releasing the GIL
during native work.

Unit 32 is complete. Native project format 3 adds a separate
`TofHistogramRecord` with explicit microsecond samples and a validated
15-coefficient TOF experiment; the existing angle-domain histogram is
unchanged. `TofLeBailProjectState` validates one optional analysis per TOF
histogram against project pattern, instrument, phase order, and labels. The
codec stores input reflection topology and intensities plus optional
Chebyshev/background and complete accepted checkpoints, including residual
history arrays, in bounded typed NPZ members. Format 1 and 2 remain readable
with empty TOF state, and summary reports distinguish `two_theta_deg` from
`tof_us`. Round-trip coverage restores a real refined checkpoint exactly.

Unit 33 is complete. `TofMultiBankInput` validates two or more unique banks
against one exact fixed-cell reflection contract while leaving grids,
instruments, masks, uncertainties, backgrounds, phase scales, and extracted
intensities local. One joint cycle performs the current-profile,
intensity-candidate, and accepted-background calculations over every bank; it
publishes state only after all banks succeed. The result retains bank-local
residual arrays plus aggregate metrics computed over the concatenated included
observations with all local intensity/background parameters counted in the
joint degrees of freedom. Runtime cancellation, events, typed checkpoints, and
continuation operate on the complete bank set. Synthetic banks with different
nonuniform grids, calibrations, masks, scales, backgrounds, and intensities
recover their local truth, and a four-cycle cancelled state resumes bitwise
identically to an uninterrupted twelve-cycle run. Unit 34 builds shared
lattice motion on this atomic cycle contract.

Unit 34 is complete. The native `TofLatticeGeometry`/`tof_lattice_geometry`
API now composes reciprocal-metric d-spacing derivatives with the bank-local
`zero + difC d + difA d^2 + difB / d` calibration derivative. Centered
finite-difference coverage spans every supported crystal system and the
rhombohedral setting, and invalid instrument/reflection inputs fail
structurally. `TofMultiBankLatticeInput` then binds selected bounded cells to
the exact multi-bank topology. One weighted dense lattice system sums fused
`dY/dd * dd/dp` columns over every included sample, solves in scaled bounded
coordinates, and backtracks on aggregate chi-square. Local extraction and
shared cells are checkpointed atomically. Distinct synthetic banks recover the
common cubic cell, full pattern columns match centered cell differences, and a
cancelled run resumes exactly. Moving finite-support integration bounds and
the width-dependent support radius now contribute analytical Leibniz terms in
both Rust and the independent NumPy reference; profile values remain
deterministic and the realistic hot-loop benchmark shows about 1.4% overhead
at `tail_log=8`.

Unit 35 is complete for fixed-cell multi-bank refinement. The public
`TofInstrumentParameter` order is identical to all 15 fused global derivative
rows, while `TofBankInstrumentModel` selects finite bounded coefficients under
stable bank IDs. One weighted system places local columns only on their bank's
sample rows, solves in scaled coordinates, and backtracks invalid or
non-improving profiles without partial acceptance. Checkpoints validate selected
bounds, physical derived profiles, and bitwise-fixed unselected coefficients.
Results report numerical Jacobian rank, maximum absolute weighted-column
correlation, and every unresolved named bank/parameter pair. Two distinct
synthetic banks recover local Zero and DIFC truth and resume exactly after
cancellation; a one-reflection Zero/DIFC selection returns rank one and its
unit correlation instead of a misleading covariance. All 15 kernel rows retain centered-difference and independent
NumPy coverage. The realistic one-cycle benchmark over two banks, 80
reflections per bank, and 4,001 samples per bank measures 426.04 ms.

Unit 36 is complete. `TofMultiBankGeometryInput` binds Unit 34's shared cells
and Unit 35's selected local instrument coefficients to the same exact bank
request. The weighted analytical system concatenates lattice and instrument
blocks only after their residual vectors agree bitwise, then applies one scaled
damped/SVD solve and one atomic backtracking decision. Checkpoints persist cells,
instruments, intensities/backgrounds, and history together. Typed diagnostics
name both lattice and instrument keys, expose cross-family correlations, and
report numerical rank. Distinct synthetic banks simultaneously recover a
displaced cubic cell and two local Zero terms while retaining the expected
lattice/calibration correlations; four accepted cycles resume exactly. The
realistic one-cycle benchmark over two banks, 80 reflections per bank, and
4,001 samples per bank measures 214.18 ms.

Unit 37a is complete. `phasesmith.refinement.tof_multibank` owns typed Python
records for bank-local observations, exact setting-aware shared cells, selected
instrument bounds, joint controls, immutable results, rank/correlation
diagnostics, accepted physical changes, progress/cancellation, and opaque
restart checkpoints. The facade delegates one complete solve while the GIL is
released. Synthetic public-API tests recover the shared cell and bank-local
Zero coefficients and reproduce uninterrupted history and arrays after a
cancelled four-cycle checkpoint resumes. Durable project state and real-data
acceptance remain separately reviewable Unit 37 slices.

Unit 37b is complete. Native project format 4 adds explicit joint multi-bank
analysis records with disjoint histogram ownership. Project validation binds
bank IDs to TOF histogram IDs and checks patterns, initial instruments, phase
order/labels, shared cells, and exact symmetry settings. The JSON/NPZ codec
round-trips bank reflection arrays, cell/instrument bounds, numerical controls,
and complete checkpoints containing accepted cells, instruments, intensities,
backgrounds, per-bank residual arrays, aggregate history, and named physical
changes. Versions 1--3 load with empty joint state. Unit 37c's native real-data
half is also complete: checksum-pinned LANL nickel banks 2--4 share the full
102-family Fm-3m topology while retaining distinct measured grids and
function-1 instruments. One cubic cell plus three local Zero terms refine
jointly over 13,293 observations to joint Rwp 0.02273306, per-bank Rwp below
0.024, a=3.52361196 A against 3.5234 A, and analytical rank 4/4. Unit 37c is now
complete: the pinned plain-array GSAS-II comparison matches reflection
calibration chains at floating-point scale, reconstructs its three peak-only
banks with measured minimum correlation 0.99999898 and maximum relative L2
0.00181021, and returns a shared cell within 0.00025472 A of PhaseSmith. The
worker materializes temporary one-bank RAW views because the pinned public
scripting loader otherwise reselects the first dataset on repeated legacy
multi-bank imports. Endpoint counts and non-comparable Le Bail Rwp procedures
remain visible rather than being forced into false parity.

Unit 38 capability assessment is complete. The review in
`docs/tof-structural-readiness.md` separates reusable structure-factor,
reciprocal-metric, TOF-profile, and multi-bank machinery from the
constant-wavelength assumptions in the current structural engine. A structural
TOF request must carry typed bank geometry and explicitly select either a
neutral/precorrected observation contract or a named correction model. Unit
38a adds the conventional one-dimensional neutron TOF Lorentz factor and an
independently documented bounded bank-angle adapter. Unit 38b adds the Rust
structural calculation with fused dense/JVP/VJP derivatives. Unit 38c composes
the guarded multi-bank solver, and Unit 38d adds Python, persistence, real-data,
and isolated-oracle gates. Incident spectrum, detector efficiency, absorption,
and extinction cannot be silently inferred from `POWGEN`, `BNKPAR`, or a TOF
profile function.

Unit 38a's correction slice is complete. The Rust and public Python
`TimeOfFlightNeutronLorentz` model implements `d^4 sin(theta_bank)` with its
analytical reciprocal-metric derivative, explicit bank-angle validation, and
zero wavelength derivative. Rust closed-form/finite-difference tests and an
independent NumPy reference agree. The constant-wavelength structural engine
explicitly rejects the TOF-only model. The remaining Unit 38a slice is typed
bank geometry plus bounded, independently documented legacy angle import.
That remaining slice is complete. Rust and Python expose
`TofBankGeometry(two_theta_deg)` with strict open-interval validation. The
bounded legacy adapter maps the second `BNKPAR` field when present while
retaining `None` for historical profile-only files. The checksum-pinned POWGEN
and LANL validation reports now gate their distinct 90.000 and 88.05 degree
angles through the same typed contract. Unit 38a is complete. Unit 38b is also
complete: the fused Rust structural TOF calculation in the
`phasesmith-engine` primitive composes reciprocal d-spacing, built-in neutron
structure factors, explicit neutral or fixed-bank TOF Lorentz correction, and
the asymmetric finite-support TOF accumulator. It returns values plus dense,
JVP, and VJP structural derivatives while retaining all 15 bank-instrument
rows. Cell motion includes both intensity and peak-position/width chains through
`dd/dp = -d^3 d(q^2)/dp / 2`; reverse products use the exact transpose of the
sparse local profile blocks. Centered finite differences, dense/JVP/VJP and
adjoint checks, invalid correction/weight boundaries, an independent NumPy
structural-TOF composition, and a 128-reflection/16-site/14,501-sample Criterion
benchmark gate the implementation. Unit 38c's guarded multi-bank objective
slice is now complete. `PreparedStructuralTofMultiBankObjective` atomically
sums the uncertainty-weighted, masked residual objective across one or more
banks. The one-bank case uses the same solver and format-5 persistence path;
the established `MultiBank` names are retained for API stability.
Symmetry-aware lattice/site parameters are shared; scale, selected
bounded instrument coefficients, and optional Chebyshev coefficients are
bank-local. Values, JVP, VJP, gradient, and normal products use the fused
structural/profile rows and exact background bases. The contract freezes
observations, geometry, correction, support, bounds, and parameter identities,
and rejects CW sample physics, dynamic topology, implicit shared scale, or
mismatched bank angles. Joint centered differences, adjoint identity, normal
product, and invalid-contract tests form the review gate. The bounded,
checkpointable Unit 38c solver is complete too. It builds the dense scaled
normal system from the analytical matrix-free product, uses guarded LU/SVD
solution with Levenberg damping, clips physical bounds, caps scaled motion, and
backtracks against the atomic all-bank objective. Runtime limits and
cancellation return the last accepted state; accepted input, parameters,
history, objective, and next damping form an exact restart checkpoint.
Synthetic two-bank recovery, exact partial/resumed equivalence,
corrupt-checkpoint rejection, cancellation, and mid-normal evaluation
exhaustion form the solver review gate. Unit 38d is next.
Unit 38d's first slice is complete. The public
`phasesmith.refinement.tof_structural` facade owns typed shared selection,
lattice bounds, bank geometry/correction, local scale/instrument/background
bounds, dense-solver controls, immutable results, progress/cancellation, and
opaque native checkpoints. It delegates directly to the Rust objective and
solver. Python tests recover local scale/Zero parameters and one shared cubic
cell, and verify exact cancellation/checkpoint continuation. The
checksum-pinned structural data/oracle gate follows.
The application-host validation layer is complete too.
`StructuralTofMultiBankProjectState` gives each analysis a stable identity,
prevents duplicate histogram ownership, matches bank arrays/instruments to the
project's TOF records, requires exactly one matching built-in phase reference,
and revalidates optional restart checkpoints. Native project format 5 now
encodes it with stable site IDs, shared structural and bank-local
selections/bounds, explicit geometry/correction, numerical controls, and
complete accepted checkpoint/history state. Project-owned bulk arrays and
structural topology are referenced instead of duplicated; loads reconstruct
and revalidate the exact request/checkpoint contract. Formats 1--4 migrate with
an empty structural-analysis list. The pinned structural real-data/oracle gate
is next.
The native structural real-data half is complete. The core now owns a
facility-neutral twelve-coefficient Maxwellian/Chebyshev incident spectrum,
including its explicit microsecond validity interval and analytical coordinate
derivative. The bounded legacy adapter maps GSAS type 4 and rejects other
non-neutral function types instead of silently dropping them. LANL nickel raw
counts and uncertainties are normalized explicitly before the structural
objective; bank 2--4 refinement of one shared cubic cell, Ni Uiso, three scales,
and three Zero terms gives Rwp 0.03278208, minimum correlation 0.99749010, and
a=3.52373113 A over 13,293 observations. Public Python exposure and the isolated
structural oracle remain separately reviewable follow-up slices.
The Python exposure is now complete. Its independent NumPy evaluator returns
immutable intensity and analytical TOF-derivative vectors, validates the
inclusive calibration domain, and normalizes observed values, one-sigma
uncertainties, and fixed background together. The public GSAS calibration
record carries the optional translated spectrum, while any facility can create
the same `TofIncidentSpectrum` directly. The isolated structural oracle is
complete. A public-scripting-only GSAS-II worker uses the exact pinned revision
and checksum-derived banks 2--4, applies the file's type-4 incident
normalization, and refines one shared cubic cell and Ni Uiso with local scale,
Zero, and twelve-term background. It reaches Rwp 0.03367362, minimum
correlation 0.99730908, a=3.52368699 A, and Uiso=0.00401813 A^2. PhaseSmith's
independently accepted result differs by 0.00004414 A and 0.00005306 A^2 in the
two shared physical quantities, while both profiles independently clear Rwp <=
0.04 and correlation >= 0.995. The Rwp delta is retained as context because
the solvers and background models are not identical. Unit 38d is complete.
The public one-bank composition boundary is complete as a separate usability
slice. `StructuralTofMultiBankInput.from_files(...)` joins one bounded reduced
TOF pattern, one selected GSAS calibration bank with required detector
geometry, and one CIF into the existing typed structural request. Incident
normalization and the integrated-intensity correction are mandatory explicit
choices: requested type-4 normalization fails when the calibration does not
provide it, and no facility or filename convention selects physics. Direct
typed construction remains the facility-neutral path for other calibration
formats and multi-bank requests.
The file-composition audit trail is complete as a separate application-layer
slice. It retains SHA-256 and byte size for the pattern, calibration, CIF, and
embedded or separate reduction record, as well as the selected bank,
normalization/correction declarations, fixed-background array hash, and the
explicit sample-correction choice. Both integrated-intensity and sample
corrections can be declared `already_applied`, which records upstream reduction
without applying a second factor; named in-engine sample models remain rejected
until their TOF value/derivative contracts exist. Hash reads obey the same
bounded I/O limits as their parsers, and the record distinguishes no supplied
background from a supplied all-zero array. Provenance remains outside the
dependency-light numerical kernel, survives immutable Python request updates,
and is compared before Python checkpoint continuation.
The explicit ORNL/POWGEN structural acceptance slice is complete. The pinned
bank-2 header records its vanadium run, bin-width scaling, and proton-charge
normalization, while the type-0 calibration supplies geometry but no incident
spectrum. The runner therefore declares already-normalized observations and a
90-degree TOF-neutron Lorentz correction, then stages geometry/background,
boron-coordinate, and isotropic-displacement motion. It reaches Rwp
0.14415481, correlation 0.97638031, a=4.15792396 A, and x(B)=0.19954072; the
cell and coordinate agree with the published POWGEN SRM-660b result within
0.000424 A and 0.000060. This is an explicit contract for this reduction, not a
facility-name default.
The structural TOF solver's special-position state audit is complete.
Symmetry-reduced `q` coordinates are local tangent changes, so accepted
physical displacements remain in the phase while the next solver/checkpoint
state resets each `q` to zero. Trials now install the change from the current
solver state exactly once, and accepted parameter state is reconstructed from
the physical request before checkpoint publication. A mirror-site regression
covers accepted coordinate motion, checkpoint validation, and exact resume.

Do not combine adjacent items merely to reduce PR count; numerical review is
easier when parameter conventions and tolerance changes remain isolated.

Unit 23 must not emulate a joint fit by alternating single-histogram results.
One accepted trial evaluates the sum of all histogram objectives. Structure,
cell, coordinates, occupancies, and atomic displacement parameters can be
shared, while radiation, scattering/intensity correction, limits, scale,
background, profile, zero/displacement geometry, masks, and uncertainties
remain explicitly histogram-local.

The existing Python API delegates built-in workflows to the same
application-neutral native implementation without losing the independent NumPy
equation references. GUI-specific state and presentation adapters belong in a
separate application repository.

## Maintainer decisions needed

Development can proceed through the next numerical units, but the following
must be decided before a public release. The project-license question was
resolved on 2026-08-05 by selecting the MIT License.

1. Whether direct `(H, eta)` remains public as a low-level API or is labeled
   explicitly as a primitive/reference interface.
2. When support-block Jacobians become the Python default.
3. Whether the first supported CW convention is strictly GSAS-II-compatible or
   a physical-unit API with a separately documented GSAS compatibility adapter.
   This plan recommends the latter.
4. Which GSAS-II-derived fixtures are lawful and useful to redistribute; fixture
   provenance must be reviewed before commit.
5. Any additional space-group/scattering dataset beyond the reviewed Unit-14
   public-domain XrayDB and `periodictable` sources needs its own source and
   license review before redistribution.

## Definition of the next completed milestone

Implementation units 0 through 24 are complete, including the Python-free
Rietveld/Le Bail workflows, persistence and application boundaries, joint
refinement, and checksum-pinned native real-data validation. Unit 25's
effective starting-profile slice and unit 26's fixed-instrument TOF Le Bail
workflow are complete. Unit 27 has delivered the independent offline
fundamental-profile target, finite axial/Soller geometry, explicit Gaussian
passband experiment, and the Rowles transferability stop gate. Its negative
transfer result is retained rather than hidden by looser compression criteria.

Unit 28's public fixed-instrument POWGEN slice is complete. Unit 30 is also
complete: the core request is facility-neutral, the production I/O boundary
normalizes three explicit bin conventions, and the independent four-bank LANL
nickel example passes on bank 2 with Rwp 0.02485869 and correlation 0.99891242.
This validates the supported profile family beyond ORNL without promising
unknown facility-specific line shapes. Unit 31 completes cooperative
cancellation/progress and restartable accepted-state checkpoints. Unit 32
completes the explicit native TOF project/persistence schema, and Unit 33
completes atomic fixed-cell multi-bank coordination, Unit 34 completes shared
analytical lattice motion, and Unit 35 completes selected fixed-cell bank-local
instrument motion, and Unit 36 combines the two geometry families in one
correlation-diagnosed system, and Unit 37a adds its Python application facade.
The next TOF increment is Unit 38d's checksum-pinned structural real-data and
isolated-oracle acceptance gate; its public Python facade and native format-5
project persistence are complete.
Structural parameter and instrument refinement must
be introduced only with analytical derivatives, finite-difference tests, and
new oracle contracts; it is not part of the completed Le Bail facade.

Unit 29 remains the next independent constant-wavelength calibration
milestone: run broader independent experimental validation on in-house
benchmark examples and define a reviewable
transferability gate. Completion requires provenance-complete physical inputs,
plain validation artifacts, explicit cross-pattern profile metrics, and a
decision supported by those results about whether any specialized profile term
belongs in production. Failed transfer remains a valid outcome; it must not be
converted into a production feature by tuning against one specimen.
