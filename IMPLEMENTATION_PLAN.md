# Rietveld Engine Implementation Plan

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
- separate external generators that never import `rietveld`;
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
   only inside the pinned GSAS-II environment and imports no `rietveld` code.
3. Add a normal-environment fixture reader under `python/rietveld/oracle/` that
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

## Implementation unit 1: scalable accumulation and derivative storage

Status: next.

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

## Implementation unit 2: GSAS-compatible TCH symmetric pseudo-Voigt

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

## Implementation unit 3: constant-wavelength U/V/W/X/Y broadening

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

## Implementation unit 4: FCJ asymmetry and K-alpha doublets

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

## Implementation unit 5: size, microstrain, and preferred orientation

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

### Goal

Accumulate several phases and their derivatives deterministically in one pattern
calculation.

### Data model

```text
PhaseBatch
  phase_id
  reflection_range
  scale
  optional phase-level correction parameters

PatternInput
  grid
  flattened reflection arrays
  phase offsets
  instrument
  sample models by phase
  optional supplied background array
```

### Work

1. Flatten phase reflection batches into contiguous arrays with explicit phase
   offsets.
2. Accumulate phase-scale derivatives as dense global rows.
3. Preserve stable phase and reflection ordering for deterministic summation and
   result labeling.
4. Allow a supplied background array to be added once, but keep background
   model evaluation outside the profile core.
5. Return optional phase-separated `y` only in a diagnostic mode; the production
   default returns the total pattern.
6. Add cancellation/negative-intensity tests even if normal physical inputs are
   non-negative, since derivative and difference calculations may contain signs.

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

## Implementation unit 9: refinement-facing API, only after profile maturity

This unit begins only after CW X-ray, CW neutron, multi-phase, and TOF kernels
are stable. It does not include a GUI or GSAS-II project compatibility layer.

Potential scope:

- parameter-vector packing with stable names and bounds;
- residual and weighted-residual evaluation in Python;
- Jacobian-vector and transpose-Jacobian-vector products;
- optional adapters to established Python optimizers;
- constraint transforms expressed as typed graphs rather than string/global
  dictionaries;
- checkpointable plain-data model serialization with its own schema.

The first refinement API should consume the profile calculator; it must not
force refinement workflow concepts back into `rietveld-core`.

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

Do not combine adjacent items merely to reduce PR count; numerical review is
easier when parameter conventions and tolerance changes remain isolated.

## Maintainer decisions needed

Development can proceed through the next numerical units, but the following
must be decided before a public release:

1. Project license. The current repository intentionally has none.
2. Whether direct `(H, eta)` remains public as a low-level API or is labeled
   explicitly as a primitive/reference interface.
3. When support-block Jacobians become the Python default.
4. Whether the first supported CW convention is strictly GSAS-II-compatible or
   a physical-unit API with a separately documented GSAS compatibility adapter.
   This plan recommends the latter.
5. Which GSAS-II-derived fixtures are lawful and useful to redistribute; fixture
   provenance must be reviewed before commit.

## Definition of the next completed milestone

The immediate next milestone comprises implementation units 0 through 3. It is
complete when:

- the pinned oracle has actually run and produced reviewed fixtures;
- sparse support derivatives replace unconditional dense per-peak storage;
- TCH Gaussian/Lorentzian component-width mapping and derivatives are validated;
- U/V/W/X/Y widths and all chain-rule derivatives are calculated in the same
  fused Rust pass;
- a whole CW reflection list is evaluated through one Python call;
- values, retained intensity, moments, reflection widths/eta, and every local and
  global derivative pass reference, finite-difference, and oracle tests;
- release benchmarks document throughput, active-support work, and memory use.
