# Public Python architecture and scripting contract

The Python package is a scientific library first. Every normal calculation and
refinement workflow must be expressible from a script without project files,
global state, callbacks per reflection, or GUI objects.

## Stable module boundaries

```text
rietveld.instrument
  Instrument response and instrument-geometry models. No phase, radiation
  spectrum, or refinement state.

rietveld.radiation
  Optional discrete wavelength/source models. A single component is the exact
  monochromatic baseline; doublets and other spectra are explicit composition.

rietveld.phase
  Crystallographic phase metadata and typed reflection batches. No observed
  pattern, optimizer, or instrument state.

rietveld.crystallography  [cells and P1 implemented]
  General cells, P1 atom sites, caller-supplied scattering amplitudes, complex
  structure factors, and dense/JVP/VJP derivatives are implemented.

rietveld.symmetry  [implemented]
  Exact symmetry operations and closure, special positions, systematic
  absences, metric parameterizations, and prepared d/Q/CW/TOF reflection
  generation. Numerical evaluation and family enumeration are Rust-owned.

rietveld.scattering  [planned]
  Typed X-ray and neutron scattering models plus versioned batch-provider
  contracts. Built-in production models execute in Rust.

rietveld.io.cif  [implemented]
  Optional CIF parsing into crystallography models. Parser objects never enter
  calculation, persistence, or refinement state.

rietveld.pattern
  Observed grids, intensities, uncertainties, masks, backgrounds, and
  calculated-pattern result containers.

rietveld.calculation
  Stateless and prepared pattern calculators that compose instrument, phase,
  sample, and pattern inputs into native batch calls.

rietveld.refinement
  Parameter selection, bounds, constraints, residuals, diagnostics, and
  optimizer adapters. Refinement methods are submodules, not mode flags.

rietveld.refinement.lebail
  First-class Le Bail intensity extraction and diagnostics.

rietveld.refinement.rietveld
  Planned structure-factor refinement orchestration after the required native
  physics layer exists. It consumes the same profile interface as Le Bail.

rietveld.integrations.dioptas
  Optional conversion between Dioptas-facing NumPy data and the public typed
  models. Dioptas is never a core or required Python dependency.

rietveld.extensions
  Versioned provider protocols and a reserved future entry-point discovery
  convenience. Providers are passed explicitly into calculations; discovery
  never creates numerical core global state.

rietveld.sample
  Built-in size, microstrain, and preferred-orientation providers. These obey
  the same public protocol as third-party implementations.
```

Low-level profile functions remain available for equation testing and advanced
use. They do not become the only way to calculate a pattern.

The implemented low-level module split already follows this boundary:

```text
rietveld.instrument.ConstantWavelengthInstrument
rietveld.instrument.FcjGeometry
rietveld.instrument.TofInstrument
rietveld.radiation.WavelengthComponents
rietveld.radiation.RadiationProbe
rietveld.radiation.MonochromaticRadiation
rietveld.radiation.ConstantWavelengthExperiment
rietveld.phase.ReflectionGeometryBatch
rietveld.phase.ReciprocalMetric
rietveld.phase.ReflectionBatch
rietveld.phase.Phase
rietveld.crystallography.UnitCell
rietveld.crystallography.AtomSiteBatch
rietveld.crystallography.calculate_p1_structure_factors
rietveld.crystallography.p1_jacobian_vector_product
rietveld.crystallography.p1_intensity_transpose_jacobian_vector_product
rietveld.symmetry.SymmetryOperation
rietveld.symmetry.SpaceGroup
rietveld.symmetry.PreparedReflectionGenerator
rietveld.symmetry.DSpacingRange
rietveld.symmetry.ScatteringVectorRange
rietveld.symmetry.CwTwoThetaRange
rietveld.symmetry.TofRange
rietveld.structure.CrystalStructure
rietveld.structure.AtomSite
rietveld.structure.AnisotropicDisplacement
rietveld.structure.structure_to_record
rietveld.structure.structure_from_record
rietveld.io.cif.read_cif
rietveld.refinement.lebail.LeBailPhase
rietveld.pattern.PowderPattern
rietveld.pattern.PatternCalculationResult
rietveld.extensions.PhysicsContribution
rietveld.extensions.ReflectionPhysicsProvider
rietveld.extensions.CompositePhysicsProvider
rietveld.sample.IsotropicSizeBroadening
rietveld.sample.IsotropicMicrostrainBroadening
rietveld.sample.MarchDollasePreferredOrientation
rietveld.sample.reciprocal_angle_geometry
rietveld.cw.cw_profile_parameters
rietveld.cw.accumulate_cw
rietveld.cw.accumulate_cw_components
rietveld.cw.accumulate_cw_contributions
rietveld.calculation.calculate_cw_pattern
rietveld.calculation.calculate_pattern
rietveld.calculation.PreparedPattern
rietveld.calculation.calculate_monochromatic_pattern
rietveld.calculation.calculate_neutron_pattern
rietveld.calculation.calculate_neutron_fcj_pattern
rietveld.fcj.profile_fcj
rietveld.fcj.accumulate_cw_fcj
rietveld.fcj.accumulate_cw_fcj_components
rietveld.tof.tof_profile_parameters
rietveld.tof.profile_tof
rietveld.tof.accumulate_tof
rietveld.results.AccumulationResult
rietveld.refinement.ParameterKey
rietveld.refinement.ParameterSpec
rietveld.refinement.ParameterSet
rietveld.refinement.ParameterChange
rietveld.refinement.ConstraintTransform
rietveld.refinement.evaluate_residuals
rietveld.refinement.jacobian_vector_product
rietveld.refinement.transpose_jacobian_vector_product
rietveld.refinement.lebail.LeBailInput
rietveld.refinement.lebail.LeBailOptions
rietveld.refinement.lebail.LeBailResult
rietveld.refinement.lebail.extract_intensities
rietveld.refinement.lebail.iterate_once
rietveld.refinement.lebail.refine
rietveld.persistence.PersistenceBundle
rietveld.persistence.PersistenceBundle.to_lebail_input
rietveld.persistence.save_bundle
rietveld.persistence.load_bundle
rietveld.integrations.dioptas.DioptasPatternData
rietveld.integrations.dioptas.DioptasDisplayResult
rietveld.integrations.dioptas.calculate
rietveld.integrations.dioptas.refine_lebail
```

Top-level imports are convenience aliases for scripts and notebooks; the
module-qualified paths above are the ownership boundary. FCJ geometry does not
contain CW coefficients, and the FCJ module composes the two models through a
single native batch call. `ReflectionGeometryBatch` owns plain `hkl`, d-spacing,
position, and base-intensity arrays. `calculate_cw_pattern` accepts it directly
and calls an optional provider exactly once before one native fused
accumulation. Future phase and refinement layers consume this calculation
surface rather than moving their state into either model.

## Data and ownership rules

- Public domain models are frozen dataclasses or similarly inspectable typed
  objects. Their numerical payloads are NumPy arrays.
- Array fields have documented shape, units, ordering, and parameter names.
- Calculation inputs never own optimizer state. Refinement state never enters
  `rietveld-core`.
- Stable string or integer IDs connect phases and reflections to derivative
  rows and diagnostics. Array position alone is not a durable external ID.
- Models support explicit plain-data serialization. Native extension objects,
  borrowed Rust views, GSAS-II objects, and GUI objects are not serialized.
- Convenience constructors perform unit conversion explicitly; the canonical
  stored units remain the documented public physical units.
- Prepared calculators may cache validated contiguous arrays, but a stateless
  one-call API remains available for notebooks, tests, and external programs.

## Calculation interface direction

The high-level call will have the conceptual form

```python
result = calculate_pattern(
    pattern=pattern,
    instrument=instrument,
    phases=phases,
    options=options,
)
```

`result.y` is the total calculated array. Named phase/background components and
reflection parameters are optional diagnostics. Analytical derivatives retain
the hybrid local/global storage implemented by the numerical core and expose
stable parameter labels.

An external program such as Dioptas should need only to provide contiguous
`x`, observed intensity, optional uncertainty/mask arrays, and typed
instrument/phase data. It receives NumPy calculated arrays and plain diagnostic
records. The adapter must not require that Dioptas adopt internal Rust types.

## Refinement methods

Refinement methods share infrastructure for parameter packing, bounds,
constraints, residual weighting, iteration records, convergence checks, and
checkpointing. Their physics and state transitions remain separate:

- Le Bail alternates profile calculation and grouped integrated-intensity
  extraction while refining selected non-structural parameters.
- Rietveld refinement obtains integrated intensities from a structure-factor
  layer and optimizes structural/profile parameters against the same pattern
  calculator.
- Future Pawley or whole-pattern methods get their own method modules and
  result types rather than conditionals in one large refinement function.

## Le Bail contract

Le Bail is the first complete refinement workflow. It must provide:

1. A typed input containing observed pattern, instrument, phases/reflections,
   background, masks, weights, and parameter selections.
2. Deterministic initialization of non-negative reflection intensities, with an
   option to supply starting values.
3. Multiplet-aware intensity partitioning for unresolved or exactly coincident
   reflections.
4. Alternating intensity extraction and bounded least-squares updates using
   analytical profile derivatives.
5. Configurable convergence thresholds and iteration limits.
6. Per-iteration residual metrics, parameter changes, intensity changes,
   warnings, and termination reason.
7. A checkpointable plain-data result containing refined parameters,
   intensities keyed by reflection ID, `Ycalc`, component arrays, and history.

The simplest supported script does not require a user to construct a parameter
vector or optimizer callback manually. Exact equations, convergence behavior,
and examples are in [`refinement.md`](refinement.md) and
[`lebail.md`](lebail.md).

## Compatibility policy

Before the first stable release, domain models may evolve with explicit release
notes. Once declared stable, field removal or unit/order changes require a
versioned migration path. New optional fields and result diagnostics may be
added compatibly. Integration adapters are kept thin so external release cycles
do not constrain the numerical core.

## Extensible physics providers

Built-in broadening and intensity models implement the same batch-oriented
provider contracts offered to downstream packages. A broadening provider is
called once with typed instrument, phase/reflection, and sample inputs and
returns contiguous per-reflection component widths or contributions, stable
parameter names, and analytical derivative-chain arrays. The generic Rust
profile kernel consumes those arrays and retains ownership of support-limited
peak/sample accumulation. This makes Python-defined size, strain, defect, or
empirical laws practical without introducing a Python callback in the hot loop.

An entirely new intrinsic line shape has two paths:

1. A vectorized Python reference provider for prototyping, validation, and
   modest workloads.
2. A separately compiled native provider implementing the versioned kernel
   trait for production. Rust's ABI is not treated as dynamically stable; a C
   ABI or other binary boundary will be introduced only with an explicit API
   version and compatibility tests.

Calculations receive provider instances explicitly. Optional Python package
entry-point discovery is a convenience in `rietveld.extensions`, not an
implicit registry used by `rietveld-core`. Persistence records provider ID,
provider version, API version, and plain-data configuration; it never pickles
live provider or native objects. Refinement methods depend on the calculator
contract, so a compatible provider is automatically usable by Le Bail and
later Rietveld workflows.

Provider API version 1 returns additive Gaussian variance, additive Lorentzian
FWHM, a multiplicative intensity correction, position chains, stable parameter
names, and parameter-major derivative chains. The precise equations and array
semantics are documented in [`sample-physics.md`](sample-physics.md). Width
contributions add; intensity modifiers compose with the full product rule.
The built-in March--Dollase provider uses the phase reciprocal metric and an
explicit preferred reciprocal-lattice axis, and participates in the same
calculation and derivative interface as broadening providers.

Multi-phase flattening, phase-scale rows, durable reflection labels, background
composition, and the prepared/stateless interfaces are specified in
[`multiphase.md`](multiphase.md). Each provider parameter is phase-prefixed, so
the same built-in or third-party model can be configured independently for
several phases without label collisions.

Typed monochromatic-neutron reuse and the deliberate exclusion of X-ray
doublets are documented in [`neutron-cw.md`](neutron-cw.md).

Neutron TOF calibration, asymmetric profiles, exact finite support, bin-center
coordinates, and the 15-row instrument Jacobian are documented in
[`tof-profile.md`](tof-profile.md). TOF uses the same `AccumulationResult`
contract as CW calculations rather than adding a refinement-specific data path.
