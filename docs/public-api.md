# Public Python architecture and scripting contract

The Python package is a scientific library first. Every normal calculation and
refinement workflow must be expressible from a script without project files,
global state, callbacks per reflection, or GUI objects.

## Stable module boundaries

```text
phasesmith.instrument
  Instrument response and instrument-geometry models. No phase, radiation
  spectrum, or refinement state.

phasesmith.radiation
  Optional discrete wavelength/source models. A single component is the exact
  monochromatic baseline; doublets and other spectra are explicit composition.

phasesmith.phase
  Crystallographic phase metadata and typed reflection batches. No observed
  pattern, optimizer, or instrument state.

phasesmith.crystallography  [general symmetry implemented]
  General cells, typed atom sites, complex structure factors, integrated
  intensities, and dense/JVP/VJP derivatives are implemented. The P1 API
  remains as a small caller-supplied-amplitude compatibility layer.

phasesmith.symmetry  [implemented]
  Exact symmetry operations and closure, special positions, systematic
  absences, metric parameterizations, and prepared d/Q/CW/TOF reflection
  generation. Numerical evaluation and family enumeration are Rust-owned.

phasesmith.scattering  [implemented]
  Typed X-ray and neutron scattering models, fixed X-ray dispersion offsets,
  and versioned batch-provider contracts. Built-in production models execute
  in Rust.

phasesmith.intensity_corrections  [implemented]
  Explicit neutral, unpolarized, and polarized monochromatic Bragg--Brentano
  integrated intensity corrections plus a batch-provider boundary.

phasesmith.io.cif  [implemented]
  Bounded native CIF parsing and pure-Rust space-group lookup into
  crystallography models. Optional parser objects never enter calculation,
  persistence, or refinement state.

phasesmith.io.powder  [implemented]
  Size-limited two/three-column, unpacked GSAS FXYE, and packed constant-step
  GSAS STD readers. The Python adapter returns immutable NumPy data plus source
  metadata through the shared Rust parser and performs no refinement. FXYE
  zero-ESD exclusions are exposed through `PowderData.mask`.

phasesmith.pattern
  Observed grids, intensities, uncertainties, masks, backgrounds, and
  calculated-pattern result containers.

phasesmith.background
  Plain-array background estimation and subtraction preprocessing. The native
  Smooth Bruckner estimator is compatible with a pinned xypattern revision;
  no GUI package is imported. Non-differentiable preprocessing is kept out of
  refinement parameter models.

phasesmith.calculation
  Stateless and prepared pattern calculators that compose instrument, phase,
  sample, and pattern inputs into native batch calls.

phasesmith.execution
  Immutable, bounded CPU execution policy. Serial execution is the embedding
  default; fixed and automatic budgets enable deterministic phase concurrency.

phasesmith.structural_calculation  [implemented]
  Stateless and prepared one-phase structural CW calculation. Built-in X-ray
  and neutron models expose fused values/JVP/VJP; custom providers retain a
  vectorized values fallback with one call per provider.

phasesmith.refinement
  Parameter selection, bounds, constraints, residuals, diagnostics, and
  optimizer adapters. Refinement methods are submodules, not mode flags.

phasesmith.refinement.lebail
  First-class Le Bail intensity extraction and diagnostics.

phasesmith.refinement.rietveld
  Implemented monochromatic structure-factor refinement orchestration. It
  consumes the same profile interface as Le Bail.

phasesmith.quantitative  [implemented]
  Hill--Howard conversion of compatible phase scales into labeled normalized
  crystalline weight fractions. This interpretation remains separate from the
  refinement solver.

phasesmith.validation  [implemented]
  Explicit checksum-pinned external dataset retrieval and reproducible
  real-data workflows. Downloads never occur at import time.

phasesmith.integrations.dioptas
  Compatibility-only conversion between Dioptas-facing NumPy data and the
  public typed models. It is not a forward roadmap target, and Dioptas is never
  a core or required Python dependency.

phasesmith.extensions
  Versioned provider protocols and a reserved future entry-point discovery
  convenience. Providers are passed explicitly into calculations; discovery
  never creates numerical core global state.

phasesmith.sample
  Built-in size, microstrain, and preferred-orientation providers. These obey
  the same public protocol as third-party implementations.
```

Low-level profile functions remain available for equation testing and advanced
use. They do not become the only way to calculate a pattern.

The implemented low-level module split already follows this boundary:

```text
phasesmith.instrument.ConstantWavelengthInstrument
phasesmith.instrument.FcjGeometry
phasesmith.instrument.TofInstrument
phasesmith.radiation.WavelengthComponents
phasesmith.radiation.RadiationProbe
phasesmith.radiation.MonochromaticRadiation
phasesmith.radiation.ConstantWavelengthExperiment
phasesmith.radiation.BraggBrentanoGeometry
phasesmith.radiation.DebyeScherrerGeometry
phasesmith.phase.ReflectionGeometryBatch
phasesmith.phase.ReciprocalMetric
phasesmith.phase.ReflectionBatch
phasesmith.phase.Phase
phasesmith.phase.StructuralReflectionBatch
phasesmith.phase.RietveldPhase
phasesmith.crystallography.UnitCell
phasesmith.crystallography.AtomSiteBatch
phasesmith.crystallography.calculate_p1_structure_factors
phasesmith.crystallography.p1_jacobian_vector_product
phasesmith.crystallography.p1_intensity_transpose_jacobian_vector_product
phasesmith.crystallography.StructureFactorValuesResult
phasesmith.crystallography.StructureFactorResult
phasesmith.crystallography.calculate_structure_factor_values
phasesmith.crystallography.calculate_structure_factors
phasesmith.symmetry.SymmetryOperation
phasesmith.symmetry.SpaceGroup
phasesmith.symmetry.PreparedReflectionGenerator
phasesmith.symmetry.DSpacingRange
phasesmith.symmetry.ScatteringVectorRange
phasesmith.symmetry.CwTwoThetaRange
phasesmith.symmetry.TofRange
phasesmith.structure.CrystalStructure
phasesmith.structure.AtomSite
phasesmith.structure.AnisotropicDisplacement
phasesmith.structure.structure_to_record
phasesmith.structure.structure_from_record
phasesmith.io.cif.read_cif
phasesmith.io.powder.read_powder_data
phasesmith.scattering.ScatteringSpecies
phasesmith.scattering.ScatteringContext
phasesmith.scattering.ScatteringFactorBatch
phasesmith.scattering.ScatteringFactorProvider
phasesmith.scattering.XrayNonResonant
phasesmith.scattering.XrayFixedDispersion
phasesmith.scattering.NeutronNuclear
phasesmith.scattering.species_from_structure
phasesmith.intensity_corrections.IntegratedIntensityCorrection
phasesmith.intensity_corrections.NeutralIntegratedIntensityCorrection
phasesmith.intensity_corrections.BraggBrentanoUnpolarizedLp
phasesmith.intensity_corrections.BraggBrentanoPolarizedLp
phasesmith.intensity_corrections.ConstantWavelengthNeutronLorentz
phasesmith.execution.ExecutionPolicy
phasesmith.refinement.lebail.LeBailPhase
phasesmith.refinement.LatticeParameterization
phasesmith.refinement.LatticeParameterBounds
phasesmith.refinement.CwLatticeReflectionDomain
phasesmith.refinement.cw_lattice_geometry
phasesmith.refinement.tof_lattice_geometry
phasesmith.pattern.PowderPattern
phasesmith.background.smooth_bruckner
phasesmith.background.SmoothBrucknerBackground
phasesmith.background.BackgroundSubtractionResult
phasesmith.quantitative.weight_fractions_from_scale
phasesmith.quantitative.quantitative_phase_analysis
phasesmith.pattern.PatternCalculationResult
phasesmith.pattern.StructuralReflectionResult
phasesmith.pattern.StructuralPatternCalculationResult
phasesmith.pattern.StructuralPatternJvpResult
phasesmith.pattern.StructuralPatternVjpResult
phasesmith.extensions.PhysicsContribution
phasesmith.extensions.ReflectionPhysicsProvider
phasesmith.extensions.CompositePhysicsProvider
phasesmith.sample.IsotropicSizeBroadening
phasesmith.sample.IsotropicMicrostrainBroadening
phasesmith.sample.MarchDollasePreferredOrientation
phasesmith.sample.reciprocal_angle_geometry
phasesmith.cw.cw_profile_parameters
phasesmith.cw.accumulate_cw
phasesmith.cw.accumulate_cw_components
phasesmith.cw.accumulate_cw_contributions
phasesmith.calculation.calculate_cw_pattern
phasesmith.calculation.calculate_pattern
phasesmith.calculation.PreparedPattern
phasesmith.calculation.calculate_monochromatic_pattern
phasesmith.calculation.calculate_neutron_pattern
phasesmith.calculation.calculate_neutron_fcj_pattern
phasesmith.structural_calculation.calculate_structural_pattern
phasesmith.structural_calculation.PreparedStructuralPattern
phasesmith.fcj.profile_fcj
phasesmith.fcj.accumulate_cw_fcj
phasesmith.fcj.accumulate_cw_fcj_components
phasesmith.tof.tof_profile_parameters
phasesmith.tof.profile_tof
phasesmith.tof.accumulate_tof
phasesmith.results.AccumulationResult
phasesmith.refinement.ParameterKey
phasesmith.refinement.ParameterSpec
phasesmith.refinement.ParameterSet
phasesmith.refinement.ParameterChange
phasesmith.refinement.ConstraintTransform
phasesmith.refinement.evaluate_residuals
phasesmith.refinement.jacobian_vector_product
phasesmith.refinement.transpose_jacobian_vector_product
phasesmith.CancellationToken
phasesmith.TerminalCancellationController
phasesmith.refinement.RefinementEvent
phasesmith.refinement.RefinementLimits
phasesmith.refinement.RefinementRuntime
phasesmith.refinement.ConsoleRefinementLogger
phasesmith.refinement.JsonLinesRefinementLogger
phasesmith.refinement.lebail.LeBailInput
phasesmith.refinement.lebail.LeBailInput.from_cif
phasesmith.refinement.lebail.LeBailOptions
phasesmith.refinement.lebail.LeBailResult
phasesmith.refinement.lebail.extract_intensities
phasesmith.refinement.lebail.iterate_once
phasesmith.refinement.lebail.refine
phasesmith.persistence.PersistenceBundle
phasesmith.persistence.PersistenceBundle.to_lebail_input
phasesmith.persistence.save_bundle
phasesmith.persistence.load_bundle
phasesmith.integrations.dioptas.DioptasPatternData
phasesmith.integrations.dioptas.DioptasDisplayResult
phasesmith.integrations.dioptas.calculate
phasesmith.integrations.dioptas.refine_lebail
```

The scripting checkpoint API above remains Python-facing while refinement is
migrated. Rust applications use the separate `phasesmith-persistence` crate to
save and load `ProjectRecord` and produce stable summary reports without
CPython. See [native project persistence](native-persistence.md).

Top-level imports are convenience aliases for scripts and notebooks; the
module-qualified paths above are the ownership boundary. FCJ geometry does not
contain CW coefficients. A `ConstantWavelengthExperiment` may own optional
`axial_geometry`, allowing structural calculation, wavelength components, and
sample-physics providers to compose through one native batch call.
`ReflectionGeometryBatch` owns plain `hkl`, d-spacing,
position, and base-intensity arrays. `calculate_cw_pattern` accepts it directly
and calls an optional provider exactly once before one native fused
accumulation. The two axial ratios and their analytical derivatives remain
explicit global rows; refinement may keep them fixed without hiding them.

## Data and ownership rules

- Public domain models are frozen dataclasses or similarly inspectable typed
  objects. Their numerical payloads are NumPy arrays.
- Array fields have documented shape, units, ordering, and parameter names.
- Calculation inputs never own optimizer state. Refinement state never enters
  `phasesmith-core`.
- Stable string or integer IDs connect phases and reflections to derivative
  rows and diagnostics. Array position alone is not a durable external ID.
- Models support explicit plain-data serialization. Native extension objects,
  borrowed Rust views, GSAS-II objects, and GUI objects are not serialized.
- Convenience constructors perform unit conversion explicitly; the canonical
  stored units remain the documented public physical units.
- Prepared calculators may cache validated contiguous arrays, but a stateless
  one-call API remains available for notebooks, tests, and external programs.

Scattering providers receive one immutable `(reflection, species)` context and
return complex amplitudes plus analytical `df/ds` matrices in one call. The
provider descriptor fixes its probe, amplitude unit, implementation version,
and provider-API version. Built-in X-ray and neutron models resolve exact table
identities during `prepare()` and perform no label lookup in the batch hot
loop. Third-party research providers use the same explicit Python protocol;
they are passed as objects and are not discovered through process-global state.

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

`CalculationOptions`, `LeBailOptions`, and `RietveldOptions` each own the same
immutable `ExecutionPolicy`. Its budget belongs to the complete public
operation, including prepared calculators and refinement backtracks; callers do
not configure module-global pools. A fused generic pattern is currently one
Python-visible task and therefore remains serial at this layer until its native
kernel receives the remaining budget.

An external program should need only to provide contiguous `x`, observed
intensity, optional uncertainty/mask arrays, and typed instrument/phase data.
It receives NumPy calculated arrays and plain diagnostic records. An adapter
must not require that the host application adopt internal Rust types.

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

## Rietveld contract

`phasesmith.refinement.rietveld` is a separate method module rather than a flag
on Le Bail. `RietveldInput.from_cif` creates a typed structural phase, scattering
model, guarded reflection domain, and selected parameter set. `refine` uses
matrix-free structural JVP/VJP products and returns the last accepted phases,
parameters, calculation, residual metrics, iteration history, covariance/rank
diagnostics, termination reason, and restart checkpoint.

`RietveldRecipe` and `RietveldStage` form a separate optional orchestration
layer. `intelligent_rietveld_recipe` proposes a deterministic cumulative plan
from only the families authorized by the input selection and records its
rationale; `run_rietveld_recipe` executes explicit stages and stops when a
stage does not meet its declared termination policy. The solver never invokes
the planner implicitly.

The initial families are CW U/V/W/X/Y coefficients, constant zero shift,
Bragg--Brentano sample height, Debye--Scherrer X/Y specimen displacement,
normalized polynomial background coefficients, phase scale,
symmetry-independent lattice parameters,
symmetry-allowed fractional coordinates, occupancy, and isotropic `U_iso`.
Multiple phases and monochromatic X-ray or neutron experiments share the same
interface. Details and a complete script are in [`rietveld.md`](rietveld.md).

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
entry-point discovery is a convenience in `phasesmith.extensions`, not an
implicit registry used by `phasesmith-core`. Persistence records provider ID,
provider version, API version, and plain-data configuration; it never pickles
live provider or native objects. Refinement methods depend on the calculator
contract, so a compatible provider is automatically usable by Le Bail and
later Rietveld workflows.

Provider API version 1 returns additive Gaussian variance, additive Lorentzian
FWHM, a multiplicative intensity correction, position chains, stable parameter
names, and parameter-major derivative chains. The precise equations and array
layout remain unchanged by execution policy. `ProviderDescriptor.thread_safe`
is an explicit concurrency capability and defaults to `False`; PhaseSmith never
infers safety from a provider's implementation language or apparent
statelessness. A provider may set it to `True` only when overlapping calls on
the same instance are supported. Composite providers are parallel-safe only
when every child is parallel-safe, and unknown custom scattering or correction
providers likewise keep their structural fallback serialized. The precise
equations and array semantics are documented in
[`sample-physics.md`](sample-physics.md). Width
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
