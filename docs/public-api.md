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
  Structure-factor refinement orchestration after the required physics layer
  exists. It consumes the same calculation interface as Le Bail.

rietveld.integrations.dioptas
  Optional conversion between Dioptas-facing NumPy data and the public typed
  models. Dioptas is never a core or required Python dependency.

rietveld.extensions
  Versioned provider protocols and optional third-party discovery. Providers
  are passed explicitly into calculations; discovery never creates numerical
  core global state.
```

Low-level profile functions remain available for equation testing and advanced
use. They do not become the only way to calculate a pattern.

The implemented low-level module split already follows this boundary:

```text
rietveld.instrument.ConstantWavelengthInstrument
rietveld.instrument.FcjGeometry
rietveld.radiation.WavelengthComponents
rietveld.cw.cw_profile_parameters
rietveld.cw.accumulate_cw
rietveld.cw.accumulate_cw_components
rietveld.fcj.profile_fcj
rietveld.fcj.accumulate_cw_fcj
rietveld.fcj.accumulate_cw_fcj_components
rietveld.results.AccumulationResult
```

Top-level imports are convenience aliases for scripts and notebooks; the
module-qualified paths above are the ownership boundary. FCJ geometry does not
contain CW coefficients, and the FCJ module composes the two models through a
single native batch call. Future phase and refinement layers consume this
calculation surface rather than moving their state into either model.

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

The simplest supported script must not require a user to construct a parameter
vector or optimizer callback manually.

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
