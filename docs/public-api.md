# Public Python architecture and scripting contract

The Python package is a scientific library first. Every normal calculation and
refinement workflow must be expressible from a script without project files,
global state, callbacks per reflection, or GUI objects.

## Stable module boundaries

```text
rietveld.instrument
  Instrument and radiation models. No phase or refinement state.

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
```

Low-level profile functions remain available for equation testing and advanced
use. They do not become the only way to calculate a pattern.

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
