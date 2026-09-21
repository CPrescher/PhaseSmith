# Python API map

PhaseSmith exposes a typed, array-oriented scripting API. Top-level imports are
convenient for domain models, calculations, and mathematical primitives;
module-qualified imports are the durable paths for adapters, automation,
reporting, and refinement methods.

```python
import phasesmith
from phasesmith.instrument import ConstantWavelengthInstrument
from phasesmith.refinement.lebail import LeBailInput, LeBailOptions, refine
```

The package ships a `py.typed` marker. Editors and type checkers can therefore
use the annotations included in the installed distribution.

## Mathematical contract

The Python API and native Rust API expose the same production mathematical
models. The complete, shared reference collects the equations, units,
normalizations, finite-support rules, and analytical derivative chains:

| Python API area | Mathematical reference |
| --- | --- |
| `phasesmith`, `phasesmith.cw`, `phasesmith.fcj`, `phasesmith.tof` | [Peak profiles](mathematics/peak-profiles.md) |
| `phasesmith.crystallography`, `phasesmith.symmetry`, `phasesmith.scattering` | [Crystallography and scattering](mathematics/crystallography.md) |
| `phasesmith.calculation`, `phasesmith.structural_calculation`, `phasesmith.sample` | [Pattern composition](mathematics/pattern-composition.md) |
| `phasesmith.background`, `phasesmith.refinement.background` | [Backgrounds](mathematics/backgrounds.md) |
| `phasesmith.refinement`, `phasesmith.quantitative` | [Refinement and quantitative analysis](mathematics/refinement.md) |

Python docstrings link to these pages instead of copying long derivations that
could diverge from the native implementation.

The [pre-1.0 API decisions](api-stability.md) distinguish stable-direction
domain APIs, public mathematical primitives, and named compatibility adapters.

## Data import and domain models

| Module | Main public entry points | Purpose |
| --- | --- | --- |
| `phasesmith.io.powder` | `read_powder_data`, `PowderData`, `PowderReadLimits` | Bounded native readers for text and GSAS powder formats. |
| `phasesmith.io.cif` | `read_cif`, `CifReadResult`, `CifReadLimits` | Native CIF ingestion and explicit backend selection. |
| `phasesmith.io.space_groups` | `space_group_by_number`, `space_group_by_symbol` | Pure-Rust space-group lookup. |
| `phasesmith.pattern` | `PowderPattern`, calculation result types | Observed data, masks, backgrounds, and calculated results. |
| `phasesmith.structure` | `CrystalStructure`, `AtomSite`, `AnisotropicDisplacement` | Parser-independent structural records. |
| `phasesmith.phase` | `Phase`, `RietveldPhase`, reflection batch types | Phase metadata and reflection geometry. |
| `phasesmith.project` | `RietveldProject` | Multi-histogram project-level orchestration. |

See [powder-file and CIF import](cif-import.md),
[symmetry and reflection generation](symmetry-reflections.md), and the
[crystallography foundation](crystallography-foundation.md).

## Instruments, radiation, and profiles

| Module | Main public entry points | Purpose |
| --- | --- | --- |
| `phasesmith.instrument` | `ConstantWavelengthInstrument`, `FcjGeometry`, `TofInstrument`, `TofIncidentSpectrum` | Instrument response, geometry, and explicit TOF normalization. |
| `phasesmith.radiation` | `MonochromaticRadiation`, `WavelengthComponents`, experiment and geometry types | Explicit source and specimen geometry. |
| `phasesmith` / `phasesmith._api` | `profile`, `accumulate`, TCH variants | Low-level symmetric pseudo-Voigt kernels. |
| `phasesmith.cw` | `accumulate_cw`, `accumulate_cw_components`, `cw_profile_parameters` | Constant-wavelength U/V/W/X/Y accumulation. |
| `phasesmith.fcj` | `profile_fcj`, `accumulate_cw_fcj` | Finger–Cox–Jephcoat axial asymmetry. |
| `phasesmith.tof` | `profile_tof`, `accumulate_tof`, `tof_profile_parameters` | Neutron time-of-flight profiles. |
| `phasesmith.fpa_calibration` | `FundamentalEmissionLine`, `GaussianSpectralPassband`, `SollerAxialGeometry`, `BraggBrentanoFundamentalProfile`, `simulate_fundamental_peaks`, `calibrate_fundamental_profile` | Offline physical laboratory targets and compression into the production CW profile. |

The native `phasesmith-workflows` crate also exposes `TofLeBailInput`,
`TofLeBailPhase`, `TofLeBailOptions`, `TofChebyshevBackground`,
`calculate_tof_lebail_pattern`, and `refine_tof_lebail`. The optional background
has a microsecond domain and exposes its sample-major analytical coefficient
basis. These records are not interchangeable with the constant-wavelength
Python refinement facade.

CW `LeBailInput.with_refinable_background` and structural
`RietveldInput.background` use the same additive convention: the analytical
model is a residual above `PowderPattern.background`, not a replacement for
that fixed baseline. Final calculations and checkpoints retain the refined
model and combined background.

Units, parameter ordering, support rules, and derivative conventions are
defined in [equations and units](equations.md) and the model-specific pages in
the navigation.

## Crystallography and scattering

| Module | Main public entry points | Purpose |
| --- | --- | --- |
| `phasesmith.crystallography` | `UnitCell`, structure-factor values/JVP/VJP functions | Cell mathematics and structural intensities. |
| `phasesmith.symmetry` | `SpaceGroup`, `SymmetryOperation`, `PreparedReflectionGenerator` | Exact symmetry, absences, families, and bounded generation. |
| `phasesmith.scattering` | `XrayNonResonant`, `XrayFixedDispersion`, `NeutronNuclear` | Built-in prepared scattering models. |
| `phasesmith.intensity_corrections` | Neutral, Bragg–Brentano, and neutron correction models | Explicit integrated-intensity corrections. |
| `phasesmith.instrument` | `ConstantWavelengthInstrument`, `TofInstrument`, `TofBankGeometry`, `TofIncidentSpectrum` | Typed profile/calibration, fixed TOF bank geometry, and incident normalization. |
| `phasesmith.sample` | Size, microstrain, and March–Dollase models | Composable sample-physics contributions. |
| `phasesmith.extensions` | Provider protocols and descriptors | Versioned third-party physics boundary. |

## Calculations and execution

| Module | Main public entry points | Purpose |
| --- | --- | --- |
| `phasesmith.calculation` | `calculate_pattern`, `calculate_cw_pattern`, `PreparedPattern` | Reflection-list profile calculation. |
| `phasesmith.structural_calculation` | `calculate_structural_pattern`, `PreparedStructuralPattern` | Structure-to-pattern calculation with fused derivatives. |
| `phasesmith.execution` | `ExecutionPolicy` | Bounded serial or fixed-thread execution. |
| `phasesmith.control` | `CancellationToken`, `ProgressEvent`, `OperationCancelled` | Cancellation and progress callbacks. |
| `phasesmith.results` | `AccumulationResult`, `PatternDerivatives`, `SupportJacobian` | Shared numerical result containers. |

## Refinement and reporting

| Module | Main public entry points | Purpose |
| --- | --- | --- |
| `phasesmith.refinement` | Parameters, constraints, residuals, JVP/VJP helpers | Shared refinement infrastructure. |
| `phasesmith.refinement.lebail` | `LeBailInput`, `LeBailOptions`, `refine` | Intensity extraction and optional profile/lattice updates. |
| `phasesmith.refinement.tof_lebail` | `TofLeBailInput`, `TofLeBailOptions`, `refine_tof_lebail` | Fixed-cell, fixed-instrument TOF extraction. |
| `phasesmith.refinement.tof_multibank` | `TofMultiBankGeometryInput`, `TofMultiBankGeometryOptions`, `refine_tof_multibank_geometry` | Joint shared-cell and bank-local TOF instrument refinement with restart and identifiability diagnostics. |
| `phasesmith.refinement.tof_structural` | `StructuralTofMultiBankInput`, `StructuralTofMultiBankProvenance`, `StructuralTofRequestProvenance`, `StructuralTofSourceDigest`, `StructuralTofRefinementOptions`, `refine_structural_tof_multibank` | Structural neutron TOF refinement with shared structure, checksum-retaining one-/multi-bank file composition, explicit bank correction contracts, bounded runtime, and exact restart. |
| `phasesmith.refinement.readiness` | `review_rietveld_input`, `RietveldReadinessReport` | Non-mutating review of import provenance, active physical models, model mismatches, and risky parameter selections. |
| `phasesmith.refinement.rietveld` | Rietveld inputs, options, recipes, and results | Structural refinement orchestration. |
| `phasesmith.refinement.runtime` | Limits, events, logs, checkpoints | Bounded execution and recovery. |
| `phasesmith.quantitative` | `quantitative_phase_analysis`, `quantitative_phase_analysis_with_covariance`, `weight_fractions_from_scale` | Hill–Howard phase fractions and scale-covariance propagation. |
| `phasesmith.reporting` | JSON and CSV Rietveld writers | Stable external reports. |
| `phasesmith.persistence` | `save_bundle`, `load_bundle`, `PersistenceBundle` | Versioned Python workflow persistence. |
| `phasesmith.automation` | `WorkflowSpec`, `ProposerProvenance`, `plan_workflow`, `advisor_packet`, `lint_recipe_proposal`, `parse_recipe_proposal`, `run_workflow`, `review_workflow_output`, `prepare_review_packet`, `replan_workflow`, `resume_workflow`, `automation_schema` | Versioned inspect/plan/advise/lint/approve/run/review/replan boundary for human- or AI-orchestrated persisted projects. |

The `phasesmith` console script exposes the same boundary as finite JSON
commands. See [AI-guided automation](ai-automation.md) for the schemas,
scientific recipe rubric, explicit approval flow, and model-facing prompt.

## Inspect exact signatures

The installed release is the final source of truth for callable signatures and
docstrings:

```python
import inspect
import phasesmith

print(inspect.signature(phasesmith.accumulate_cw))
help(phasesmith.refinement.lebail.refine)
```

The [public Python architecture](public-api.md) defines compatibility and data
ownership in more detail. Source links in the page header lead to the exact
implementation for the selected documentation version.

Release review does not rely on documentation pages alone. The versioned
machine-readable snapshot in `api/python-public-api-v0.7.0.json` covers the
explicit top-level, I/O, refinement, integration, oracle, and validation
exports. Run `python scripts/public_api_snapshot.py --check` to compare it with
the installed package; any name, target, kind, or callable-signature change is
reported as a unified diff.

Version 0.6 removes method-owned refinement names, dataset-specific converters,
and adapter/orchestration aliases from aggregate `__all__` lists. Explicit 0.5
imports remain warning-backed aliases until 1.0. See [Migrating to
0.6](migration-0.6.md) for exact replacements.

## Pawley workflows

[CW Pawley refinement](pawley.md) supports monochromatic and fixed-spectrum
family areas; [TOF Pawley](tof-pawley.md) supports single banks and joint banks
with shared cells. Both provide bounded/tied parameters, analytical derivatives,
dense or matrix-free solving, cancellation and accepted-state restart.
Use `phasesmith.refinement.pawley` or `phasesmith.refinement.tof_pawley` in Python,
and `phasesmith::workflows` in Rust. Standalone CW version 2 and TOF version 1
codecs also integrate into lossless [native format-7 bundles](native-persistence.md).
See [validation](pawley-validation.md) for measured gates and model limitations.
