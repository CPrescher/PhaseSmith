# Python API map

PhaseSmith exposes a typed, array-oriented scripting API. Top-level imports are
convenient for notebooks and short scripts; module-qualified imports make
ownership clearer in applications and reusable packages.

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
| `phasesmith.instrument` | `ConstantWavelengthInstrument`, `FcjGeometry`, `TofInstrument` | Instrument response and geometry. |
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
| `phasesmith.instrument` | `ConstantWavelengthInstrument`, `TofInstrument`, `TofBankGeometry` | Typed profile/calibration and fixed TOF bank geometry. |
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
| `phasesmith.refinement.tof_structural` | `StructuralTofMultiBankInput`, `StructuralTofRefinementOptions`, `refine_structural_tof_multibank` | Structural neutron TOF refinement with shared structure, explicit bank correction contracts, bounded runtime, and exact restart. |
| `phasesmith.refinement.rietveld` | Rietveld inputs, options, recipes, and results | Structural refinement orchestration. |
| `phasesmith.refinement.runtime` | Limits, events, logs, checkpoints | Bounded execution and recovery. |
| `phasesmith.quantitative` | `quantitative_phase_analysis`, `quantitative_phase_analysis_with_covariance`, `weight_fractions_from_scale` | Hill–Howard phase fractions and scale-covariance propagation. |
| `phasesmith.reporting` | JSON and CSV Rietveld writers | Stable external reports. |
| `phasesmith.persistence` | `save_bundle`, `load_bundle`, `PersistenceBundle` | Versioned Python workflow persistence. |

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
