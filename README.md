# PhaseSmith

PhaseSmith is an early-stage powder-diffraction computation library with a
Rust numerical core and a typed Python/NumPy API. The current implementation
includes symmetric TCH, CW U/V/W/X/Y broadening, FCJ asymmetry, wavelength
components, extensible sample physics, multi-phase CW X-ray/neutron and neutron
TOF calculation, plus a first-class scripted Le Bail workflow. Analytical
derivatives are computed during fused peak accumulation. The crystallography
foundation includes native general-cell mathematics, P1 complex structure
factors, exact symmetry, bounded reflection generation, and prepared X-ray and
neutron scattering factors with analytical derivatives. General-symmetry
structural intensities and monochromatic structural patterns now have a fused
native values/JVP/VJP path, direct fixed CIF anisotropic displacement, and a
scriptable `RietveldPhase` API.
CIF-backed Le Bail can refine setting-aware lattice parameters with analytical
derivatives and guarded, stable-ID reflection-domain regeneration.
The first full CIF-backed Rietveld workflow now refines CW profile/background,
phase scale, lattice, symmetry-allowed coordinates, occupancy, and isotropic
displacement through matrix-free Rust JVP/VJP products with safe checkpoints
and structured logs.
Model-independent preprocessing now includes a native Smooth Bruckner
background implementation compatible with pinned xypattern/Dioptas behavior,
plus optional Chebyshev compression and plain NumPy subtraction results.

Use the [documentation index](docs/index.md) to follow the shortest path from
powder data and a CIF to background subtraction, Le Bail extraction, Rietveld
refinement, reports, and persistence. The architecture and non-negotiable
numerical rules are in [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

## Development

Requires Rust 1.85 or newer, Python 3.11 or newer, `uv`, and `maturin`.

```shell
uv venv
uv pip install -e '.[dev]'
maturin develop --uv
cargo test --workspace --all-features
uv run pytest
```

Run the Rust benchmarks with `cargo bench -p phasesmith-core`. For a comparable
optimized Python-to-Rust measurement, build the release extension and require
release mode explicitly:

```shell
maturin develop --release --uv
uv run python benchmarks/profile.py --require-release
uv run python benchmarks/lebail.py --require-release
uv run python benchmarks/scattering.py --require-release
uv run python benchmarks/structural_pattern.py --require-release
uv run python benchmarks/lattice_refinement.py --require-release
uv run python benchmarks/background.py --require-release
```

The pinned external-oracle environment can compare the same support-limited CW
profile-and-derivative workload against GSAS-II. A second benchmark starts from
the crystal structure and compares structure factors, integrated intensities,
and the composed structural CW pattern. Numerical agreement is checked before
timings are reported:

```shell
uv run python benchmarks/compare_gsasii.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory

uv run python benchmarks/compare_gsasii_structural.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory
```

The exact scope and interpretation are documented in
[docs/gsasii-performance.md](docs/gsasii-performance.md). This is a kernel-level
comparison, not a claim about complete refinement workflow speed.

Background estimation is an explicit preprocessing step:

```python
import phasesmith

subtracted = phasesmith.SmoothBrucknerBackground(
    smooth_width=0.1,
    iterations=50,
    chebyshev_order=50,
).subtract(two_theta, measured)

pattern = phasesmith.PowderPattern(
    two_theta,
    observed_y=measured,
    background=subtracted.background,
)
```

Set `chebyshev_order=None` to use the raw Bruckner envelope. Neither xypattern
nor Dioptas is required at runtime.

```python
import numpy as np
from phasesmith import accumulate

x = np.linspace(20.0, 30.0, 10_001)
result = accumulate(
    x,
    positions=[24.0, 26.0],
    intensities=[100.0, 80.0],
    fwhms=[0.08, 0.1],
    etas=[0.3, 0.5],
)
print(result.y.shape, result.derivatives.local.values.shape)

# Dense materialization is explicit and intended for small compatibility uses.
dense = result.derivatives.local.to_dense(result.y.size)
print(dense.shape)  # (peak, parameter, sample)
```

Use `profile_tch` or `accumulate_tch` when Gaussian and Lorentzian component
FWHMs are the direct inputs. Explicit `tch_shape_from_gaussian_sigma` and
`profile_tch_from_gaussian_sigma` helpers are provided when Gaussian width is a
standard deviation; width conventions are never inferred.

An entire constant-wavelength reflection list, including all local and shared
instrument derivatives, is also one array-oriented call:

```python
from phasesmith import ConstantWavelengthInstrument, accumulate_cw

instrument = ConstantWavelengthInstrument(
    wavelength_angstrom=1.54056,
    u_deg2=2e-4,
    v_deg2=-1e-4,
    w_deg2=1.2e-4,
    x_deg=1.5e-3,
    y_deg=3e-3,
)
cw = accumulate_cw(x, [24.0, 26.0], [100.0, 80.0], instrument)
print(cw.derivatives.local_parameter_names)  # intensity, position
print(cw.derivatives.global_parameter_names)  # U, V, W, X, Y
```

Atomic scattering models are also script-first and can be prepared once for a
structure. X-ray species select exact neutral/ionic table states, while neutron
species retain natural/isotope identity:

```python
from phasesmith import ScatteringSpecies, XrayNonResonant

species = (
    ScatteringSpecies("Si"),
    ScatteringSpecies("O"),
    ScatteringSpecies("Fe", charge=3),
)
scattering = XrayNonResonant().prepare(species).evaluate([0.0, 0.5, 1.0])
print(scattering.amplitudes.shape)  # (reflection, site)
print(scattering.d_amplitudes_d_s.shape)  # analytical df/ds, same shape
```

A typed structure can be evaluated without assembling scattering arrays or
looping over atoms/reflections in Python. Correction geometry is explicit; the
default neutral model returns raw multiplicity-weighted structural intensity:

```python
from phasesmith import XrayNonResonant, calculate_structure_factor_values
from phasesmith.io.cif import read_cif

structure = read_cif("phase.cif").structure
structural = calculate_structure_factor_values(
    structure,
    hkl=[[1, 0, 0], [1, 1, 0]],
    multiplicity=[6, 12],
    scattering=XrayNonResonant(),
    scale=1.0,
)
print(structural.f, structural.integrated_intensity)
```

Use `calculate_structure_factors` when the bounded dense analytical structural
Jacobian is also required.

Axial divergence is a separate typed model and composes with CW broadening
without expanding reflections in Python:

```python
from phasesmith import FcjGeometry, accumulate_cw_fcj

geometry = FcjGeometry(sample_over_radius=0.012, detector_over_radius=0.012)
asymmetric = accumulate_cw_fcj(x, [24.0, 26.0], [100.0, 80.0], instrument, geometry)
print(asymmetric.derivatives.global_parameter_names)
# U, V, W, X, Y, sample_over_radius, detector_over_radius
```

Discrete radiation components are optional. The ordinary CW calls above are
monochromatic; a K-alpha doublet is an explicit model:

```python
from phasesmith import WavelengthComponents, accumulate_cw_fcj_components

radiation = WavelengthComponents.doublet(
    reference_wavelength_angstrom=1.54056,
    secondary_wavelength_angstrom=1.54439,
    secondary_to_reference_intensity=0.5,
)
doublet = accumulate_cw_fcj_components(
    x, [24.0, 26.0], [100.0, 80.0], instrument, radiation, geometry
)
```

Sample physics is explicit and provider-based. Built-in and third-party models
return the same vectorized contribution schema, and Python is never called from
the native peak/sample loop:

```python
from phasesmith import (
    CompositePhysicsProvider,
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    ReflectionGeometryBatch,
    calculate_cw_pattern,
)

reflections = ReflectionGeometryBatch(
    hkl=[[1, 0, 0], [1, 1, 0]],
    d_spacing_angstrom=[3.72, 2.64],
    two_theta_deg=[24.0, 34.0],
    base_integrated_intensity=[100.0, 80.0],
)
sample = CompositePhysicsProvider(
    (
        IsotropicSizeBroadening(crystallite_size_nm=50.0),
        IsotropicMicrostrainBroadening(rms_microstrain=5e-4),
    )
)
calculated = calculate_cw_pattern(x, reflections, instrument, physics=sample)
print(calculated.derivatives.global_parameter_names)
```

The high-level script interface assigns durable IDs and calculates all phases
through one flattened native call:

```python
from phasesmith import Phase, PowderPattern, ReflectionBatch, calculate_pattern

alpha_reflections = ReflectionBatch(
    reflection_ids=["alpha-100", "alpha-110"],
    hkl=[[1, 0, 0], [1, 1, 0]],
    d_spacing_angstrom=[3.72, 2.64],
    two_theta_deg=[24.0, 34.0],
    integrated_intensity=[100.0, 80.0],
)
alpha = Phase(
    phase_id="alpha",
    name="Alpha phase",
    reflections=alpha_reflections,
    scale=1.0,
    physics=sample,
)
pattern_result = calculate_pattern(PowderPattern(x), instrument, [alpha])
print(pattern_result.reflection_keys)
print(pattern_result.derivatives.global_parameter_names)
```

Neutron CW is an explicit monochromatic probe configuration and cannot receive
an X-ray K-alpha doublet:

```python
from phasesmith import ConstantWavelengthExperiment, calculate_neutron_pattern

neutron = ConstantWavelengthExperiment.neutron(instrument)
neutron_result = calculate_neutron_pattern(PowderPattern(x), neutron, [alpha])
```

TOF reflections use d-spacing as their durable local coordinate. Values and
all local/shared derivatives are accumulated in one native call:

```python
from phasesmith import TofInstrument, accumulate_tof

tof_instrument = TofInstrument(
    zero_us=-0.773,
    difc_us_per_angstrom=5084.83,
    difa_us_per_angstrom2=-2.63,
    difb_us_angstrom=0.0,
    alpha_coefficient=5.0,
    beta0_per_us=0.0333,
    beta1_angstrom4_per_us=0.000964,
    betaq_angstrom2_per_us=0.0,
    sigma0_us2=0.0,
    sigma1_us2_per_angstrom2=15.14,
    sigma2_us2_per_angstrom4=0.0,
    sigmaq_us2_per_angstrom=0.0,
    x_us_per_angstrom=0.0,
    y_us_per_angstrom2=0.0,
    z_us=0.0,
)
tof_x = np.linspace(2_000.0, 20_000.0, 6_001)  # bin centers, microseconds
tof_result = accumulate_tof(tof_x, [0.8, 1.5], [100.0, 80.0], tof_instrument)
print(tof_result.derivatives.local_parameter_names)  # intensity, d_spacing
print(tof_result.derivatives.global_parameter_names)  # 15 instrument rows
```

A complete Le Bail extraction uses the same typed pattern, instrument, and
phase models and requires no project file or hand-written optimizer callback:

```python
from phasesmith.refinement import lebail

observed = PowderPattern(
    x,
    observed_y=measured_y,
    background=background_y,
    uncertainty=sigma_y,
)
result = lebail.refine(lebail.LeBailInput(observed, instrument, (alpha,)))
print(result.termination_reason, result.metrics.rwp)
print([(item.reflection_id, item.integrated_intensity) for item in result.intensities])
```

For a CIF-backed single phase, the convenience constructor creates the
generated reflection domain and the symmetry-allowed bounded lattice parameter
set directly from the observed grid:

```python
request = lebail.LeBailInput.from_cif(
    observed,
    instrument,
    "phase.cif",
    phase_id="alpha",
)
result = lebail.refine(request)
print(result.phases[0].structure.cell)
```

A monochromatic structural refinement is likewise constructed directly from a
CIF. Parameter families are explicit and no GSAS-II installation is involved:

```python
from phasesmith.refinement import rietveld

request = rietveld.RietveldInput.from_cif(
    observed,
    experiment,
    "phase.cif",
    phase_id="alpha",
    selection=rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=True,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    ),
)
result = rietveld.refine(request)
print(result.termination_reason, result.metrics.rwp)
print(result.phases[0].structure.cell)
```

Multiphase calculations and Rietveld refinement can use a bounded worker pool.
The default is one thread so embedding applications retain control of CPU use;
choose a fixed budget for predictable GUI/script behavior, or `None` to use the
available logical CPUs:

```python
import phasesmith
from phasesmith.refinement import rietveld

options = rietveld.RietveldOptions(
    execution=phasesmith.ExecutionPolicy(threads=2),
)
result = rietveld.refine(request, options)
```

Independent phases execute concurrently, while their values and analytical
derivatives are combined in the original phase order.

For long-running scripts and application integration, the small project facade
keeps restart state, cooperative stop control, persistence, and plain reports
together while leaving `RietveldInput` fully accessible:

```python
from dataclasses import replace

from phasesmith import BraggBrentanoGeometry, RietveldParameterSelection, RietveldProject

experiment = replace(
    experiment,
    zero_shift_deg=0.0,
    geometry=BraggBrentanoGeometry(240.0, sample_displacement_mm=0.0),
)
selection = RietveldParameterSelection(
    phase_scale=True,
    lattice=True,
    sample_physics=True,
    instrument_parameters=("zero_shift_deg", "sample_displacement_mm"),
    background=True,
)
request = replace(request, experiment=experiment, selection=selection)
project = RietveldProject(request)
result = project.refine(logger=my_event_logger)
project.write_reports(json_path="result.json", csv_path="pattern.csv")
project.save("run-state")

# Another thread or a callback can stop safely; the accepted state is resumable.
project.stop("user_requested")
```

Refinable backgrounds share one analytical interface. Built-ins include power
and Chebyshev series, fixed-knot linear interpolation, broad normalized
Gaussian amorphous components, and ordered composites. Smooth Bruckner remains
an explicit preprocessing operation and is never inserted into refinement
automatically.

GSAS-II is used only as the optional pinned validation oracle described in
[`oracle/README.md`](oracle/README.md).

## License

PhaseSmith is licensed under the [MIT License](LICENSE). GSAS-II is
separately licensed, is used only as an optional external validation oracle,
and is not redistributed here.
