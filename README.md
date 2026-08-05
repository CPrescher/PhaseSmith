# Rietveld Engine

Rietveld Engine is an early-stage powder-diffraction computation library with a
Rust numerical core and a typed Python/NumPy API. The current implementation
includes symmetric TCH, CW U/V/W/X/Y broadening, FCJ asymmetry, wavelength
components, extensible sample physics, multi-phase CW X-ray/neutron and neutron
TOF calculation, plus a first-class scripted Le Bail workflow. Analytical
derivatives are computed during fused peak accumulation.

The architecture and roadmap are in [PROJECT_BRIEF.md](PROJECT_BRIEF.md); the
equations and parameter conventions are in [docs/equations.md](docs/equations.md).
The component-width TCH transform and chain-rule derivatives are documented in
[docs/tch-profile.md](docs/tch-profile.md); constant-wavelength U/V/W/X/Y
broadening is documented in [docs/cw-profile.md](docs/cw-profile.md). FCJ
geometry, quadrature, derivatives, and asymmetric support are documented in
[docs/fcj-profile.md](docs/fcj-profile.md). The script-first module boundaries,
including the planned first-class Le Bail and Dioptas integration layers, are
in [docs/public-api.md](docs/public-api.md).

Optional monochromatic and multi-wavelength radiation composition is documented
in [docs/wavelength-components.md](docs/wavelength-components.md).
The versioned physics-provider contract, isotropic size/microstrain equations,
and preferred-orientation convention are documented in
[docs/sample-physics.md](docs/sample-physics.md).
Typed multi-phase composition, scale derivatives, supplied backgrounds, and
prepared calculations are documented in [docs/multiphase.md](docs/multiphase.md).
Explicit monochromatic-neutron configuration and shared CW/FCJ behavior are
documented in [docs/neutron-cw.md](docs/neutron-cw.md).
Neutron TOF calibration, back-to-back exponential TCH profiles, derivatives,
and support semantics are documented in
[docs/tof-profile.md](docs/tof-profile.md).
Shared refinement contracts and Le Bail extraction are documented in
[docs/refinement.md](docs/refinement.md) and [docs/lebail.md](docs/lebail.md).
Versioned JSON+NPZ persistence and the optional Dioptas-facing NumPy boundary
are documented in [docs/persistence.md](docs/persistence.md) and
[docs/dioptas-integration.md](docs/dioptas-integration.md).

## Development

Requires Rust 1.85 or newer, Python 3.11 or newer, `uv`, and `maturin`.

```shell
uv venv
uv pip install -e '.[dev]'
maturin develop --uv
cargo test --workspace --all-features
uv run pytest
```

Run the Rust benchmarks with `cargo bench -p rietveld-core`. For a comparable
optimized Python-to-Rust measurement, build the release extension and require
release mode explicitly:

```shell
maturin develop --release --uv
uv run python benchmarks/profile.py --require-release
uv run python benchmarks/lebail.py --require-release
```

```python
import numpy as np
from rietveld import accumulate

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
from rietveld import ConstantWavelengthInstrument, accumulate_cw

instrument = ConstantWavelengthInstrument(
    wavelength_angstrom=1.54056,
    u_deg2=2e-4,
    v_deg2=-1e-4,
    w_deg2=1.2e-4,
    x_deg=1.5e-3,
    y_deg=3e-3,
)
cw = accumulate_cw(x, [24.0, 26.0], [100.0, 80.0], instrument)
print(cw.derivatives.local_parameter_names)   # intensity, position
print(cw.derivatives.global_parameter_names)  # U, V, W, X, Y
```

Axial divergence is a separate typed model and composes with CW broadening
without expanding reflections in Python:

```python
from rietveld import FcjGeometry, accumulate_cw_fcj

geometry = FcjGeometry(sample_over_radius=0.012, detector_over_radius=0.012)
asymmetric = accumulate_cw_fcj(
    x, [24.0, 26.0], [100.0, 80.0], instrument, geometry
)
print(asymmetric.derivatives.global_parameter_names)
# U, V, W, X, Y, sample_over_radius, detector_over_radius
```

Discrete radiation components are optional. The ordinary CW calls above are
monochromatic; a K-alpha doublet is an explicit model:

```python
from rietveld import WavelengthComponents, accumulate_cw_fcj_components

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
from rietveld import (
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
from rietveld import Phase, PowderPattern, ReflectionBatch, calculate_pattern

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
from rietveld import ConstantWavelengthExperiment, calculate_neutron_pattern

neutron = ConstantWavelengthExperiment.neutron(instrument)
neutron_result = calculate_neutron_pattern(PowderPattern(x), neutron, [alpha])
```

TOF reflections use d-spacing as their durable local coordinate. Values and
all local/shared derivatives are accumulated in one native call:

```python
from rietveld import TofInstrument, accumulate_tof

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
print(tof_result.derivatives.local_parameter_names)   # intensity, d_spacing
print(tof_result.derivatives.global_parameter_names)  # 15 instrument rows
```

A complete Le Bail extraction uses the same typed pattern, instrument, and
phase models and requires no project file or hand-written optimizer callback:

```python
from rietveld.refinement import lebail

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

GSAS-II is used only as the optional pinned validation oracle described in
[`oracle/README.md`](oracle/README.md).

## License

The project license has not yet been selected. Contributions should not be
accepted or releases published until maintainers add an explicit open-source
license. GSAS-II is separately licensed and is not redistributed here.
