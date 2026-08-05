# Rietveld Engine

Rietveld Engine is an early-stage powder-diffraction computation library with a
Rust numerical core and a typed Python/NumPy API. The current profile layer
includes finite-support symmetric TCH pseudo-Voigt, constant-wavelength
U/V/W/X/Y broadening, and FCJ axial asymmetry with analytical derivatives
computed during fused peak accumulation.

The architecture and roadmap are in [PROJECT_BRIEF.md](PROJECT_BRIEF.md); the
equations and parameter conventions are in [docs/equations.md](docs/equations.md).
The component-width TCH transform and chain-rule derivatives are documented in
[docs/tch-profile.md](docs/tch-profile.md); constant-wavelength U/V/W/X/Y
broadening is documented in [docs/cw-profile.md](docs/cw-profile.md). FCJ
geometry, quadrature, derivatives, and asymmetric support are documented in
[docs/fcj-profile.md](docs/fcj-profile.md). The script-first module boundaries,
including the planned first-class Le Bail and Dioptas integration layers, are
in [docs/public-api.md](docs/public-api.md).

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
    wavelength_angstrom=1.5406,
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

GSAS-II is used only as the optional pinned validation oracle described in
[`oracle/README.md`](oracle/README.md).

## License

The project license has not yet been selected. Contributions should not be
accepted or releases published until maintainers add an explicit open-source
license. GSAS-II is separately licensed and is not redistributed here.
