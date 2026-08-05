# Rietveld Engine

Rietveld Engine is an early-stage powder-diffraction computation library with a
Rust numerical core and a typed Python/NumPy API. The current vertical slice is
a finite-support symmetric pseudo-Voigt profile with analytical derivatives
computed during fused peak accumulation.

The architecture and roadmap are in [PROJECT_BRIEF.md](PROJECT_BRIEF.md); the
equations and parameter conventions are in [docs/equations.md](docs/equations.md).
The component-width TCH transform and chain-rule derivatives are documented in
[docs/tch-profile.md](docs/tch-profile.md).

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

GSAS-II is used only as the optional pinned validation oracle described in
[`oracle/README.md`](oracle/README.md).

## License

The project license has not yet been selected. Contributions should not be
accepted or releases published until maintainers add an explicit open-source
license. GSAS-II is separately licensed and is not redistributed here.
