# Getting started

PhaseSmith provides a compiled Python package for Python 3.11 and newer. The
numerical implementation is native Rust; using the Python package does not
require a separate Rust installation.

## Install from PyPI

Create an isolated environment and install the released wheel:

```shell
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install phasesmith
```

Confirm which release and installation are active:

```shell
python -c "import phasesmith; print(phasesmith.__version__, phasesmith.__file__)"
```

Optional dependencies are explicit:

```shell
python -m pip install "phasesmith[cif]"         # Gemmi-backed CIF conveniences
python -m pip install "phasesmith[refinement]"  # SciPy optimizer adapter
```

The native CIF and powder readers, calculation kernels, and built-in
refinement workflows do not require these optional packages.

## First profile calculation

```python
import numpy as np
from phasesmith import accumulate

x = np.linspace(20.0, 30.0, 10_001)
result = accumulate(
    x,
    positions=[24.0, 26.0],
    intensities=[100.0, 80.0],
    fwhms=[0.08, 0.10],
    etas=[0.30, 0.50],
)

print(result.y.shape)
print(result.derivatives.local_parameter_names)
```

`result.y` is the calculated profile. Analytical derivative storage remains
sparse until dense materialization is explicitly requested:

```python
dense = result.derivatives.local.to_dense(result.y.size)
print(dense.shape)  # (peak, parameter, sample)
```

## Choose the next workflow

- Start from measured powder data and a structure with
  [powder-file and CIF import](cif-import.md).
- Estimate a non-refinable preprocessing envelope with
  [background subtraction](background-subtraction.md).
- Extract reflection intensities with [Le Bail](lebail.md).
- Refine a structural model with [Rietveld refinement](rietveld.md).
- Use a persisted project from a coding agent or language model through the
  constrained [AI-guided automation](ai-automation.md) boundary.
- Save scripted workflow state with [Python persistence](persistence.md), or
  use the Rust-only project boundary described in
  [native persistence](native-persistence.md).

The [Python API map](api-reference.md) lists the stable module ownership and
the main public entry points. Native Rust applications should start with the
[Rust API](rust-api.md).

## Compatibility expectations

PhaseSmith is currently pre-1.0. Public behavior is typed and tested, but minor
releases may still revise APIs when scientific conventions or ownership
boundaries need correction. Release notes and versioned Read the Docs pages
are the authoritative contract for a published version.
