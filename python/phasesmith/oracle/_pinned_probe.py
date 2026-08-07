"""Narrow, revision-gated access to GSAS-II internals.

Use this only when the public scripting adapter cannot expose a value needed for
an oracle comparison. Every supported probe name is explicit so GSAS-II's data
tree cannot leak into the rest of the project.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

PINNED_REVISION: Final = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"

_HISTOGRAM_PATHS: Final[dict[str, tuple[str, ...]]] = {
    "instrument_parameters": ("Instrument Parameters",),
    "sample_parameters": ("Sample Parameters",),
    "limits": ("Limits",),
}


def detected_revision(gsas_module: ModuleType) -> str:
    """Read the exact Git revision containing an imported GSAS-II module."""

    module_path = Path(gsas_module.__file__).resolve()
    repository = next(
        (parent for parent in module_path.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError("private probes require a Git checkout of the pinned GSAS-II revision")
    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def probe_histogram(histogram: Any, gsas_module: ModuleType, *names: str) -> dict[str, Any]:
    """Copy a fixed set of internal histogram values after checking the pin."""

    revision = detected_revision(gsas_module)
    if revision != PINNED_REVISION:
        raise RuntimeError(
            f"private GSAS-II probe requires {PINNED_REVISION}, detected {revision}"
        )

    unsupported = set(names).difference(_HISTOGRAM_PATHS)
    if unsupported:
        choices = ", ".join(sorted(_HISTOGRAM_PATHS))
        raise KeyError(f"unsupported probe(s) {sorted(unsupported)!r}; supported: {choices}")

    extracted: dict[str, Any] = {}
    for name in names:
        value: Any = histogram.data
        for key in _HISTOGRAM_PATHS[name]:
            value = value[key]
        extracted[name] = copy.deepcopy(value)
    return extracted


def probe_symmetric_profile(
    gsas_pwd_module: ModuleType,
    x: ArrayLike,
    *,
    position: float,
    gaussian_sigma: float,
    lorentzian_fwhm: float,
) -> NDArray[np.float64]:
    """Evaluate GSAS-II's private simple pseudo-Voigt in physical degree units.

    GSAS-II accepts Gaussian variance in centidegree squared, Lorentzian FWHM
    in centidegrees, and returns density per centidegree. This adapter converts
    both parameters and output to degrees.
    """

    revision = detected_revision(gsas_pwd_module)
    if revision != PINNED_REVISION:
        raise RuntimeError(
            f"private GSAS-II probe requires {PINNED_REVISION}, detected {revision}"
        )
    x = np.ascontiguousarray(x, dtype=np.float64)
    sigma_centidegrees_squared = (100.0 * gaussian_sigma) ** 2
    gamma_centidegrees = 100.0 * lorentzian_fwhm
    values, _integral = gsas_pwd_module.getPsVoigt(
        float(position), sigma_centidegrees_squared, gamma_centidegrees, x
    )
    return 100.0 * np.asarray(values, dtype=np.float64)
