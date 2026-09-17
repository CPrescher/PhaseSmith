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
        raise RuntimeError(f"private GSAS-II probe requires {PINNED_REVISION}, detected {revision}")

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
        raise RuntimeError(f"private GSAS-II probe requires {PINNED_REVISION}, detected {revision}")
    x = np.ascontiguousarray(x, dtype=np.float64)
    sigma_centidegrees_squared = (100.0 * gaussian_sigma) ** 2
    gamma_centidegrees = 100.0 * lorentzian_fwhm
    values, _integral = gsas_pwd_module.getPsVoigt(
        float(position), sigma_centidegrees_squared, gamma_centidegrees, x
    )
    return 100.0 * np.asarray(values, dtype=np.float64)


def initialize_pawley(
    phase: Any,
    histogram: Any,
    gsas_module: ModuleType,
    reflections: ArrayLike,
    observed: ArrayLike,
    *,
    initial_fraction: float = 0.5,
    starting_cell: ArrayLike | None = None,
) -> None:
    """Initialize documented Pawley rows on the exact pinned external oracle.

    The public scripting API supplies reflection records but has no Pawley-table
    initializer at this revision. This adapter writes records only; it neither
    evaluates profiles nor implements or copies GSAS-II's optimization method.
    Row convention: h, k, l, multiplicity, d, refine, F-squared, sigma(F-squared).
    """
    revision = detected_revision(gsas_module)
    if revision != PINNED_REVISION:
        raise RuntimeError(f"private GSAS-II probe requires {PINNED_REVISION}, detected {revision}")
    if any(np.iscomplexobj(v) for v in (reflections, observed, starting_cell) if v is not None):
        raise TypeError("Pawley oracle arrays must be real-valued")
    refs = np.asarray(reflections, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    if (
        refs.ndim != 2
        or refs.shape[0] == 0
        or refs.shape[1] != 15
        or not np.isfinite(refs).all()
        or np.any(refs[:, :4] != np.rint(refs[:, :4]))
        or np.any(refs[:, 3:5] <= 0)
        or y.ndim != 1
        or not np.isfinite(y).all()
        or not np.isfinite(initial_fraction)
        or initial_fraction <= 0
    ):
        raise ValueError("invalid pinned CW Pawley reflection/observation schema")
    general = phase.data.get("General")
    if not isinstance(general, dict) or not isinstance(phase.data.get("Pawley ref"), list):
        raise RuntimeError("unexpected pinned GSAS-II phase schema")
    values = histogram.data["data"][1]
    if len(values) < 3 or any(np.asarray(v).shape != y.shape for v in values[:3]):
        raise RuntimeError("unexpected pinned GSAS-II histogram schema")
    cell = None
    if starting_cell is not None:
        cell = np.asarray(starting_cell, dtype=np.float64)
        if (
            cell.shape != (6,)
            or not np.isfinite(cell).all()
            or np.any(cell[:3] <= 0)
            or np.any((cell[3:] <= 0) | (cell[3:] >= 180))
        ):
            raise ValueError("invalid starting cell")
        cosines = np.cos(np.deg2rad(cell[3:]))
        volume_factor = 1 + 2 * np.prod(cosines) - np.sum(cosines**2)
        if volume_factor <= 0 or len(general["Cell"]) != 8:
            raise ValueError("invalid starting metric or pinned cell schema")
    # All validation precedes writes to the isolated oracle project.
    general["doPawley"] = True
    general["Pawley dmin"] = float(np.min(refs[:, 4])) * 0.99
    general["Pawley dmax"] = float(np.max(refs[:, 4])) * 1.01
    general["Pawley neg wt"] = 0.0
    phase.data["Pawley ref"] = [
        [
            int(r[0]),
            int(r[1]),
            int(r[2]),
            int(r[3]),
            float(r[4]),
            True,
            float(r[9]) * initial_fraction,
            0.0,
        ]
        for r in refs
    ]
    if cell is not None:
        general["Cell"][1:7] = cell.tolist()
        general["Cell"][7] = float(np.prod(cell[:3]) * np.sqrt(volume_factor))
    values[1][:] = y
    values[2][:] = 1.0
