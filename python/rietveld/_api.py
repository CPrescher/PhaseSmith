"""Public, typed Python API over the native numerical kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.PARAMETER_ORDER)


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """A unit-area profile and derivatives at a vector of offsets."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_fwhm: NDArray[np.float64]
    d_eta: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AccumulationResult:
    """Calculated pattern and per-peak analytical Jacobian.

    ``jacobian`` has shape ``(peak, parameter, sample)`` and uses
    :data:`PARAMETER_ORDER`.
    """

    y: NDArray[np.float64]
    jacobian: NDArray[np.float64]


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def profile(delta: ArrayLike, fwhm: float, eta: float) -> ProfileResult:
    """Evaluate the normalized symmetric pseudo-Voigt and its derivatives."""

    arrays = _core.profile(_vector(delta, "delta"), float(fwhm), float(eta))
    return ProfileResult(*arrays)


def accumulate(
    x: ArrayLike,
    positions: ArrayLike,
    intensities: ArrayLike,
    fwhms: ArrayLike,
    etas: ArrayLike,
    *,
    support_fwhm: float = 20.0,
) -> AccumulationResult:
    """Accumulate finite-support peaks and their derivatives in one native pass.

    Samples at exactly ``support_fwhm * fwhm`` from a peak center are included.
    Derivatives hold that active sample set fixed.
    """

    y, jacobian = _core.accumulate(
        _vector(x, "x"),
        _vector(positions, "positions"),
        _vector(intensities, "intensities"),
        _vector(fwhms, "fwhms"),
        _vector(etas, "etas"),
        float(support_fwhm),
    )
    return AccumulationResult(y=y, jacobian=jacobian)
