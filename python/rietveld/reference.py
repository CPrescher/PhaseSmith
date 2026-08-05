"""Independent NumPy reference for the first profile vertical slice.

This implementation favors transparent equations over speed and must not call
the Rust extension. It is the first differential-testing layer for native code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FOUR_LN_2 = 4.0 * np.log(2.0)
GAUSSIAN_NORMALIZATION = np.sqrt(FOUR_LN_2 / np.pi)
PARAMETER_COUNT = 4


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Reference profile values and first derivatives."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_fwhm: NDArray[np.float64]
    d_eta: NDArray[np.float64]


def profile(delta: ArrayLike, fwhm: float, eta: float) -> ReferenceProfile:
    """Evaluate the symmetric pseudo-Voigt directly from its component equations."""

    delta = np.asarray(delta, dtype=np.float64)
    inverse_fwhm = 1.0 / fwhm
    z_squared = np.square(delta * inverse_fwhm)

    gaussian = GAUSSIAN_NORMALIZATION * inverse_fwhm * np.exp(-FOUR_LN_2 * z_squared)
    denominator = 1.0 + 4.0 * z_squared
    lorentzian = 2.0 * inverse_fwhm / (np.pi * denominator)

    d_gaussian_delta = gaussian * (-2.0 * FOUR_LN_2 * delta * inverse_fwhm**2)
    d_lorentzian_delta = lorentzian * (-8.0 * delta * inverse_fwhm**2 / denominator)
    d_gaussian_fwhm = gaussian * inverse_fwhm * (-1.0 + 2.0 * FOUR_LN_2 * z_squared)
    d_lorentzian_fwhm = lorentzian * inverse_fwhm * (
        -1.0 + 8.0 * z_squared / denominator
    )

    return ReferenceProfile(
        value=eta * lorentzian + (1.0 - eta) * gaussian,
        d_delta=eta * d_lorentzian_delta + (1.0 - eta) * d_gaussian_delta,
        d_fwhm=eta * d_lorentzian_fwhm + (1.0 - eta) * d_gaussian_fwhm,
        d_eta=lorentzian - gaussian,
    )


def accumulate(
    x: ArrayLike,
    positions: ArrayLike,
    intensities: ArrayLike,
    fwhms: ArrayLike,
    etas: ArrayLike,
    *,
    support_fwhm: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Loop over peaks in Python as a clear differential reference."""

    x = np.asarray(x, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    fwhms = np.asarray(fwhms, dtype=np.float64)
    etas = np.asarray(etas, dtype=np.float64)
    y = np.zeros_like(x)
    jacobian = np.zeros((positions.size, PARAMETER_COUNT, x.size), dtype=np.float64)

    for peak_index, (position, intensity, fwhm, eta) in enumerate(
        zip(positions, intensities, fwhms, etas, strict=True)
    ):
        active = np.abs(x - position) <= support_fwhm * fwhm
        delta = x[active] - position
        evaluated = profile(delta, float(fwhm), float(eta))
        y[active] += intensity * evaluated.value
        jacobian[peak_index, 0, active] = evaluated.value
        jacobian[peak_index, 1, active] = -intensity * evaluated.d_delta
        jacobian[peak_index, 2, active] = intensity * evaluated.d_fwhm
        jacobian[peak_index, 3, active] = intensity * evaluated.d_eta
    return y, jacobian
