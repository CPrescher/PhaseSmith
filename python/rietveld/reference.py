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
GAUSSIAN_FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Reference profile values and first derivatives."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_fwhm: NDArray[np.float64]
    d_eta: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceTchShape:
    """Reference TCH width transform and component-width derivatives."""

    total_fwhm: float
    eta: float
    d_total_fwhm_d_gaussian_fwhm: float
    d_total_fwhm_d_lorentzian_fwhm: float
    d_eta_d_gaussian_fwhm: float
    d_eta_d_lorentzian_fwhm: float


@dataclass(frozen=True, slots=True)
class ReferenceTchProfile:
    """Reference TCH profile values and direct-input derivatives."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_gaussian_fwhm: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]


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


def tch_shape_from_fwhm(
    gaussian_fwhm: float, lorentzian_fwhm: float
) -> ReferenceTchShape:
    """Evaluate the published TCH transform directly from component FWHMs."""

    width_scale = max(float(gaussian_fwhm), float(lorentzian_fwhm))
    gaussian = float(gaussian_fwhm) / width_scale
    lorentzian = float(lorentzian_fwhm) / width_scale
    polynomial = (
        gaussian**5
        + 2.69269 * gaussian**4 * lorentzian
        + 2.42843 * gaussian**3 * lorentzian**2
        + 4.47163 * gaussian**2 * lorentzian**3
        + 0.07842 * gaussian * lorentzian**4
        + lorentzian**5
    )
    normalized_total_fwhm = polynomial**0.2
    total_fwhm = width_scale * normalized_total_fwhm
    d_polynomial_d_gaussian = (
        5.0 * gaussian**4
        + 4.0 * 2.69269 * gaussian**3 * lorentzian
        + 3.0 * 2.42843 * gaussian**2 * lorentzian**2
        + 2.0 * 4.47163 * gaussian * lorentzian**3
        + 0.07842 * lorentzian**4
    )
    d_polynomial_d_lorentzian = (
        2.69269 * gaussian**4
        + 2.0 * 2.42843 * gaussian**3 * lorentzian
        + 3.0 * 4.47163 * gaussian**2 * lorentzian**2
        + 4.0 * 0.07842 * gaussian * lorentzian**3
        + 5.0 * lorentzian**4
    )
    derivative_scale = 1.0 / (5.0 * normalized_total_fwhm**4)
    d_total_d_gaussian = d_polynomial_d_gaussian * derivative_scale
    d_total_d_lorentzian = d_polynomial_d_lorentzian * derivative_scale

    ratio = lorentzian / normalized_total_fwhm
    eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3
    d_eta_d_ratio = 1.36603 - 2.0 * 0.47719 * ratio + 3.0 * 0.11116 * ratio**2
    d_ratio_d_gaussian = -ratio * d_total_d_gaussian / total_fwhm
    d_ratio_d_lorentzian = (1.0 - ratio * d_total_d_lorentzian) / total_fwhm
    return ReferenceTchShape(
        total_fwhm=total_fwhm,
        eta=eta,
        d_total_fwhm_d_gaussian_fwhm=d_total_d_gaussian,
        d_total_fwhm_d_lorentzian_fwhm=d_total_d_lorentzian,
        d_eta_d_gaussian_fwhm=d_eta_d_ratio * d_ratio_d_gaussian,
        d_eta_d_lorentzian_fwhm=d_eta_d_ratio * d_ratio_d_lorentzian,
    )


def tch_shape_from_gaussian_sigma(
    gaussian_sigma: float, lorentzian_fwhm: float
) -> ReferenceTchShape:
    """Evaluate TCH after an explicit Gaussian sigma-to-FWHM conversion."""

    return tch_shape_from_fwhm(
        GAUSSIAN_FWHM_PER_SIGMA * gaussian_sigma, lorentzian_fwhm
    )


def profile_tch(
    delta: ArrayLike, gaussian_fwhm: float, lorentzian_fwhm: float
) -> ReferenceTchProfile:
    """Compose the independent TCH transform with the primitive profile."""

    shape = tch_shape_from_fwhm(gaussian_fwhm, lorentzian_fwhm)
    primitive = profile(delta, shape.total_fwhm, shape.eta)
    return ReferenceTchProfile(
        value=primitive.value,
        d_delta=primitive.d_delta,
        d_gaussian_fwhm=(
            primitive.d_fwhm * shape.d_total_fwhm_d_gaussian_fwhm
            + primitive.d_eta * shape.d_eta_d_gaussian_fwhm
        ),
        d_lorentzian_fwhm=(
            primitive.d_fwhm * shape.d_total_fwhm_d_lorentzian_fwhm
            + primitive.d_eta * shape.d_eta_d_lorentzian_fwhm
        ),
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


def accumulate_tch(
    x: ArrayLike,
    positions: ArrayLike,
    intensities: ArrayLike,
    gaussian_fwhms: ArrayLike,
    lorentzian_fwhms: ArrayLike,
    *,
    support_fwhm: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Loop over TCH peaks as a transparent differential reference."""

    x = np.asarray(x, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    gaussian_fwhms = np.asarray(gaussian_fwhms, dtype=np.float64)
    lorentzian_fwhms = np.asarray(lorentzian_fwhms, dtype=np.float64)
    y = np.zeros_like(x)
    jacobian = np.zeros((positions.size, PARAMETER_COUNT, x.size), dtype=np.float64)
    for peak_index, (position, intensity, gaussian, lorentzian) in enumerate(
        zip(
            positions,
            intensities,
            gaussian_fwhms,
            lorentzian_fwhms,
            strict=True,
        )
    ):
        shape = tch_shape_from_fwhm(float(gaussian), float(lorentzian))
        active = np.abs(x - position) <= support_fwhm * shape.total_fwhm
        evaluated = profile_tch(x[active] - position, float(gaussian), float(lorentzian))
        y[active] += intensity * evaluated.value
        jacobian[peak_index, 0, active] = evaluated.value
        jacobian[peak_index, 1, active] = -intensity * evaluated.d_delta
        jacobian[peak_index, 2, active] = intensity * evaluated.d_gaussian_fwhm
        jacobian[peak_index, 3, active] = intensity * evaluated.d_lorentzian_fwhm
    return y, jacobian
