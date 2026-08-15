"""Public, typed Python API over the native numerical kernel.

Profile equations and derivative conventions:
https://phasesmith.readthedocs.io/en/latest/mathematics/peak-profiles/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from .results import AccumulationResult, _build_accumulation_result

PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.PARAMETER_ORDER)
TCH_PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.TCH_PARAMETER_ORDER)
GAUSSIAN_FWHM_PER_SIGMA: Final[float] = float(np.sqrt(8.0 * np.log(2.0)))


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """A unit-area profile and derivatives at a vector of offsets."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_fwhm: NDArray[np.float64]
    d_eta: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TchFwhmShape:
    """TCH transform with derivatives with respect to component FWHMs."""

    total_fwhm: float
    eta: float
    d_total_fwhm_d_gaussian_fwhm: float
    d_total_fwhm_d_lorentzian_fwhm: float
    d_eta_d_gaussian_fwhm: float
    d_eta_d_lorentzian_fwhm: float


@dataclass(frozen=True, slots=True)
class TchSigmaShape:
    """TCH transform with derivatives with respect to Gaussian sigma and Lorentzian FWHM."""

    total_fwhm: float
    eta: float
    d_total_fwhm_d_gaussian_sigma: float
    d_total_fwhm_d_lorentzian_fwhm: float
    d_eta_d_gaussian_sigma: float
    d_eta_d_lorentzian_fwhm: float


@dataclass(frozen=True, slots=True)
class TchProfileResult:
    """TCH profile and derivatives with respect to its direct FWHM inputs."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_gaussian_fwhm: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TchSigmaProfileResult:
    """TCH profile with a Gaussian-sigma derivative."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_gaussian_sigma: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have a real floating-point or integer dtype")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def profile(delta: ArrayLike, fwhm: float, eta: float) -> ProfileResult:
    """Evaluate the low-level normalized ``H, eta`` primitive and derivatives."""

    arrays = _core.profile(_vector(delta, "delta"), float(fwhm), float(eta))
    return ProfileResult(*arrays)


def tch_shape_from_fwhm(gaussian_fwhm: float, lorentzian_fwhm: float) -> TchFwhmShape:
    """Transform explicitly named Gaussian and Lorentzian component FWHMs."""

    return TchFwhmShape(*_core.tch_shape_from_fwhm(float(gaussian_fwhm), float(lorentzian_fwhm)))


def _gaussian_fwhm_from_sigma(gaussian_sigma: float) -> float:
    sigma = float(gaussian_sigma)
    if not np.isfinite(sigma):
        raise ValueError("Gaussian sigma must be finite")
    if sigma < 0.0:
        raise ValueError("Gaussian sigma must be non-negative")
    return GAUSSIAN_FWHM_PER_SIGMA * sigma


def tch_shape_from_gaussian_sigma(gaussian_sigma: float, lorentzian_fwhm: float) -> TchSigmaShape:
    """Transform Gaussian standard deviation and Lorentzian FWHM."""

    shape = tch_shape_from_fwhm(_gaussian_fwhm_from_sigma(gaussian_sigma), lorentzian_fwhm)
    return TchSigmaShape(
        total_fwhm=shape.total_fwhm,
        eta=shape.eta,
        d_total_fwhm_d_gaussian_sigma=(
            shape.d_total_fwhm_d_gaussian_fwhm * GAUSSIAN_FWHM_PER_SIGMA
        ),
        d_total_fwhm_d_lorentzian_fwhm=shape.d_total_fwhm_d_lorentzian_fwhm,
        d_eta_d_gaussian_sigma=(shape.d_eta_d_gaussian_fwhm * GAUSSIAN_FWHM_PER_SIGMA),
        d_eta_d_lorentzian_fwhm=shape.d_eta_d_lorentzian_fwhm,
    )


def profile_tch(delta: ArrayLike, gaussian_fwhm: float, lorentzian_fwhm: float) -> TchProfileResult:
    """Evaluate TCH using explicitly named component FWHMs."""

    arrays = _core.profile_tch(
        _vector(delta, "delta"), float(gaussian_fwhm), float(lorentzian_fwhm)
    )
    return TchProfileResult(*arrays)


def profile_tch_from_gaussian_sigma(
    delta: ArrayLike, gaussian_sigma: float, lorentzian_fwhm: float
) -> TchSigmaProfileResult:
    """Evaluate TCH with derivatives with respect to Gaussian sigma."""

    result = profile_tch(delta, _gaussian_fwhm_from_sigma(gaussian_sigma), lorentzian_fwhm)
    return TchSigmaProfileResult(
        value=result.value,
        d_delta=result.d_delta,
        d_gaussian_sigma=result.d_gaussian_fwhm * GAUSSIAN_FWHM_PER_SIGMA,
        d_lorentzian_fwhm=result.d_lorentzian_fwhm,
    )


def accumulate(
    x: ArrayLike,
    positions: ArrayLike,
    intensities: ArrayLike,
    fwhms: ArrayLike,
    etas: ArrayLike,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Accumulate low-level ``H, eta`` peaks and derivatives in one native pass.

    Samples at exactly ``support_fwhm * fwhm`` from a peak center are included.
    Derivatives hold that active sample set fixed.
    """

    if jacobian_layout not in ("support", "dense"):
        raise ValueError("jacobian_layout must be 'support' or 'dense'")

    y, starts, offsets, values, global_values = _core.accumulate(
        _vector(x, "x"),
        _vector(positions, "positions"),
        _vector(intensities, "intensities"),
        _vector(fwhms, "fwhms"),
        _vector(etas, "etas"),
        float(support_fwhm),
    )
    return _build_accumulation_result(
        y,
        starts,
        offsets,
        values,
        global_values,
        PARAMETER_ORDER,
        (),
        jacobian_layout,
    )


def accumulate_tch(
    x: ArrayLike,
    positions: ArrayLike,
    intensities: ArrayLike,
    gaussian_fwhms: ArrayLike,
    lorentzian_fwhms: ArrayLike,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Accumulate TCH peaks and component-FWHM derivatives in one native pass."""

    if jacobian_layout not in ("support", "dense"):
        raise ValueError("jacobian_layout must be 'support' or 'dense'")
    y, starts, offsets, values, global_values = _core.accumulate_tch(
        _vector(x, "x"),
        _vector(positions, "positions"),
        _vector(intensities, "intensities"),
        _vector(gaussian_fwhms, "gaussian_fwhms"),
        _vector(lorentzian_fwhms, "lorentzian_fwhms"),
        float(support_fwhm),
    )
    return _build_accumulation_result(
        y,
        starts,
        offsets,
        values,
        global_values,
        TCH_PARAMETER_ORDER,
        (),
        jacobian_layout,
    )
