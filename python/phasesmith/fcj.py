"""Finger-Cox-Jephcoat axial-asymmetry calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import _vector
from .instrument import ConstantWavelengthInstrument, FcjGeometry
from .radiation import WavelengthComponents, component_global_parameter_names
from .results import AccumulationResult, _build_accumulation_result

FCJ_PARAMETER_ORDER: Final[tuple[str, ...]] = (
    "position",
    "gaussian_fwhm",
    "lorentzian_fwhm",
    "sample_over_radius",
    "detector_over_radius",
)
CW_FCJ_LOCAL_PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.CW_LOCAL_PARAMETER_ORDER)
CW_FCJ_GLOBAL_PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.CW_FCJ_GLOBAL_PARAMETER_ORDER)


@dataclass(frozen=True, slots=True)
class FcjProfileResult:
    """FCJ-convolved profile values and direct analytical derivatives."""

    value: NDArray[np.float64]
    d_position: NDArray[np.float64]
    d_gaussian_fwhm: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]
    d_sample_over_radius: NDArray[np.float64]
    d_detector_over_radius: NDArray[np.float64]


def profile_fcj(
    x_deg: ArrayLike,
    position_deg: float,
    gaussian_fwhm_deg: float,
    lorentzian_fwhm_deg: float,
    geometry: FcjGeometry,
) -> FcjProfileResult:
    """Evaluate one normalized FCJ-convolved TCH profile."""

    arrays = _core.profile_fcj(
        _vector(x_deg, "x_deg"),
        float(position_deg),
        float(gaussian_fwhm_deg),
        float(lorentzian_fwhm_deg),
        geometry.sample_over_radius,
        geometry.detector_over_radius,
    )
    return FcjProfileResult(*arrays)


def accumulate_cw_fcj(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    integrated_intensities: ArrayLike,
    instrument: ConstantWavelengthInstrument,
    geometry: FcjGeometry,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Accumulate an FCJ-asymmetric CW reflection batch in one native call.

    Local derivative columns belong to each logical reflection and are
    integrated intensity and ideal position. Shared rows are U, V, W, X, Y,
    sample/radius, and detector/radius. The position derivative includes both
    FCJ translation/geometry and reflection-dependent width changes.
    """

    if jacobian_layout not in ("support", "dense"):
        raise ValueError("jacobian_layout must be 'support' or 'dense'")
    y, starts, offsets, local_values, global_values = _core.accumulate_cw_fcj(
        _vector(x, "x"),
        _vector(two_theta_deg, "two_theta_deg"),
        _vector(integrated_intensities, "integrated_intensities"),
        instrument.wavelength_angstrom,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
        geometry.sample_over_radius,
        geometry.detector_over_radius,
        float(support_fwhm),
    )
    return _build_accumulation_result(
        y,
        starts,
        offsets,
        local_values,
        global_values,
        CW_FCJ_LOCAL_PARAMETER_ORDER,
        CW_FCJ_GLOBAL_PARAMETER_ORDER,
        jacobian_layout,
    )


def accumulate_cw_fcj_components(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    integrated_intensities: ArrayLike,
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponents,
    geometry: FcjGeometry,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Accumulate discrete CW wavelengths with FCJ asymmetry in one call."""

    if jacobian_layout not in ("support", "dense"):
        raise ValueError("jacobian_layout must be 'support' or 'dense'")
    y, starts, offsets, local_values, global_values = _core.accumulate_cw_components(
        _vector(x, "x"),
        _vector(two_theta_deg, "two_theta_deg"),
        _vector(integrated_intensities, "integrated_intensities"),
        components.wavelengths_angstrom,
        components.relative_intensities,
        instrument.wavelength_angstrom,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
        True,
        geometry.sample_over_radius,
        geometry.detector_over_radius,
        float(support_fwhm),
    )
    return _build_accumulation_result(
        y,
        starts,
        offsets,
        local_values,
        global_values,
        CW_FCJ_LOCAL_PARAMETER_ORDER,
        component_global_parameter_names(components, include_fcj=True),
        jacobian_layout,
    )
