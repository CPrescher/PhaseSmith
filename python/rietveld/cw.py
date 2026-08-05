"""Array-oriented constant-wavelength profile calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import _vector
from .instrument import ConstantWavelengthInstrument
from .radiation import WavelengthComponents, component_global_parameter_names
from .results import AccumulationResult, _build_accumulation_result

CW_LOCAL_PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.CW_LOCAL_PARAMETER_ORDER)
CW_GLOBAL_PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.CW_GLOBAL_PARAMETER_ORDER)


@dataclass(frozen=True, slots=True)
class CwProfileParameters:
    """Reflection-dependent component widths and analytical derivatives."""

    gaussian_variance_deg2: NDArray[np.float64]
    gaussian_fwhm_deg: NDArray[np.float64]
    lorentzian_fwhm_deg: NDArray[np.float64]
    total_fwhm_deg: NDArray[np.float64]
    eta: NDArray[np.float64]
    d_gaussian_fwhm_d_instrument: NDArray[np.float64]
    d_lorentzian_fwhm_d_instrument: NDArray[np.float64]
    d_component_fwhm_d_two_theta: NDArray[np.float64]


def cw_profile_parameters(
    two_theta_deg: ArrayLike,
    instrument: ConstantWavelengthInstrument,
) -> CwProfileParameters:
    """Derive widths and derivatives for a reflection-position array."""

    arrays = _core.cw_profile_parameters(
        _vector(two_theta_deg, "two_theta_deg"),
        instrument.wavelength_angstrom,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
    )
    return CwProfileParameters(*arrays)


def accumulate_cw(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    integrated_intensities: ArrayLike,
    instrument: ConstantWavelengthInstrument,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Accumulate a complete CW reflection batch in one native call.

    Local derivative columns are reflection intensity and position. Dense
    global rows are U, V, W, X, and Y. Samples on support boundaries are
    included, and derivatives hold that active set fixed.
    """

    if jacobian_layout not in ("support", "dense"):
        raise ValueError("jacobian_layout must be 'support' or 'dense'")
    y, starts, offsets, local_values, global_values = _core.accumulate_cw(
        _vector(x, "x"),
        _vector(two_theta_deg, "two_theta_deg"),
        _vector(integrated_intensities, "integrated_intensities"),
        instrument.wavelength_angstrom,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
        float(support_fwhm),
    )
    return _build_accumulation_result(
        y,
        starts,
        offsets,
        local_values,
        global_values,
        CW_LOCAL_PARAMETER_ORDER,
        CW_GLOBAL_PARAMETER_ORDER,
        jacobian_layout,
    )


def accumulate_cw_components(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    integrated_intensities: ArrayLike,
    instrument: ConstantWavelengthInstrument,
    components: WavelengthComponents,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Accumulate optional discrete wavelengths without FCJ asymmetry.

    A one-component model is exactly the monochromatic :func:`accumulate_cw`
    path. Secondary wavelength/intensity ratios add shared Jacobian rows while
    every crystallographic reflection retains one intensity/position block.
    """

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
        False,
        0.0,
        0.0,
        float(support_fwhm),
    )
    return _build_accumulation_result(
        y,
        starts,
        offsets,
        local_values,
        global_values,
        CW_LOCAL_PARAMETER_ORDER,
        component_global_parameter_names(components, include_fcj=False),
        jacobian_layout,
    )
