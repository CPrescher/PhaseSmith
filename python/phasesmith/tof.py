"""Time-of-flight profile parameters and fused accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import _vector
from .execution import ExecutionPolicy
from .instrument import TofInstrument
from .results import AccumulationResult, _build_accumulation_result

TOF_LOCAL_PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.TOF_LOCAL_PARAMETER_ORDER)
TOF_GLOBAL_PARAMETER_ORDER: Final[tuple[str, ...]] = tuple(_core.TOF_GLOBAL_PARAMETER_ORDER)


@dataclass(frozen=True, slots=True)
class TofProfileResult:
    """TOF profile values and direct analytical derivatives."""

    value: NDArray[np.float64]
    d_position: NDArray[np.float64]
    d_alpha: NDArray[np.float64]
    d_beta: NDArray[np.float64]
    d_gaussian_fwhm: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TofProfileParameters:
    """Derived TOF calibration, rates, component widths, and TCH shape."""

    position_us: NDArray[np.float64]
    alpha_per_us: NDArray[np.float64]
    beta_per_us: NDArray[np.float64]
    gaussian_variance_us2: NDArray[np.float64]
    gaussian_fwhm_us: NDArray[np.float64]
    lorentzian_fwhm_us: NDArray[np.float64]
    total_fwhm_us: NDArray[np.float64]
    eta: NDArray[np.float64]


def profile_tof(
    x_us: ArrayLike,
    position_us: float,
    alpha_per_us: float,
    beta_per_us: float,
    gaussian_fwhm_us: float,
    lorentzian_fwhm_us: float,
    *,
    tail_log: float = 20.0,
) -> TofProfileResult:
    """Evaluate the normalized truncated double-exponential TCH convolution."""

    arrays = _core.profile_tof(
        _vector(x_us, "x_us"),
        float(position_us),
        float(alpha_per_us),
        float(beta_per_us),
        float(gaussian_fwhm_us),
        float(lorentzian_fwhm_us),
        float(tail_log),
    )
    return TofProfileResult(*arrays)


def tof_profile_parameters(
    d_spacing_angstrom: ArrayLike,
    instrument: TofInstrument,
) -> TofProfileParameters:
    """Derive TOF calibration and profile parameters for a reflection batch."""

    return TofProfileParameters(
        *_core.tof_profile_parameters(
            _vector(d_spacing_angstrom, "d_spacing_angstrom"), *instrument.as_tuple()
        )
    )


def accumulate_tof(
    x_us: ArrayLike,
    d_spacing_angstrom: ArrayLike,
    integrated_intensities: ArrayLike,
    instrument: TofInstrument,
    *,
    support_fwhm: float = 20.0,
    tail_log: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
    execution: ExecutionPolicy | None = None,
) -> AccumulationResult:
    """Accumulate TOF reflections in one native finite-support pass.

    Input coordinates are bin centers in microseconds. Local derivative columns
    are integrated intensity and d-spacing. Dense shared rows follow
    :data:`TOF_GLOBAL_PARAMETER_ORDER`.
    """

    if jacobian_layout not in ("support", "dense"):
        raise ValueError("jacobian_layout must be 'support' or 'dense'")
    selected_execution = ExecutionPolicy() if execution is None else execution
    if not isinstance(selected_execution, ExecutionPolicy):
        raise TypeError("execution must be an ExecutionPolicy")
    arrays = _core.accumulate_tof(
        _vector(x_us, "x_us"),
        _vector(d_spacing_angstrom, "d_spacing_angstrom"),
        _vector(integrated_intensities, "integrated_intensities"),
        *instrument.as_tuple(),
        float(support_fwhm),
        float(tail_log),
        selected_execution.resolved_budget(),
    )
    return _build_accumulation_result(
        *arrays,
        TOF_LOCAL_PARAMETER_ORDER,
        TOF_GLOBAL_PARAMETER_ORDER,
        jacobian_layout,
    )
