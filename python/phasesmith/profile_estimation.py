"""Effective CW starting-profile estimation from one dominant phase.

This workflow uses the native Le Bail engine for width fitting.  Optional
lattice alignment uses the existing bounded CIF-backed Le Bail phase first;
wavelength is never selected in either step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from . import _core
from .instrument import ConstantWavelengthInstrument
from .phase import ReflectionBatch
from .refinement.lebail import (
    LeBailInput,
    LeBailOptions,
    LeBailPhase,
    build_parameter_set,
    refine,
)

_GAUSSIAN_FWHM_PER_SIGMA = 2.354_820_045_030_949_3


class ProfileEstimationMode(StrEnum):
    """Requested complexity for effective-profile estimation."""

    W_ONLY = "w_only"
    UVW = "uvw"
    UVWXY = "uvwxy"
    AUTOMATIC = "automatic"


@dataclass(frozen=True, slots=True)
class ProfileEstimationOptions:
    """Controls for staged profile estimation and conservative model selection."""

    mode: ProfileEstimationMode = ProfileEstimationMode.AUTOMATIC
    align_lattice: bool = False
    minimum_relative_rwp_improvement: float = 0.002
    minimum_absolute_rwp_improvement: float = 1.0e-5
    maximum_absolute_correlation: float = 0.98
    lebail: LeBailOptions = field(default_factory=LeBailOptions)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ProfileEstimationMode):
            raise TypeError("mode must be ProfileEstimationMode")
        if not isinstance(self.align_lattice, bool):
            raise TypeError("align_lattice must be boolean")
        if not np.isfinite(self.minimum_relative_rwp_improvement) or (
            self.minimum_relative_rwp_improvement < 0.0
        ):
            raise ValueError("minimum_relative_rwp_improvement must be non-negative and finite")
        if not np.isfinite(self.minimum_absolute_rwp_improvement) or (
            self.minimum_absolute_rwp_improvement < 0.0
        ):
            raise ValueError("minimum_absolute_rwp_improvement must be non-negative and finite")
        if not np.isfinite(self.maximum_absolute_correlation) or not (
            0.0 <= self.maximum_absolute_correlation < 1.0
        ):
            raise ValueError("maximum_absolute_correlation must lie in [0, 1)")
        if not isinstance(self.lebail, LeBailOptions):
            raise TypeError("lebail must be LeBailOptions")


@dataclass(frozen=True, slots=True)
class ProfileEstimationStage:
    """Audit record for one attempted lattice/profile stage."""

    kind: str
    instrument_parameters: tuple[str, ...]
    accepted: bool
    decision: str
    rwp: float
    maximum_absolute_correlation: float | None


@dataclass(frozen=True, slots=True)
class ProfileEstimationResult:
    """Effective starting profile plus diagnostics and fitted reflection intensities."""

    instrument: ConstantWavelengthInstrument
    active_parameters: tuple[str, ...]
    phase: LeBailPhase
    integrated_intensity: NDArray[np.float64]
    calculated_y: NDArray[np.float64]
    rp: float
    rwp: float
    chi_square: float
    termination_reason: str
    stages: tuple[ProfileEstimationStage, ...]
    warnings: tuple[str, ...]


def starting_profile_from_fwhm(
    wavelength_angstrom: float,
    fwhm_deg: float,
) -> ConstantWavelengthInstrument:
    """Return a valid all-Gaussian optimizer seed from an approximate FWHM."""

    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0.0:
        raise ValueError("wavelength_angstrom must be positive and finite")
    if not np.isfinite(fwhm_deg) or fwhm_deg <= 0.0:
        raise ValueError("fwhm_deg must be positive and finite")
    return ConstantWavelengthInstrument(
        wavelength_angstrom=float(wavelength_angstrom),
        u_deg2=0.0,
        v_deg2=0.0,
        w_deg2=float((fwhm_deg / _GAUSSIAN_FWHM_PER_SIGMA) ** 2),
        x_deg=0.0,
        y_deg=0.0,
    )


def estimate_effective_profile(
    input_data: LeBailInput,
    options: ProfileEstimationOptions | None = None,
) -> ProfileEstimationResult:
    """Estimate a reusable starting profile from a predominantly single-phase pattern.

    Construct ``input_data`` with :meth:`LeBailInput.from_cif`.  A wavelength
    already calibrated by Dioptas/PONI belongs in its instrument.  Use the
    pattern mask to exclude known diamond, gasket, impurity, or detector-artifact
    regions.  The returned coefficients are effective widths, not guaranteed
    instrument-only resolution parameters.
    """

    if not isinstance(input_data, LeBailInput):
        raise TypeError("input_data must be LeBailInput")
    selected = ProfileEstimationOptions() if options is None else options
    if not isinstance(selected, ProfileEstimationOptions):
        raise TypeError("options must be ProfileEstimationOptions")
    if len(input_data.phases) != 1 or not isinstance(input_data.phases[0], LeBailPhase):
        raise ValueError("profile estimation requires exactly one CIF-backed LeBailPhase")

    phase = input_data.phases[0]
    instrument = input_data.instrument
    prefix_stages: list[ProfileEstimationStage] = []
    if selected.align_lattice:
        if phase.reflection_domain is None:
            raise ValueError(
                "lattice alignment requires refine_lattice=True in LeBailInput.from_cif"
            )
        parameters = build_parameter_set(
            instrument,
            (phase,),
            lattice_parameters=True,
        )
        aligned = refine(
            LeBailInput(
                input_data.pattern,
                instrument,
                (phase,),
                parameters,
                input_data.constraints,
            ),
            selected.lebail,
        )
        if aligned.instrument.wavelength_angstrom.hex() != instrument.wavelength_angstrom.hex():
            raise RuntimeError("fixed wavelength changed during lattice alignment")
        phase = aligned.phases[0]
        if not isinstance(phase, LeBailPhase):  # pragma: no cover - solver preserves subtype
            raise RuntimeError("lattice alignment did not preserve LeBailPhase")
        prefix_stages.append(
            ProfileEstimationStage(
                kind="lattice_alignment",
                instrument_parameters=(),
                accepted=True,
                decision="accepted bounded nuisance lattice alignment",
                rwp=float(aligned.metrics.rwp),
                maximum_absolute_correlation=_maximum_correlation(aligned.covariance),
            )
        )

    pattern = input_data.pattern
    native = _core._estimate_effective_profile(
        pattern.x,
        pattern.observed_y,
        pattern.uncertainty,
        pattern.mask,
        pattern.background,
        instrument.wavelength_angstrom,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
        phase.phase_id,
        phase.name,
        list(phase.reflections.reflection_ids),
        np.ascontiguousarray(phase.reflections.hkl, dtype=np.int64),
        phase.reflections.d_spacing_angstrom,
        phase.reflections.two_theta_deg,
        phase.reflections.integrated_intensity,
        phase.scale,
        selected.mode.value,
        selected.minimum_relative_rwp_improvement,
        selected.minimum_absolute_rwp_improvement,
        selected.maximum_absolute_correlation,
        selected.lebail.max_iterations,
        selected.lebail.min_iterations,
        selected.lebail.intensity_tolerance,
        selected.lebail.rwp_tolerance,
        selected.lebail.redistribution_damping,
        selected.lebail.minimum_calculated,
        selected.lebail.initial_intensity_floor,
        selected.lebail.use_uncertainty,
        selected.lebail.profile_damping,
        selected.lebail.max_scaled_parameter_step,
        selected.lebail.max_profile_backtracks,
        selected.lebail.unresolved_correlation,
        selected.lebail.diagnose_rank_deficiency,
        selected.lebail.support_fwhm,
        selected.lebail.execution._native,
    )
    fitted_instrument = ConstantWavelengthInstrument(
        native["wavelength_angstrom"],
        native["u_deg2"],
        native["v_deg2"],
        native["w_deg2"],
        native["x_deg"],
        native["y_deg"],
    )
    if fitted_instrument.wavelength_angstrom.hex() != instrument.wavelength_angstrom.hex():
        raise RuntimeError("fixed wavelength changed during profile estimation")
    intensities = _readonly_float64(native["integrated_intensity"])
    calculated = _readonly_float64(native["calculated_y"])
    fitted_phase = replace(
        phase,
        reflections=ReflectionBatch(
            list(phase.reflections.reflection_ids),
            phase.reflections.hkl,
            phase.reflections.d_spacing_angstrom,
            phase.reflections.two_theta_deg,
            intensities,
        ),
    )
    native_stages = tuple(
        ProfileEstimationStage(
            kind=item["kind"],
            instrument_parameters=tuple(item["instrument_parameters"]),
            accepted=bool(item["accepted"]),
            decision=item["decision"],
            rwp=float(item["rwp"]),
            maximum_absolute_correlation=item["maximum_absolute_correlation"],
        )
        for item in native["stages"]
    )
    return ProfileEstimationResult(
        instrument=fitted_instrument,
        active_parameters=tuple(native["active_parameters"]),
        phase=fitted_phase,
        integrated_intensity=intensities,
        calculated_y=calculated,
        rp=float(native["rp"]),
        rwp=float(native["rwp"]),
        chi_square=float(native["chi_square"]),
        termination_reason=native["termination_reason"],
        stages=tuple(prefix_stages) + native_stages,
        warnings=tuple(native["warnings"]),
    )


def _readonly_float64(values: object) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    array.flags.writeable = False
    return array


def _maximum_correlation(covariance: NDArray[np.float64] | None) -> float | None:
    if covariance is None:
        return None
    if covariance.size == 0:
        return 0.0
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0):
        return None
    correlation = covariance / np.sqrt(np.outer(diagonal, diagonal))
    return float(np.max(np.abs(correlation - np.eye(correlation.shape[0]))))
