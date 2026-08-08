"""Script-friendly stateless powder-profile calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .control import (
    CancellationCallback,
    ProgressCallback,
    ProgressEvent,
    check_cancelled,
    report_progress,
)
from .cw import accumulate_cw, accumulate_cw_contributions
from .execution import ExecutionPolicy
from .extensions import (
    PhysicsContext,
    PhysicsContribution,
    ReflectionPhysicsProvider,
    evaluate_provider,
)
from .fcj import accumulate_cw_fcj
from .instrument import ConstantWavelengthInstrument, FcjGeometry
from .pattern import PatternCalculationResult, PhasePatternComponent, PowderPattern
from .phase import Phase, ReflectionGeometryBatch
from .radiation import ConstantWavelengthExperiment, RadiationProbe
from .results import AccumulationResult


def calculate_cw_pattern(
    x: ArrayLike,
    reflections: ReflectionGeometryBatch,
    instrument: ConstantWavelengthInstrument,
    *,
    physics: ReflectionPhysicsProvider | None = None,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Calculate a monochromatic CW profile with an optional batch provider.

    The provider is called exactly once. Its arrays are then consumed by one
    native fused accumulation call; no Python callback occurs per reflection or
    per profile sample.
    """

    if physics is None:
        return accumulate_cw(
            x,
            reflections.two_theta_deg,
            reflections.base_integrated_intensity,
            instrument,
            support_fwhm=support_fwhm,
            jacobian_layout=jacobian_layout,
        )
    contribution = evaluate_provider(physics, PhysicsContext(reflections, instrument))
    return accumulate_cw_contributions(
        x,
        reflections.two_theta_deg,
        reflections.base_integrated_intensity,
        instrument,
        contribution,
        support_fwhm=support_fwhm,
        jacobian_layout=jacobian_layout,
    )


@dataclass(frozen=True, slots=True)
class CalculationOptions:
    """Deterministic controls for a high-level pattern calculation."""

    support_fwhm: float = 20.0
    jacobian_layout: Literal["support", "dense"] = "support"
    return_phase_components: bool = False
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def __post_init__(self) -> None:
        """Validate support and derivative-layout selection early."""

        if not np.isfinite(self.support_fwhm) or self.support_fwhm <= 0.0:
            raise ValueError("support_fwhm must be positive and finite")
        if self.jacobian_layout not in ("support", "dense"):
            raise ValueError("jacobian_layout must be 'support' or 'dense'")
        if not isinstance(self.execution, ExecutionPolicy):
            raise TypeError("execution must be ExecutionPolicy")


@dataclass(frozen=True, slots=True)
class _FlattenedPhases:
    positions: NDArray[np.float64]
    base_intensities: NDArray[np.float64]
    contributions: PhysicsContribution
    phase_ids: tuple[str, ...]
    phase_offsets: NDArray[np.int64]
    reflection_keys: tuple[tuple[str, str], ...]


def _phase_parameter_name(phase_id: str, parameter: str) -> str:
    return f"phase[{phase_id}].{parameter}"


def _phase_physics_parameter_name(phase_id: str, parameter: str) -> str:
    return f"phase[{phase_id}].physics.{parameter}"


def _flatten_phases(
    phases: tuple[Phase, ...], instrument: ConstantWavelengthInstrument
) -> _FlattenedPhases:
    if not phases:
        raise ValueError("at least one phase is required")
    phase_ids = tuple(phase.phase_id for phase in phases)
    if len(set(phase_ids)) != len(phase_ids):
        raise ValueError("phase_id values must be unique")
    counts = np.asarray([phase.reflections.reflection_count for phase in phases], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    reflection_count = int(offsets[-1])
    if reflection_count == 0:
        raise ValueError("at least one reflection is required")

    positions = np.concatenate(tuple(phase.reflections.geometry.two_theta_deg for phase in phases))
    base_intensities = np.concatenate(
        tuple(phase.reflections.integrated_intensity for phase in phases)
    )
    evaluated = tuple(
        PhysicsContribution.neutral(phase.reflections.reflection_count)
        if phase.physics is None
        else evaluate_provider(
            phase.physics,
            PhysicsContext(phase.reflections.geometry, instrument),
        )
        for phase in phases
    )
    parameter_names = tuple(
        name
        for phase, contribution in zip(phases, evaluated, strict=True)
        for name in (
            _phase_parameter_name(phase.phase_id, "scale"),
            *(
                _phase_physics_parameter_name(phase.phase_id, parameter)
                for parameter in contribution.parameter_names
            ),
        )
    )
    parameter_count = len(parameter_names)
    d_gaussian = np.zeros((parameter_count, reflection_count), dtype=np.float64)
    d_lorentzian = np.zeros((parameter_count, reflection_count), dtype=np.float64)
    d_multiplier = np.zeros((parameter_count, reflection_count), dtype=np.float64)
    gaussian_parts = []
    lorentzian_parts = []
    multiplier_parts = []
    d_gaussian_position_parts = []
    d_lorentzian_position_parts = []
    d_multiplier_position_parts = []
    reflection_keys = []
    parameter_offset = 0
    for phase_index, (phase, contribution) in enumerate(zip(phases, evaluated, strict=True)):
        start = int(offsets[phase_index])
        stop = int(offsets[phase_index + 1])
        row = parameter_offset
        d_multiplier[row, start:stop] = contribution.intensity_multiplier
        provider_rows = len(contribution.parameter_names)
        if provider_rows:
            selection = slice(row + 1, row + 1 + provider_rows)
            d_gaussian[selection, start:stop] = contribution.d_gaussian_variance_d_parameters
            d_lorentzian[selection, start:stop] = contribution.d_lorentzian_fwhm_d_parameters
            d_multiplier[selection, start:stop] = (
                phase.scale * contribution.d_intensity_multiplier_d_parameters
            )
        parameter_offset += 1 + provider_rows
        gaussian_parts.append(contribution.gaussian_variance_deg2)
        lorentzian_parts.append(contribution.lorentzian_fwhm_deg)
        multiplier_parts.append(phase.scale * contribution.intensity_multiplier)
        d_gaussian_position_parts.append(contribution.d_gaussian_variance_d_position)
        d_lorentzian_position_parts.append(contribution.d_lorentzian_fwhm_d_position)
        d_multiplier_position_parts.append(
            phase.scale * contribution.d_intensity_multiplier_d_position
        )
        reflection_keys.extend(
            (phase.phase_id, reflection_id) for reflection_id in phase.reflections.reflection_ids
        )
    positions = np.ascontiguousarray(positions)
    base_intensities = np.ascontiguousarray(base_intensities)
    positions.flags.writeable = False
    base_intensities.flags.writeable = False
    offsets.flags.writeable = False
    return _FlattenedPhases(
        positions=positions,
        base_intensities=base_intensities,
        contributions=PhysicsContribution(
            gaussian_variance_deg2=np.concatenate(gaussian_parts),
            lorentzian_fwhm_deg=np.concatenate(lorentzian_parts),
            intensity_multiplier=np.concatenate(multiplier_parts),
            d_gaussian_variance_d_position=np.concatenate(d_gaussian_position_parts),
            d_lorentzian_fwhm_d_position=np.concatenate(d_lorentzian_position_parts),
            d_intensity_multiplier_d_position=np.concatenate(d_multiplier_position_parts),
            parameter_names=parameter_names,
            d_gaussian_variance_d_parameters=d_gaussian,
            d_lorentzian_fwhm_d_parameters=d_lorentzian,
            d_intensity_multiplier_d_parameters=d_multiplier,
        ),
        phase_ids=phase_ids,
        phase_offsets=offsets,
        reflection_keys=tuple(reflection_keys),
    )


def _phase_components(
    accumulation: AccumulationResult,
    flattened: _FlattenedPhases,
) -> tuple[PhasePatternComponent, ...]:
    """Reconstruct diagnostic phase curves from the fused local support blocks."""

    local = accumulation.derivatives.local
    components = []
    for phase_index, phase_id in enumerate(flattened.phase_ids):
        phase_y = np.zeros(accumulation.y.size, dtype=np.float64)
        first = int(flattened.phase_offsets[phase_index])
        last = int(flattened.phase_offsets[phase_index + 1])
        for reflection in range(first, last):
            begin = int(local.offsets[reflection])
            end = int(local.offsets[reflection + 1])
            start = int(local.starts[reflection])
            stop = start + end - begin
            phase_y[start:stop] += (
                flattened.base_intensities[reflection] * local.values[begin:end, 0]
            )
        phase_y.flags.writeable = False
        components.append(PhasePatternComponent(phase_id=phase_id, y=phase_y))
    return tuple(components)


def _calculate_prepared(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    flattened: _FlattenedPhases,
    options: CalculationOptions,
) -> PatternCalculationResult:
    accumulation = accumulate_cw_contributions(
        pattern.x,
        flattened.positions,
        flattened.base_intensities,
        instrument,
        flattened.contributions,
        support_fwhm=options.support_fwhm,
        jacobian_layout=options.jacobian_layout,
    )
    profile_y = np.array(accumulation.y, copy=True)
    total_y = profile_y + pattern.background
    profile_y.flags.writeable = False
    total_y.flags.writeable = False
    components = (
        _phase_components(accumulation, flattened) if options.return_phase_components else ()
    )
    return PatternCalculationResult(
        y=total_y,
        profile_y=profile_y,
        background=pattern.background,
        accumulation=accumulation,
        reflection_keys=flattened.reflection_keys,
        phase_offsets=flattened.phase_offsets,
        phase_components=components,
    )


def calculate_pattern(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...] | list[Phase],
    *,
    options: CalculationOptions | None = None,
    progress: ProgressCallback | None = None,
    cancellation: CancellationCallback | None = None,
) -> PatternCalculationResult:
    """Calculate all CW phases through one flattened native accumulation call."""

    if not isinstance(pattern, PowderPattern):
        raise TypeError("pattern must be a PowderPattern")
    phase_tuple = tuple(phases)
    if any(not isinstance(phase, Phase) for phase in phase_tuple):
        raise TypeError("phases must contain only Phase objects")
    selected_options = CalculationOptions() if options is None else options
    if not isinstance(selected_options, CalculationOptions):
        raise TypeError("options must be CalculationOptions")
    check_cancelled(cancellation)
    report_progress(progress, ProgressEvent("calculation", 0, 1))
    flattened = _flatten_phases(phase_tuple, instrument)
    check_cancelled(cancellation)
    result = _calculate_prepared(pattern, instrument, flattened, selected_options)
    check_cancelled(cancellation)
    report_progress(progress, ProgressEvent("calculation", 1, 1))
    return result


@dataclass(frozen=True, slots=True, init=False)
class PreparedPattern:
    """Reusable validated and flattened CW pattern calculation."""

    pattern: PowderPattern
    instrument: ConstantWavelengthInstrument
    options: CalculationOptions
    _flattened: _FlattenedPhases

    def __init__(
        self,
        pattern: PowderPattern,
        instrument: ConstantWavelengthInstrument,
        phases: tuple[Phase, ...] | list[Phase],
        *,
        options: CalculationOptions | None = None,
    ) -> None:
        """Evaluate providers once and retain immutable flat native inputs."""

        if not isinstance(pattern, PowderPattern):
            raise TypeError("pattern must be a PowderPattern")
        phase_tuple = tuple(phases)
        if any(not isinstance(phase, Phase) for phase in phase_tuple):
            raise TypeError("phases must contain only Phase objects")
        selected_options = CalculationOptions() if options is None else options
        if not isinstance(selected_options, CalculationOptions):
            raise TypeError("options must be CalculationOptions")
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "options", selected_options)
        object.__setattr__(self, "_flattened", _flatten_phases(phase_tuple, instrument))

    def calculate(
        self,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationCallback | None = None,
    ) -> PatternCalculationResult:
        """Run one native call using the prepared immutable arrays."""

        check_cancelled(cancellation)
        report_progress(progress, ProgressEvent("calculation", 0, 1))
        result = _calculate_prepared(
            self.pattern,
            self.instrument,
            self._flattened,
            self.options,
        )
        check_cancelled(cancellation)
        report_progress(progress, ProgressEvent("calculation", 1, 1))
        return result


def calculate_monochromatic_cw_pattern(
    x: ArrayLike,
    reflections: ReflectionGeometryBatch,
    experiment: ConstantWavelengthExperiment,
    *,
    physics: ReflectionPhysicsProvider | None = None,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Calculate one typed monochromatic X-ray or neutron reflection batch."""

    if not isinstance(experiment, ConstantWavelengthExperiment):
        raise TypeError("experiment must be a ConstantWavelengthExperiment")
    return calculate_cw_pattern(
        x,
        reflections,
        experiment.instrument,
        physics=physics,
        support_fwhm=support_fwhm,
        jacobian_layout=jacobian_layout,
    )


def calculate_monochromatic_pattern(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[Phase, ...] | list[Phase],
    *,
    options: CalculationOptions | None = None,
) -> PatternCalculationResult:
    """Calculate all phases for an explicitly typed monochromatic experiment."""

    if not isinstance(experiment, ConstantWavelengthExperiment):
        raise TypeError("experiment must be a ConstantWavelengthExperiment")
    return calculate_pattern(pattern, experiment.instrument, phases, options=options)


def calculate_neutron_pattern(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[Phase, ...] | list[Phase],
    *,
    options: CalculationOptions | None = None,
) -> PatternCalculationResult:
    """Calculate a typed monochromatic-neutron CW pattern."""

    if not isinstance(experiment, ConstantWavelengthExperiment):
        raise TypeError("experiment must be a ConstantWavelengthExperiment")
    if experiment.radiation.probe is not RadiationProbe.NEUTRON:
        raise ValueError("calculate_neutron_pattern requires neutron radiation")
    return calculate_pattern(pattern, experiment.instrument, phases, options=options)


def calculate_neutron_fcj_pattern(
    x: ArrayLike,
    reflections: ReflectionGeometryBatch,
    experiment: ConstantWavelengthExperiment,
    geometry: FcjGeometry,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Calculate typed monochromatic-neutron CW reflections with FCJ asymmetry."""

    if not isinstance(experiment, ConstantWavelengthExperiment):
        raise TypeError("experiment must be a ConstantWavelengthExperiment")
    if experiment.radiation.probe is not RadiationProbe.NEUTRON:
        raise ValueError("calculate_neutron_fcj_pattern requires neutron radiation")
    return accumulate_cw_fcj(
        x,
        reflections.two_theta_deg,
        reflections.base_integrated_intensity,
        experiment.instrument,
        geometry,
        support_fwhm=support_fwhm,
        jacobian_layout=jacobian_layout,
    )
