"""Deterministic first-class Le Bail integrated-intensity extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..calculation import CalculationOptions, calculate_pattern
from ..control import (
    CancellationCallback,
    ProgressCallback,
    ProgressEvent,
    report_progress,
)
from ..instrument import ConstantWavelengthInstrument
from ..pattern import PatternCalculationResult, PowderPattern
from ..phase import Phase, ReflectionBatch
from ..structure import CrystalStructure
from ..symmetry import CwTwoThetaRange, PreparedReflectionGenerator
from .core import (
    AffineConstraint,
    Bounds,
    Constraint,
    ConstraintTransform,
    LeastSquaresOptimizer,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
    ResidualEvaluation,
    ResidualOptions,
    TerminationReason,
    evaluate_residuals,
)

if TYPE_CHECKING:
    from ..io.cif import CifBackend, CifReadLimits

INSTRUMENT_ROWS = {
    "u_deg2": "u",
    "v_deg2": "v",
    "w_deg2": "w",
    "x_deg": "x",
    "y_deg": "y",
}


@dataclass(frozen=True, slots=True)
class LeBailPhase(Phase):
    """A reflection-extraction phase retaining its source crystal structure."""

    structure: CrystalStructure | None = None

    def __post_init__(self) -> None:
        """Validate the generic phase contract and required source structure."""

        Phase.__post_init__(self)
        if not isinstance(self.structure, CrystalStructure):
            raise TypeError("LeBailPhase structure must be a CrystalStructure")

    @classmethod
    def from_structure(
        cls,
        structure: CrystalStructure,
        *,
        phase_id: str,
        wavelength_angstrom: float,
        two_theta_min_deg: float,
        two_theta_max_deg: float,
        name: str | None = None,
        scale: float = 1.0,
        initial_intensity: float = 1.0,
        merge_friedel: bool = True,
        max_candidates: int = 50_000_000,
    ) -> LeBailPhase:
        """Generate fixed-cell monochromatic reflections from a typed structure."""

        if not isinstance(structure, CrystalStructure):
            raise TypeError("structure must be a CrystalStructure")
        if not np.isfinite(initial_intensity) or initial_intensity < 0.0:
            raise ValueError("initial_intensity must be non-negative and finite")
        generated = PreparedReflectionGenerator(
            structure.space_group,
            merge_friedel=merge_friedel,
            max_candidates=max_candidates,
        ).generate(
            structure.cell,
            CwTwoThetaRange(
                two_theta_min_deg,
                two_theta_max_deg,
                wavelength_angstrom,
            ),
        )
        if generated.hkl.shape[0] == 0:
            raise ValueError("no allowed reflections lie within the requested 2theta range")
        argument = np.clip(
            0.5 * wavelength_angstrom * generated.reciprocal_length_inverse_angstrom,
            -1.0,
            1.0,
        )
        two_theta = 2.0 * np.degrees(np.arcsin(argument))
        reflections = ReflectionBatch(
            list(generated.reflection_ids),
            generated.hkl,
            generated.d_spacing_angstrom,
            two_theta,
            np.full(generated.hkl.shape[0], initial_intensity),
        )
        return cls(
            phase_id=phase_id,
            name=structure.name if name is None else name,
            reflections=reflections,
            scale=scale,
            structure=structure,
        )

    @classmethod
    def from_cif(
        cls,
        path_or_text: str | Path,
        *,
        phase_id: str,
        wavelength_angstrom: float,
        two_theta_min_deg: float,
        two_theta_max_deg: float,
        block: str | None = None,
        strict: bool = True,
        name: str | None = None,
        scale: float = 1.0,
        initial_intensity: float = 1.0,
        merge_friedel: bool = True,
        max_candidates: int = 50_000_000,
        limits: CifReadLimits | None = None,
        backend: CifBackend | None = None,
    ) -> LeBailPhase:
        """Read an optional CIF backend and construct a fixed-cell Le Bail phase."""

        from ..io.cif import read_cif

        imported = read_cif(
            path_or_text,
            block=block,
            strict=strict,
            limits=limits,
            backend=backend,
        )
        return cls.from_structure(
            imported.structure,
            phase_id=phase_id,
            wavelength_angstrom=wavelength_angstrom,
            two_theta_min_deg=two_theta_min_deg,
            two_theta_max_deg=two_theta_max_deg,
            name=name,
            scale=scale,
            initial_intensity=initial_intensity,
            merge_friedel=merge_friedel,
            max_candidates=max_candidates,
        )


def instrument_parameter_key(name: str) -> ParameterKey:
    """Return the standard key for one supported CW instrument scalar."""

    if name not in INSTRUMENT_ROWS:
        raise ValueError(f"unsupported Le Bail instrument parameter {name!r}")
    return ParameterKey("instrument", "cw", name)


def phase_scale_key(phase_id: str) -> ParameterKey:
    """Return the standard key for a phase scale."""

    return ParameterKey("phase", phase_id, "scale")


def reflection_position_key(phase_id: str, reflection_id: str) -> ParameterKey:
    """Return the standard key for one independent reflection position."""

    return ParameterKey("reflection", f"{phase_id}/{reflection_id}", "two_theta_deg")


def build_parameter_set(
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...] | list[Phase],
    *,
    instrument_parameters: tuple[str, ...] = (),
    phase_scales: bool = False,
    reflection_positions: bool = False,
) -> ParameterSet:
    """Build bounded typed specifications for selected Le Bail parameters."""

    specs = []
    for name in instrument_parameters:
        key = instrument_parameter_key(name)
        value = getattr(instrument, name)
        specs.append(
            ParameterSpec(
                key,
                value,
                "degree^2" if name.endswith("deg2") else "degree",
                Bounds(),
                max(abs(value), 1.0e-5 if name.endswith("deg2") else 1.0e-4),
            )
        )
    for phase in phases:
        if phase_scales:
            specs.append(
                ParameterSpec(
                    phase_scale_key(phase.phase_id),
                    phase.scale,
                    "dimensionless",
                    Bounds(0.0, np.inf),
                    max(phase.scale, 1.0),
                )
            )
        if reflection_positions:
            specs.extend(
                ParameterSpec(
                    reflection_position_key(phase.phase_id, reflection_id),
                    float(position),
                    "degree_2theta",
                    Bounds(np.nextafter(0.0, 1.0), np.nextafter(180.0, 0.0)),
                    0.01,
                )
                for reflection_id, position in zip(
                    phase.reflections.reflection_ids,
                    phase.reflections.two_theta_deg,
                    strict=True,
                )
            )
    return ParameterSet(specs)


@dataclass(frozen=True, slots=True)
class LeBailInput:
    """Observed pattern, current models, and optional profile parameter set."""

    pattern: PowderPattern
    instrument: ConstantWavelengthInstrument
    phases: tuple[Phase, ...]
    parameters: ParameterSet | None = None
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        """Validate complete script-facing Le Bail input."""

        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not isinstance(self.pattern, PowderPattern) or self.pattern.observed_y is None:
            raise ValueError("Le Bail input requires a PowderPattern with observed_y")
        if not isinstance(self.instrument, ConstantWavelengthInstrument):
            raise TypeError("instrument must be ConstantWavelengthInstrument")
        if not self.phases or any(not isinstance(phase, Phase) for phase in self.phases):
            raise TypeError("phases must be a non-empty tuple of Phase objects")
        if self.parameters is not None:
            _domain_parameter_values(self.instrument, self.phases, self.parameters)
            ConstraintTransform(self.parameters, self.constraints)
        elif self.constraints:
            raise ValueError("constraints require a parameter set")


@dataclass(frozen=True, slots=True)
class LeBailOptions:
    """Controls for non-negative redistribution and profile updates."""

    max_iterations: int = 50
    min_iterations: int = 2
    intensity_tolerance: float = 1.0e-6
    rwp_tolerance: float = 1.0e-8
    redistribution_damping: float = 1.0
    minimum_calculated: float = 1.0e-15
    initial_intensity_floor: float = 1.0e-12
    use_uncertainty: bool = True
    profile_damping: float = 1.0e-10
    max_scaled_parameter_step: float = 0.25
    max_profile_backtracks: int = 8
    unresolved_correlation: float = 1.0 - 1.0e-10
    support_fwhm: float = 20.0

    def __post_init__(self) -> None:
        """Validate finite convergence, support, and damping controls."""

        if self.max_iterations <= 0 or self.min_iterations <= 0:
            raise ValueError("iteration counts must be positive")
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations must not exceed max_iterations")
        positive = (
            self.intensity_tolerance,
            self.rwp_tolerance,
            self.minimum_calculated,
            self.initial_intensity_floor,
            self.max_scaled_parameter_step,
            self.support_fwhm,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Le Bail tolerances, floors, steps, and support must be positive")
        if not np.isfinite(self.redistribution_damping) or not (
            0.0 < self.redistribution_damping <= 1.0
        ):
            raise ValueError("redistribution_damping must lie in (0, 1]")
        if not np.isfinite(self.profile_damping) or self.profile_damping < 0.0:
            raise ValueError("profile_damping must be non-negative and finite")
        if self.max_profile_backtracks < 0:
            raise ValueError("max_profile_backtracks must be non-negative")
        if not 0.0 <= self.unresolved_correlation <= 1.0:
            raise ValueError("unresolved_correlation must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class CoincidentReflectionGroup:
    """A numerically rank-deficient group of reflection profile columns."""

    reflection_keys: tuple[tuple[str, str], ...]
    rank: int


@dataclass(frozen=True, slots=True)
class IntensityExtractionResult:
    """One non-negative Le Bail redistribution step."""

    intensities: NDArray[np.float64]
    maximum_relative_change: float
    unobserved_reflections: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ParameterChange:
    """One physical parameter change accepted during an iteration."""

    key: ParameterKey
    before: float
    after: float
    scaled_change: float

    def __post_init__(self) -> None:
        """Require a typed key and finite, auditable scalar values."""

        if not isinstance(self.key, ParameterKey):
            raise TypeError("parameter change key must be a ParameterKey")
        if not np.isfinite([self.before, self.after, self.scaled_change]).all():
            raise ValueError("parameter change values must be finite")


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """Immutable diagnostics after one full Le Bail iteration."""

    iteration: int
    rp: float
    rwp: float
    chi_square: float
    reduced_chi_square: float
    maximum_relative_intensity_change: float
    scaled_profile_step_norm: float
    parameter_changes: tuple[ParameterChange, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze sequences and reject malformed diagnostics."""

        object.__setattr__(self, "parameter_changes", tuple(self.parameter_changes))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.iteration <= 0:
            raise ValueError("iteration number must be positive")
        if any(not isinstance(change, ParameterChange) for change in self.parameter_changes):
            raise TypeError("parameter_changes must contain ParameterChange values")
        if any(not isinstance(warning, str) or not warning for warning in self.warnings):
            raise ValueError("iteration warnings must be non-empty strings")


@dataclass(frozen=True, slots=True)
class ReflectionIntensity:
    """Final non-negative integrated intensity with durable ownership."""

    phase_id: str
    reflection_id: str
    integrated_intensity: float


@dataclass(frozen=True, slots=True)
class LeBailResult:
    """Final calculation, extracted intensities, parameters, and history."""

    calculation: PatternCalculationResult
    instrument: ConstantWavelengthInstrument
    phases: tuple[Phase, ...]
    intensities: tuple[ReflectionIntensity, ...]
    metrics: ResidualEvaluation
    history: tuple[IterationRecord, ...]
    termination_reason: TerminationReason
    rank_deficient_groups: tuple[CoincidentReflectionGroup, ...]
    parameters: ParameterSet | None
    covariance: NDArray[np.float64] | None
    checkpoint: LeBailCheckpoint


@dataclass(frozen=True, slots=True)
class LeBailCheckpoint:
    """Immutable in-memory state sufficient for deterministic continuation."""

    completed_iterations: int
    instrument: ConstantWavelengthInstrument
    phases: tuple[Phase, ...]
    intensities: NDArray[np.float64]
    parameters: ParameterSet | None
    previous_rwp: float
    history: tuple[IterationRecord, ...]

    def __post_init__(self) -> None:
        """Validate continuation counters and immutable intensity storage."""

        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "history", tuple(self.history))
        intensities = np.array(self.intensities, dtype=np.float64, copy=True, order="C")
        intensities.flags.writeable = False
        object.__setattr__(self, "intensities", intensities)
        if self.completed_iterations < 0 or self.completed_iterations != len(self.history):
            raise ValueError("checkpoint iteration count must match its history")
        if self.intensities.ndim != 1 or not np.isfinite(self.intensities).all():
            raise ValueError("checkpoint intensities must be a finite vector")
        if np.any(self.intensities < 0.0):
            raise ValueError("checkpoint intensities must be non-negative")
        if np.isnan(self.previous_rwp) or np.isneginf(self.previous_rwp):
            raise ValueError("checkpoint previous_rwp must be finite or positive infinity")


def _bin_integration_weights(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    if x.size == 1:
        return np.ones(1, dtype=np.float64)
    widths = np.empty_like(x)
    widths[0] = 0.5 * (x[1] - x[0])
    widths[-1] = 0.5 * (x[-1] - x[-2])
    widths[1:-1] = 0.5 * (x[2:] - x[:-2])
    return widths


def _flat_intensities(phases: tuple[Phase, ...]) -> NDArray[np.float64]:
    return np.concatenate(tuple(phase.reflections.integrated_intensity for phase in phases))


def _replace_intensities(
    phases: tuple[Phase, ...], intensities: NDArray[np.float64]
) -> tuple[Phase, ...]:
    updated = []
    offset = 0
    for phase in phases:
        count = phase.reflections.reflection_count
        values = intensities[offset : offset + count]
        reflections = ReflectionBatch(
            list(phase.reflections.reflection_ids),
            phase.reflections.hkl,
            phase.reflections.d_spacing_angstrom,
            phase.reflections.two_theta_deg,
            values,
        )
        updated.append(replace(phase, reflections=reflections))
        offset += count
    return tuple(updated)


def initialize_intensities(
    input_data: LeBailInput, options: LeBailOptions | None = None
) -> NDArray[np.float64]:
    """Return deterministic positive starting intensities in reflection order."""

    selected_options = LeBailOptions() if options is None else options
    values = _flat_intensities(input_data.phases)
    if np.any(values < 0.0):
        raise ValueError("Le Bail starting intensities must be non-negative")
    if np.any(values > 0.0):
        return np.maximum(values, selected_options.initial_intensity_floor)
    net = np.maximum(input_data.pattern.observed_y - input_data.pattern.background, 0.0)
    area = float(np.sum(net * _bin_integration_weights(input_data.pattern.x)))
    starting = max(area / max(values.size, 1), selected_options.initial_intensity_floor)
    return np.full(values.size, starting, dtype=np.float64)


def extract_intensities(
    pattern: PowderPattern,
    calculation: PatternCalculationResult,
    current_intensities: ArrayLike,
    options: LeBailOptions | None = None,
) -> IntensityExtractionResult:
    """Perform one non-negative multiplicative Le Bail redistribution step."""

    selected_options = LeBailOptions() if options is None else options
    if pattern.observed_y is None:
        raise ValueError("observed_y is required for Le Bail extraction")
    current = np.asarray(current_intensities, dtype=np.float64)
    reflection_count = calculation.derivatives.local.peak_count
    if current.shape != (reflection_count,) or not np.isfinite(current).all():
        raise ValueError("current_intensities must match the reflection count")
    if np.any(current < 0.0):
        raise ValueError("current_intensities must be non-negative")
    included = np.ones(pattern.x.size, dtype=np.bool_) if pattern.mask is None else pattern.mask
    observation = np.maximum(pattern.observed_y - pattern.background, 0.0)
    ratio = np.zeros_like(observation)
    valid = included & (calculation.profile_y > selected_options.minimum_calculated)
    ratio[valid] = observation[valid] / calculation.profile_y[valid]
    weights = _bin_integration_weights(pattern.x)
    if selected_options.use_uncertainty and pattern.uncertainty is not None:
        weights = weights / np.square(pattern.uncertainty)
    weights = np.where(included, weights, 0.0)
    local = calculation.derivatives.local
    updated = np.zeros_like(current)
    unobserved = []
    for reflection, key in enumerate(calculation.reflection_keys):
        begin = int(local.offsets[reflection])
        end = int(local.offsets[reflection + 1])
        start = int(local.starts[reflection])
        stop = start + end - begin
        profile = local.values[begin:end, 0]
        weighted_profile = weights[start:stop] * profile
        denominator = float(np.sum(weighted_profile))
        if denominator <= 0.0:
            unobserved.append(key)
            continue
        factor = float(weighted_profile @ ratio[start:stop] / denominator)
        raw = max(0.0, current[reflection] * factor)
        updated[reflection] = current[reflection] + selected_options.redistribution_damping * (
            raw - current[reflection]
        )
    denominator = np.maximum(np.abs(current), selected_options.initial_intensity_floor)
    relative = np.abs(updated - current) / denominator
    updated.flags.writeable = False
    return IntensityExtractionResult(
        intensities=updated,
        maximum_relative_change=float(np.max(relative, initial=0.0)),
        unobserved_reflections=tuple(unobserved),
    )


def _domain_parameter_values(
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...],
    parameters: ParameterSet,
) -> dict[ParameterKey, float]:
    values: dict[ParameterKey, float] = {}
    phase_by_id = {phase.phase_id: phase for phase in phases}
    reflection_lookup = {
        f"{phase.phase_id}/{reflection_id}": float(position)
        for phase in phases
        for reflection_id, position in zip(
            phase.reflections.reflection_ids,
            phase.reflections.two_theta_deg,
            strict=True,
        )
    }
    for spec in parameters.specs:
        key = spec.key
        if key.module == "instrument" and key.owner_id == "cw" and key.name in INSTRUMENT_ROWS:
            value = float(getattr(instrument, key.name))
        elif key.module == "phase" and key.name == "scale" and key.owner_id in phase_by_id:
            value = float(phase_by_id[key.owner_id].scale)
        elif (
            key.module == "reflection"
            and key.name == "two_theta_deg"
            and key.owner_id in reflection_lookup
        ):
            value = reflection_lookup[key.owner_id]
        else:
            raise ValueError(f"unsupported or unknown Le Bail parameter key {key.label}")
        if not spec.bounds.contains(value):
            raise ValueError(f"domain value for {key.label} lies outside its bounds")
        values[key] = value
    return values


def _parameter_columns(
    calculation: PatternCalculationResult, parameters: ParameterSet
) -> NDArray[np.float64]:
    columns = []
    global_names = calculation.derivatives.global_parameter_names
    local_positions: NDArray[np.float64] | None = None
    reflection_index = {key: index for index, key in enumerate(calculation.reflection_keys)}
    for spec in parameters.specs:
        key = spec.key
        if key.module == "instrument":
            row = global_names.index(INSTRUMENT_ROWS[key.name])
            columns.append(calculation.derivatives.global_jacobian[row])
        elif key.module == "phase":
            row = global_names.index(f"phase[{key.owner_id}].scale")
            columns.append(calculation.derivatives.global_jacobian[row])
        else:
            phase_id, reflection_id = key.owner_id.split("/", 1)
            index = reflection_index[(phase_id, reflection_id)]
            if local_positions is None:
                local_positions = calculation.derivatives.local.to_dense(calculation.y.size)[:, 1]
            columns.append(local_positions[index])
    if not columns:
        return np.empty((calculation.y.size, 0), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack(columns))


def _constraint_matrix(transform: ConstraintTransform) -> NDArray[np.float64]:
    """Return the exact physical-to-scaled-free affine derivative matrix."""

    specs = transform.parameters.specs
    row_for_key = {spec.key: row for row, spec in enumerate(specs)}
    matrix = np.zeros((len(specs), len(transform.free_keys)), dtype=np.float64)
    for column, key in enumerate(transform.free_keys):
        matrix[row_for_key[key], column] = transform.parameters.spec(key).scale
    for constraint in transform.constraints:
        if isinstance(constraint, AffineConstraint):
            matrix[row_for_key[constraint.target]] = (
                constraint.multiplier * matrix[row_for_key[constraint.source]]
            )
    return matrix


def _apply_parameter_values(
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...],
    values: dict[ParameterKey, float],
) -> tuple[ConstantWavelengthInstrument, tuple[Phase, ...]]:
    instrument_updates = {
        key.name: value for key, value in values.items() if key.module == "instrument"
    }
    updated_instrument = replace(instrument, **instrument_updates)
    updated_phases = []
    for phase in phases:
        phase_updates = {}
        scale_key = phase_scale_key(phase.phase_id)
        if scale_key in values:
            phase_updates["scale"] = values[scale_key]
        positions = np.array(phase.reflections.two_theta_deg, copy=True)
        positions_changed = False
        for index, reflection_id in enumerate(phase.reflections.reflection_ids):
            key = reflection_position_key(phase.phase_id, reflection_id)
            if key in values:
                positions[index] = values[key]
                positions_changed = True
        if positions_changed:
            phase_updates["reflections"] = ReflectionBatch(
                list(phase.reflections.reflection_ids),
                phase.reflections.hkl,
                phase.reflections.d_spacing_angstrom,
                positions,
                phase.reflections.integrated_intensity,
            )
        updated_phases.append(replace(phase, **phase_updates))
    return updated_instrument, tuple(updated_phases)


def _profile_update(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...],
    calculation: PatternCalculationResult,
    parameters: ParameterSet,
    constraints: tuple[Constraint, ...],
    options: LeBailOptions,
    optimizer: LeastSquaresOptimizer | None,
) -> tuple[
    ConstantWavelengthInstrument,
    tuple[Phase, ...],
    PatternCalculationResult,
    ParameterSet,
    float,
    tuple[ParameterChange, ...],
    tuple[str, ...],
]:
    domain_values = _domain_parameter_values(instrument, phases, parameters)
    current_parameters = parameters.replace_values(domain_values)
    transform = ConstraintTransform(current_parameters, constraints)
    if not transform.free_keys:
        return instrument, phases, calculation, current_parameters, 0.0, (), ()
    physical_columns = _parameter_columns(calculation, current_parameters)
    chain = _constraint_matrix(transform)
    jacobian = physical_columns @ chain
    residual = pattern.observed_y - calculation.y
    included = np.ones(pattern.x.size, dtype=np.bool_) if pattern.mask is None else pattern.mask
    selected_jacobian = jacobian[included]
    selected_residual = residual[included]
    if options.use_uncertainty and pattern.uncertainty is not None:
        selected_jacobian = selected_jacobian / pattern.uncertainty[included, None]
        selected_residual = selected_residual / pattern.uncertainty[included]
    normal = selected_jacobian.T @ selected_jacobian
    rank = int(np.linalg.matrix_rank(normal))
    warnings: tuple[str, ...] = (
        () if rank == normal.shape[0] else ("profile Jacobian is rank deficient",)
    )
    if any(
        spec.key.module == "phase" and spec.key.name == "scale" for spec in current_parameters.specs
    ):
        warnings = (
            *warnings,
            "phase scale is not identifiable independently of extracted Le Bail intensities",
        )
    base = transform.pack()
    lower = np.full(base.shape, -options.max_scaled_parameter_step)
    upper = np.full(base.shape, options.max_scaled_parameter_step)
    for index, key in enumerate(transform.free_keys):
        spec = current_parameters.spec(key)
        lower[index] = max(lower[index], spec.bounds.lower / spec.scale - base[index])
        upper[index] = min(upper[index], spec.bounds.upper / spec.scale - base[index])
    if optimizer is None:
        regularized = normal + options.profile_damping * np.eye(normal.shape[0])
        try:
            step = np.linalg.solve(regularized, selected_jacobian.T @ selected_residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(selected_jacobian, selected_residual, rcond=None)[0]
            warnings = (*warnings, "profile normal equations used least-squares fallback")
        step = np.clip(step, lower, upper)
    else:
        step = np.asarray(
            optimizer.solve(-selected_residual, selected_jacobian, lower, upper),
            dtype=np.float64,
        )
        if step.shape != base.shape or not np.isfinite(step).all():
            raise ValueError("optimizer returned a non-finite step with an invalid shape")
        tolerance = 32.0 * np.finfo(np.float64).eps
        if np.any(step < lower - tolerance) or np.any(step > upper + tolerance):
            raise ValueError("optimizer returned a step outside the supplied bounds")
        step = np.clip(step, lower, upper)
    baseline_metrics = evaluate_residuals(
        pattern,
        calculation.y,
        ResidualOptions(
            use_uncertainty=options.use_uncertainty,
            parameter_count=len(transform.free_keys),
        ),
    )
    factor = 1.0
    for _ in range(options.max_profile_backtracks + 1):
        try:
            values = transform.unpack(base + factor * step, clip=True)
            candidate_instrument, candidate_phases = _apply_parameter_values(
                instrument, phases, values
            )
            candidate_calculation = calculate_pattern(
                pattern,
                candidate_instrument,
                candidate_phases,
                options=CalculationOptions(
                    support_fwhm=options.support_fwhm,
                    return_phase_components=True,
                ),
            )
            candidate_metrics = evaluate_residuals(
                pattern,
                candidate_calculation.y,
                ResidualOptions(
                    use_uncertainty=options.use_uncertainty,
                    parameter_count=len(transform.free_keys),
                ),
            )
        except ValueError:
            factor *= 0.5
            continue
        if candidate_metrics.chi_square < baseline_metrics.chi_square:
            candidate_parameters = current_parameters.replace_values(values)
            changes = tuple(
                ParameterChange(
                    spec.key,
                    spec.value,
                    candidate_parameters.spec(spec.key).value,
                    (candidate_parameters.spec(spec.key).value - spec.value) / spec.scale,
                )
                for spec in current_parameters.specs
                if candidate_parameters.spec(spec.key).value != spec.value
            )
            return (
                candidate_instrument,
                candidate_phases,
                candidate_calculation,
                candidate_parameters,
                float(np.linalg.norm(factor * step)),
                changes,
                warnings,
            )
        factor *= 0.5
    return (
        instrument,
        phases,
        calculation,
        current_parameters,
        0.0,
        (),
        (*warnings, "profile step rejected by backtracking"),
    )


def _rank_deficient_groups(
    calculation: PatternCalculationResult, threshold: float
) -> tuple[CoincidentReflectionGroup, ...]:
    local = calculation.derivatives.local
    count = local.peak_count
    parents = list(range(count))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    norms = np.sqrt(
        [
            float(
                local.values[int(local.offsets[index]) : int(local.offsets[index + 1]), 0]
                @ local.values[int(local.offsets[index]) : int(local.offsets[index + 1]), 0]
            )
            for index in range(count)
        ]
    )
    for left in range(count):
        left_begin = int(local.offsets[left])
        left_end = int(local.offsets[left + 1])
        left_start = int(local.starts[left])
        left_stop = left_start + left_end - left_begin
        for right in range(left + 1, count):
            right_begin = int(local.offsets[right])
            right_end = int(local.offsets[right + 1])
            right_start = int(local.starts[right])
            right_stop = right_start + right_end - right_begin
            start = max(left_start, right_start)
            stop = min(left_stop, right_stop)
            if start >= stop or norms[left] == 0.0 or norms[right] == 0.0:
                continue
            left_values = local.values[
                left_begin + start - left_start : left_begin + stop - left_start, 0
            ]
            right_values = local.values[
                right_begin + start - right_start : right_begin + stop - right_start, 0
            ]
            correlation = float(left_values @ right_values / (norms[left] * norms[right]))
            if correlation >= threshold:
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(root(index), []).append(index)
    output = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        first_start = min(int(local.starts[index]) for index in indices)
        last_stop = max(
            int(local.starts[index]) + int(local.offsets[index + 1]) - int(local.offsets[index])
            for index in indices
        )
        matrix = np.zeros((last_stop - first_start, len(indices)))
        for column, index in enumerate(indices):
            begin = int(local.offsets[index])
            end = int(local.offsets[index + 1])
            start = int(local.starts[index]) - first_start
            matrix[start : start + end - begin, column] = local.values[begin:end, 0]
        output.append(
            CoincidentReflectionGroup(
                reflection_keys=tuple(calculation.reflection_keys[index] for index in indices),
                rank=int(np.linalg.matrix_rank(matrix)),
            )
        )
    return tuple(output)


def _covariance(
    pattern: PowderPattern,
    calculation: PatternCalculationResult,
    parameters: ParameterSet | None,
    constraints: tuple[Constraint, ...],
    use_uncertainty: bool,
) -> NDArray[np.float64] | None:
    if parameters is None:
        return None
    transform = ConstraintTransform(parameters, constraints)
    if not transform.free_keys:
        return np.empty((0, 0), dtype=np.float64)
    chain = _constraint_matrix(transform)
    for row, spec in enumerate(parameters.specs):
        if spec.key.module == "phase" and spec.key.name == "scale" and np.any(chain[row] != 0.0):
            return None
    jacobian = _parameter_columns(calculation, parameters) @ chain
    included = np.ones(pattern.x.size, dtype=np.bool_) if pattern.mask is None else pattern.mask
    selected = jacobian[included]
    if use_uncertainty and pattern.uncertainty is not None:
        selected = selected / pattern.uncertainty[included, None]
    normal = selected.T @ selected
    if np.linalg.matrix_rank(normal) != normal.shape[0]:
        return None
    covariance = np.linalg.inv(normal)
    covariance.flags.writeable = False
    return covariance


def refine(
    input_data: LeBailInput,
    options: LeBailOptions | None = None,
    *,
    checkpoint: LeBailCheckpoint | None = None,
    optimizer: LeastSquaresOptimizer | None = None,
    progress: ProgressCallback | None = None,
    cancellation: CancellationCallback | None = None,
) -> LeBailResult:
    """Run deterministic Le Bail extraction and optional profile updates."""

    if not isinstance(input_data, LeBailInput):
        raise TypeError("input_data must be LeBailInput")
    selected_options = LeBailOptions() if options is None else options
    if not isinstance(selected_options, LeBailOptions):
        raise TypeError("options must be LeBailOptions")
    if optimizer is not None and not isinstance(optimizer, LeastSquaresOptimizer):
        raise TypeError("optimizer must implement LeastSquaresOptimizer")
    if checkpoint is None:
        instrument = input_data.instrument
        intensities = initialize_intensities(input_data, selected_options)
        phases = _replace_intensities(input_data.phases, intensities)
        parameters = input_data.parameters
        history = []
        previous_rwp = np.inf
        first_iteration = 1
    else:
        if not isinstance(checkpoint, LeBailCheckpoint):
            raise TypeError("checkpoint must be LeBailCheckpoint")
        input_keys = tuple(
            (phase.phase_id, reflection_id)
            for phase in input_data.phases
            for reflection_id in phase.reflections.reflection_ids
        )
        checkpoint_keys = tuple(
            (phase.phase_id, reflection_id)
            for phase in checkpoint.phases
            for reflection_id in phase.reflections.reflection_ids
        )
        if input_keys != checkpoint_keys:
            raise ValueError("checkpoint reflection identities do not match Le Bail input")
        instrument = checkpoint.instrument
        intensities = np.array(checkpoint.intensities, copy=True)
        phases = checkpoint.phases
        parameters = checkpoint.parameters
        history = list(checkpoint.history)
        previous_rwp = checkpoint.previous_rwp
        first_iteration = checkpoint.completed_iterations + 1
    termination = TerminationReason.MAX_ITERATIONS
    calculation = calculate_pattern(
        input_data.pattern,
        instrument,
        phases,
        options=CalculationOptions(
            support_fwhm=selected_options.support_fwhm,
            return_phase_components=True,
        ),
    )
    for iteration in range(first_iteration, selected_options.max_iterations + 1):
        if cancellation is not None and cancellation():
            termination = TerminationReason.CANCELLED
            break
        extraction = extract_intensities(
            input_data.pattern,
            calculation,
            intensities,
            selected_options,
        )
        intensities = extraction.intensities
        phases = _replace_intensities(phases, intensities)
        calculation = calculate_pattern(
            input_data.pattern,
            instrument,
            phases,
            options=CalculationOptions(
                support_fwhm=selected_options.support_fwhm,
                return_phase_components=True,
            ),
        )
        profile_step = 0.0
        parameter_changes: tuple[ParameterChange, ...] = ()
        warnings = ()
        if parameters is not None:
            (
                instrument,
                phases,
                calculation,
                parameters,
                profile_step,
                parameter_changes,
                warnings,
            ) = _profile_update(
                input_data.pattern,
                instrument,
                phases,
                calculation,
                parameters,
                input_data.constraints,
                selected_options,
                optimizer,
            )
        if extraction.unobserved_reflections:
            warnings = (
                *warnings,
                f"{len(extraction.unobserved_reflections)} reflections have no included support",
            )
        parameter_count = (
            0
            if parameters is None
            else len(ConstraintTransform(parameters, input_data.constraints).free_keys)
        )
        metrics = evaluate_residuals(
            input_data.pattern,
            calculation.y,
            ResidualOptions(
                use_uncertainty=selected_options.use_uncertainty,
                parameter_count=parameter_count,
            ),
        )
        history.append(
            IterationRecord(
                iteration=iteration,
                rp=metrics.rp,
                rwp=metrics.rwp,
                chi_square=metrics.chi_square,
                reduced_chi_square=metrics.reduced_chi_square,
                maximum_relative_intensity_change=extraction.maximum_relative_change,
                scaled_profile_step_norm=profile_step,
                parameter_changes=parameter_changes,
                warnings=warnings,
            )
        )
        report_progress(
            progress,
            ProgressEvent(
                "lebail_iteration",
                iteration,
                selected_options.max_iterations,
                (
                    ("rwp", metrics.rwp),
                    (
                        "maximum_relative_intensity_change",
                        extraction.maximum_relative_change,
                    ),
                ),
            ),
        )
        if (
            iteration >= selected_options.min_iterations
            and extraction.maximum_relative_change < selected_options.intensity_tolerance
            and abs(previous_rwp - metrics.rwp) < selected_options.rwp_tolerance
        ):
            termination = TerminationReason.CONVERGED
            break
        previous_rwp = metrics.rwp
    final_metrics = evaluate_residuals(
        input_data.pattern,
        calculation.y,
        ResidualOptions(
            use_uncertainty=selected_options.use_uncertainty,
            parameter_count=(
                0
                if parameters is None
                else len(ConstraintTransform(parameters, input_data.constraints).free_keys)
            ),
        ),
    )
    labeled_intensities = tuple(
        ReflectionIntensity(phase_id, reflection_id, float(intensity))
        for (phase_id, reflection_id), intensity in zip(
            calculation.reflection_keys, intensities, strict=True
        )
    )
    final_intensity_array = np.array(intensities, copy=True)
    final_intensity_array.flags.writeable = False
    final_checkpoint = LeBailCheckpoint(
        completed_iterations=len(history),
        instrument=instrument,
        phases=phases,
        intensities=final_intensity_array,
        parameters=parameters,
        previous_rwp=(
            previous_rwp if termination is TerminationReason.CANCELLED else final_metrics.rwp
        ),
        history=tuple(history),
    )
    return LeBailResult(
        calculation=calculation,
        instrument=instrument,
        phases=phases,
        intensities=labeled_intensities,
        metrics=final_metrics,
        history=tuple(history),
        termination_reason=termination,
        rank_deficient_groups=_rank_deficient_groups(
            calculation, selected_options.unresolved_correlation
        ),
        parameters=parameters,
        covariance=_covariance(
            input_data.pattern,
            calculation,
            parameters,
            input_data.constraints,
            selected_options.use_uncertainty,
        ),
        checkpoint=final_checkpoint,
    )


def iterate_once(
    input_data: LeBailInput,
    options: LeBailOptions | None = None,
    *,
    checkpoint: LeBailCheckpoint | None = None,
    optimizer: LeastSquaresOptimizer | None = None,
    progress: ProgressCallback | None = None,
    cancellation: CancellationCallback | None = None,
) -> LeBailResult:
    """Advance exactly one Le Bail iteration for custom orchestration.

    Pass the returned checkpoint to the next call. Cancellation may return
    without advancing, preserving the supplied checkpoint unchanged.
    """

    selected = LeBailOptions() if options is None else options
    if not isinstance(selected, LeBailOptions):
        raise TypeError("options must be LeBailOptions")
    completed = 0 if checkpoint is None else checkpoint.completed_iterations
    single_iteration = replace(
        selected,
        min_iterations=completed + 1,
        max_iterations=completed + 1,
    )
    return refine(
        input_data,
        single_iteration,
        checkpoint=checkpoint,
        optimizer=optimizer,
        progress=progress,
        cancellation=cancellation,
    )
