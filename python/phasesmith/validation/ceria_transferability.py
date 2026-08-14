"""IUCr ceria round-robin calibration/holdout transferability gate.

The Birmingham measurements are three contiguous angular ranges for each of
two specimens, not repeat scans.  The annealed, narrow-line specimen calibrates
an empirical instrument profile.  That profile is then frozen while the
broadened specimen receives only isotropic size/microstrain terms and a
separately estimated position nuisance.  Missing optics metadata remains an
explicit stop for specialized physical-profile terms.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..background import SmoothBrucknerBackground
from ..extensions import CompositePhysicsProvider
from ..instrument import ConstantWavelengthInstrument
from ..pattern import PowderPattern
from ..phase import ReflectionBatch
from ..profile_estimation import (
    ProfileEstimationMode,
    ProfileEstimationOptions,
    estimate_effective_profile,
    starting_profile_from_fwhm,
)
from ..refinement import lebail
from ..sample import IsotropicMicrostrainBroadening, IsotropicSizeBroadening
from .datasets import validation_dataset, verify_validation_dataset

DATASET_ID = "iucr-ceria-size-strain-round-robin"
WAVELENGTH_1_ANGSTROM = 1.5406
WAVELENGTH_2_ANGSTROM = 1.5444
K_ALPHA2_OVER_K_ALPHA1 = 0.016
POLARIZATION_FACTOR_P = 0.8
REFERENCE_CELL_ANGSTROM = 5.41134

_SHARP_FILES = ("langfsh1.xy", "langfsh2.xy", "langfsh3.xy")
_BROAD_FILES = ("langfbr1.xy", "langfbr2.xy", "langfbr3.xy")
_PROFILE_STARTS_DEG = (0.04, 0.10, 0.20)
_BROADENING_STARTS = ((15.0, 0.002), (35.0, 0.0005), (80.0, 0.0))
_UNKNOWN_OPTICS = (
    "goniometer radius",
    "equatorial receiving aperture or detector geometry",
    "source, specimen, and receiver axial dimensions",
    "incident and diffracted Soller angles",
    "Ge(111) monochromator passband and coupled dispersion",
    "specimen mounting geometry and displacement metrology",
)
_BLOCKED_SPECIALIZED_TERMS = (
    "lpsd_defocusing",
    "tube_tail",
    "continuum",
    "coupled_dispersion",
)


@dataclass(frozen=True, slots=True)
class CeriaPositionFit:
    """Cubic cell plus constant two-theta nuisance inferred from peak centroids."""

    lattice_angstrom: float
    zero_shift_deg: float
    reflection_count: int
    centroid_rms_error_deg: float
    centroid_max_error_deg: float


@dataclass(frozen=True, slots=True)
class CeriaProfileStart:
    """One dispersed-start calibration fit."""

    starting_fwhm_deg: float
    active_parameters: tuple[str, ...]
    weighted_rank: int
    free_parameter_count: int
    maximum_absolute_correlation: float
    u_deg2: float
    v_deg2: float
    w_deg2: float
    x_deg: float
    y_deg: float
    poisson_rwp: float
    profile_correlation: float
    termination_reason: str


@dataclass(frozen=True, slots=True)
class CeriaBroadeningStart:
    """One dispersed-start fit of the pre-existing sample-width model."""

    starting_size_nm: float
    starting_rms_microstrain: float
    crystallite_size_nm: float
    rms_microstrain: float
    poisson_rwp: float
    profile_correlation: float


@dataclass(frozen=True, slots=True)
class CeriaPeakMetric:
    """Local observed/calculated shape comparison for one cubic reflection position."""

    squared_hkl_norm: int
    two_theta_deg: float
    observed_area: float
    calculated_area: float
    relative_area_error: float
    observed_centroid_deg: float
    calculated_centroid_deg: float
    centroid_error_deg: float
    observed_fwhm_deg: float
    calculated_fwhm_deg: float
    observed_rms_width_deg: float
    calculated_rms_width_deg: float
    rms_width_error_deg: float
    normalized_l1: float


@dataclass(frozen=True, slots=True)
class CeriaTransferabilityResult:
    """Machine-readable result of the external empirical transferability gate."""

    dataset_id: str
    status: str
    source_provenance: dict[str, Any]
    calibration_sample_count: int
    holdout_sample_count: int
    calibration_position: CeriaPositionFit
    holdout_position: CeriaPositionFit
    profile_starts: tuple[CeriaProfileStart, ...]
    frozen_instrument: dict[str, float]
    calibration_poisson_rwp: float
    calibration_profile_correlation: float
    calibration_relative_width_spread: float
    calibration_peak_metrics: tuple[CeriaPeakMetric, ...]
    holdout_base_poisson_rwp: float
    holdout_base_profile_correlation: float
    broadening_starts: tuple[CeriaBroadeningStart, ...]
    holdout_poisson_rwp: float
    holdout_profile_correlation: float
    holdout_peak_metrics: tuple[CeriaPeakMetric, ...]
    weighted_sse_improvement_fraction: float
    broadening_rwp_spread: float
    broadening_size_spread_nm: float
    broadening_microstrain_spread: float
    poisson_resampling_count: int
    poisson_improvement_standard_deviation: float
    poisson_improvement_maximum_deviation: float
    instrument_configuration: dict[str, Any]
    range_semantics: str
    promotion_decision: str
    blocked_specialized_terms: tuple[str, ...]
    missing_physical_inputs: tuple[str, ...]
    qualifications: tuple[str, ...]
    checks: dict[str, bool]
    conclusion: str

    def __post_init__(self) -> None:
        if self.dataset_id != DATASET_ID or self.status not in {"passed", "failed"}:
            raise ValueError("invalid ceria transferability identity or status")
        _require_finite(self.to_record())

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-ready record containing no NaN or infinity."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PreparedPattern:
    pattern: PowderPattern
    range_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BroadeningEvaluation:
    size_nm: float | None
    rms_microstrain: float
    rwp: float
    chi_square: float
    correlation: float


def _require_finite(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("ceria transferability record must contain only finite values")


def _read_two_column_range(path: Path) -> NDArray[np.float64]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            x_value, count = (float(field) for field in fields)
        except ValueError:
            continue
        rows.append((x_value, count))
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 100:
        raise ValueError(f"{path.name} does not contain the expected two-column scan range")
    if (
        not np.isfinite(values).all()
        or np.any(values[:, 1] < 0.0)
        or np.any(np.diff(values[:, 0]) <= 0.0)
    ):
        raise ValueError(f"{path.name} contains invalid coordinates or counts")
    step = np.diff(values[:, 0])
    if not np.allclose(step, step[0], rtol=0.0, atol=2.0e-10):
        raise ValueError(f"{path.name} is not one uniformly stepped scan range")
    return np.ascontiguousarray(values)


def _prepare_pattern(
    root: Path,
    filenames: tuple[str, ...],
    *,
    smooth_width_deg: float,
) -> _PreparedPattern:
    x_parts: list[NDArray[np.float64]] = []
    y_parts: list[NDArray[np.float64]] = []
    background_parts: list[NDArray[np.float64]] = []
    range_lengths = []
    previous_end: float | None = None
    for filename in filenames:
        values = _read_two_column_range(root / filename)
        background = SmoothBrucknerBackground(
            smooth_width=smooth_width_deg,
            iterations=50,
            chebyshev_order=30,
        ).estimate(values[:, 0], values[:, 1])
        keep = np.ones(values.shape[0], dtype=np.bool_)
        if previous_end is not None:
            if values[0, 0] != previous_end:
                raise ValueError("adjacent ceria scan ranges must share exactly one boundary")
            # The adjacent ranges have different steps/count scales.  Keep the
            # closing observation of the earlier range and discard the duplicate
            # opening coordinate of the later range.
            keep[0] = False
        selected = values[keep]
        selected_background = background[keep]
        x_parts.append(np.ascontiguousarray(selected[:, 0]))
        y_parts.append(np.ascontiguousarray(selected[:, 1]))
        background_parts.append(np.ascontiguousarray(selected_background))
        range_lengths.append(int(selected.shape[0]))
        previous_end = float(values[-1, 0])
    x = np.ascontiguousarray(np.concatenate(x_parts))
    observed = np.ascontiguousarray(np.concatenate(y_parts))
    fixed_background = np.ascontiguousarray(np.concatenate(background_parts))
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=np.sqrt(np.maximum(observed, 1.0)),
        background=fixed_background,
    )
    return _PreparedPattern(pattern, tuple(range_lengths))


def _ceria_cif(lattice_angstrom: float) -> str:
    return f"""data_ceria
_chemical_formula_sum 'Ce O2'
_cell_length_a {lattice_angstrom:.15g}
_cell_length_b {lattice_angstrom:.15g}
_cell_length_c {lattice_angstrom:.15g}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Ce1 Ce 0 0 0 1
O1 O 0.25 0.25 0.25 1
"""


def _phase_for_pattern(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    position: CeriaPositionFit,
) -> lebail.LeBailPhase:
    request = lebail.LeBailInput.from_cif(
        pattern,
        instrument,
        _ceria_cif(position.lattice_angstrom),
        phase_id="ceria",
        strict=False,
        refine_lattice=False,
    )
    phase = request.phases[0]
    if not isinstance(phase, lebail.LeBailPhase):  # pragma: no cover - constructor contract
        raise RuntimeError("ceria phase construction did not return a LeBailPhase")
    reflections = phase.reflections
    shifted = ReflectionBatch(
        list(reflections.reflection_ids),
        reflections.hkl,
        reflections.d_spacing_angstrom,
        reflections.two_theta_deg + position.zero_shift_deg,
        reflections.integrated_intensity,
    )
    return replace(phase, reflections=shifted)


def _fit_position(pattern: PowderPattern, *, centroid_window_deg: float) -> CeriaPositionFit:
    seed = starting_profile_from_fwhm(WAVELENGTH_1_ANGSTROM, 0.10)
    request = lebail.LeBailInput.from_cif(
        pattern,
        seed,
        _ceria_cif(REFERENCE_CELL_ANGSTROM),
        phase_id="ceria",
        strict=False,
        refine_lattice=False,
    )
    reflections = request.phases[0].reflections
    squared_norms: list[int] = []
    for hkl in reflections.hkl:
        squared_norm = int(np.dot(hkl, hkl))
        if squared_norm not in squared_norms:
            squared_norms.append(squared_norm)
    observed_centroids = []
    used_squared_norms = []
    peak_only = pattern.observed_y - pattern.background
    for squared_norm in squared_norms:
        argument = WAVELENGTH_1_ANGSTROM * math.sqrt(squared_norm) / (2.0 * REFERENCE_CELL_ANGSTROM)
        predicted = math.degrees(2.0 * math.asin(argument))
        selected = np.abs(pattern.x - predicted) <= centroid_window_deg
        weights = np.maximum(peak_only[selected], 0.0)
        if weights.size < 3 or float(np.sum(weights)) <= 0.0:
            continue
        observed_centroids.append(float(np.sum(pattern.x[selected] * weights) / np.sum(weights)))
        used_squared_norms.append(squared_norm)
    if len(used_squared_norms) < 8:
        raise ValueError("too few ceria reflections support the position nuisance fit")
    squared = np.asarray(used_squared_norms, dtype=np.float64)
    observed = np.asarray(observed_centroids, dtype=np.float64)
    lattice = REFERENCE_CELL_ANGSTROM
    zero = 0.0
    for _ in range(12):
        argument = WAVELENGTH_1_ANGSTROM * np.sqrt(squared) / (2.0 * lattice)
        predicted = np.degrees(2.0 * np.arcsin(argument)) + zero
        derivative_lattice = (
            -(360.0 / np.pi) * argument / (lattice * np.sqrt(1.0 - np.square(argument)))
        )
        jacobian = np.column_stack((derivative_lattice, np.ones_like(derivative_lattice)))
        step, *_ = np.linalg.lstsq(jacobian, observed - predicted, rcond=None)
        lattice += float(step[0])
        zero += float(step[1])
    argument = WAVELENGTH_1_ANGSTROM * np.sqrt(squared) / (2.0 * lattice)
    residual = observed - (np.degrees(2.0 * np.arcsin(argument)) + zero)
    return CeriaPositionFit(
        lattice_angstrom=lattice,
        zero_shift_deg=zero,
        reflection_count=len(used_squared_norms),
        centroid_rms_error_deg=float(np.sqrt(np.mean(np.square(residual)))),
        centroid_max_error_deg=float(np.max(np.abs(residual))),
    )


def _profile_correlation(pattern: PowderPattern, profile_y: NDArray[np.float64]) -> float:
    correlation = float(np.corrcoef(pattern.observed_y - pattern.background, profile_y)[0, 1])
    if not math.isfinite(correlation):
        raise ValueError("profile correlation is not finite")
    return correlation


def _sampled_fwhm(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    maximum_index = int(np.argmax(y))
    half_height = 0.5 * float(y[maximum_index])
    above = np.flatnonzero(y >= half_height)
    if above.size == 0 or above[0] == 0 or above[-1] == y.size - 1:
        raise ValueError("peak window does not bracket its sampled half maximum")

    def crossing(left: int, right: int) -> float:
        fraction = (half_height - y[left]) / (y[right] - y[left])
        return float(x[left] + fraction * (x[right] - x[left]))

    left = crossing(int(above[0] - 1), int(above[0]))
    right = crossing(int(above[-1]), int(above[-1] + 1))
    return right - left


def _peak_metrics(
    pattern: PowderPattern,
    phase: lebail.LeBailPhase,
    profile_y: NDArray[np.float64],
    *,
    half_window_deg: float,
) -> tuple[CeriaPeakMetric, ...]:
    observed_profile = np.maximum(pattern.observed_y - pattern.background, 0.0)
    metrics = []
    seen_positions: set[int] = set()
    for hkl, position in zip(
        phase.reflections.hkl,
        phase.reflections.two_theta_deg,
        strict=True,
    ):
        squared_norm = int(np.dot(hkl, hkl))
        if squared_norm in seen_positions:
            continue
        seen_positions.add(squared_norm)
        selected = np.abs(pattern.x - position) <= half_window_deg
        x = pattern.x[selected]
        observed = observed_profile[selected]
        calculated = profile_y[selected]
        if x.size < 5:
            raise ValueError("ceria peak metric window contains too few samples")
        observed_area = float(np.trapezoid(observed, x))
        calculated_area = float(np.trapezoid(calculated, x))
        if observed_area <= 0.0 or calculated_area <= 0.0:
            raise ValueError("ceria peak metric requires positive observed/calculated area")
        observed_centroid = float(np.trapezoid(x * observed, x) / observed_area)
        calculated_centroid = float(np.trapezoid(x * calculated, x) / calculated_area)
        observed_rms = float(
            np.sqrt(np.trapezoid(np.square(x - observed_centroid) * observed, x) / observed_area)
        )
        calculated_rms = float(
            np.sqrt(
                np.trapezoid(np.square(x - calculated_centroid) * calculated, x) / calculated_area
            )
        )
        metrics.append(
            CeriaPeakMetric(
                squared_hkl_norm=squared_norm,
                two_theta_deg=float(position),
                observed_area=observed_area,
                calculated_area=calculated_area,
                relative_area_error=(calculated_area - observed_area) / observed_area,
                observed_centroid_deg=observed_centroid,
                calculated_centroid_deg=calculated_centroid,
                centroid_error_deg=calculated_centroid - observed_centroid,
                observed_fwhm_deg=_sampled_fwhm(x, observed),
                calculated_fwhm_deg=_sampled_fwhm(x, calculated),
                observed_rms_width_deg=observed_rms,
                calculated_rms_width_deg=calculated_rms,
                rms_width_error_deg=calculated_rms - observed_rms,
                normalized_l1=float(np.trapezoid(np.abs(observed - calculated), x) / observed_area),
            )
        )
    return tuple(metrics)


def _calibrate_profile(
    pattern: PowderPattern,
    position: CeriaPositionFit,
) -> tuple[tuple[CeriaProfileStart, ...], ConstantWavelengthInstrument]:
    results = []
    instruments = []
    for starting_fwhm in _PROFILE_STARTS_DEG:
        instrument = starting_profile_from_fwhm(WAVELENGTH_1_ANGSTROM, starting_fwhm)
        phase = _phase_for_pattern(pattern, instrument, position)
        result = estimate_effective_profile(
            lebail.LeBailInput(pattern, instrument, (phase,)),
            ProfileEstimationOptions(
                mode=ProfileEstimationMode.AUTOMATIC,
                lebail=lebail.LeBailOptions(
                    max_iterations=100,
                    min_iterations=4,
                    support_fwhm=30.0,
                    max_scaled_parameter_step=0.30,
                    use_uncertainty=True,
                    diagnose_rank_deficiency=True,
                ),
            ),
        )
        fitted = result.instrument
        accepted_profile_stage = next(
            stage
            for stage in reversed(result.stages)
            if stage.accepted and stage.instrument_parameters
        )
        maximum_correlation = accepted_profile_stage.maximum_absolute_correlation
        if maximum_correlation is None:  # pragma: no cover - accepted fit is diagnosed
            raise RuntimeError("accepted ceria profile fit did not report correlation diagnostics")
        instruments.append(fitted)
        results.append(
            CeriaProfileStart(
                starting_fwhm_deg=starting_fwhm,
                active_parameters=result.active_parameters,
                weighted_rank=len(result.active_parameters),
                free_parameter_count=len(result.active_parameters),
                maximum_absolute_correlation=maximum_correlation,
                u_deg2=fitted.u_deg2,
                v_deg2=fitted.v_deg2,
                w_deg2=fitted.w_deg2,
                x_deg=fitted.x_deg,
                y_deg=fitted.y_deg,
                poisson_rwp=float(result.rwp),
                profile_correlation=_profile_correlation(
                    pattern, result.calculated_y - pattern.background
                ),
                termination_reason=result.termination_reason,
            )
        )
    return tuple(results), ConstantWavelengthInstrument(
        WAVELENGTH_1_ANGSTROM,
        float(np.median([item.u_deg2 for item in instruments])),
        float(np.median([item.v_deg2 for item in instruments])),
        float(np.median([item.w_deg2 for item in instruments])),
        float(np.median([item.x_deg for item in instruments])),
        float(np.median([item.y_deg for item in instruments])),
    )


def _broadening_physics(size_nm: float | None, rms_microstrain: float) -> object | None:
    providers: list[object] = []
    if size_nm is not None:
        providers.append(IsotropicSizeBroadening(size_nm, shape_factor=1.0))
    if rms_microstrain > 0.0:
        providers.append(IsotropicMicrostrainBroadening(rms_microstrain))
    if not providers:
        return None
    if len(providers) == 1:
        return providers[0]
    return CompositePhysicsProvider(tuple(providers))


def _evaluate_broadening(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    phase: lebail.LeBailPhase,
    size_nm: float | None,
    rms_microstrain: float,
    *,
    max_iterations: int = 60,
) -> _BroadeningEvaluation:
    result = lebail.refine(
        lebail.LeBailInput(
            pattern,
            instrument,
            (replace(phase, physics=_broadening_physics(size_nm, rms_microstrain)),),
        ),
        lebail.LeBailOptions(
            max_iterations=max_iterations,
            min_iterations=4,
            support_fwhm=30.0,
            use_uncertainty=True,
        ),
    )
    return _BroadeningEvaluation(
        size_nm=size_nm,
        rms_microstrain=rms_microstrain,
        rwp=float(result.metrics.rwp),
        chi_square=float(result.metrics.chi_square),
        correlation=_profile_correlation(pattern, result.calculation.profile_y),
    )


def _fixed_lebail_result(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    phase: lebail.LeBailPhase,
    size_nm: float | None,
    rms_microstrain: float,
) -> lebail.LeBailResult:
    return lebail.refine(
        lebail.LeBailInput(
            pattern,
            instrument,
            (replace(phase, physics=_broadening_physics(size_nm, rms_microstrain)),),
        ),
        lebail.LeBailOptions(
            max_iterations=100,
            min_iterations=4,
            support_fwhm=30.0,
            use_uncertainty=True,
        ),
    )


def _fit_broadening_start(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    phase: lebail.LeBailPhase,
    start: tuple[float, float],
) -> _BroadeningEvaluation:
    cache: dict[tuple[float, float], _BroadeningEvaluation] = {}

    def evaluate(log_size: float, microstrain: float) -> _BroadeningEvaluation:
        size = float(np.exp(log_size))
        strain = float(max(microstrain, 0.0))
        key = (round(size, 12), round(strain, 14))
        if key not in cache:
            cache[key] = _evaluate_broadening(pattern, instrument, phase, size, strain)
        return cache[key]

    log_size = math.log(start[0])
    microstrain = start[1]
    log_step = math.log(1.8)
    strain_step = 0.001
    for _ in range(22):
        candidates = []
        for size_direction in (-1.0, 0.0, 1.0):
            for strain_direction in (-1.0, 0.0, 1.0):
                candidate_log_size = float(
                    np.clip(
                        log_size + size_direction * log_step,
                        math.log(5.0),
                        math.log(200.0),
                    )
                )
                candidate_strain = float(
                    np.clip(microstrain + strain_direction * strain_step, 0.0, 0.005)
                )
                candidate = evaluate(candidate_log_size, candidate_strain)
                candidates.append((candidate.rwp, candidate_log_size, candidate_strain, candidate))
        _, next_log_size, next_strain, best = min(candidates, key=lambda item: item[:3])
        moved = abs(next_log_size - log_size) > 1.0e-14 or abs(next_strain - microstrain) > 1.0e-16
        log_size = next_log_size
        microstrain = next_strain
        if not moved:
            log_step *= 0.5
            strain_step *= 0.5
    return best


def _poisson_variation(
    pattern: PowderPattern,
    instrument: ConstantWavelengthInstrument,
    phase: lebail.LeBailPhase,
    best: _BroadeningEvaluation,
    original_improvement: float,
    *,
    count: int = 12,
) -> tuple[float, float]:
    generator = np.random.default_rng(20260814)
    improvements = []
    for _ in range(count):
        observed = np.ascontiguousarray(generator.poisson(pattern.observed_y).astype(np.float64))
        resampled = PowderPattern(
            pattern.x,
            observed_y=observed,
            uncertainty=np.sqrt(np.maximum(observed, 1.0)),
            background=pattern.background,
        )
        base = _evaluate_broadening(resampled, instrument, phase, None, 0.0, max_iterations=50)
        candidate = _evaluate_broadening(
            resampled,
            instrument,
            phase,
            best.size_nm,
            best.rms_microstrain,
            max_iterations=50,
        )
        improvements.append(1.0 - candidate.chi_square / base.chi_square)
    values = np.asarray(improvements, dtype=np.float64)
    return float(np.std(values, ddof=1)), float(np.max(np.abs(values - original_improvement)))


def run_ceria_profile_transferability(
    dataset_directory: str | Path,
) -> CeriaTransferabilityResult:
    """Run the frozen calibration-only/frozen-profile holdout protocol."""

    root = Path(dataset_directory)
    dataset = validation_dataset(DATASET_ID)
    verify_validation_dataset(DATASET_ID, root)
    calibration = _prepare_pattern(root, _SHARP_FILES, smooth_width_deg=0.5)
    calibration_position = _fit_position(calibration.pattern, centroid_window_deg=0.25)
    profile_starts, frozen_instrument = _calibrate_profile(
        calibration.pattern, calibration_position
    )
    calibration_phase = _phase_for_pattern(
        calibration.pattern, frozen_instrument, calibration_position
    )
    calibration_refined = _fixed_lebail_result(
        calibration.pattern,
        frozen_instrument,
        calibration_phase,
        None,
        0.0,
    )
    calibration_peak_metrics = _peak_metrics(
        calibration.pattern,
        calibration_phase,
        calibration_refined.calculation.profile_y,
        half_window_deg=0.35,
    )
    # Do not read or preprocess the holdout until the instrument profile has
    # been calibrated and frozen.
    holdout = _prepare_pattern(root, _BROAD_FILES, smooth_width_deg=1.0)
    holdout_position = _fit_position(holdout.pattern, centroid_window_deg=0.50)
    phase = _phase_for_pattern(holdout.pattern, frozen_instrument, holdout_position)
    base = _evaluate_broadening(holdout.pattern, frozen_instrument, phase, None, 0.0)
    fitted = tuple(
        _fit_broadening_start(holdout.pattern, frozen_instrument, phase, start)
        for start in _BROADENING_STARTS
    )
    best = min(fitted, key=lambda item: (item.rwp, item.size_nm or math.inf, item.rms_microstrain))
    if best.size_nm is None:  # pragma: no cover - optimizer always fits finite size
        raise RuntimeError("ceria broadened specimen fit did not retain finite size")
    holdout_refined = _fixed_lebail_result(
        holdout.pattern,
        frozen_instrument,
        phase,
        best.size_nm,
        best.rms_microstrain,
    )
    holdout_peak_metrics = _peak_metrics(
        holdout.pattern,
        phase,
        holdout_refined.calculation.profile_y,
        half_window_deg=0.80,
    )
    weighted_sse_improvement = 1.0 - best.chi_square / base.chi_square
    bootstrap_standard_deviation, bootstrap_maximum_deviation = _poisson_variation(
        holdout.pattern,
        frozen_instrument,
        phase,
        best,
        weighted_sse_improvement,
    )
    profile_widths = np.asarray([item.w_deg2 for item in profile_starts])
    profile_rwp = np.asarray([item.poisson_rwp for item in profile_starts])
    broadening_rwp = np.asarray([item.rwp for item in fitted])
    broadening_size = np.asarray([item.size_nm for item in fitted], dtype=np.float64)
    broadening_strain = np.asarray([item.rms_microstrain for item in fitted])
    relative_width_spread = float(np.ptp(profile_widths) / np.median(profile_widths))
    checks = {
        "calibration_position_residual": calibration_position.centroid_max_error_deg <= 0.005,
        "holdout_position_residual": holdout_position.centroid_max_error_deg <= 0.015,
        "calibration_profile_quality": bool(
            np.max(profile_rwp) <= 0.20
            and min(item.profile_correlation for item in profile_starts) >= 0.94
        ),
        "calibration_profile_identifiable": all(
            item.weighted_rank == item.free_parameter_count
            and item.maximum_absolute_correlation <= 0.98
            for item in profile_starts
        ),
        "calibration_start_stability": relative_width_spread <= 1.0e-5,
        "holdout_profile_quality": best.rwp <= 0.06 and best.correlation >= 0.985,
        "broadening_start_stability": bool(
            np.ptp(broadening_rwp) <= 5.0e-5
            and np.ptp(broadening_size) <= 0.10
            and np.ptp(broadening_strain) <= 5.0e-5
        ),
        "holdout_weighted_sse_improvement": weighted_sse_improvement >= 0.90,
        "holdout_improvement_exceeds_resampling": weighted_sse_improvement
        > 3.0 * bootstrap_maximum_deviation,
        "specialized_term_inputs_complete": False,
    }
    empirical_checks = {
        key: value for key, value in checks.items() if key != "specialized_term_inputs_complete"
    }
    status = "passed" if all(empirical_checks.values()) else "failed"
    qualifications = (
        "The Birmingham files numbered 1-3 are contiguous angular ranges, not repeat scans.",
        "Poisson uncertainties are inferred as sqrt(max(counts, 1)); none are deposited.",
        "The 1.6% K-alpha2 component is documented but the current Le Bail profile estimator "
        "uses the dominant K-alpha1 wavelength, so this is an empirical baseline gate rather "
        "than a fixed-spectrum fundamental-parameters calibration.",
        "The approximate round-robin CeO2 cell and a constant two-theta nuisance are fitted "
        "from peak centroids independently for the two specimens before width fitting.",
        "The source page states no redistribution license; original bytes remain external.",
    )
    conclusion = (
        "The existing frozen empirical instrument profile plus isotropic size/microstrain "
        "transfers to the broadened ceria holdout. Missing optics metadata prevents a fair "
        "test of LPSD, tube-tail, continuum, or coupled-dispersion terms, so none is promoted."
    )
    return CeriaTransferabilityResult(
        dataset_id=DATASET_ID,
        status=status,
        source_provenance={
            "title": dataset.title,
            "source_url": dataset.source_url,
            "citation": dataset.citation,
            "license_note": dataset.license_note,
            "files": tuple(
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "role": "calibration" if item.name in _SHARP_FILES else "untouched_holdout",
                }
                for item in dataset.files
            ),
        },
        calibration_sample_count=calibration.pattern.x.size,
        holdout_sample_count=holdout.pattern.x.size,
        calibration_position=calibration_position,
        holdout_position=holdout_position,
        profile_starts=profile_starts,
        frozen_instrument={
            "wavelength_angstrom": frozen_instrument.wavelength_angstrom,
            "u_deg2": frozen_instrument.u_deg2,
            "v_deg2": frozen_instrument.v_deg2,
            "w_deg2": frozen_instrument.w_deg2,
            "x_deg": frozen_instrument.x_deg,
            "y_deg": frozen_instrument.y_deg,
        },
        calibration_poisson_rwp=float(np.median(profile_rwp)),
        calibration_profile_correlation=float(
            np.median([item.profile_correlation for item in profile_starts])
        ),
        calibration_relative_width_spread=relative_width_spread,
        calibration_peak_metrics=calibration_peak_metrics,
        holdout_base_poisson_rwp=base.rwp,
        holdout_base_profile_correlation=base.correlation,
        broadening_starts=tuple(
            CeriaBroadeningStart(
                starting_size_nm=start[0],
                starting_rms_microstrain=start[1],
                crystallite_size_nm=float(result.size_nm),
                rms_microstrain=result.rms_microstrain,
                poisson_rwp=result.rwp,
                profile_correlation=result.correlation,
            )
            for start, result in zip(_BROADENING_STARTS, fitted, strict=True)
        ),
        holdout_poisson_rwp=best.rwp,
        holdout_profile_correlation=best.correlation,
        holdout_peak_metrics=holdout_peak_metrics,
        weighted_sse_improvement_fraction=weighted_sse_improvement,
        broadening_rwp_spread=float(np.ptp(broadening_rwp)),
        broadening_size_spread_nm=float(np.ptp(broadening_size)),
        broadening_microstrain_spread=float(np.ptp(broadening_strain)),
        poisson_resampling_count=12,
        poisson_improvement_standard_deviation=bootstrap_standard_deviation,
        poisson_improvement_maximum_deviation=bootstrap_maximum_deviation,
        instrument_configuration={
            "source": "sealed-tube laboratory X-ray",
            "incident_monochromator": "Ge(111)",
            "wavelengths_angstrom": (WAVELENGTH_1_ANGSTROM, WAVELENGTH_2_ANGSTROM),
            "k_alpha2_over_k_alpha1": K_ALPHA2_OVER_K_ALPHA1,
            "polarization_factor_p": POLARIZATION_FACTOR_P,
        },
        range_semantics=(
            "three contiguous ranges per specimen with steps 0.01/0.02/0.02 degree "
            "for sharp and 0.02/0.04/0.05 degree for broadened; duplicated range "
            "boundaries retain the earlier range observation"
        ),
        promotion_decision="no_specialized_term_promoted",
        blocked_specialized_terms=_BLOCKED_SPECIALIZED_TERMS,
        missing_physical_inputs=_UNKNOWN_OPTICS,
        qualifications=qualifications,
        checks=checks,
        conclusion=conclusion,
    )
