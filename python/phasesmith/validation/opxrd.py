"""Checksum-pinned opXRD robustness and common-model diagnostics.

The campaign deliberately separates archive/schema robustness, model-independent
background and peak metrics, lattice-label diagnostics, and a small structural
common-model subset.  Missing experimental metadata never becomes an acceptance
target.  The full archive remains an opt-in external input.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..background import SmoothBrucknerBackground
from ..execution import ExecutionPolicy
from ..extensions import CompositePhysicsProvider
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..intensity_corrections import BraggBrentanoPolarizedLp
from ..pattern import PowderPattern
from ..radiation import ConstantWavelengthExperiment, WavelengthComponents
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ..sample import IsotropicLorentzianMicrostrainBroadening, IsotropicSizeBroadening
from ..symmetry import CwTwoThetaRange, PreparedReflectionGenerator
from ._rietveld_parity import refine_nonlinear_block, solve_linear_profile_block
from .real_data import _qarr_initial_scales

MetadataClass = Literal["full_structure", "partial_structure", "unlabeled"]
CaseRole = Literal[
    "structural_common_model",
    "lattice_diagnostic",
    "metadata_boundary",
    "unsupported_geometry",
    "negative_intensity",
    "robustness_only",
]

_EXPECTED_ARCHIVE_KEYS = {
    "filename",
    "member_count",
    "sha256",
    "size_bytes",
    "zenodo_record",
}
_EXPECTED_CASE_KEYS = {
    "case_id",
    "expected_findings",
    "member_path",
    "member_sha256",
    "metadata_class",
    "role",
    "source_group",
}
_METADATA_CLASSES = {"full_structure", "partial_structure", "unlabeled"}
_CASE_ROLES = {
    "structural_common_model",
    "lattice_diagnostic",
    "metadata_boundary",
    "unsupported_geometry",
    "negative_intensity",
    "robustness_only",
}
_ELEMENT = re.compile(r"^([A-Z][a-z]?)")


@dataclass(frozen=True, slots=True)
class OpxrdCaseSpec:
    """One immutable selected archive member and its declared scientific role."""

    case_id: str
    member_path: str
    member_sha256: str
    source_group: str
    metadata_class: MetadataClass
    role: CaseRole
    expected_findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpxrdResidualMetrics:
    """Complementary residual measures with explicit Poisson applicability."""

    relative_l2: float
    normalized_l1: float
    cosine_similarity: float
    pearson_correlation: float | None
    poisson_rwp: float | None
    poisson_block_reason: str | None


@dataclass(frozen=True, slots=True)
class OpxrdPositionAlignment:
    """Deterministic bounded zero-shift scan followed by a local zero-only polish."""

    status: Literal["aligned", "large_offset_warning", "scan_boundary_warning"]
    scan_range_deg: tuple[float, float]
    scan_step_deg: float
    candidate_count: int
    initial_poisson_rwp: float
    coarse_zero_shift_deg: float
    coarse_poisson_rwp: float
    polished_zero_shift_deg: float
    selected_zero_shift_deg: float
    selected_poisson_rwp: float
    poisson_rwp_improvement: float


@dataclass(frozen=True, slots=True)
class OpxrdStructuralResult:
    """Finite result of one explicitly assumed common structural model."""

    sample_count: int
    reflection_count: int
    phase_count: int
    wavelengths_angstrom: tuple[float, ...]
    wavelength_order_corrected: bool
    assumed_relative_intensities: tuple[float, ...]
    position_alignment: OpxrdPositionAlignment
    profile_parameters: dict[str, float]
    maximum_gaussian_fwhm_deg: float
    parameter_plausibility: Literal["plausible", "warning"]
    residual_metrics: OpxrdResidualMetrics
    profile_correlation: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class OpxrdCaseResult:
    """One selected pattern's integrity, robustness, and optional physics diagnostics."""

    case_id: str
    member_path: str
    source_group: str
    metadata_class: MetadataClass
    role: CaseRole
    sample_count: int
    two_theta_range_deg: tuple[float, float]
    median_step_deg: float
    maximum_relative_step_deviation: float
    intensity_range: tuple[float, float]
    negative_fraction: float
    zero_fraction: float
    decoded_phase_count: int
    wavelength_count: int
    wavelength_order_corrected: bool
    phasesmith_background_status: Literal["evaluated", "rejected_nonuniform_grid"]
    background_deterministic: bool | None
    smooth_points: int | None
    bkg_to_positive_signal_area: float | None
    background_agreement_normalized_l1: float | None
    bkg_only_metrics: OpxrdResidualMetrics | None
    quantile_only_metrics: OpxrdResidualMetrics
    detected_peak_count: int
    strongest_peak_positions_deg: tuple[float, ...]
    lattice_diagnostic: dict[str, Any] | None
    structural_result: OpxrdStructuralResult | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpxrdRejectedCaseResult:
    """A selected member rejected at an explicit, lossless input boundary."""

    case_id: str
    member_path: str
    source_group: str
    metadata_class: MetadataClass
    role: CaseRole
    sample_count: int
    two_theta_range_deg: tuple[float, float]
    intensity_range: tuple[float, float]
    negative_fraction: float
    zero_fraction: float
    decoded_phase_count: int
    wavelength_count: int
    wavelength_order_corrected: bool
    rejection_status: Literal["rejected_nonmonotonic_grid"]
    nonincreasing_step_count: int
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpxrdCampaignResult:
    """Stable, finite aggregate for a pinned selection manifest."""

    schema_version: int
    dataset_id: str
    archive_sha256: str
    selection_sha256: str
    archive_member_count: int
    selected_case_count: int
    source_group_count: int
    metadata_class_counts: dict[str, int]
    role_counts: dict[str, int]
    checks: dict[str, bool]
    cases: tuple[OpxrdCaseResult | OpxrdRejectedCaseResult, ...]
    elapsed_seconds: float
    interpretation: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-ready record without NumPy or path objects."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _DecodedPattern:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    label: dict[str, Any]
    metadata: dict[str, Any]
    phases: tuple[dict[str, Any], ...]
    xray: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validate_procedure(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "background",
        "independent_baseline",
        "peak_detection",
        "structural_common_model",
    }:
        raise ValueError("opXRD procedure fields are invalid")
    background = value["background"]
    if not isinstance(background, dict) or set(background) != {
        "chebyshev_order",
        "iterations",
        "smooth_width_deg",
    }:
        raise ValueError("opXRD background procedure fields are invalid")
    if (
        background["chebyshev_order"] is not None
        or not isinstance(background["iterations"], int)
        or isinstance(background["iterations"], bool)
        or background["iterations"] <= 0
        or _number(background["smooth_width_deg"], "smooth_width_deg") <= 0.0
    ):
        raise ValueError("opXRD background procedure values are invalid")
    baseline = value["independent_baseline"]
    if not isinstance(baseline, dict) or set(baseline) != {"bin_count", "quantile"}:
        raise ValueError("opXRD independent-baseline fields are invalid")
    if (
        not isinstance(baseline["bin_count"], int)
        or isinstance(baseline["bin_count"], bool)
        or baseline["bin_count"] < 2
        or not 0.0 <= _number(baseline["quantile"], "quantile") <= 1.0
    ):
        raise ValueError("opXRD independent-baseline values are invalid")
    peaks = value["peak_detection"]
    if not isinstance(peaks, dict) or set(peaks) != {
        "minimum_separation_deg",
        "noise_multiplier",
        "relative_height_floor",
    }:
        raise ValueError("opXRD peak-detection fields are invalid")
    if (
        _number(peaks["minimum_separation_deg"], "minimum_separation_deg") <= 0.0
        or _number(peaks["noise_multiplier"], "noise_multiplier") < 0.0
        or not 0.0 <= _number(peaks["relative_height_floor"], "relative_height_floor") <= 1.0
    ):
        raise ValueError("opXRD peak-detection values are invalid")
    common = value["structural_common_model"]
    if not isinstance(common, dict) or set(common) != {
        "axial_detector_over_radius",
        "axial_sample_over_radius",
        "background_terms",
        "doublet_secondary_to_primary_intensity",
        "initial_profile_deg2",
        "large_gaussian_fwhm_warning_deg",
        "large_zero_shift_warning_deg",
        "polarization_fraction",
        "support_fwhm",
        "zero_alignment_max_deg",
        "zero_alignment_min_deg",
        "zero_alignment_step_deg",
    }:
        raise ValueError("opXRD structural common-model fields are invalid")
    profile = common["initial_profile_deg2"]
    numeric_profile = (
        isinstance(profile, list)
        and len(profile) == 3
        and all(math.isfinite(_number(item, "initial_profile_deg2")) for item in profile)
    )
    if (
        not isinstance(common["background_terms"], int)
        or isinstance(common["background_terms"], bool)
        or common["background_terms"] <= 0
        or not numeric_profile
        or _number(
            common["large_gaussian_fwhm_warning_deg"],
            "large_gaussian_fwhm_warning_deg",
        )
        <= 0.0
        or _number(common["large_zero_shift_warning_deg"], "large_zero_shift_warning_deg") <= 0.0
        or _number(common["axial_sample_over_radius"], "axial_sample_over_radius") < 0.0
        or _number(common["axial_detector_over_radius"], "axial_detector_over_radius") < 0.0
        or _number(
            common["doublet_secondary_to_primary_intensity"],
            "doublet_secondary_to_primary_intensity",
        )
        <= 0.0
        or not 0.0 <= _number(common["polarization_fraction"], "polarization_fraction") <= 1.0
        or _number(common["support_fwhm"], "support_fwhm") <= 0.0
        or _number(common["zero_alignment_min_deg"], "zero_alignment_min_deg") >= 0.0
        or _number(common["zero_alignment_max_deg"], "zero_alignment_max_deg") <= 0.0
        or _number(common["zero_alignment_step_deg"], "zero_alignment_step_deg") <= 0.0
    ):
        raise ValueError("opXRD structural common-model values are invalid")
    zero_span = _number(common["zero_alignment_max_deg"], "zero_alignment_max_deg") - _number(
        common["zero_alignment_min_deg"], "zero_alignment_min_deg"
    )
    zero_step = _number(common["zero_alignment_step_deg"], "zero_alignment_step_deg")
    if zero_span / zero_step > 401:
        raise ValueError("opXRD zero-alignment scan may not exceed 401 intervals")


def _load_selection(path: Path) -> tuple[dict[str, Any], tuple[OpxrdCaseSpec, ...], str]:
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("opXRD selection manifest must be valid JSON") from error
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("opXRD selection manifest must use schema version 1")
    if set(record) != {
        "archive",
        "cases",
        "dataset_id",
        "procedure",
        "schema_version",
        "selection_policy",
    }:
        raise ValueError("opXRD selection manifest has unexpected top-level fields")
    if (
        not isinstance(record["dataset_id"], str)
        or not record["dataset_id"].replace("-", "").isalnum()
        or not isinstance(record["selection_policy"], dict)
        or not record["selection_policy"]
    ):
        raise ValueError("opXRD dataset ID or selection policy is invalid")
    _validate_procedure(record["procedure"])
    archive = record["archive"]
    if not isinstance(archive, dict) or set(archive) != _EXPECTED_ARCHIVE_KEYS:
        raise ValueError("opXRD archive manifest fields are invalid")
    _hex_digest(archive["sha256"], "archive.sha256")
    if (
        archive["filename"] != "opxrd.zip"
        or not isinstance(archive["size_bytes"], int)
        or archive["size_bytes"] <= 0
        or not isinstance(archive["member_count"], int)
        or archive["member_count"] <= 0
        or not isinstance(archive["zenodo_record"], int)
        or archive["zenodo_record"] <= 0
    ):
        raise ValueError("opXRD archive manifest values are invalid")
    cases_raw = record["cases"]
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("opXRD selection must contain cases")
    cases = []
    for item in cases_raw:
        if not isinstance(item, dict) or set(item) != _EXPECTED_CASE_KEYS:
            raise ValueError("opXRD case manifest fields are invalid")
        case_id = item["case_id"]
        member_path = item["member_path"]
        source_group = item["source_group"]
        findings = item["expected_findings"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or not case_id.replace("-", "").isalnum()
            or not isinstance(member_path, str)
            or Path(member_path).is_absolute()
            or ".." in Path(member_path).parts
            or not member_path.endswith(".json")
            or not isinstance(source_group, str)
            or member_path.split("/", 1)[0] != source_group
            or item["metadata_class"] not in _METADATA_CLASSES
            or item["role"] not in _CASE_ROLES
            or not isinstance(findings, list)
            or not findings
            or any(not isinstance(value, str) or not value for value in findings)
        ):
            raise ValueError(f"invalid opXRD case manifest {case_id!r}")
        cases.append(
            OpxrdCaseSpec(
                case_id,
                member_path,
                _hex_digest(item["member_sha256"], "member_sha256"),
                source_group,
                item["metadata_class"],
                item["role"],
                tuple(findings),
            )
        )
    if len({case.case_id for case in cases}) != len(cases) or len(
        {case.member_path for case in cases}
    ) != len(cases):
        raise ValueError("opXRD case IDs and member paths must be unique")
    return record, tuple(cases), hashlib.sha256(raw).hexdigest()


def _decode_embedded_json(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"opXRD {name} must be an encoded JSON object")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"opXRD {name} contains invalid embedded JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"opXRD {name} must decode to an object")
    return decoded


def _decode_pattern(payload: bytes) -> _DecodedPattern:
    try:
        outer = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("opXRD member contains invalid JSON") from error
    if not isinstance(outer, dict) or set(outer) != {
        "two_theta_values",
        "intensities",
        "label",
        "metadata",
    }:
        raise ValueError("opXRD member has unexpected fields")
    x = np.ascontiguousarray(outer["two_theta_values"], dtype=np.float64)
    y = np.ascontiguousarray(outer["intensities"], dtype=np.float64)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or x.size < 50
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("opXRD pattern arrays must be finite, aligned one-dimensional vectors")
    label = _decode_embedded_json(outer["label"], "label")
    metadata = _decode_embedded_json(outer["metadata"], "metadata")
    phase_values = label.get("phases")
    if not isinstance(phase_values, list) or not phase_values:
        raise ValueError("opXRD label must contain at least one encoded phase")
    phases = tuple(_decode_embedded_json(value, "phase") for value in phase_values)
    xray = _decode_embedded_json(label.get("xray_info"), "xray_info")
    x.flags.writeable = False
    y.flags.writeable = False
    return _DecodedPattern(x, y, label, metadata, phases, xray)


def _quantile_background(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bins: int,
    q: float,
) -> NDArray[np.float64]:
    if bins < 2 or not 0.0 <= q <= 1.0:
        raise ValueError("independent baseline settings are invalid")
    edges = np.linspace(0, x.size, min(bins, x.size), endpoint=False, dtype=np.int64)
    edges = np.append(edges, x.size)
    centers = []
    levels = []
    for start, stop in itertools.pairwise(edges):
        if stop <= start:
            continue
        centers.append(float(np.mean(x[start:stop])))
        levels.append(float(np.quantile(y[start:stop], q)))
    result = np.interp(x, np.asarray(centers), np.asarray(levels))
    return np.ascontiguousarray(result)


def _trapezoid(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """Integrate without depending on the NumPy 2-only ``trapezoid`` alias."""

    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _residual_metrics(
    observed: NDArray[np.float64], calculated: NDArray[np.float64]
) -> OpxrdResidualMetrics:
    residual = calculated - observed
    observed_norm = float(np.linalg.norm(observed))
    relative_l2 = float(np.linalg.norm(residual) / max(observed_norm, np.finfo(float).tiny))
    normalized_l1 = float(
        np.sum(np.abs(residual)) / max(float(np.sum(np.abs(observed))), np.finfo(float).tiny)
    )
    calculated_norm = float(np.linalg.norm(calculated))
    cosine = float(
        np.dot(observed, calculated) / max(observed_norm * calculated_norm, np.finfo(float).tiny)
    )
    if np.std(observed) > 0.0 and np.std(calculated) > 0.0:
        correlation: float | None = float(np.corrcoef(observed, calculated)[0, 1])
    else:
        correlation = None
    if np.any(observed < 0.0):
        poisson = None
        block_reason = "observed intensities contain negative values"
    elif np.any(observed > 0.0):
        sigma = np.sqrt(np.maximum(observed, 1.0))
        denominator = float(np.sum((observed / sigma) ** 2))
        poisson = float(np.sqrt(np.sum((residual / sigma) ** 2) / denominator))
        block_reason = None
    else:
        poisson = None
        block_reason = "observed intensities contain no positive signal"
    return OpxrdResidualMetrics(
        relative_l2,
        normalized_l1,
        cosine,
        correlation,
        poisson,
        block_reason,
    )


def _required_poisson_rwp(metrics: OpxrdResidualMetrics) -> float:
    if metrics.poisson_rwp is None:
        raise ValueError("structural common-model alignment requires Poisson-compatible counts")
    return metrics.poisson_rwp


def _align_zero_shift(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[rietveld.RietveldPhase, ...],
    background: ChebyshevBackground,
    execution: ExecutionPolicy,
    common: dict[str, Any],
) -> tuple[
    ConstantWavelengthExperiment,
    tuple[rietveld.RietveldPhase, ...],
    ChebyshevBackground,
    OpxrdPositionAlignment,
]:
    """Bridge non-overlapping peaks with a bounded global scan before local refinement."""

    if pattern.observed_y is None:
        raise ValueError("zero alignment requires observations")
    support_fwhm = float(common["support_fwhm"])

    def solve_at_zero(
        zero_shift_deg: float,
    ) -> tuple[
        float,
        ConstantWavelengthExperiment,
        tuple[rietveld.RietveldPhase, ...],
        ChebyshevBackground,
    ]:
        candidate_experiment = replace(experiment, zero_shift_deg=zero_shift_deg)
        candidate_phases, candidate_background = solve_linear_profile_block(
            pattern,
            candidate_experiment,
            phases,
            background,
            execution,
            support_fwhm=support_fwhm,
        )
        calculation = rietveld.calculate(
            pattern,
            candidate_experiment,
            candidate_phases,
            background=candidate_background,
            support_fwhm=support_fwhm,
            execution=execution,
        )
        score = _required_poisson_rwp(_residual_metrics(pattern.observed_y, calculation.y))
        return (
            score,
            candidate_experiment,
            candidate_phases,
            candidate_background,
        )

    minimum = float(common["zero_alignment_min_deg"])
    maximum = float(common["zero_alignment_max_deg"])
    step = float(common["zero_alignment_step_deg"])
    interval_count = round((maximum - minimum) / step)
    candidates = np.linspace(minimum, maximum, interval_count + 1)
    initial = solve_at_zero(0.0)
    evaluated = [solve_at_zero(float(value)) for value in candidates]
    coarse = min(evaluated, key=lambda item: (item[0], abs(item[1].zero_shift_deg)))
    coarse_score, coarse_experiment, coarse_phases, coarse_background = coarse
    zero_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        instrument_parameters=("zero_shift_deg",),
        background=False,
    )
    polished = refine_nonlinear_block(
        pattern,
        coarse_experiment,
        coarse_phases,
        coarse_background,
        zero_selection,
        execution,
        iterations=30,
        step=0.1,
        support_fwhm=support_fwhm,
    )
    polished_experiment = polished.experiment
    polished_phases, polished_background = solve_linear_profile_block(
        pattern,
        polished_experiment,
        polished.phases,
        coarse_background,
        execution,
        support_fwhm=support_fwhm,
    )
    polished_calculation = rietveld.calculate(
        pattern,
        polished_experiment,
        polished_phases,
        background=polished_background,
        support_fwhm=support_fwhm,
        execution=execution,
    )
    polished_score = _required_poisson_rwp(
        _residual_metrics(pattern.observed_y, polished_calculation.y)
    )
    polished_zero = polished_experiment.zero_shift_deg
    polish_is_admissible = minimum <= polished_zero <= maximum and polished_score <= coarse_score
    if polish_is_admissible:
        selected_experiment = polished_experiment
        selected_phases = polished_phases
        selected_background = polished_background
        selected_score = polished_score
    else:
        selected_experiment = coarse_experiment
        selected_phases = coarse_phases
        selected_background = coarse_background
        selected_score = coarse_score
    selected_zero = selected_experiment.zero_shift_deg
    if abs(selected_zero - minimum) <= 0.5 * step or abs(selected_zero - maximum) <= 0.5 * step:
        status: Literal["aligned", "large_offset_warning", "scan_boundary_warning"] = (
            "scan_boundary_warning"
        )
    elif abs(selected_zero) > float(common["large_zero_shift_warning_deg"]):
        status = "large_offset_warning"
    else:
        status = "aligned"
    alignment = OpxrdPositionAlignment(
        status,
        (minimum, maximum),
        step,
        int(candidates.size),
        initial[0],
        coarse_experiment.zero_shift_deg,
        coarse_score,
        polished_zero,
        selected_zero,
        selected_score,
        initial[0] - selected_score,
    )
    return selected_experiment, selected_phases, selected_background, alignment


def _detect_peaks(
    x: NDArray[np.float64],
    corrected: NDArray[np.float64],
    *,
    minimum_separation_deg: float,
    noise_multiplier: float,
    relative_height_floor: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    difference = np.diff(corrected)
    centered_difference = difference - np.median(difference)
    noise = float(np.median(np.abs(centered_difference)) / 0.954) if difference.size else 0.0
    robust_range = max(
        float(np.quantile(corrected, 0.99) - np.quantile(corrected, 0.01)),
        np.finfo(float).tiny,
    )
    threshold = max(noise_multiplier * noise, relative_height_floor * robust_range)
    indices = (
        np.flatnonzero(
            (corrected[1:-1] > corrected[:-2])
            & (corrected[1:-1] >= corrected[2:])
            & (corrected[1:-1] > threshold)
        )
        + 1
    )
    selected: list[int] = []
    for index in sorted(indices.tolist(), key=lambda value: (-corrected[value], value)):
        separated = all(
            abs(float(x[index] - x[previous])) >= minimum_separation_deg for previous in selected
        )
        if separated:
            selected.append(index)
    selected.sort()
    return x[selected], corrected[selected]


def _finite_cell(phase: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
    lengths = phase.get("lengths")
    angles = phase.get("angles")
    if (
        not isinstance(lengths, list)
        or not isinstance(angles, list)
        or len(lengths) != 3
        or len(angles) != 3
    ):
        return None
    try:
        values = tuple(float(value) for value in (*lengths, *angles))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(values).all() or any(value <= 0.0 for value in values[:3]):
        return None
    return values  # type: ignore[return-value]


def _wavelengths(xray: dict[str, Any]) -> tuple[tuple[float, ...], bool]:
    raw = [xray.get("primary_wavelength"), xray.get("secondary_wavelength")]
    values = []
    for value in raw:
        if value is None:
            continue
        converted = float(value)
        if not math.isfinite(converted) or converted <= 0.0:
            raise ValueError("opXRD wavelengths must be positive and finite")
        values.append(converted)
    if not values:
        return (), False
    corrected = len(values) == 2 and values[0] > values[1]
    return tuple(sorted(values)), corrected


def _lattice_diagnostic(
    pattern: _DecodedPattern,
    peak_positions: NDArray[np.float64],
    median_step: float,
) -> dict[str, Any] | None:
    wavelengths, order_corrected = _wavelengths(pattern.xray)
    if not wavelengths:
        return None
    predicted_batches = []
    used_phases = 0
    try:
        import gemmi

        from ..crystallography import UnitCell
        from ..io._gemmi import _space_group_from_gemmi
    except ImportError:
        return {"status": "blocked", "reason": "gemmi validation extra is unavailable"}
    for phase in pattern.phases:
        cell_values = _finite_cell(phase)
        number = phase.get("spacegroup")
        if cell_values is None or not isinstance(number, int):
            continue
        group = gemmi.find_spacegroup_by_number(number)
        if group is None:
            continue
        space_group = _space_group_from_gemmi(group)
        generator = PreparedReflectionGenerator(space_group, merge_friedel=True)
        cell = UnitCell(*cell_values)
        for wavelength in wavelengths:
            generated = generator.generate(
                cell,
                CwTwoThetaRange(float(pattern.x[0]), float(pattern.x[-1]), wavelength),
            )
            sine_theta = wavelength / (2.0 * generated.d_spacing_angstrom)
            predicted_batches.append(np.rad2deg(2.0 * np.arcsin(sine_theta)))
        used_phases += 1
    if not predicted_batches:
        return None
    predicted = np.unique(np.round(np.concatenate(predicted_batches), decimals=8))
    tolerance = max(0.1, 4.0 * median_step)
    if peak_positions.size:
        peak_delta = np.min(np.abs(peak_positions[:, None] - predicted[None, :]), axis=1)
        observed_coverage = float(np.mean(peak_delta <= tolerance))
        median_delta = float(np.median(peak_delta))
        q90_delta = float(np.quantile(peak_delta, 0.9))
        predicted_delta = np.min(np.abs(predicted[:, None] - peak_positions[None, :]), axis=1)
        predicted_coverage = float(np.mean(predicted_delta <= tolerance))
    else:
        observed_coverage = 0.0
        predicted_coverage = 0.0
        median_delta = None
        q90_delta = None
    return {
        "status": "diagnostic",
        "phase_count": used_phases,
        "predicted_position_count": int(predicted.size),
        "detected_peak_count": int(peak_positions.size),
        "match_tolerance_deg": tolerance,
        "observed_peak_coverage": observed_coverage,
        "predicted_position_coverage": predicted_coverage,
        "median_observed_to_predicted_delta_deg": median_delta,
        "q90_observed_to_predicted_delta_deg": q90_delta,
        "wavelength_order_corrected": order_corrected,
        "claim": "position-only diagnostic; unweighted predicted reflections are not ground truth",
    }


def _decode_base(phase: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = phase.get("base")
    if not isinstance(raw, str):
        return ()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(rows, list):
        return ()
    return tuple(_decode_embedded_json(value, "atom") for value in rows)


def _p1_cif(pattern: _DecodedPattern) -> str:
    if len(pattern.phases) != 1:
        raise ValueError("structural common model requires exactly one labeled phase")
    phase = pattern.phases[0]
    cell = _finite_cell(phase)
    atoms = _decode_base(phase)
    if cell is None or not atoms:
        raise ValueError("structural common model requires a finite cell and expanded atoms")
    lines = [
        "data_opxrd_common_model",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        f"_cell_length_a {cell[0]:.12g}",
        f"_cell_length_b {cell[1]:.12g}",
        f"_cell_length_c {cell[2]:.12g}",
        f"_cell_angle_alpha {cell[3]:.12g}",
        f"_cell_angle_beta {cell[4]:.12g}",
        f"_cell_angle_gamma {cell[5]:.12g}",
        "loop_",
        "_space_group_symop_operation_xyz",
        "'x,y,z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_U_iso_or_equiv",
    ]
    for index, atom in enumerate(atoms, start=1):
        match = _ELEMENT.match(str(atom.get("symbol", "")))
        try:
            xyz = tuple(float(atom[name]) for name in ("x", "y", "z"))
            occupancy = float(atom.get("occupancy", 1.0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("opXRD expanded atom is invalid") from error
        if match is None or not np.isfinite((*xyz, occupancy)).all():
            raise ValueError("opXRD expanded atom is invalid")
        element = match.group(1)
        lines.append(
            f"{element}{index} {element} {float(xyz[0]):.12g} {float(xyz[1]):.12g} "
            f"{float(xyz[2]):.12g} {float(occupancy):.12g} 0.005"
        )
    return "\n".join(lines) + "\n"


def write_opxrd_common_model_bundle(
    archive_path: str | Path,
    selection_path: str | Path,
    case_id: str,
    output_directory: str | Path,
) -> Path:
    """Write plain XYE/CIF/model files for an isolated black-box comparator."""

    archive = Path(archive_path)
    selection, cases, selection_sha256 = _load_selection(Path(selection_path))
    selected = next((case for case in cases if case.case_id == case_id), None)
    if selected is None:
        raise KeyError(f"unknown opXRD case {case_id!r}")
    if selected.role != "structural_common_model":
        raise ValueError("only structural_common_model cases can produce a bundle")
    with zipfile.ZipFile(archive) as source:
        payload = source.read(selected.member_path)
    if hashlib.sha256(payload).hexdigest() != selected.member_sha256:
        raise ValueError("selected opXRD member digest mismatch")
    pattern = _decode_pattern(payload)
    if np.any(pattern.y < 0.0):
        raise ValueError("structural common model requires non-negative observations")
    wavelengths, order_corrected = _wavelengths(pattern.xray)
    if not wavelengths:
        raise ValueError("structural common model requires a wavelength")
    common = selection["procedure"]["structural_common_model"]
    relative = (
        (1.0,)
        if len(wavelengths) == 1
        else (
            1.0,
            float(common["doublet_secondary_to_primary_intensity"]),
        )
    )
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        root / "pattern.xye",
        np.column_stack((pattern.x, pattern.y, np.sqrt(np.maximum(pattern.y, 1.0)))),
        fmt="%.12g",
    )
    (root / "phase.cif").write_text(_p1_cif(pattern), encoding="utf-8")
    model = {
        "schema_version": 1,
        "dataset_id": selection["dataset_id"],
        "selection_sha256": selection_sha256,
        "case_id": case_id,
        "member_path": selected.member_path,
        "member_sha256": selected.member_sha256,
        "wavelengths_angstrom": wavelengths,
        "wavelength_order_corrected": order_corrected,
        "relative_intensities": relative,
        "common_model": common,
        "assumptions": [
            "Expanded opXRD atoms are represented in P1 to avoid inventing an asymmetric unit.",
            "Missing displacement values are fixed at Uiso=0.005 angstrom^2.",
            "Unknown geometry uses an assumed Bragg-Brentano polarized correction.",
            "A present secondary wavelength uses an assumed 0.5 intensity ratio.",
            "A bounded global zero scan precedes local zero-only and width refinement.",
            "The fitted result is a cross-program capability diagnostic, not structural truth.",
        ],
    }
    model_path = root / "model.json"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_path


def run_opxrd_structural_common_model(
    bundle_directory: str | Path,
    *,
    cycles: int = 2,
    execution: ExecutionPolicy | None = None,
) -> OpxrdStructuralResult:
    """Run one disclosed P1-expanded common model through PhaseSmith."""

    started = perf_counter()
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    root = Path(bundle_directory)
    model = json.loads((root / "model.json").read_text(encoding="utf-8"))
    data = np.loadtxt(root / "pattern.xye")
    if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] < 50:
        raise ValueError("opXRD common-model pattern must be a three-column XYE file")
    x, observed, uncertainty = np.ascontiguousarray(data.T)
    common = model["common_model"]
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=uncertainty,
        background=np.zeros_like(observed),
    )
    wavelengths = tuple(float(value) for value in model["wavelengths_angstrom"])
    relative = tuple(float(value) for value in model["relative_intensities"])
    reference = wavelengths[0]
    u, v, w = map(float, common["initial_profile_deg2"])
    instrument = ConstantWavelengthInstrument(reference, u, v, w, 0.0, 0.0)
    axial = FcjGeometry(
        float(common["axial_sample_over_radius"]),
        float(common["axial_detector_over_radius"]),
    )
    if len(wavelengths) == 1:
        experiment = ConstantWavelengthExperiment.x_ray(instrument, axial_geometry=axial)
    else:
        experiment = ConstantWavelengthExperiment.x_ray_components(
            instrument,
            WavelengthComponents(wavelengths, relative),
            axial_geometry=axial,
        )
    selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    loaded = rietveld.RietveldInput.from_cif(
        pattern,
        experiment,
        root / "phase.cif",
        phase_id="opxrd",
        selection=selection,
        intensity_correction=BraggBrentanoPolarizedLp(
            reference, float(common["polarization_fraction"])
        ),
    )
    physics = CompositePhysicsProvider(
        (
            IsotropicSizeBroadening(250.0, shape_factor=1.0),
            IsotropicLorentzianMicrostrainBroadening(1.0e-3),
        )
    )
    phase = replace(loaded.phases[0], physics=physics)
    phase = replace(phase, scale=_qarr_initial_scales(pattern, experiment, (phase,))[0])
    background = ChebyshevBackground(
        "opxrd_residual",
        (0.0,) * int(common["background_terms"]),
        (float(x[0]), float(x[-1])),
    )
    selected_execution = ExecutionPolicy(threads=1) if execution is None else execution
    phases, background = solve_linear_profile_block(
        pattern,
        experiment,
        (phase,),
        background,
        selected_execution,
        support_fwhm=float(common["support_fwhm"]),
    )
    experiment, phases, background, position_alignment = _align_zero_shift(
        pattern,
        experiment,
        phases,
        background,
        selected_execution,
        common,
    )
    instrument_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2"),
        background=False,
    )
    sample_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=True,
        background=False,
    )
    for _ in range(cycles):
        instrument_result = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            instrument_selection,
            selected_execution,
            iterations=20,
            step=0.1,
        )
        experiment = instrument_result.experiment
        phases, background = solve_linear_profile_block(
            pattern,
            experiment,
            instrument_result.phases,
            background,
            selected_execution,
            support_fwhm=float(common["support_fwhm"]),
        )
        sample_result = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            sample_selection,
            selected_execution,
            iterations=20,
            step=0.08,
        )
        experiment = sample_result.experiment
        phases, background = solve_linear_profile_block(
            pattern,
            experiment,
            sample_result.phases,
            background,
            selected_execution,
            support_fwhm=float(common["support_fwhm"]),
        )
    calculation = rietveld.calculate(
        pattern,
        experiment,
        phases,
        background=background,
        support_fwhm=float(common["support_fwhm"]),
        execution=selected_execution,
    )
    metrics = _residual_metrics(observed, calculation.y)
    observed_signal = observed - calculation.background
    if np.std(observed_signal) > 0.0 and np.std(calculation.profile_y) > 0.0:
        correlation = float(np.corrcoef(observed_signal, calculation.profile_y)[0, 1])
    else:
        correlation = 0.0
    profile = {
        "u_deg2": experiment.instrument.u_deg2,
        "v_deg2": experiment.instrument.v_deg2,
        "w_deg2": experiment.instrument.w_deg2,
        "x_deg": experiment.instrument.x_deg,
        "y_deg": experiment.instrument.y_deg,
        "zero_shift_deg": experiment.zero_shift_deg,
    }
    tangent = np.tan(np.deg2rad(x / 2.0))
    gaussian_variance = (
        experiment.instrument.u_deg2 * tangent**2
        + experiment.instrument.v_deg2 * tangent
        + experiment.instrument.w_deg2
    )
    maximum_gaussian_fwhm = float(np.sqrt(max(float(np.max(gaussian_variance)), 0.0)))
    plausibility: Literal["plausible", "warning"] = (
        "warning"
        if position_alignment.status != "aligned"
        or maximum_gaussian_fwhm > float(common["large_gaussian_fwhm_warning_deg"])
        else "plausible"
    )
    values = (
        *profile.values(),
        correlation,
        metrics.relative_l2,
        metrics.normalized_l1,
        position_alignment.initial_poisson_rwp,
        position_alignment.coarse_zero_shift_deg,
        position_alignment.coarse_poisson_rwp,
        position_alignment.polished_zero_shift_deg,
        position_alignment.selected_zero_shift_deg,
        position_alignment.selected_poisson_rwp,
        position_alignment.poisson_rwp_improvement,
        maximum_gaussian_fwhm,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("opXRD structural common model produced non-finite output")
    return OpxrdStructuralResult(
        int(x.size),
        phases[0].reflections.reflection_count,
        len(phases),
        wavelengths,
        bool(model["wavelength_order_corrected"]),
        relative,
        position_alignment,
        profile,
        maximum_gaussian_fwhm,
        plausibility,
        metrics,
        correlation,
        perf_counter() - started,
    )


def run_opxrd_robustness_campaign(
    archive_path: str | Path,
    selection_path: str | Path,
    *,
    structural: bool = False,
    structural_cycles: int = 2,
    case_ids: tuple[str, ...] | None = None,
    work_directory: str | Path | None = None,
) -> OpxrdCampaignResult:
    """Verify the archive and run the fixed stratified opXRD procedure."""

    started = perf_counter()
    if not isinstance(structural, bool):
        raise TypeError("structural must be a bool")
    archive = Path(archive_path)
    selection, specs, selection_sha256 = _load_selection(Path(selection_path))
    expected_archive = selection["archive"]
    if archive.name != expected_archive["filename"]:
        raise ValueError("opXRD archive filename does not match the selection manifest")
    if archive.stat().st_size != expected_archive["size_bytes"]:
        raise ValueError("opXRD archive size mismatch")
    archive_sha256 = _sha256_file(archive)
    if archive_sha256 != expected_archive["sha256"]:
        raise ValueError("opXRD archive SHA-256 mismatch")
    if case_ids is not None:
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("opXRD case_ids must be unique")
        by_id = {spec.case_id: spec for spec in specs}
        try:
            specs = tuple(by_id[value] for value in case_ids)
        except KeyError as error:
            raise KeyError(f"unknown opXRD case {error.args[0]!r}") from error
    procedure = selection["procedure"]
    background_settings = procedure["background"]
    baseline_settings = procedure["independent_baseline"]
    peak_settings = procedure["peak_detection"]
    results = []
    structural_root = Path(work_directory) if work_directory is not None else None
    with zipfile.ZipFile(archive) as source:
        file_members = tuple(info for info in source.infolist() if not info.is_dir())
        if len(file_members) != expected_archive["member_count"]:
            raise ValueError("opXRD archive member count mismatch")
        available = {info.filename for info in file_members}
        for spec in specs:
            if spec.member_path not in available:
                raise ValueError(f"selected opXRD member is missing: {spec.member_path}")
            payload = source.read(spec.member_path)
            if hashlib.sha256(payload).hexdigest() != spec.member_sha256:
                raise ValueError(f"selected opXRD member digest mismatch: {spec.case_id}")
            pattern = _decode_pattern(payload)
            wavelengths, order_corrected = _wavelengths(pattern.xray)
            spacing = np.diff(pattern.x)
            if np.any(spacing <= 0.0):
                diagnostics = [
                    "PhaseSmith rejected the source grid because 2theta is not strictly "
                    "increasing; the pattern was not sorted, segmented, or resampled"
                ]
                if order_corrected:
                    diagnostics.append("wavelength components reordered from shortest to longest")
                results.append(
                    OpxrdRejectedCaseResult(
                        spec.case_id,
                        spec.member_path,
                        spec.source_group,
                        spec.metadata_class,
                        spec.role,
                        int(pattern.x.size),
                        (float(np.min(pattern.x)), float(np.max(pattern.x))),
                        (float(np.min(pattern.y)), float(np.max(pattern.y))),
                        float(np.mean(pattern.y < 0.0)),
                        float(np.mean(pattern.y == 0.0)),
                        len(pattern.phases),
                        len(wavelengths),
                        order_corrected,
                        "rejected_nonmonotonic_grid",
                        int(np.count_nonzero(spacing <= 0.0)),
                        tuple(diagnostics),
                    )
                )
                continue
            median_step = float(np.median(spacing))
            step_deviation = float(np.max(np.abs(spacing - median_step)) / median_step)
            estimator = SmoothBrucknerBackground(
                smooth_width=float(background_settings["smooth_width_deg"]),
                iterations=int(background_settings["iterations"]),
                chebyshev_order=background_settings["chebyshev_order"],
            )
            diagnostics: list[str] = []
            subtraction = None
            try:
                subtraction = estimator.subtract(pattern.x, pattern.y)
                repeated = estimator.subtract(pattern.x, pattern.y)
                deterministic = bool(
                    np.array_equal(subtraction.background, repeated.background)
                    and np.array_equal(subtraction.corrected_y, repeated.corrected_y)
                )
            except ValueError as error:
                if "uniformly spaced" not in str(error):
                    raise ValueError(
                        f"{spec.case_id}: PhaseSmith background rejected the pattern"
                    ) from error
                deterministic = None
                diagnostics.append(
                    "PhaseSmith physical-width background rejected the nonuniform grid; "
                    "the source grid was not resampled"
                )
            independent = _quantile_background(
                pattern.x,
                pattern.y,
                int(baseline_settings["bin_count"]),
                float(baseline_settings["quantile"]),
            )
            scale = max(float(np.sum(np.abs(pattern.y))), np.finfo(float).tiny)
            if subtraction is None:
                background_status = "rejected_nonuniform_grid"
                background_agreement = None
                bkg_ratio = None
                bkg_metrics = None
                corrected = pattern.y - independent
                smooth_points = None
            else:
                background_status = "evaluated"
                background_agreement = float(
                    np.sum(np.abs(subtraction.background - independent)) / scale
                )
                positive_signal_area = _trapezoid(
                    np.maximum(subtraction.corrected_y, 0.0), pattern.x
                )
                background_area = _trapezoid(np.abs(subtraction.background), pattern.x)
                bkg_ratio = background_area / max(positive_signal_area, np.finfo(float).tiny)
                bkg_metrics = _residual_metrics(pattern.y, subtraction.background)
                corrected = subtraction.corrected_y
                smooth_points = subtraction.smooth_points
            peaks, peak_heights = _detect_peaks(
                pattern.x,
                corrected,
                minimum_separation_deg=float(peak_settings["minimum_separation_deg"]),
                noise_multiplier=float(peak_settings["noise_multiplier"]),
                relative_height_floor=float(peak_settings["relative_height_floor"]),
            )
            strongest = tuple(
                float(peaks[index])
                for index in sorted(
                    range(peaks.size), key=lambda index: (-peak_heights[index], peaks[index])
                )[:10]
            )
            lattice = _lattice_diagnostic(pattern, peaks, median_step)
            if np.any(pattern.y < 0.0):
                diagnostics.append("negative intensities make Poisson Rwp scientifically invalid")
            if order_corrected:
                diagnostics.append("wavelength components reordered from shortest to longest")
            if spec.role in {"metadata_boundary", "unsupported_geometry", "robustness_only"}:
                diagnostics.append("no structural-fit claim is made for this selected role")
            structural_result = None
            if structural and spec.role == "structural_common_model":
                if structural_root is None:
                    raise ValueError("structural campaign requires work_directory")
                bundle = structural_root / spec.case_id
                write_opxrd_common_model_bundle(archive, selection_path, spec.case_id, bundle)
                structural_result = run_opxrd_structural_common_model(
                    bundle, cycles=structural_cycles
                )
                if structural_result.position_alignment.status == "large_offset_warning":
                    diagnostics.append(
                        "structural fit requires a large zero offset; treat it as a "
                        "calibration warning"
                    )
                if structural_result.position_alignment.status == "scan_boundary_warning":
                    diagnostics.append(
                        "structural zero alignment reached the declared scan boundary"
                    )
                if structural_result.parameter_plausibility == "warning":
                    diagnostics.append(
                        "structural parameter plausibility warning is separate from residuals"
                    )
            results.append(
                OpxrdCaseResult(
                    spec.case_id,
                    spec.member_path,
                    spec.source_group,
                    spec.metadata_class,
                    spec.role,
                    int(pattern.x.size),
                    (float(pattern.x[0]), float(pattern.x[-1])),
                    median_step,
                    step_deviation,
                    (float(np.min(pattern.y)), float(np.max(pattern.y))),
                    float(np.mean(pattern.y < 0.0)),
                    float(np.mean(pattern.y == 0.0)),
                    len(pattern.phases),
                    len(wavelengths),
                    order_corrected,
                    background_status,
                    deterministic,
                    smooth_points,
                    bkg_ratio,
                    background_agreement,
                    bkg_metrics,
                    _residual_metrics(pattern.y, independent),
                    int(peaks.size),
                    strongest,
                    lattice,
                    structural_result,
                    tuple(diagnostics),
                )
            )
    metadata_counts = {
        name: sum(result.metadata_class == name for result in results)
        for name in sorted(_METADATA_CLASSES)
    }
    role_counts = {
        name: sum(result.role == name for result in results) for name in sorted(_CASE_ROLES)
    }
    checks = {
        "archive_verified": True,
        "selected_members_verified": True,
        "all_selected_members_accounted_for": len(results) == len(specs),
        "evaluated_backgrounds_are_bitwise_deterministic": all(
            result.background_deterministic
            for result in results
            if isinstance(result, OpxrdCaseResult)
            and result.phasesmith_background_status == "evaluated"
        ),
        "background_outcomes_are_explicit": all(
            result.phasesmith_background_status in {"evaluated", "rejected_nonuniform_grid"}
            for result in results
            if isinstance(result, OpxrdCaseResult)
        ),
        "nonmonotonic_grids_are_explicitly_rejected": all(
            result.rejection_status == "rejected_nonmonotonic_grid"
            for result in results
            if isinstance(result, OpxrdRejectedCaseResult)
        ),
        "negative_cases_block_poisson": all(
            result.quantile_only_metrics.poisson_rwp is None
            for result in results
            if isinstance(result, OpxrdCaseResult)
            if result.negative_fraction > 0.0
        ),
        "nonnegative_cases_have_poisson": all(
            result.quantile_only_metrics.poisson_rwp is not None
            for result in results
            if isinstance(result, OpxrdCaseResult)
            if result.negative_fraction == 0.0
        ),
        "requested_structural_results_are_finite": (not structural)
        or all(
            result.structural_result is not None
            for result in results
            if isinstance(result, OpxrdCaseResult)
            if result.role == "structural_common_model"
        ),
        "structural_position_alignment_is_nonworsening": (not structural)
        or all(
            result.structural_result.position_alignment.selected_poisson_rwp
            <= result.structural_result.position_alignment.initial_poisson_rwp
            for result in results
            if isinstance(result, OpxrdCaseResult)
            if result.structural_result is not None
        ),
        "structural_position_scans_do_not_hit_bounds": (not structural)
        or all(
            result.structural_result.position_alignment.status != "scan_boundary_warning"
            for result in results
            if isinstance(result, OpxrdCaseResult)
            if result.structural_result is not None
        ),
    }
    return OpxrdCampaignResult(
        1,
        selection["dataset_id"],
        archive_sha256,
        selection_sha256,
        int(expected_archive["member_count"]),
        len(results),
        len({result.source_group for result in results}),
        metadata_counts,
        role_counts,
        checks,
        tuple(results),
        perf_counter() - started,
        (
            "The selected patterns are grouped by contributor source; no pattern is counted "
            "as an independent scientific replicate of another.",
            "Smooth-Bruckner and piecewise-quantile background metrics are diagnostics, "
            "not competing ground-truth backgrounds.",
            "Poisson Rwp is omitted for negative observations instead of shifting or "
            "clipping source intensities.",
            "Position-only lattice coverage does not weight calculated reflection "
            "intensities and is never an acceptance oracle.",
            "Structural common-model results use disclosed missing-metadata assumptions "
            "and are suitable only for same-input cross-program comparison.",
            "Structural fits align zero before widths and report large-offset or broad-profile "
            "warnings separately from residual metrics.",
        ),
    )
