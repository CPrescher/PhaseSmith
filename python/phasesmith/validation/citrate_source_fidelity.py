"""Source-deposited reflection-intensity fidelity for the Rb-citrate holdout."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..crystallography import calculate_structure_factor_values
from ..io import read_cif
from ..scattering import (
    ScatteringContext,
    ScatteringFactorBatch,
    ScatteringProviderDescriptor,
    XrayNonResonant,
)

_SCOPE = "iucr_anhydrous_trirubidium_citrate_silicon_holdout"
_LOW_ANGLE_LIMITS_DEG = (17.004916, 30.0)
_SOURCE_CROMER_MANN = {
    "C": ((2.31000, 1.02000, 1.58860, 0.86500), (20.8439, 10.2075, 0.56870, 51.6512), 0.21560),
    "H": ((0.49300, 0.32291, 0.14019, 0.04081), (10.5109, 26.1257, 3.14236, 57.7997), 0.00304),
    "O": ((3.04850, 2.28680, 1.54630, 0.86700), (13.2771, 5.70110, 0.32390, 32.9089), 0.25080),
    "Rb": ((17.1784, 9.64350, 5.13990, 1.52920), (1.78880, 17.3151, 0.27480, 164.934), 3.48730),
    "Si": ((6.29150, 3.03530, 1.98910, 1.54100), (2.43860, 32.3337, 0.67850, 81.6937), 1.14070),
}
_SOURCE_DESCRIPTOR = ScatteringProviderDescriptor(
    "phasesmith.validation.iucr_cromer_mann",
    "vn2123",
    "xray",
    "electrons",
    thread_safe=True,
)


class _SourceCromerMann:
    descriptor = _SOURCE_DESCRIPTOR

    def evaluate(self, context: ScatteringContext) -> ScatteringFactorBatch:
        values = []
        derivatives = []
        s = context.s_inverse_angstrom
        for species in context.species:
            try:
                source_a, source_b, source_c = _SOURCE_CROMER_MANN[species.element_symbol]
            except KeyError as error:
                raise ValueError(
                    f"source pdCIF has no Cromer-Mann row for {species.element_symbol}"
                ) from error
            a = np.asarray(source_a, dtype=np.float64)
            b = np.asarray(source_b, dtype=np.float64)
            exponent = np.exp(-(s[:, None] ** 2) * b)
            values.append(np.sum(exponent * a, axis=1) + source_c)
            derivatives.append(np.sum(exponent * (-2.0 * s[:, None] * a * b), axis=1))
        return ScatteringFactorBatch(
            np.asarray(values, dtype=np.float64).T,
            np.asarray(derivatives, dtype=np.float64).T,
            self.descriptor,
        )


@dataclass(frozen=True, slots=True)
class ReflectionFidelityMetrics:
    reflection_count: int
    profile_correlation: float | None
    median_abs_relative_error: float
    p95_abs_relative_error: float
    maximum_abs_relative_error: float
    median_calculated_over_source: float
    rms_log_ratio: float
    source_weighted_l1_relative_error: float
    source_normalized_rms_error: float


@dataclass(frozen=True, slots=True)
class CitrateSourceReflectionFidelityResult:
    source_dataset_id: str
    source_row_count: int
    unique_reflection_count: int
    duplicate_wavelength_f_squared_max_abs_delta: float
    low_angle_limits_deg: tuple[float, float]
    phases: dict[str, dict[str, Any]]
    review: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(source: np.ndarray, calculated: np.ndarray) -> ReflectionFidelityMetrics:
    positive = source > 0.0
    if source.shape != calculated.shape or source.ndim != 1 or not np.any(positive):
        raise ValueError("reflection fidelity arrays are invalid")
    source_value = source[positive]
    calculated_value = calculated[positive]
    ratio = calculated_value / source_value
    relative = np.abs(ratio - 1.0)
    correlation = (
        None
        if source_value.size < 2 or np.std(source_value) == 0.0 or np.std(calculated_value) == 0.0
        else float(np.corrcoef(source_value, calculated_value)[0, 1])
    )
    return ReflectionFidelityMetrics(
        reflection_count=int(source_value.size),
        profile_correlation=correlation,
        median_abs_relative_error=float(np.median(relative)),
        p95_abs_relative_error=float(np.quantile(relative, 0.95)),
        maximum_abs_relative_error=float(np.max(relative)),
        median_calculated_over_source=float(np.median(ratio)),
        rms_log_ratio=float(np.sqrt(np.mean(np.log(ratio) ** 2))),
        source_weighted_l1_relative_error=float(
            np.sum(np.abs(calculated_value - source_value)) / np.sum(source_value)
        ),
        source_normalized_rms_error=float(
            np.sqrt(np.sum((calculated_value - source_value) ** 2) / np.sum(source_value**2))
        ),
    )


def _unique_reflections(rows: np.ndarray) -> tuple[np.ndarray, float]:
    by_key: dict[tuple[int, int, int, int], np.ndarray] = {}
    duplicate_delta = 0.0
    for row in rows:
        key = tuple(int(value) for value in row[:4])
        previous = by_key.get(key)
        if previous is not None:
            duplicate_delta = max(duplicate_delta, abs(float(previous[6] - row[6])))
            if abs(float(previous[8] - row[8])) > 5.0e-6:
                raise ValueError("source wavelength rows disagree on d-spacing")
        else:
            by_key[key] = row
    return np.asarray(list(by_key.values()), dtype=np.float64), duplicate_delta


def run_citrate_source_reflection_fidelity(
    bundle_directory: str | Path,
) -> CitrateSourceReflectionFidelityResult:
    """Compare converted structures with source-deposited reflection F-squared values."""

    root = Path(bundle_directory)
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != _SCOPE:
        raise ValueError("source-reflection audit requires the reviewed rubidium bundle")
    reflection_record = manifest.get("source_reflections", {})
    expected_columns = tuple(reflection_record.get("columns", ()))
    if expected_columns != (
        "h",
        "k",
        "l",
        "phase_id",
        "wavelength_id",
        "f_squared_measured",
        "f_squared_calculated",
        "phase_deg",
        "d_spacing_angstrom",
        "i100_measured",
    ):
        raise ValueError("source-reflection bundle columns are incompatible")
    rows = np.loadtxt(root / str(reflection_record["file"]), delimiter=",", skiprows=1)
    if (
        rows.shape != (int(reflection_record["row_count"]), len(expected_columns))
        or not np.isfinite(rows).all()
    ):
        raise ValueError("source-reflection bundle array is invalid")
    unique, duplicate_delta = _unique_reflections(rows)
    wavelength = float(manifest["instrument"]["wavelengths_angstrom"][0])
    phases = {}
    for phase_id_text, phase_name in reflection_record["phase_id_to_name"].items():
        phase_id = int(phase_id_text)
        selected = unique[unique[:, 3] == phase_id]
        hkl = np.ascontiguousarray(selected[:, :3], dtype=np.int64)
        source_f_squared = np.ascontiguousarray(selected[:, 6], dtype=np.float64)
        source_d = np.ascontiguousarray(selected[:, 8], dtype=np.float64)
        structure = read_cif(root / f"{phase_name}.cif", strict=False).structure
        multiplicity = np.ones(hkl.shape[0], dtype=np.int64)
        source_native = calculate_structure_factor_values(
            structure, hkl, multiplicity, _SourceCromerMann()
        )
        production = calculate_structure_factor_values(
            structure, hkl, multiplicity, XrayNonResonant()
        )
        calculated_d = 1.0 / np.sqrt(source_native.q_squared_inverse_angstrom2)
        two_theta = np.rad2deg(2.0 * np.arcsin(wavelength / (2.0 * source_d)))
        low = (two_theta >= _LOW_ANGLE_LIMITS_DEG[0]) & (two_theta < _LOW_ANGLE_LIMITS_DEG[1])
        positive = source_f_squared > 0.0
        strong_threshold = float(np.quantile(source_f_squared[positive], 0.25))
        strong = source_f_squared >= strong_threshold
        phases[str(phase_name)] = {
            "source_phase_id": phase_id,
            "reflection_count": int(hkl.shape[0]),
            "low_angle_reflection_count": int(np.count_nonzero(low)),
            "strong_f_squared_threshold": strong_threshold,
            "d_spacing_max_abs_error_angstrom": float(np.max(np.abs(calculated_d - source_d))),
            "source_cromer_mann": {
                "all_positive": asdict(_metrics(source_f_squared, source_native.f_squared)),
                "strong": asdict(
                    _metrics(source_f_squared[strong], source_native.f_squared[strong])
                ),
                "low_angle": asdict(_metrics(source_f_squared[low], source_native.f_squared[low])),
            },
            "production_waasmaier_kirfel": {
                "all_positive": asdict(_metrics(source_f_squared, production.f_squared)),
                "strong": asdict(_metrics(source_f_squared[strong], production.f_squared[strong])),
                "low_angle": asdict(_metrics(source_f_squared[low], production.f_squared[low])),
            },
        }
    rubidium = phases["trirubidium_citrate"]
    source_low = rubidium["source_cromer_mann"]["low_angle"]
    review = {
        "duplicate_wavelength_rows_are_consistent": duplicate_delta <= 1.0e-12,
        "converted_d_spacings_match_source": all(
            phase["d_spacing_max_abs_error_angstrom"] < 1.0e-4 for phase in phases.values()
        ),
        "source_native_low_angle_f_squared_is_faithful": (
            source_low["median_abs_relative_error"] < 0.005
            and source_low["source_weighted_l1_relative_error"] < 0.001
        ),
        "conclusion": (
            "The converted structures, symmetry, displacement values, and source-deposited "
            "Cromer-Mann scattering reproduce the low-angle calculated F-squared values. "
            "Reflection-intensity conversion is not the dominant residual source."
        ),
    }
    if not all(value is True for key, value in review.items() if key != "conclusion"):
        raise RuntimeError("source-reflection fidelity review did not pass")
    return CitrateSourceReflectionFidelityResult(
        source_dataset_id=str(manifest["source_dataset_id"]),
        source_row_count=int(rows.shape[0]),
        unique_reflection_count=int(unique.shape[0]),
        duplicate_wavelength_f_squared_max_abs_delta=duplicate_delta,
        low_angle_limits_deg=_LOW_ANGLE_LIMITS_DEG,
        phases=phases,
        review=review,
    )
