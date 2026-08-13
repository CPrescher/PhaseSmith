#!/usr/bin/env python3
"""Low-angle component forensics for the anhydrous Rb-citrate/Si holdout.

This external-only worker imports no PhaseSmith module. It uses the pinned
GSAS-II public scripting API to refine controlled low-angle subsets, then uses
plain NumPy projections and peak-group moments to distinguish residual
background, integrated-intensity, symmetric-width, FCJ-asymmetry, and position
effects. It emits plain JSON and never exposes GSAS-II objects.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
LOW_ANGLE_LIMITS_DEG = (17.004916, 30.0)
PHYSICAL_SCANS = {
    "gaussian_width_W": (0.5, 1.0, 2.5, 5.109, 10.0, 20.0, 40.0, 80.0),
    "lorentzian_size_X": (0.5, 1.0, 2.0, 3.634, 5.0, 8.0),
    # The pinned GSAS-II profile evaluator applies max(SH/L, 0.002). Start at
    # that observable boundary instead of reporting a misleading zero value.
    "fcj_asymmetry": (0.002, 0.005, 0.0097, 0.0194, 0.03, 0.05, 0.08),
}
SCAN_PARAMETER = {
    "gaussian_width_W": "W",
    "lorentzian_size_X": "X",
    "fcj_asymmetry": "SH/L",
}


def _load_source_worker() -> Any:
    path = Path(__file__).with_name("benchmark_citrate_residual_forensics.py")
    spec = importlib.util.spec_from_file_location("citrate_residual_source_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load citrate residual-forensics worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_worker()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def weighted_metrics(
    observed: np.ndarray,
    calculated: np.ndarray,
    background: np.ndarray,
) -> dict[str, float]:
    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (observed, calculated, background)
    )
    if (
        any(value.ndim != 1 for value in arrays)
        or len({value.size for value in arrays}) != 1
        or arrays[0].size < 3
        or not all(np.isfinite(value).all() for value in arrays)
        or np.any(arrays[0] < 0.0)
    ):
        raise ValueError("metric arrays must be finite aligned one-dimensional values")
    observed_value, calculated_value, background_value = arrays
    weight = 1.0 / np.maximum(observed_value, 1.0)
    residual = observed_value - calculated_value
    denominator = float(np.sum(weight * observed_value**2))
    observed_signal = observed_value - background_value
    calculated_signal = calculated_value - background_value
    return {
        "weighted_sse": float(np.sum(weight * residual**2)),
        "poisson_rwp": math.sqrt(float(np.sum(weight * residual**2)) / denominator),
        "profile_correlation": float(np.corrcoef(observed_signal, calculated_signal)[0, 1]),
    }


def peak_group_segments(
    x: np.ndarray,
    signal: np.ndarray,
    *,
    minimum_relative_height: float = 0.005,
    minimum_separation_deg: float = 0.10,
) -> tuple[tuple[int, int, int], ...]:
    x_value = np.asarray(x, dtype=np.float64)
    signal_value = np.asarray(signal, dtype=np.float64)
    if (
        x_value.ndim != 1
        or signal_value.shape != x_value.shape
        or x_value.size < 5
        or not np.isfinite(x_value).all()
        or not np.isfinite(signal_value).all()
        or np.any(np.diff(x_value) <= 0.0)
        or not 0.0 < minimum_relative_height < 1.0
        or minimum_separation_deg <= 0.0
    ):
        raise ValueError("peak-group inputs are invalid")
    threshold = minimum_relative_height * float(np.max(signal_value))
    candidates = (
        np.flatnonzero(
            (signal_value[1:-1] > signal_value[:-2])
            & (signal_value[1:-1] >= signal_value[2:])
            & (signal_value[1:-1] >= threshold)
        )
        + 1
    )
    selected: list[int] = []
    for candidate in candidates:
        if not selected or x_value[candidate] - x_value[selected[-1]] >= minimum_separation_deg:
            selected.append(int(candidate))
        elif signal_value[candidate] > signal_value[selected[-1]]:
            selected[-1] = int(candidate)
    if not selected:
        raise ValueError("no significant peak groups were found")
    boundaries = [0]
    for left, right in itertools.pairwise(selected):
        valley = left + int(np.argmin(signal_value[left : right + 1]))
        boundaries.append(valley + 1)
    boundaries.append(x_value.size)
    return tuple(
        (int(start), int(stop), int(peak))
        for start, stop, peak in zip(boundaries[:-1], boundaries[1:], selected, strict=True)
    )


def _weighted_lstsq(
    design: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, int, float]:
    root_weight = np.sqrt(weight)
    weighted_design = design * root_weight[:, None]
    weighted_target = target * root_weight
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_design, weighted_target, rcond=None
    )
    condition = (
        math.inf
        if singular_values.size == 0 or singular_values[-1] <= 0.0
        else float(singular_values[0] / singular_values[-1])
    )
    return coefficients, int(rank), condition


def fit_peak_group_amplitudes(
    x: np.ndarray,
    observed: np.ndarray,
    calculated: np.ndarray,
    background: np.ndarray,
    segments: tuple[tuple[int, int, int], ...],
) -> dict[str, Any]:
    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (x, observed, calculated, background)
    )
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise ValueError("projection arrays must be aligned one-dimensional values")
    x_value, observed_value, calculated_value, background_value = arrays
    signal = calculated_value - background_value
    peak_columns = np.zeros((x_value.size, len(segments)), dtype=np.float64)
    for column, (start, stop, peak) in enumerate(segments):
        if not 0 <= start <= peak < stop <= x_value.size:
            raise ValueError("peak-group segment is invalid")
        peak_columns[start:stop, column] = signal[start:stop]
    centered_x = (x_value - np.mean(x_value)) / np.ptp(x_value)
    background_columns = np.column_stack((np.ones_like(x_value), centered_x))
    target = observed_value - background_value
    weight = 1.0 / np.maximum(observed_value, 1.0)
    active = list(range(len(segments)))
    while True:
        design = np.column_stack((peak_columns[:, active], background_columns))
        coefficients, rank, condition = _weighted_lstsq(design, target, weight)
        peak_coefficients = coefficients[: len(active)]
        negative = np.flatnonzero(peak_coefficients < 0.0)
        if negative.size == 0:
            break
        del active[int(negative[np.argmin(peak_coefficients[negative])])]
        if not active:
            raise RuntimeError("non-negative peak-group projection removed every group")
    amplitudes = np.zeros(len(segments), dtype=np.float64)
    amplitudes[active] = coefficients[: len(active)]
    fitted = background_value + peak_columns @ amplitudes + background_columns @ coefficients[-2:]
    metrics = weighted_metrics(observed_value, fitted, background_value)
    return {
        **metrics,
        "group_count": len(segments),
        "active_group_count": len(active),
        "rank": rank,
        "condition_number": condition,
        "amplitudes": amplitudes.tolist(),
        "amplitude_median": float(np.median(amplitudes)),
        "amplitude_minimum": float(np.min(amplitudes)),
        "amplitude_maximum": float(np.max(amplitudes)),
        "background_offset_and_slope": coefficients[-2:].tolist(),
    }


def fit_residual_background(
    x: np.ndarray,
    observed: np.ndarray,
    calculated: np.ndarray,
    background: np.ndarray,
    *,
    degree: int = 3,
) -> dict[str, Any]:
    """Project the residual onto a stable low-angle Legendre basis."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (x, observed, calculated, background)
    )
    if (
        any(value.ndim != 1 for value in arrays)
        or len({value.size for value in arrays}) != 1
        or arrays[0].size < degree + 2
        or not all(np.isfinite(value).all() for value in arrays)
        or np.any(np.diff(arrays[0]) <= 0.0)
        or degree < 0
    ):
        raise ValueError("background projection inputs are invalid")
    x_value, observed_value, calculated_value, background_value = arrays
    normalized_x = 2.0 * (x_value - x_value[0]) / (x_value[-1] - x_value[0]) - 1.0
    design = np.polynomial.legendre.legvander(normalized_x, degree)
    weight = 1.0 / np.maximum(observed_value, 1.0)
    coefficients, rank, condition = _weighted_lstsq(
        design, observed_value - calculated_value, weight
    )
    correction = design @ coefficients
    fitted_background = background_value + correction
    return {
        **weighted_metrics(observed_value, calculated_value + correction, fitted_background),
        "basis": "Legendre on x mapped to [-1, 1]",
        "degree": degree,
        "coefficients": coefficients.tolist(),
        "rank": rank,
        "condition_number": condition,
        "residual_background_summary": {
            "minimum": float(np.min(correction)),
            "maximum": float(np.max(correction)),
            "rms": float(np.sqrt(np.mean(correction**2))),
        },
    }


def _moments(x: np.ndarray, signal: np.ndarray) -> tuple[float, float, float, float]:
    positive = np.maximum(np.asarray(signal, dtype=np.float64), 0.0)
    area = float(np.trapezoid(positive, x))
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("peak-group area must be positive")
    centroid = float(np.trapezoid(positive * x, x) / area)
    centered = x - centroid
    variance = float(np.trapezoid(positive * centered**2, x) / area)
    if variance <= 0.0:
        raise ValueError("peak-group variance must be positive")
    width = math.sqrt(variance)
    skewness = float(np.trapezoid(positive * centered**3, x) / area / width**3)
    return area, centroid, width, skewness


def peak_group_moment_comparison(
    x: np.ndarray,
    current_signal: np.ndarray,
    target_signal: np.ndarray,
    segments: tuple[tuple[int, int, int], ...],
) -> dict[str, Any]:
    rows = []
    for start, stop, peak in segments:
        x_group = x[start:stop]
        current = _moments(x_group, current_signal[start:stop])
        target = _moments(x_group, target_signal[start:stop])
        rows.append(
            {
                "range_deg": [float(x_group[0]), float(x_group[-1])],
                "peak_deg": float(x[peak]),
                "target_over_current_area": target[0] / current[0],
                "target_minus_current_centroid_deg": target[1] - current[1],
                "target_over_current_rms_width": target[2] / current[2],
                "target_minus_current_skewness": target[3] - current[3],
            }
        )
    return {
        "groups": rows,
        "median_target_over_current_area": float(
            np.median([row["target_over_current_area"] for row in rows])
        ),
        "median_abs_centroid_delta_deg": float(
            np.median([abs(row["target_minus_current_centroid_deg"]) for row in rows])
        ),
        "median_target_over_current_rms_width": float(
            np.median([row["target_over_current_rms_width"] for row in rows])
        ),
        "median_abs_skewness_delta": float(
            np.median([abs(row["target_minus_current_skewness"]) for row in rows])
        ),
    }


def _json_variant(variant: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    arrays = variant.pop("arrays")
    low_metrics = weighted_metrics(
        arrays["observed"][mask], arrays["calculated"][mask], arrays["background"][mask]
    )
    return {**variant, "low_angle": low_metrics, "arrays": arrays}


def main() -> None:
    arguments = parse_args()
    if arguments.cycles <= 0:
        raise ValueError("cycles must be positive")
    actual_revision = revision(arguments.gsas_root)
    if actual_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II revision mismatch: expected {PINNED_REVISION}, got {actual_revision}"
        )
    sys.path.insert(0, str(arguments.gsas_root))
    if arguments.binary_dir is not None:
        sys.path.insert(0, str(arguments.binary_dir))
    from GSASII import GSASIIpath

    if arguments.binary_dir is not None:
        GSASIIpath.binaryPath = str(arguments.binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable as G2sc

    root = arguments.data_directory
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != "iucr_anhydrous_trirubidium_citrate_silicon_holdout":
        raise ValueError("worker requires the reviewed anhydrous rubidium bundle")
    data = np.loadtxt(root / "pattern.csv", delimiter=",", skiprows=1)
    x, observed, legacy_calculated, legacy_background = data.T
    mask = (x >= LOW_ANGLE_LIMITS_DEG[0]) & (x < LOW_ANGLE_LIMITS_DEG[1])
    source_factors = frozenset(("legacy_lx_size_axis", "sample_shift", "stephens"))
    with tempfile.TemporaryDirectory(prefix="phasesmith-rb-low-angle-") as name:
        work = Path(name)
        full_range = _json_variant(
            SOURCE._run_variant(
                G2sc,
                root,
                work,
                manifest,
                x,
                observed,
                legacy_background,
                source_factors,
                arguments.cycles,
                case_suffix="_low_audit_full",
                include_arrays=True,
            ),
            mask,
        )
        base = _json_variant(
            SOURCE._run_variant(
                G2sc,
                root,
                work,
                manifest,
                x,
                observed,
                legacy_background,
                source_factors,
                arguments.cycles,
                case_suffix="_low_audit_base",
                fit_window_deg=LOW_ANGLE_LIMITS_DEG,
                include_arrays=True,
            ),
            mask,
        )
        scans = {}
        for scan_index, (factor, values) in enumerate(PHYSICAL_SCANS.items()):
            parameter = SCAN_PARAMETER[factor]
            cases = []
            for value_index, value in enumerate(values):
                variant = _json_variant(
                    SOURCE._run_variant(
                        G2sc,
                        root,
                        work,
                        manifest,
                        x,
                        observed,
                        legacy_background,
                        source_factors,
                        arguments.cycles,
                        case_suffix=f"_low_audit_scan_{scan_index}_{value_index}",
                        fit_window_deg=LOW_ANGLE_LIMITS_DEG,
                        instrument_overrides={parameter: value},
                        include_arrays=True,
                    ),
                    mask,
                )
                variant.pop("arrays")
                cases.append(variant)
            scans[factor] = cases
        best_scan_cases = {
            factor: min(cases, key=lambda case: float(case["low_angle"]["weighted_sse"]))
            for factor, cases in scans.items()
        }
        best_shape_overrides = {
            SCAN_PARAMETER[factor]: float(case["instrument_overrides"][SCAN_PARAMETER[factor]])
            for factor, case in best_scan_cases.items()
        }
        combined_shape = _json_variant(
            SOURCE._run_variant(
                G2sc,
                root,
                work,
                manifest,
                x,
                observed,
                legacy_background,
                source_factors,
                arguments.cycles,
                case_suffix="_low_audit_combined_shape",
                fit_window_deg=LOW_ANGLE_LIMITS_DEG,
                instrument_overrides=best_shape_overrides,
                include_arrays=True,
            ),
            mask,
        )
        shift_probe = _json_variant(
            SOURCE._run_variant(
                G2sc,
                root,
                work,
                manifest,
                x,
                observed,
                legacy_background,
                source_factors,
                arguments.cycles,
                case_suffix="_low_audit_shift",
                fit_window_deg=LOW_ANGLE_LIMITS_DEG,
                sample_refinements=("Shift",),
                include_arrays=True,
            ),
            mask,
        )

    base_arrays = base["arrays"]
    combined_shape_arrays = combined_shape["arrays"]
    x_low = x[mask]
    current_signal = base_arrays["calculated"][mask] - base_arrays["background"][mask]
    segments = peak_group_segments(x_low, current_signal)
    intensity_projection = fit_peak_group_amplitudes(
        x_low,
        observed[mask],
        base_arrays["calculated"][mask],
        base_arrays["background"][mask],
        segments,
    )
    moment_comparison = peak_group_moment_comparison(
        x_low,
        current_signal,
        legacy_calculated[mask] - legacy_background[mask],
        segments,
    )
    deposited_low_angle = weighted_metrics(
        observed[mask], legacy_calculated[mask], legacy_background[mask]
    )
    background_shape = fit_residual_background(
        x_low,
        observed[mask],
        base_arrays["calculated"][mask],
        base_arrays["background"][mask],
    )
    combined_shape_background = fit_residual_background(
        x_low,
        observed[mask],
        combined_shape_arrays["calculated"][mask],
        combined_shape_arrays["background"][mask],
    )
    base_sse = float(base["low_angle"]["weighted_sse"])
    deposited_sse = float(deposited_low_angle["weighted_sse"])
    gap = base_sse - deposited_sse

    def gap_closed(result: dict[str, Any]) -> float:
        return (base_sse - float(result["weighted_sse"])) / gap

    scan_summary = {}
    for factor, cases in scans.items():
        best_case = best_scan_cases[factor]
        scan_summary[factor] = {
            "parameter": SCAN_PARAMETER[factor],
            "cases": [
                {
                    "value": case["instrument_overrides"][SCAN_PARAMETER[factor]],
                    "low_angle": case["low_angle"],
                }
                for case in cases
            ],
            "best": {
                "value": best_case["instrument_overrides"][SCAN_PARAMETER[factor]],
                "low_angle": best_case["low_angle"],
                "fraction_of_base_to_deposited_sse_gap_closed": gap_closed(best_case["low_angle"]),
            },
        }
    for result in (base, combined_shape):
        result.pop("arrays")
    full_range.pop("arrays")
    shift_probe.pop("arrays")
    report = {
        "schema_version": 1,
        "scope": "iucr_trirubidium_citrate_low_angle_component_forensics",
        "source_dataset_id": manifest["source_dataset_id"],
        "revision": actual_revision,
        "limits_deg": list(LOW_ANGLE_LIMITS_DEG),
        "sample_count": int(np.count_nonzero(mask)),
        "oracle_boundary": {
            "public_api": [
                "GSASIIscriptable.G2Project.add_powder_histogram/add_phase/do_refinements",
                "GSASIIscriptable.G2PwdrData.set_refinements/getdata/ComputeMassFracs",
                "GSASIIscriptable.G2Phase.getHAPentryList/setHAPentryValue",
            ],
            "private_probe": False,
        },
        "source_model": {
            "enabled_factors": [name for name in SOURCE.FACTORS if name in source_factors],
            "fixed_transparency": False,
            "refined_in_every_low_angle_case": [
                "trirubidium-citrate scale",
                "silicon scale",
                "constant residual background",
            ],
        },
        "full_range_source_model_low_angle": full_range["low_angle"],
        "low_angle_local_base": {
            **base["low_angle"],
            "full_range_phase_scales": full_range["phase_scales"],
            "low_angle_phase_scales": base["phase_scales"],
            "residual_background_summary": base["fit_window_residual_background_summary"],
        },
        "physical_profile_scans": {
            "method": (
                "Fixed physically admissible one-dimensional scans; two phase scales and "
                "a constant residual background are re-refined in every case."
            ),
            "rejected_unconstrained_fit": (
                "The preliminary unconstrained GSAS-II refinement drove W, X, and SH/L "
                "negative in multiple branches and is excluded from scientific results."
            ),
            "gsasii_sh_over_l_boundary": (
                "Pinned GSAS-II evaluates max(SH/L, 0.002); the scan therefore "
                "reports 0.002 as its observable lower boundary."
            ),
            "scan_ranges": {name: list(values) for name, values in PHYSICAL_SCANS.items()},
            "results": scan_summary,
            "combined_best_physical_shape": {
                "instrument_overrides": best_shape_overrides,
                "low_angle": combined_shape["low_angle"],
                "fraction_of_base_to_deposited_sse_gap_closed": gap_closed(
                    combined_shape["low_angle"]
                ),
            },
            "combined_best_physical_shape_and_background": {
                "instrument_overrides": best_shape_overrides,
                "low_angle": {
                    name: combined_shape_background[name]
                    for name in ("weighted_sse", "poisson_rwp", "profile_correlation")
                },
                "background_projection": {
                    name: combined_shape_background[name]
                    for name in (
                        "basis",
                        "degree",
                        "coefficients",
                        "rank",
                        "condition_number",
                        "residual_background_summary",
                    )
                },
                "fraction_of_base_to_deposited_sse_gap_closed": gap_closed(
                    combined_shape_background
                ),
            },
        },
        "background_shape_probe": {
            **background_shape,
            "fraction_of_base_to_deposited_sse_gap_closed": gap_closed(background_shape),
        },
        "position_probe": {
            "low_angle": shift_probe["low_angle"],
            "source_shift_micrometre": SOURCE.legacy_shft_to_gsasii_micrometre(-8.7503, 141.5),
            "refined_shift_micrometre": shift_probe["sample_shift_micrometre"],
            "fraction_of_base_to_deposited_sse_gap_closed": gap_closed(shift_probe["low_angle"]),
        },
        "peak_group_intensity_projection": {
            **intensity_projection,
            "fraction_of_base_to_deposited_sse_gap_closed": gap_closed(intensity_projection),
        },
        "deposited_to_current_peak_group_moments": moment_comparison,
        "deposited_curve_low_angle": deposited_low_angle,
        "review": {
            "no_single_component_closes_half_the_local_base_to_deposited_sse_gap": all(
                value < 0.5
                for value in (
                    gap_closed(background_shape),
                    gap_closed(shift_probe["low_angle"]),
                    gap_closed(intensity_projection),
                    *(
                        item["best"]["fraction_of_base_to_deposited_sse_gap_closed"]
                        for item in scan_summary.values()
                    ),
                )
            ),
            "fcj_scan_hits_lower_boundary": (
                scan_summary["fcj_asymmetry"]["best"]["value"]
                == min(PHYSICAL_SCANS["fcj_asymmetry"])
            ),
            "combined_shape_and_background_still_above_deposited_curve": (
                combined_shape_background["poisson_rwp"] > deposited_low_angle["poisson_rwp"]
            ),
            "conclusion": (
                "The remaining low-angle gap is coupled across angle-dependent phase "
                "intensity/scale, background, position, symmetric width, and axial-profile "
                "compression. No isolated component justifies a new production model."
            ),
        },
    }
    finite_values = (
        report["full_range_source_model_low_angle"]["poisson_rwp"],
        report["deposited_curve_low_angle"]["poisson_rwp"],
        intensity_projection["poisson_rwp"],
        *(case["low_angle"]["poisson_rwp"] for cases in scans.values() for case in cases),
        combined_shape["low_angle"]["poisson_rwp"],
        combined_shape_background["poisson_rwp"],
        background_shape["poisson_rwp"],
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise RuntimeError("low-angle forensic report contains non-finite values")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    if arguments.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {arguments.report}")
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
