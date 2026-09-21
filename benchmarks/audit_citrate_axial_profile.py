#!/usr/bin/env python3
"""Compare source-native FCJ shapes with source-specific pinned GSAS-II probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import phasesmith
from phasesmith._numpy_compat import trapezoid

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", required=True, type=Path)
    parser.add_argument("--oracle-arrays", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _moments(x: np.ndarray, profile: np.ndarray) -> dict[str, float]:
    area = float(trapezoid(profile, x))
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("profile sampled area must be positive and finite")
    centroid = float(trapezoid(profile * x, x) / area)
    centered = x - centroid
    variance = float(trapezoid(profile * centered**2, x) / area)
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("profile sampled variance must be positive and finite")
    width = math.sqrt(variance)
    skewness = float(trapezoid(profile * centered**3, x) / area / width**3)
    return {
        "sampled_area": area,
        "centroid_deg": centroid,
        "rms_width_deg": width,
        "skewness": skewness,
        "mode_deg": float(x[int(np.argmax(profile))]),
        "maximum_per_deg": float(np.max(profile)),
    }


def profile_difference(
    x: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    if (
        x.ndim != 1
        or reference.shape != x.shape
        or candidate.shape != x.shape
        or x.size < 5
        or not np.isfinite(x).all()
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
        or np.any(np.diff(x) <= 0.0)
        or np.any(reference < 0.0)
        or np.any(candidate < 0.0)
    ):
        raise ValueError("profile comparison arrays are invalid")
    reference_moments = _moments(x, reference)
    candidate_moments = _moments(x, candidate)
    reference_normalized = reference / reference_moments["sampled_area"]
    candidate_normalized = candidate / candidate_moments["sampled_area"]
    difference = candidate_normalized - reference_normalized
    return {
        "normalized_l1_distance": float(trapezoid(np.abs(difference), x)),
        "normalized_rms_distance": float(
            math.sqrt(trapezoid(difference**2, x) / trapezoid(reference_normalized**2, x))
        ),
        "normalized_maximum_abs_error": float(
            np.max(np.abs(difference)) / np.max(reference_normalized)
        ),
        "candidate_minus_reference_centroid_deg": (
            candidate_moments["centroid_deg"] - reference_moments["centroid_deg"]
        ),
        "candidate_over_reference_rms_width": (
            candidate_moments["rms_width_deg"] / reference_moments["rms_width_deg"]
        ),
        "candidate_minus_reference_skewness": (
            candidate_moments["skewness"] - reference_moments["skewness"]
        ),
        "candidate_minus_reference_mode_deg": (
            candidate_moments["mode_deg"] - reference_moments["mode_deg"]
        ),
        "reference_moments": reference_moments,
        "candidate_moments": candidate_moments,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {arguments.report}")
    oracle = json.loads(arguments.oracle_report.read_text(encoding="utf-8"))
    if oracle.get("revision") != PINNED_REVISION:
        raise ValueError("axial-profile oracle report has the wrong pinned revision")
    if oracle.get("arrays_sha256") != _sha256_file(arguments.oracle_arrays):
        raise ValueError("axial-profile oracle array hash is invalid")
    source = oracle["source_profile"]
    source_geometry = phasesmith.FcjGeometry(float(source["S/L"]), float(source["H/L"]))
    formal = format(float(source["S/L"]) + float(source["H/L"]), ".12g")
    results = []
    formal_phase_smith_identity = []
    with np.load(arguments.oracle_arrays, allow_pickle=False) as arrays:
        for case in oracle["cases"]:
            x = np.asarray(arrays[case["x_array"]], dtype=np.float64)
            source_native = phasesmith.profile_fcj(
                x,
                float(case["position_deg"]),
                float(case["gaussian_fwhm_deg"]),
                float(case["lorentzian_fwhm_deg"]),
                source_geometry,
            ).value
            comparisons = {}
            for sh_over_l_text, array_name in case["profile_arrays_by_sh_over_l"].items():
                sh_over_l = float(sh_over_l_text)
                gsasii = np.asarray(arrays[array_name], dtype=np.float64)
                equal_height = phasesmith.profile_fcj(
                    x,
                    float(case["position_deg"]),
                    float(case["gaussian_fwhm_deg"]),
                    float(case["lorentzian_fwhm_deg"]),
                    phasesmith.FcjGeometry(sh_over_l / 2.0, sh_over_l / 2.0),
                ).value
                comparisons[sh_over_l_text] = {
                    "gsasii_against_source_native": profile_difference(x, source_native, gsasii),
                    "gsasii_against_published_equal_height_same_parameter": profile_difference(
                        x, equal_height, gsasii
                    ),
                }
                if sh_over_l_text == formal:
                    formal_phase_smith_identity.append(
                        bool(np.array_equal(source_native, equal_height))
                    )
            results.append(
                {
                    "hkl": case["hkl"],
                    "position_deg": case["position_deg"],
                    "gaussian_fwhm_deg": case["gaussian_fwhm_deg"],
                    "lorentzian_fwhm_deg": case["lorentzian_fwhm_deg"],
                    "source_native_moments": _moments(x, source_native),
                    "comparisons_by_gsasii_sh_over_l": comparisons,
                }
            )
    if len(formal_phase_smith_identity) != len(results):
        raise ValueError("oracle cases do not contain the formal GSAS-II SH/L sum")
    l1_by_parameter = {
        value: [
            result["comparisons_by_gsasii_sh_over_l"][value]["gsasii_against_source_native"][
                "normalized_l1_distance"
            ]
            for result in results
        ]
        for value in ("0.002", "0.0097", formal)
    }
    same_parameter = [
        result["comparisons_by_gsasii_sh_over_l"][formal][
            "gsasii_against_published_equal_height_same_parameter"
        ]
        for result in results
    ]
    closest_by_case = [
        min(l1_by_parameter, key=lambda value: l1_by_parameter[value][index])
        for index in range(len(results))
    ]
    review = {
        "source_native_equals_formal_phasesmith_mapping": all(formal_phase_smith_identity),
        "formal_sum_is_closest_gsasii_parameter": all(
            formal_value <= min(values)
            for formal_value, *values in zip(
                l1_by_parameter[formal],
                l1_by_parameter["0.0097"],
                l1_by_parameter["0.002"],
                strict=True,
            )
        ),
        "closest_tested_gsasii_parameter_by_case": closest_by_case,
        "closest_tested_maximum_l1_distance": max(
            l1_by_parameter[value][index] for index, value in enumerate(closest_by_case)
        ),
        "formal_sum_maximum_l1_distance": max(l1_by_parameter[formal]),
        "formal_sum_maximum_normalized_profile_error": max(
            item["normalized_maximum_abs_error"] for item in same_parameter
        ),
        "formal_sum_maximum_abs_centroid_delta_deg": max(
            abs(item["candidate_minus_reference_centroid_deg"]) for item in same_parameter
        ),
        "conclusion": (
            "The native PhaseSmith conversion preserves the deposited separate S/L and H/L "
            "geometry exactly. The documented GSAS-II formal-sum field is not shape-faithful "
            "for these low-angle source peaks under the pinned discretized kernel; its "
            "empirically closer tested value must not replace the physical conversion."
        ),
    }
    report = {
        "schema_version": 1,
        "scope": "iucr_trirubidium_citrate_source_axial_profile_fidelity",
        "source_dataset_id": oracle["source_dataset_id"],
        "revision": oracle["revision"],
        "source_geometry": {"S/L": source["S/L"], "H/L": source["H/L"]},
        "oracle_boundary": oracle["oracle_boundary"],
        "interpretation": {
            "phasesmith_conversion": "retain separate source S/L and H/L ratios",
            "gsasii_formal_sum_parameter": float(formal),
            "gsasii_parameter_action": (
                "retain the documented formal sum for declared common-input comparisons; "
                "do not retune it to imitate the deposited curve"
            ),
            "remaining_gap": (
                "the pinned GSAS-II one-parameter discretization cannot establish the "
                "deposited legacy profile-function-4 shape at this geometry"
            ),
        },
        "cases": results,
        "review": review,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
