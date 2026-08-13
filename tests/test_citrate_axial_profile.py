from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/audit_citrate_axial_profile.py"
SPEC = importlib.util.spec_from_file_location("audit_citrate_axial_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
profile_difference = MODULE.profile_difference


def test_profile_difference_is_zero_for_identical_profiles() -> None:
    x = np.linspace(-1.0, 1.0, 1001)
    profile = np.exp(-((x / 0.2) ** 2))

    result = profile_difference(x, profile, profile.copy())

    assert result["normalized_l1_distance"] == 0.0
    assert result["normalized_rms_distance"] == 0.0
    assert result["normalized_maximum_abs_error"] == 0.0
    assert result["candidate_minus_reference_centroid_deg"] == 0.0
    assert result["candidate_over_reference_rms_width"] == 1.0


def test_profile_difference_detects_a_shift() -> None:
    x = np.linspace(-1.0, 1.0, 4001)
    reference = np.exp(-((x / 0.2) ** 2))
    candidate = np.exp(-(((x - 0.03) / 0.2) ** 2))

    result = profile_difference(x, reference, candidate)

    assert result["normalized_l1_distance"] > 0.1
    assert result["candidate_minus_reference_centroid_deg"] == pytest.approx(0.03, abs=1.0e-10)
    assert result["candidate_over_reference_rms_width"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "x, reference, candidate",
    [
        (np.array([0.0, 1.0, 0.5, 2.0, 3.0]), np.ones(5), np.ones(5)),
        (np.arange(5.0), np.ones(4), np.ones(5)),
        (np.arange(5.0), np.ones(5), np.array([1.0, 1.0, -1.0, 1.0, 1.0])),
        (np.arange(5.0), np.zeros(5), np.ones(5)),
    ],
)
def test_profile_difference_rejects_invalid_arrays(
    x: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> None:
    with pytest.raises(ValueError, match=r"invalid|positive"):
        profile_difference(x, reference, candidate)


def test_reviewed_axial_report_preserves_native_geometry_and_rejects_oracle_shape() -> None:
    report_path = (
        Path(__file__).resolve().parents[1]
        / "validation/results/2026-08-13-citrate-rubidium-source-axial-profile-fidelity.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["revision"] == "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
    assert report["source_geometry"] == {"S/L": 0.0097, "H/L": 0.0097}
    assert report["interpretation"]["gsasii_formal_sum_parameter"] == 0.0194
    review = report["review"]
    assert review["source_native_equals_formal_phasesmith_mapping"] is True
    assert review["formal_sum_is_closest_gsasii_parameter"] is False
    assert review["closest_tested_gsasii_parameter_by_case"] == ["0.0097"] * 3
    assert review["closest_tested_maximum_l1_distance"] == pytest.approx(0.04246592362537831)
    assert review["formal_sum_maximum_l1_distance"] == pytest.approx(0.17529864315052218)
    assert review["formal_sum_maximum_normalized_profile_error"] == pytest.approx(
        0.1537823560994232
    )
    assert review["formal_sum_maximum_abs_centroid_delta_deg"] == pytest.approx(
        0.008469418902411263
    )
