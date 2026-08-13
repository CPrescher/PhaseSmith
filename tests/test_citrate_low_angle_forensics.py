from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "oracle/scripts/benchmark_citrate_low_angle_forensics.py"
)
SPEC = importlib.util.spec_from_file_location("citrate_low_angle_forensics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_physical_profile_scans_are_positive_and_include_source_values() -> None:
    assert set(MODULE.PHYSICAL_SCANS) == set(MODULE.SCAN_PARAMETER)
    assert all(value >= 0.0 for values in MODULE.PHYSICAL_SCANS.values() for value in values)
    assert 5.109 in MODULE.PHYSICAL_SCANS["gaussian_width_W"]
    assert 3.634 in MODULE.PHYSICAL_SCANS["lorentzian_size_X"]
    assert 0.0194 in MODULE.PHYSICAL_SCANS["fcj_asymmetry"]
    assert min(MODULE.PHYSICAL_SCANS["fcj_asymmetry"]) == 0.002


def test_peak_group_segments_split_significant_groups_at_valleys() -> None:
    x = np.linspace(17.0, 30.0, 1_301)
    signal = (
        100.0 * np.exp(-0.5 * ((x - 20.0) / 0.08) ** 2)
        + 80.0 * np.exp(-0.5 * ((x - 23.0) / 0.10) ** 2)
        + 60.0 * np.exp(-0.5 * ((x - 27.0) / 0.12) ** 2)
    )
    segments = MODULE.peak_group_segments(x, signal)
    assert len(segments) == 3
    assert [x[peak] for _, _, peak in segments] == pytest.approx([20.0, 23.0, 27.0])
    assert segments[0][0] == 0
    assert segments[-1][1] == x.size


def test_peak_group_amplitude_projection_recovers_nonnegative_scales() -> None:
    x = np.linspace(17.0, 30.0, 1_301)
    first = 100.0 * np.exp(-0.5 * ((x - 20.0) / 0.08) ** 2)
    second = 80.0 * np.exp(-0.5 * ((x - 25.0) / 0.11) ** 2)
    background = 20.0 + 0.1 * (x - np.mean(x))
    calculated = background + first + second
    observed = background + 1.4 * first + 0.6 * second
    segments = MODULE.peak_group_segments(x, calculated - background)
    result = MODULE.fit_peak_group_amplitudes(x, observed, calculated, background, segments)
    assert result["poisson_rwp"] < 1.0e-12
    assert result["amplitudes"] == pytest.approx([1.4, 0.6], abs=1.0e-10)
    assert result["active_group_count"] == 2


def test_peak_group_moments_distinguish_area_width_and_centroid() -> None:
    x = np.linspace(17.0, 30.0, 1_301)
    current = np.exp(-0.5 * ((x - 23.0) / 0.10) ** 2)
    target = 1.5 * np.exp(-0.5 * ((x - 23.02) / 0.13) ** 2)
    segments = ((0, x.size, int(np.argmax(current))),)
    result = MODULE.peak_group_moment_comparison(x, current, target, segments)
    row = result["groups"][0]
    assert row["target_over_current_area"] == pytest.approx(1.95, rel=1.0e-6)
    assert row["target_minus_current_centroid_deg"] == pytest.approx(0.02, abs=1.0e-10)
    assert row["target_over_current_rms_width"] == pytest.approx(1.3, rel=1.0e-6)


def test_residual_background_projection_recovers_legendre_shape() -> None:
    x = np.linspace(17.0, 30.0, 301)
    background = np.full_like(x, 20.0)
    calculated = background + 100.0 * np.exp(-0.5 * ((x - 23.0) / 0.3) ** 2)
    normalized_x = 2.0 * (x - x[0]) / (x[-1] - x[0]) - 1.0
    correction = np.polynomial.legendre.legvander(normalized_x, 3) @ np.array([4.0, -2.0, 1.0, 0.5])
    result = MODULE.fit_residual_background(x, calculated + correction, calculated, background)
    assert result["poisson_rwp"] < 1.0e-14
    assert result["rank"] == 4
    assert result["condition_number"] < 3.0
    assert result["coefficients"] == pytest.approx([4.0, -2.0, 1.0, 0.5], abs=1e-12)


def test_metrics_reject_invalid_arrays() -> None:
    values = np.ones(5)
    with pytest.raises(ValueError, match="finite aligned"):
        MODULE.weighted_metrics(values, values[:-1], values)


def test_reviewed_report_records_identifiable_coupled_gap() -> None:
    report_path = (
        Path(__file__).resolve().parents[1]
        / "validation/results/2026-08-13-citrate-rubidium-low-angle-forensics.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["revision"] == MODULE.PINNED_REVISION
    assert report["limits_deg"] == pytest.approx(MODULE.LOW_ANGLE_LIMITS_DEG)
    assert report["sample_count"] == 643
    assert all(value is True for key, value in report["review"].items() if key != "conclusion")

    combined = report["physical_profile_scans"]["combined_best_physical_shape_and_background"]
    assert combined["instrument_overrides"] == {"SH/L": 0.002, "W": 20.0, "X": 5.0}
    assert combined["background_projection"]["rank"] == 4
    assert combined["background_projection"]["condition_number"] < 3.0
    assert combined["fraction_of_base_to_deposited_sse_gap_closed"] == pytest.approx(
        0.8047262134145285
    )
    assert combined["low_angle"]["poisson_rwp"] > report["deposited_curve_low_angle"]["poisson_rwp"]
    assert report["peak_group_intensity_projection"]["group_count"] == 16
