from __future__ import annotations

from pathlib import Path

import phasesmith
import pytest
from phasesmith.validation import (
    run_pbso4_cw_validation,
    run_qarr_1g_validation,
    run_sucrose_lebail_validation,
    verify_validation_dataset,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "validation" / "data"


def available_dataset(dataset_id: str) -> Path:
    directory = DATA_ROOT / dataset_id
    try:
        verify_validation_dataset(dataset_id, directory)
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"checksum-pinned real dataset is unavailable: {error}")
    return directory


@pytest.mark.real_data
def test_sucrose_real_pattern_regression() -> None:
    report = run_sucrose_lebail_validation(available_dataset("aps-sucrose-11bmb"))

    assert report.status == "passed"
    assert report.sample_count == 23_003
    assert report.reflection_count == 811
    assert {check.check_id for check in report.checks} == {
        "profile_improvement",
        "smoke_rwp",
        "profile_correlation",
        "integrated_intensities",
    }


@pytest.mark.real_data
def test_qarr_real_pattern_two_thread_regression() -> None:
    report = run_qarr_1g_validation(
        available_dataset("iucr-qarr-1g"),
        execution=phasesmith.ExecutionPolicy(threads=2),
    )

    assert report.status == "passed"
    assert report.sample_count == 7_251
    assert report.reflection_count == 110
    measurements = {
        check.check_id: check.measured for check in report.checks if check.measured is not None
    }
    assert measurements["poisson_rwp"] <= 0.20
    assert measurements["unit_weight_rwp"] <= 0.15
    assert measurements["profile_correlation"] >= 0.98
    assert measurements["qpa_weight_fraction"] <= 0.02


@pytest.mark.real_data
@pytest.mark.parametrize(
    ("probe", "samples"),
    [
        (phasesmith.RadiationProbe.X_RAY, 5_697),
        (phasesmith.RadiationProbe.NEUTRON, 2_681),
    ],
)
def test_pbso4_real_pattern_regression(probe: phasesmith.RadiationProbe, samples: int) -> None:
    report = run_pbso4_cw_validation(
        available_dataset("gsasii-pbso4-cw"),
        probe,
    )

    assert report.status == "passed"
    assert report.sample_count == samples
    measurements = {
        check.check_id: check.measured for check in report.checks if check.measured is not None
    }
    assert measurements["poisson_rwp"] <= (
        0.11 if probe is phasesmith.RadiationProbe.X_RAY else 0.05
    )
    assert measurements["unit_weight_rwp"] <= (
        0.10 if probe is phasesmith.RadiationProbe.X_RAY else 0.06
    )
    assert measurements["profile_correlation"] >= 0.99
    assert measurements["reference_cell_relative_error"] <= 0.005
    assert any(
        "fixed native Smooth Bruckner estimate plus a refined three-term Chebyshev" in note
        for note in report.notes
    )
    if probe is phasesmith.RadiationProbe.NEUTRON:
        assert any(
            "Debye-Scherrer geometry: fixed radius=650.000 mm" in note for note in report.notes
        )
