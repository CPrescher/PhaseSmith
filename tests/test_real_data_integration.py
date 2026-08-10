from __future__ import annotations

from pathlib import Path

import phasesmith
import pytest
from phasesmith.io import convert_rowles_topas_bundle
from phasesmith.validation import (
    run_echidna_lab6_validation,
    run_nist_srm660c_validation,
    run_pbso4_cw_validation,
    run_powgen_tof_validation,
    run_qarr_1g_validation,
    run_qarr_1h_validation,
    run_rowles_qpa_workflow,
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
def test_echidna_real_pattern_regression_is_deterministic() -> None:
    directory = available_dataset("ansto-echidna-lab6-cw-neutron")
    first = run_echidna_lab6_validation(directory)
    second = run_echidna_lab6_validation(directory)

    assert first.status == "passed"
    assert first.sample_count == 2_111
    assert first.reflection_count == 13
    assert first.to_record() | {"elapsed_seconds": 0.0} == second.to_record() | {
        "elapsed_seconds": 0.0
    }


@pytest.mark.real_data
def test_nist_archive_integrity_regression() -> None:
    report = run_nist_srm660c_validation(available_dataset("nist-srm660c-lab6-xray"))

    assert report.status == "passed"
    assert report.sample_count == 106_640
    assert report.reflection_count == 24
    assert {check.check_id for check in report.checks} == {
        "archive_specimens",
        "pdcif_profiles",
        "certified_lattice_interval",
        "nist_reference_profile",
        "nist_specimen_correlation",
        "nist_specimen_rwp",
    }


@pytest.mark.real_data
def test_qarr_1h_holdout_preserves_the_reviewed_failure_signal() -> None:
    report = run_qarr_1h_validation(available_dataset("iucr-qarr-1h"))
    checks = {check.check_id: check for check in report.checks}

    assert report.status == "failed"
    assert checks["qpa_weight_fraction"].status == "passed"
    assert checks["poisson_rwp"].status == "failed"
    assert checks["unit_weight_rwp"].status == "failed"


@pytest.mark.real_data
def test_powgen_tof_complete_workflow_passes() -> None:
    report = run_powgen_tof_validation(available_dataset("powgen-lab6-tof-calibration"))
    checks = {check.check_id: check for check in report.checks}

    assert report.status == "passed"
    assert report.sample_count == 6_824
    assert report.reflection_count == 330
    assert checks["tof_fxye_grid"].status == "passed"
    assert checks["tof_position_kernel"].status == "passed"
    assert checks["tof_position_derivative"].status == "passed"
    assert checks["tof_calibration_range"].status == "passed"
    assert checks["tof_profile_improvement"].status == "passed"
    assert checks["tof_profile_correlation"].status == "passed"
    assert checks["tof_integrated_intensities"].status == "passed"
    assert checks["tof_reflection_coverage"].status == "passed"
    assert checks["tof_analytical_derivatives"].status == "passed"
    assert checks["tof_chebyshev_background"].status == "passed"


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
    assert measurements["qpa_covariance"] >= 0.0


@pytest.mark.real_data
@pytest.mark.parametrize(
    ("sample", "maximum_qpa_error", "maximum_rwp"),
    [("1a", 0.01, 0.095), ("1e", 0.03, 0.09)],
)
def test_rowles_topas_conversion_and_phasesmith_qpa_regression(
    tmp_path: Path, sample: str, maximum_qpa_error: float, maximum_rwp: float
) -> None:
    source = available_dataset("curtin-rowles-qpa-topas")
    bundle = tmp_path / "converted"
    convert_rowles_topas_bundle(source, bundle)

    result = run_rowles_qpa_workflow(
        bundle, sample, execution=phasesmith.ExecutionPolicy(threads=1)
    )

    assert result.sample_count in {14_066, 14_091}
    assert result.reflection_count == 109
    assert result.maximum_weight_fraction_error <= maximum_qpa_error
    assert result.poisson_rwp <= maximum_rwp
    assert result.profile_correlation >= 0.995


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
