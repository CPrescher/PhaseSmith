from pathlib import Path

import pytest
from phasesmith.refinement.core import TerminationReason
from phasesmith.validation import (
    QARR_1G_CUKA_FIXED_DISPERSION,
    qarr_1g_readiness,
    run_echidna_lab6_validation,
    run_nist_srm660c_validation,
    run_powgen_tof_readiness,
    run_powgen_tof_validation,
    run_qarr_1g_validation,
    run_qarr_1h_validation,
)
from phasesmith.validation.real_data import (
    RealDataValidationReport,
    ValidationCheck,
    _native_validation_report,
    _pbso4_stage_termination_is_safe,
    _qarr_cancelled_report,
)


@pytest.mark.parametrize(
    ("reason", "accepted_iterations", "starting_rwp", "final_rwp", "expected"),
    [
        (TerminationReason.REPEATED_REJECTIONS, 48, 0.15848610, 0.10346042, True),
        (TerminationReason.REPEATED_REJECTIONS, 0, 0.15848610, 0.10346042, False),
        (TerminationReason.REPEATED_REJECTIONS, 3, 0.15848610, 0.15848609, False),
        (TerminationReason.NUMERICAL_FAILURE, 48, 0.15848610, 0.10346042, False),
        (TerminationReason.CONVERGED, 3, 0.2, 0.1, True),
    ],
)
def test_pbso4_repeated_rejection_policy_requires_accepted_material_improvement(
    reason: TerminationReason,
    accepted_iterations: int,
    starting_rwp: float,
    final_rwp: float,
    expected: bool,
) -> None:
    assert (
        _pbso4_stage_termination_is_safe(
            reason, accepted_iterations, starting_rwp, final_rwp
        )
        is expected
    )


def test_qarr_readiness_reports_structural_doublet_capability(tmp_path: Path) -> None:
    rows = "\n".join(f"{5.0 + 0.02 * index:.3f} 100" for index in range(7_251))
    (tmp_path / "cpd-1g.prn").write_text(rows + "\n", encoding="utf-8")
    (tmp_path / "cuka.instprm").write_text(
        "Lam1:1.54051\nLam2:1.54433\nI(L2)/I(L1):0.5\n", encoding="utf-8"
    )

    report = qarr_1g_readiness(tmp_path)

    assert report.status == "passed"
    assert report.sample_count == 7_251
    assert report.to_record()["status"] == "passed"
    assert report.checks[-1].check_id == "structural_doublet_refinement"
    assert report.checks[-1].status == "passed"


def test_qarr_structural_runner_and_fixed_dispersion_are_public() -> None:
    assert callable(run_echidna_lab6_validation)
    assert callable(run_qarr_1g_validation)
    assert callable(run_qarr_1h_validation)
    assert callable(run_nist_srm660c_validation)
    assert callable(run_powgen_tof_readiness)
    assert callable(run_powgen_tof_validation)
    assert set(QARR_1G_CUKA_FIXED_DISPERSION) == {"Al", "O", "Zn", "Ca", "F"}
    assert all(value.imag > 0.0 for value in QARR_1G_CUKA_FIXED_DISPERSION.values())


def test_qarr_cancellation_is_a_blocked_last_accepted_report() -> None:
    report = _qarr_cancelled_report(
        sample_count=7_251,
        reflection_count=110,
        elapsed_seconds=0.5,
        stage="stage 2",
        rwp=0.2,
    )
    assert report.status == "blocked"
    assert report.reflection_count == 110
    assert report.checks[0].check_id == "cooperative_cancellation"
    assert "stage 2" in report.checks[0].detail
    assert report.notes == ("Last accepted Poisson-weighted Rwp=0.20000000.",)


def test_native_validation_json_reconstructs_the_public_report(monkeypatch) -> None:
    record = {
        "dataset_id": "native-case",
        "status": "passed",
        "sample_count": 2,
        "reflection_count": 1,
        "elapsed_seconds": 0.25,
        "checks": [
            {
                "check_id": "gate",
                "status": "passed",
                "detail": "native report preserved",
                "measured": 0.5,
                "criterion": "finite",
            }
        ],
        "notes": ["Rust-owned calculation."],
    }
    monkeypatch.setattr(
        "phasesmith.validation.real_data._core._run_native_validation",
        lambda runner, directory: __import__("json").dumps(record),
    )
    report = _native_validation_report("native-case", Path("dataset"))
    assert report.to_record() == record


def test_report_status_must_summarize_check_statuses() -> None:
    with pytest.raises(ValueError, match="summarize"):
        RealDataValidationReport(
            dataset_id="case",
            status="passed",
            sample_count=1,
            reflection_count=None,
            elapsed_seconds=0.0,
            checks=(
                ValidationCheck(
                    "blocked", "blocked", "not implemented", criterion="complete workflow"
                ),
            ),
        )


def test_report_rejects_untyped_checks_and_empty_notes() -> None:
    with pytest.raises(TypeError, match="ValidationCheck"):
        RealDataValidationReport(
            dataset_id="case",
            status="passed",
            sample_count=1,
            reflection_count=None,
            elapsed_seconds=0.0,
            checks=("not a check",),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="notes"):
        RealDataValidationReport(
            dataset_id="case",
            status="passed",
            sample_count=1,
            reflection_count=None,
            elapsed_seconds=0.0,
            checks=(ValidationCheck("ok", "passed", "complete", criterion="finite result"),),
            notes=("",),
        )


def test_report_rejects_duplicate_check_ids_and_missing_criteria() -> None:
    with pytest.raises(ValueError, match="criterion"):
        ValidationCheck("missing", "passed", "no acceptance criterion")
    check = ValidationCheck("same", "passed", "first", criterion="finite")
    with pytest.raises(ValueError, match="unique"):
        RealDataValidationReport(
            dataset_id="case",
            status="passed",
            sample_count=1,
            reflection_count=None,
            elapsed_seconds=0.0,
            checks=(check, check),
        )
