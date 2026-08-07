from pathlib import Path

import pytest
from phasesmith.validation import (
    QARR_1G_CUKA_FIXED_DISPERSION,
    qarr_1g_readiness,
    run_qarr_1g_validation,
)
from phasesmith.validation.real_data import (
    RealDataValidationReport,
    ValidationCheck,
    _qarr_cancelled_report,
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
    assert callable(run_qarr_1g_validation)
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


def test_report_status_must_summarize_check_statuses() -> None:
    with pytest.raises(ValueError, match="summarize"):
        RealDataValidationReport(
            dataset_id="case",
            status="passed",
            sample_count=1,
            reflection_count=None,
            elapsed_seconds=0.0,
            checks=(ValidationCheck("blocked", "blocked", "not implemented"),),
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
            checks=(ValidationCheck("ok", "passed", "complete"),),
            notes=("",),
        )
