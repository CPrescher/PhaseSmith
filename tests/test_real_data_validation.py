from pathlib import Path

import pytest
from phasesmith.validation import qarr_1g_readiness
from phasesmith.validation.real_data import RealDataValidationReport, ValidationCheck


def test_qarr_readiness_reports_structural_doublet_blocker(tmp_path: Path) -> None:
    rows = "\n".join(f"{5.0 + 0.02 * index:.3f} 100" for index in range(7_251))
    (tmp_path / "cpd-1g.prn").write_text(rows + "\n", encoding="utf-8")
    (tmp_path / "cuka.instprm").write_text(
        "Lam1:1.54051\nLam2:1.54433\nI(L2)/I(L1):0.5\n", encoding="utf-8"
    )

    report = qarr_1g_readiness(tmp_path)

    assert report.status == "blocked"
    assert report.sample_count == 7_251
    assert report.to_record()["status"] == "blocked"
    assert report.checks[-1].check_id == "structural_doublet_refinement"


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
