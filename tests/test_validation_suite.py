from __future__ import annotations

from dataclasses import replace

import pytest
from phasesmith.validation import (
    VALIDATION_CASES,
    RealDataValidationReport,
    ValidationCheck,
    build_validation_suite_record,
    compare_validation_suite_records,
)


def report(dataset_id: str, status: str) -> RealDataValidationReport:
    return RealDataValidationReport(
        dataset_id=dataset_id,
        status=status,  # type: ignore[arg-type]
        sample_count=10,
        reflection_count=2,
        elapsed_seconds=1.25,
        checks=(
            ValidationCheck(
                "scientific_gate",
                status,  # type: ignore[arg-type]
                "reviewed outcome",
                measured=0.125,
                criterion="value <= 0.2",
            ),
        ),
    )


def suite(case, result):
    return build_validation_suite_record(
        ((case, result),),
        generated_at_utc="2026-08-10T00:00:00+00:00",
        phasesmith_version="0.1.0",
        phasesmith_revision="0" * 40,
        python_version="3.13",
        platform="test",
    )


def test_reviewed_failure_and_new_tof_acceptance_make_a_passing_suite() -> None:
    holdout = next(case for case in VALIDATION_CASES if case.case_id == "iucr-qarr-1h")
    tof_acceptance = next(
        case for case in VALIDATION_CASES if case.case_id == "powgen-lab6-tof-calibration"
    )
    record = build_validation_suite_record(
        (
            (holdout, report(holdout.dataset_id, "failed")),
            (tof_acceptance, report(tof_acceptance.dataset_id, "passed")),
        ),
        generated_at_utc="2026-08-10T00:00:00+00:00",
        phasesmith_version="0.1.0",
        phasesmith_revision=None,
        python_version="3.13",
        platform="test",
    )
    assert record["status"] == "passed"
    assert all(case["expectation_met"] for case in record["cases"])


def test_fingerprint_excludes_timing_and_suite_comparison_reports_changes() -> None:
    case = VALIDATION_CASES[0]
    first_report = report(case.dataset_id, "passed")
    second_report = replace(first_report, elapsed_seconds=99.0)
    first = suite(case, first_report)
    second = suite(case, second_report)
    assert (
        first["cases"][0]["scientific_fingerprint_sha256"]
        == second["cases"][0]["scientific_fingerprint_sha256"]
    )
    assert compare_validation_suite_records(first, second) == ()

    changed_report = RealDataValidationReport(
        dataset_id=case.dataset_id,
        status="passed",
        sample_count=11,
        reflection_count=2,
        elapsed_seconds=1.0,
        checks=first_report.checks,
    )
    changed = suite(case, changed_report)
    assert compare_validation_suite_records(first, changed) == (
        f"{case.case_id}: scientific result changed",
    )


def test_split_cases_share_manifest_but_keep_distinct_report_identity() -> None:
    xray = next(case for case in VALIDATION_CASES if case.case_id == "gsasii-pbso4-cw-x-ray")
    neutron = next(case for case in VALIDATION_CASES if case.case_id == "gsasii-pbso4-cw-neutron")
    assert xray.dataset_id == neutron.dataset_id == "gsasii-pbso4-cw"

    first = suite(xray, report(xray.case_id, "passed"))
    second = suite(neutron, report(neutron.case_id, "passed"))
    assert first["cases"][0]["report"]["dataset_id"] == xray.case_id
    assert second["cases"][0]["report"]["dataset_id"] == neutron.case_id
    with pytest.raises(ValueError, match="wrong dataset identity"):
        suite(xray, report(xray.dataset_id, "passed"))
