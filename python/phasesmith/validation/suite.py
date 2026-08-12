"""Versioned orchestration and scientific fingerprints for real-data validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ..radiation import RadiationProbe
from .datasets import validation_dataset
from .real_data import (
    RealDataValidationReport,
    run_echidna_lab6_validation,
    run_nickel_tof_validation,
    run_nist_srm660c_validation,
    run_pbso4_cw_validation,
    run_powgen_tof_validation,
    run_qarr_1g_validation,
    run_qarr_1h_validation,
    run_sucrose_lebail_validation,
)

ValidationPurpose = Literal["acceptance", "oracle_integrity", "holdout", "capability"]
ValidationStatus = Literal["passed", "failed", "blocked"]


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """One stable runner configuration and its reviewed expected outcome."""

    case_id: str
    dataset_id: str
    purpose: ValidationPurpose
    expected_status: ValidationStatus

    def __post_init__(self) -> None:
        dataset = validation_dataset(self.dataset_id)
        if self.purpose != dataset.purpose or self.expected_status != dataset.expected_status:
            raise ValueError("validation case policy must match its dataset manifest")
        if not self.case_id or not self.case_id.replace("-", "").isalnum():
            raise ValueError("validation case ID must be alphanumeric with hyphens")


VALIDATION_CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        "ansto-echidna-lab6-cw-neutron",
        "ansto-echidna-lab6-cw-neutron",
        "acceptance",
        "passed",
    ),
    ValidationCase("aps-sucrose-11bmb", "aps-sucrose-11bmb", "acceptance", "passed"),
    ValidationCase("gsasii-pbso4-cw-x-ray", "gsasii-pbso4-cw", "acceptance", "passed"),
    ValidationCase("gsasii-pbso4-cw-neutron", "gsasii-pbso4-cw", "acceptance", "passed"),
    ValidationCase("iucr-qarr-1g", "iucr-qarr-1g", "acceptance", "passed"),
    ValidationCase("iucr-qarr-1h", "iucr-qarr-1h", "holdout", "failed"),
    ValidationCase(
        "nist-srm660c-lab6-xray",
        "nist-srm660c-lab6-xray",
        "oracle_integrity",
        "passed",
    ),
    ValidationCase(
        "lanl-nickel-tof",
        "lanl-nickel-tof",
        "acceptance",
        "passed",
    ),
    ValidationCase(
        "powgen-lab6-tof-calibration",
        "powgen-lab6-tof-calibration",
        "acceptance",
        "passed",
    ),
)


def run_validation_case(
    case: ValidationCase, dataset_directory: str | Path
) -> RealDataValidationReport:
    """Run one registered case without changing its reviewed policy."""

    directory = Path(dataset_directory)
    if case.case_id == "ansto-echidna-lab6-cw-neutron":
        return run_echidna_lab6_validation(directory)
    if case.case_id == "aps-sucrose-11bmb":
        return run_sucrose_lebail_validation(directory)
    if case.case_id == "gsasii-pbso4-cw-x-ray":
        return run_pbso4_cw_validation(directory, RadiationProbe.X_RAY)
    if case.case_id == "gsasii-pbso4-cw-neutron":
        return run_pbso4_cw_validation(directory, RadiationProbe.NEUTRON)
    if case.case_id == "iucr-qarr-1g":
        return run_qarr_1g_validation(directory)
    if case.case_id == "iucr-qarr-1h":
        return run_qarr_1h_validation(directory)
    if case.case_id == "nist-srm660c-lab6-xray":
        return run_nist_srm660c_validation(directory)
    if case.case_id == "lanl-nickel-tof":
        return run_nickel_tof_validation(directory)
    if case.case_id == "powgen-lab6-tof-calibration":
        return run_powgen_tof_validation(directory)
    raise ValueError(f"unknown validation case {case.case_id!r}")


def scientific_record(case: ValidationCase, report: RealDataValidationReport) -> dict[str, object]:
    """Return deterministic scientific state without host timing."""

    # A manifest can contain more than one independently reported case (the
    # combined PbSO4 archive is the current example), so report identity is the
    # case identity rather than the shared download-manifest identity.
    if report.dataset_id != case.case_id:
        raise ValueError("validation report returned the wrong dataset identity")
    dataset = validation_dataset(case.dataset_id)
    report_record = report.to_record()
    report_record.pop("elapsed_seconds")
    return {
        "case": asdict(case),
        "dataset_files": [
            {"name": item.name, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in dataset.files
        ],
        "report": report_record,
    }


def scientific_fingerprint(record: dict[str, object]) -> str:
    """Hash one canonical scientific record with finite round-trip JSON."""

    encoded = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_validation_suite_record(
    case_reports: tuple[tuple[ValidationCase, RealDataValidationReport], ...],
    *,
    generated_at_utc: str,
    phasesmith_version: str,
    phasesmith_revision: str | None,
    python_version: str,
    platform: str,
) -> dict[str, object]:
    """Build schema version 2 with policy-aware suite status and fingerprints."""

    case_records: list[dict[str, object]] = []
    expectations_met = True
    for case, report in case_reports:
        scientific = scientific_record(case, report)
        matches = report.status == case.expected_status
        expectations_met &= matches
        case_records.append(
            {
                **scientific,
                "elapsed_seconds": report.elapsed_seconds,
                "expectation_met": matches,
                "scientific_fingerprint_sha256": scientific_fingerprint(scientific),
            }
        )
    return {
        "schema_version": 2,
        "status": "passed" if expectations_met else "failed",
        "generated_at_utc": generated_at_utc,
        "phasesmith_version": phasesmith_version,
        "phasesmith_revision": phasesmith_revision,
        "environment": {"python_version": python_version, "platform": platform},
        "cases": case_records,
    }


def compare_validation_suite_records(
    baseline: dict[str, object], candidate: dict[str, object]
) -> tuple[str, ...]:
    """Return stable case-level scientific differences between schema-2 suites."""

    if baseline.get("schema_version") != 2 or candidate.get("schema_version") != 2:
        raise ValueError("validation suite comparison requires schema version 2")
    baseline_cases = {
        item["case"]["case_id"]: item
        for item in baseline.get("cases", [])
        if isinstance(item, dict) and isinstance(item.get("case"), dict)
    }
    candidate_cases = {
        item["case"]["case_id"]: item
        for item in candidate.get("cases", [])
        if isinstance(item, dict) and isinstance(item.get("case"), dict)
    }
    differences = []
    for case_id in sorted(baseline_cases.keys() | candidate_cases.keys()):
        if case_id not in baseline_cases:
            differences.append(f"{case_id}: added")
        elif case_id not in candidate_cases:
            differences.append(f"{case_id}: missing")
        elif baseline_cases[case_id].get("scientific_fingerprint_sha256") != candidate_cases[
            case_id
        ].get("scientific_fingerprint_sha256"):
            differences.append(f"{case_id}: scientific result changed")
    return tuple(differences)
