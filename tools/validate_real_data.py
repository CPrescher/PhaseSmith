#!/usr/bin/env python3
"""Fetch/verify and run PhaseSmith's external real-data validation cases."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from phasesmith import TerminalCancellationController
from phasesmith.refinement import ConsoleRefinementLogger
from phasesmith.validation import (
    VALIDATION_CASES,
    build_validation_suite_record,
    compare_validation_suite_records,
    fetch_validation_dataset,
    run_qarr_1g_validation,
    run_validation_case,
    verify_validation_dataset,
)


def repository_revision() -> str | None:
    """Return the checked-out revision when the source tree has Git metadata."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and len(revision) == 40 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("validation/data"))
    parser.add_argument("--fetch", action="store_true", help="download missing pinned data")
    parser.add_argument("--output", type=Path, help="write the complete JSON record")
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="also run reviewed expected-failure and blocked cases",
    )
    parser.add_argument("--compare", type=Path, help="compare scientific fingerprints to schema 2")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing --output")
    parser.add_argument(
        "--interactive-qarr",
        action="store_true",
        help="run QARR 1g through the cancellable independent Python path",
    )
    args = parser.parse_args()

    cases = tuple(
        case
        for case in VALIDATION_CASES
        if args.include_diagnostics or case.expected_status == "passed"
    )
    case_reports = []
    prepared: set[str] = set()
    for case in cases:
        destination = args.data_directory / case.dataset_id
        if case.dataset_id not in prepared:
            if args.fetch:
                fetch_validation_dataset(case.dataset_id, destination)
            else:
                verify_validation_dataset(case.dataset_id, destination)
            prepared.add(case.dataset_id)
        if case.case_id == "iucr-qarr-1g" and args.interactive_qarr:
            with TerminalCancellationController() as controller:
                report = run_qarr_1g_validation(
                    destination,
                    cancellation=controller.token,
                    logger=ConsoleRefinementLogger(),
                )
        else:
            report = run_validation_case(case, destination)
        case_reports.append((case, report))
        expectation = "matched" if report.status == case.expected_status else "MISMATCH"
        print(f"{case.case_id}: {report.status} (expected {case.expected_status}, {expectation})")
        for check in report.checks:
            measured = "" if check.measured is None else f" measured={check.measured:.8g}"
            print(f"  {check.status:7s} {check.check_id}{measured}")

    try:
        package_version = version("phasesmith")
    except PackageNotFoundError:
        package_version = "source-tree"
    record = build_validation_suite_record(
        tuple(case_reports),
        generated_at_utc=datetime.now(UTC).isoformat(),
        phasesmith_version=package_version,
        phasesmith_revision=repository_revision(),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
    differences: tuple[str, ...] = ()
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        differences = compare_validation_suite_records(baseline, record)
        for difference in differences:
            print(f"DIFF {difference}")
    if args.output is not None:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to replace {args.output}; pass --overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 1 if record["status"] != "passed" or differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
