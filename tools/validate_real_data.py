#!/usr/bin/env python3
"""Fetch/verify and run PhaseSmith's external real-data validation cases."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from phasesmith import TerminalCancellationController
from phasesmith.radiation import RadiationProbe
from phasesmith.refinement import ConsoleRefinementLogger
from phasesmith.validation import (
    fetch_validation_dataset,
    run_pbso4_cw_validation,
    run_qarr_1g_validation,
    run_sucrose_lebail_validation,
    verify_validation_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("validation/data"))
    parser.add_argument("--fetch", action="store_true", help="download missing pinned data")
    parser.add_argument("--output", type=Path, help="write the complete JSON record")
    args = parser.parse_args()

    reports = []
    workflows = (
        ("aps-sucrose-11bmb", run_sucrose_lebail_validation),
        ("iucr-qarr-1g", run_qarr_1g_validation),
        (
            "gsasii-pbso4-cw",
            lambda directory: run_pbso4_cw_validation(directory, RadiationProbe.X_RAY),
        ),
        (
            "gsasii-pbso4-cw",
            lambda directory: run_pbso4_cw_validation(directory, RadiationProbe.NEUTRON),
        ),
    )
    prepared: set[str] = set()
    for dataset_id, runner in workflows:
        destination = args.data_directory / dataset_id
        if dataset_id not in prepared:
            if args.fetch:
                fetch_validation_dataset(dataset_id, destination)
            else:
                verify_validation_dataset(dataset_id, destination)
            prepared.add(dataset_id)
        if dataset_id == "iucr-qarr-1g":
            with TerminalCancellationController() as controller:
                report = run_qarr_1g_validation(
                    destination,
                    cancellation=controller.token,
                    logger=ConsoleRefinementLogger(),
                )
        else:
            report = runner(destination)
        reports.append(report)
        print(f"{report.dataset_id}: {report.status}")
        for check in report.checks:
            measured = "" if check.measured is None else f" measured={check.measured:.8g}"
            print(f"  {check.status:7s} {check.check_id}{measured}")

    try:
        package_version = version("phasesmith")
    except PackageNotFoundError:
        package_version = "source-tree"
    record = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phasesmith_version": package_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "reports": [report.to_record() for report in reports],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 1 if any(report.status != "passed" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
