#!/usr/bin/env python3
"""Run the checksum-pinned, stratified opXRD robustness campaign."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from phasesmith.validation.opxrd import (
    OpxrdCaseResult,
    run_opxrd_robustness_campaign,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=(REPOSITORY_ROOT / "validation" / "data" / "opxrd-robustness-v1" / "opxrd.zip"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "validation" / "opxrd-robustness-v1.json",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--structural", action="store_true")
    parser.add_argument("--structural-cycles", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    selected = None if arguments.case_ids is None else tuple(arguments.case_ids)
    with tempfile.TemporaryDirectory(prefix="phasesmith-opxrd-") as name:
        result = run_opxrd_robustness_campaign(
            arguments.archive,
            arguments.selection,
            structural=arguments.structural,
            structural_cycles=arguments.structural_cycles,
            case_ids=selected,
            work_directory=name,
        )
        record = result.to_record()
    evaluated = sum(isinstance(case, OpxrdCaseResult) for case in result.cases)
    structural = sum(
        isinstance(case, OpxrdCaseResult) and case.structural_result is not None
        for case in result.cases
    )
    print(
        f"opXRD: {result.selected_case_count} selected, {evaluated} metric-evaluated, "
        f"{result.selected_case_count - evaluated} rejected at an explicit input boundary, "
        f"{structural} structural common-model fits"
    )
    failed = [name for name, passed in result.checks.items() if not passed]
    print("checks: " + ("all passed" if not failed else ", ".join(failed)))
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
