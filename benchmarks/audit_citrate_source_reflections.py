#!/usr/bin/env python3
"""Audit Rb-citrate converted structures against source-deposited F-squared values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phasesmith.validation import run_citrate_source_reflection_fidelity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = run_citrate_source_reflection_fidelity(arguments.data_directory)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    if arguments.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {arguments.report}")
    arguments.report.write_text(
        json.dumps(result.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
