#!/usr/bin/env python3
"""Run one controlled citrate/Si isotropic-broadening ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phasesmith.validation import run_citrate_isotropic_broadening_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("tripotassium", "trirubidium"))
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {arguments.report}")
    result = run_citrate_isotropic_broadening_ablation(
        arguments.data_directory,
        arguments.case,
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(result.to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.to_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
