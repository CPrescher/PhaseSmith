#!/usr/bin/env python3
"""Run the controlled citrate deposited-shape Stephens ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phasesmith.validation.citrate_stephens_ablation import run_citrate_stephens_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("tripotassium", "trirubidium"))
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {arguments.report}")
    result = run_citrate_stephens_ablation(arguments.data_directory, arguments.case)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    record = result.to_record()
    arguments.report.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
