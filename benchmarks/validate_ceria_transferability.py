#!/usr/bin/env python3
"""Run the checksum-pinned IUCr ceria profile-transferability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phasesmith.validation import run_ceria_profile_transferability

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=(REPOSITORY_ROOT / "validation" / "data" / "iucr-ceria-size-strain-round-robin"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = run_ceria_profile_transferability(arguments.dataset_directory)
    record = result.to_record()
    output = json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if arguments.json_output is not None:
        if arguments.json_output.exists() and not arguments.overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing result {arguments.json_output}; pass --overwrite"
            )
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(output, encoding="utf-8")
    if result.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
