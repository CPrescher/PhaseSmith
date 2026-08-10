#!/usr/bin/env python3
"""Compare deterministic scientific fingerprints in two schema-2 validation suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phasesmith.validation import compare_validation_suite_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()

    baseline = json.loads(arguments.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(arguments.candidate.read_text(encoding="utf-8"))
    differences = compare_validation_suite_records(baseline, candidate)
    if not differences:
        print("validation scientific fingerprints match")
        return 0
    for difference in differences:
        print(difference)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
