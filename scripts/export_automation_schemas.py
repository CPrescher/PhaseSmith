"""Export or verify PhaseSmith's versioned automation JSON Schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phasesmith.automation import automation_schema, automation_schema_names


def _encoded(name: str) -> str:
    return json.dumps(automation_schema(name), allow_nan=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "schemas" / "automation",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()

    failures = []
    for name in automation_schema_names():
        path = directory / f"{name}-v1.schema.json"
        expected = _encoded(name)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                failures.append(str(path))
            continue
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to replace existing schema: {path}")
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")

    if failures:
        raise SystemExit(
            "automation schema files differ from the live API:\n" + "\n".join(failures)
        )
    if args.check:
        print(f"{len(automation_schema_names())} automation schemas match {directory}")
    else:
        print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
