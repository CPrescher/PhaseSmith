"""Verify a local single-crystal dataset without copying or downloading it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "validation/enstatite-mgsio3-local-v1.json"


def verify_file(root: Path, record: dict) -> dict:
    """Check a manifest-relative regular file using bounded-memory hashing."""
    root = root.resolve(strict=True)
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"dataset path must be relative: {relative}")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"dataset file escapes root or is not a file: {relative}")
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"dataset byte-size mismatch: {relative}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != record["sha256"]:
        raise ValueError(f"dataset SHA-256 mismatch: {relative}")
    return {"path": record["path"], "sha256": actual, "size_bytes": record["size_bytes"]}


def verify_dataset(manifest: dict, root: Path) -> dict:
    """Return input verification only; no scientific workflow is executed."""
    if manifest.get("schema") != "phasesmith.local-single-crystal-data.v1":
        raise ValueError("unsupported local single-crystal manifest schema")
    records = manifest["files"]
    if not records or len({record["path"] for record in records}) != len(records):
        raise ValueError("dataset files must be nonempty and unique")
    verified = [verify_file(root, record) for record in records]
    if sum(record["size_bytes"] for record in verified) != manifest["total_size_bytes"]:
        raise ValueError("dataset total byte size differs from manifest")
    return {
        "dataset_id": manifest["dataset_id"],
        "input_verification": "passed",
        "scientific_workflow": "not_executed",
        "files": verified,
        "total_size_bytes": manifest["total_size_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify_dataset(json.loads(args.manifest.read_text()), args.data_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Input verification failed: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
