#!/usr/bin/env python3
"""Generate or verify PhaseSmith's deterministic public Python API snapshot."""

from __future__ import annotations

import argparse
import difflib
import enum
import importlib
import inspect
import json
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ADDRESS = re.compile(r"0x[0-9A-Fa-f]+")
PUBLIC_MODULES = (
    ("phasesmith", "domain"),
    ("phasesmith.io", "adapter"),
    ("phasesmith.refinement", "domain"),
    ("phasesmith.integrations", "adapter"),
    ("phasesmith.oracle", "validation"),
    ("phasesmith.validation", "validation"),
)


def project_version(root: Path = ROOT) -> str:
    """Read the Python project version without requiring an installed wheel."""

    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    return str(project["project"]["version"])


def default_snapshot_path(root: Path = ROOT) -> Path:
    """Return the versioned snapshot path for the current source tree."""

    return root / "api" / f"python-public-api-v{project_version(root)}.json"


def _kind(value: object) -> str:
    if inspect.ismodule(value):
        return "module"
    if inspect.isclass(value):
        return "class"
    if inspect.isroutine(value):
        return "function"
    if callable(value):
        return "callable"
    return "constant"


def _target(value: object) -> str:
    if isinstance(value, ModuleType):
        return value.__name__
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _signature(value: object) -> str | None:
    if not callable(value):
        return None
    if isinstance(value, enum.EnumMeta):
        return None
    try:
        signature = str(inspect.signature(value, follow_wrapped=False, eval_str=False))
    except (TypeError, ValueError):
        return None
    if ADDRESS.search(signature) is not None:
        raise ValueError(f"public signature contains a process-local address: {signature}")
    return signature


def _module_record(module_name: str, tier: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    exported = getattr(module, "__all__", None)
    if not isinstance(exported, list) or not all(isinstance(name, str) for name in exported):
        raise ValueError(f"{module_name} must define __all__ as a list of strings")
    if len(exported) != len(set(exported)):
        raise ValueError(f"{module_name}.__all__ contains duplicate names")
    entries = []
    for name in sorted(exported):
        if not hasattr(module, name):
            raise ValueError(f"{module_name}.__all__ exports missing name {name!r}")
        value = getattr(module, name)
        entries.append(
            {
                "name": name,
                "kind": _kind(value),
                "target": _target(value),
                "signature": _signature(value),
            }
        )
    return {"module": module_name, "tier": tier, "exports": entries}


def build_snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Build a deterministic JSON-compatible record from explicit exports."""

    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": project_version(root),
        "scope": (
            "Names in __all__ for the listed modules; callable signatures use "
            "inspect.signature without evaluating string annotations"
        ),
        "modules": [_module_record(module_name, tier) for module_name, tier in PUBLIC_MODULES],
    }


def encoded_snapshot(snapshot: dict[str, Any]) -> str:
    """Serialize a snapshot with stable ordering and no non-finite values."""

    return json.dumps(snapshot, allow_nan=False, indent=2, sort_keys=True) + "\n"


def verify_snapshot(path: Path, root: Path = ROOT) -> None:
    """Raise ``ValueError`` when *path* differs from the live public API."""

    expected = path.read_text(encoding="utf-8")
    actual = encoded_snapshot(build_snapshot(root))
    if expected == actual:
        return
    difference = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=str(path),
            tofile="live-public-api",
        )
    )
    raise ValueError(f"public API snapshot differs from the live package:\n{difference}")


def main() -> int:
    """Run the API snapshot command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare the snapshot to live exports")
    mode.add_argument("--write", action="store_true", help="write the live export snapshot")
    parser.add_argument("--snapshot", type=Path, help="override the versioned snapshot path")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing a snapshot")
    arguments = parser.parse_args()
    path = default_snapshot_path() if arguments.snapshot is None else arguments.snapshot
    try:
        if arguments.check:
            if arguments.overwrite:
                raise ValueError("--overwrite is valid only with --write")
            verify_snapshot(path)
            print(f"public API snapshot matches {path}")
            return 0
        if path.exists() and not arguments.overwrite:
            raise FileExistsError(f"refusing to overwrite existing snapshot {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded_snapshot(build_snapshot()), encoding="utf-8")
        print(path)
        return 0
    except (
        AttributeError,
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"public API snapshot failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
