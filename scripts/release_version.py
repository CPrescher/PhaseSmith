#!/usr/bin/env python3
"""Validate the version shared by PhaseSmith release surfaces."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")
PUBLIC_CRATES = {
    "phasesmith",
    "phasesmith-core",
    "phasesmith-crystallography",
    "phasesmith-engine",
    "phasesmith-execution",
    "phasesmith-io",
    "phasesmith-model",
    "phasesmith-persistence",
    "phasesmith-workflows",
    "phasesmith-single-crystal",
    "phasesmith-hdf5",
}


def _toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def validate_api_snapshot(root: Path, version: str) -> None:
    """Require the schema-1 public API snapshot for *version*."""

    api_snapshot_path = root / "api" / f"python-public-api-v{version}.json"
    try:
        api_snapshot = json.loads(api_snapshot_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"public API snapshot is missing: {api_snapshot_path}") from error
    if api_snapshot.get("schema_version") != 1:
        raise ValueError("public API snapshot must use schema_version=1")
    if api_snapshot.get("package_version") != version:
        raise ValueError("public API snapshot version does not match the release version")


def _citation_version(root: Path) -> str:
    citation_path = root / "CITATION.cff"
    text = citation_path.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*[\"']?([^\"'\s]+)[\"']?\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"CITATION.cff has no version field: {citation_path}")
    return match.group(1)


def _is_internal_dependency(name: str) -> bool:
    return name == "phasesmith" or name.startswith("phasesmith-")


def release_version(root: Path = ROOT) -> str:
    """Return the validated release version or raise ``ValueError``."""

    cargo = _toml(root / "Cargo.toml")
    pyproject = _toml(root / "pyproject.toml")
    versions = {
        "Cargo workspace": cargo["workspace"]["package"]["version"],
        "Python project": pyproject["project"]["version"],
        "CITATION.cff": _citation_version(root),
    }
    unique = set(versions.values())
    if len(unique) != 1:
        details = ", ".join(f"{name}={value}" for name, value in versions.items())
        raise ValueError(f"release versions disagree: {details}")
    version = str(unique.pop())
    if SEMVER.fullmatch(version) is None:
        raise ValueError(f"release version is not SemVer: {version}")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(rf"^## {re.escape(version)}$", changelog, re.MULTILINE) is None:
        raise ValueError(f"CHANGELOG.md has no '## {version}' release section")
    validate_api_snapshot(root, version)
    workspace_deps = cargo.get("workspace", {}).get("dependencies", {})
    if isinstance(workspace_deps, dict):
        for dependency_name, specification in workspace_deps.items():
            if not _is_internal_dependency(dependency_name):
                continue
            if not isinstance(specification, dict) or specification.get("version") != version:
                raise ValueError(
                    f"Cargo.toml workspace.dependencies.{dependency_name} "
                    f"must specify version={version}"
                )
    manifests = [root / "Cargo.toml", *sorted((root / "crates").glob("*/Cargo.toml"))]
    package_names = set()
    for manifest in manifests:
        record = _toml(manifest)
        package = record.get("package")
        if not isinstance(package, dict):
            continue
        package_name = str(package["name"])
        package_names.add(package_name)
        if package_name in PUBLIC_CRATES:
            packaged_license = manifest.parent / "LICENSE"
            if (
                not packaged_license.is_file()
                or packaged_license.read_bytes() != (root / "LICENSE").read_bytes()
            ):
                raise ValueError(f"public crate {package_name} must include the canonical LICENSE")
        if package_name in PUBLIC_CRATES and package.get("publish") is False:
            raise ValueError(f"public crate {package_name} is marked publish=false")
        for table_name in ("dependencies", "dev-dependencies", "build-dependencies"):
            dependencies = record.get(table_name, {})
            if not isinstance(dependencies, dict):
                continue
            for dependency_name, specification in dependencies.items():
                if not _is_internal_dependency(dependency_name):
                    continue
                if isinstance(specification, dict) and specification.get("workspace") is True:
                    ws_spec = (
                        workspace_deps.get(dependency_name)
                        if isinstance(workspace_deps, dict)
                        else None
                    )
                    if not isinstance(ws_spec, dict) or ws_spec.get("version") != version:
                        relative = manifest.relative_to(root)
                        raise ValueError(
                            f"{relative} {table_name}.{dependency_name} "
                            "inherits missing or mismatched workspace version"
                        )
                elif not isinstance(specification, dict) or specification.get("version") != version:
                    relative = manifest.relative_to(root)
                    raise ValueError(
                        f"{relative} {table_name}.{dependency_name} must specify version={version}"
                    )
    missing = PUBLIC_CRATES.difference(package_names)
    if missing:
        raise ValueError(f"public crate manifests are missing: {', '.join(sorted(missing))}")
    return version


def main() -> int:
    """Run the release-version command-line check."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Require an exact v<version> release tag")
    parser.add_argument("--github-output", type=Path, help="Append version output for Actions")
    args = parser.parse_args()
    try:
        version = release_version()
        if args.tag is not None and args.tag != f"v{version}":
            raise ValueError(f"tag {args.tag!r} does not match release version v{version}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"release version check failed: {error}", file=sys.stderr)
        return 1
    if args.github_output is not None:
        with args.github_output.open("a") as stream:
            stream.write(f"version={version}\n")
            stream.write(f"tag=v{version}\n")
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
