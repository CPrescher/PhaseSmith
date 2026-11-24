from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/release_version.py"


def _release_module() -> object:
    specification = importlib.util.spec_from_file_location("release_version", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_release_version_matches_every_checked_surface() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT, "--tag", "v0.9.0"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.9.0"


def test_supported_python_versions_include_314() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["requires-python"] == ">=3.11"
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]


def test_release_version_rejects_a_mismatched_tag() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT, "--tag", "v9.9.9"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "does not match release version v0.9.0" in result.stderr


def test_api_snapshot_release_metadata_is_versioned(tmp_path: Path) -> None:
    module = _release_module()
    snapshot = tmp_path / "api" / "python-public-api-v1.2.3.json"
    snapshot.parent.mkdir()
    snapshot.write_text(
        json.dumps({"schema_version": 1, "package_version": "1.2.3"}),
        encoding="utf-8",
    )
    module.validate_api_snapshot(tmp_path, "1.2.3")

    snapshot.write_text(
        json.dumps({"schema_version": 1, "package_version": "1.2.4"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version does not match"):
        module.validate_api_snapshot(tmp_path, "1.2.3")


def test_published_crates_avoid_exact_equality_external_pins() -> None:
    module = _release_module()
    root_cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    workspace_deps = root_cargo.get("workspace", {}).get("dependencies", {})
    assert isinstance(workspace_deps, dict)
    for name, spec in workspace_deps.items():
        version = spec if isinstance(spec, str) else spec.get("version", "")
        assert not str(version).startswith("="), (
            f"workspace.dependencies.{name} uses exact pin {version!r}"
        )
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        record = tomllib.loads(manifest.read_text(encoding="utf-8"))
        package = record.get("package", {})
        if package.get("name") not in module.PUBLIC_CRATES:
            continue
        for table_name in ("dependencies", "dev-dependencies", "build-dependencies"):
            deps = record.get(table_name, {})
            if not isinstance(deps, dict):
                continue
            for dep_name, spec in deps.items():
                version = spec if isinstance(spec, str) else spec.get("version", "")
                relative = manifest.relative_to(ROOT)
                assert not str(version).startswith("="), (
                    f"{relative} {table_name}.{dep_name} uses exact pin {version!r}"
                )
