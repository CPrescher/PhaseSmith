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
    # Scientific database crates whose exact version is exported in public provenance constants.
    allowed_exact_pins = {"moyo"}
    root_cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    workspace_deps = root_cargo.get("workspace", {}).get("dependencies", {})
    assert isinstance(workspace_deps, dict)
    for name, spec in workspace_deps.items():
        if name in allowed_exact_pins:
            continue
        version = spec if isinstance(spec, str) else spec.get("version", "")
        assert not str(version).startswith("="), (
            f"workspace.dependencies.{name} uses exact pin {version!r}"
        )
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        record = tomllib.loads(manifest.read_text(encoding="utf-8"))
        package = record.get("package", {})
        if package.get("name") not in module.PUBLIC_CRATES:
            continue
        for table_name, deps in module._iter_dependency_tables(record):
            for dep_name, spec in deps.items():
                if dep_name in allowed_exact_pins:
                    continue
                version = spec if isinstance(spec, str) else spec.get("version", "")
                relative = manifest.relative_to(ROOT)
                assert not str(version).startswith("="), (
                    f"{relative} {table_name}.{dep_name} uses exact pin {version!r}"
                )


def _write_minimal_release_tree(
    root: Path,
    *,
    version: str = "1.2.3",
    citation_text: str | None = None,
    workspace_dep_version: str = "1.2.3",
    py_facade_spec: str = "{ workspace = true }",
) -> None:
    module = _release_module()
    root.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n## {version}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "phasesmith"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        citation_text if citation_text is not None else f"cff-version: 1.2.0\nversion: {version}\n",
        encoding="utf-8",
    )
    snapshot = root / "api" / f"python-public-api-v{version}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(
        json.dumps({"schema_version": 1, "package_version": version}),
        encoding="utf-8",
    )
    ws_lines = [
        "[workspace]",
        "members = []",
        "[workspace.package]",
        f'version = "{version}"',
        "[workspace.dependencies]",
    ]
    for crate_name in sorted(module.PUBLIC_CRATES):
        entry = f'{{ version = "{workspace_dep_version}", path = "crates/{crate_name}" }}'
        ws_lines.append(f"{crate_name} = {entry}")
        crate_dir = root / "crates" / crate_name
        crate_dir.mkdir(parents=True, exist_ok=True)
        (crate_dir / "LICENSE").write_text("MIT\n", encoding="utf-8")
        (crate_dir / "Cargo.toml").write_text(
            f'[package]\nname = "{crate_name}"\nversion.workspace = true\n',
            encoding="utf-8",
        )
    py_dir = root / "crates" / "phasesmith-py"
    py_dir.mkdir(parents=True, exist_ok=True)
    (py_dir / "Cargo.toml").write_text(
        '[package]\nname = "phasesmith-py"\nversion.workspace = true\npublish = false\n'
        f"[dependencies]\nphasesmith = {py_facade_spec}\n",
        encoding="utf-8",
    )
    (root / "Cargo.toml").write_text("\n".join(ws_lines) + "\n", encoding="utf-8")


def test_release_version_checks_citation_workspace_deps_and_bare_facade(tmp_path: Path) -> None:
    module = _release_module()

    ok_root = tmp_path / "ok"
    _write_minimal_release_tree(
        ok_root,
        citation_text='cff-version: 1.2.0\nversion: "1.2.3" # release tag\n',
    )
    assert module.release_version(ok_root) == "1.2.3"

    bad_citation = tmp_path / "bad-citation"
    _write_minimal_release_tree(
        bad_citation,
        citation_text="cff-version: 1.2.0\nversion: 0.1.0\n",
    )
    with pytest.raises(ValueError, match=r"CITATION\.cff=0\.1\.0"):
        module.release_version(bad_citation)

    missing_citation = tmp_path / "missing-citation"
    _write_minimal_release_tree(missing_citation, citation_text="cff-version: 1.2.0\n")
    with pytest.raises(ValueError, match=r"CITATION\.cff has no version field"):
        module.release_version(missing_citation)

    bad_ws = tmp_path / "bad-ws"
    _write_minimal_release_tree(bad_ws, workspace_dep_version="1.2.2")
    with pytest.raises(ValueError, match=r"workspace\.dependencies"):
        module.release_version(bad_ws)

    bad_facade = tmp_path / "bad-facade"
    _write_minimal_release_tree(
        bad_facade,
        py_facade_spec='{ version = "1.2.2", path = "../phasesmith" }',
    )
    with pytest.raises(ValueError, match=r"dependencies\.phasesmith must specify version=1\.2\.3"):
        module.release_version(bad_facade)
