from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "public_api_snapshot.py"
SNAPSHOT = ROOT / "api" / "python-public-api-v0.8.1.json"


def _snapshot_module() -> object:
    specification = importlib.util.spec_from_file_location("public_api_snapshot", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_checked_in_public_api_snapshot_matches_live_exports() -> None:
    module = _snapshot_module()
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert module.build_snapshot() == expected
    assert expected["schema_version"] == 1
    assert expected["package_version"] == "0.8.1"
    assert [item["module"] for item in expected["modules"]][:6] == [
        "phasesmith",
        "phasesmith.io",
        "phasesmith.refinement",
        "phasesmith.integrations",
        "phasesmith.oracle",
        "phasesmith.validation",
    ]
    assert all(item["exports"] for item in expected["modules"])
    top_level = {item["name"]: item for item in expected["modules"][0]["exports"]}
    assert top_level["ProfileEstimationMode"]["signature"] is None
    assert top_level["RadiationProbe"]["signature"] is None
    assert "WorkflowSpec" not in top_level
    assert "read_cif" not in top_level
    assert "review_rietveld_input" not in top_level

    io_exports = {item["name"] for item in expected["modules"][1]["exports"]}
    assert {"cif", "powder", "space_groups", "tof_instrument"} <= io_exports
    assert "convert_rowles_topas_bundle" not in io_exports

    refinement_exports = {item["name"] for item in expected["modules"][2]["exports"]}
    assert {"lebail", "readiness", "rietveld", "tof_lebail", "tof_multibank"} <= (
        refinement_exports
    )
    assert "refine" not in refinement_exports
    assert "LeBailInput" not in refinement_exports


def test_snapshot_cli_refuses_implicit_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "snapshot.json"
    first = subprocess.run(
        [sys.executable, SCRIPT, "--write", "--snapshot", output],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.stdout.strip() == str(output)

    second = subprocess.run(
        [sys.executable, SCRIPT, "--write", "--snapshot", output],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 1
    assert "refusing to overwrite existing snapshot" in second.stderr


def test_snapshot_cli_reports_a_reviewable_diff(tmp_path: Path) -> None:
    changed = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    changed["modules"][0]["exports"][0]["name"] = "removed_or_renamed"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, SCRIPT, "--check", "--snapshot", path],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "public API snapshot differs from the live package" in result.stderr
    assert "removed_or_renamed" in result.stderr
    assert "live-public-api" in result.stderr


def test_pawley_supplemental_snapshot_matches_live_exports() -> None:
    module = _snapshot_module()
    expected = json.loads((ROOT / "api" / "python-pawley-api-v0.7.0.json").read_text())
    actual = module.build_snapshot(
        modules=(
            ("phasesmith.refinement.pawley", "workflow"),
            ("phasesmith.project_bundle", "workflow"),
            ("phasesmith.refinement.tof_pawley", "workflow"),
        ),
    )
    actual["package_version"] = expected["package_version"]
    assert actual == expected


def test_retired_0_5_exports_are_absent_from_0_8_namespaces() -> None:
    # Use immutable release records rather than a second hand-maintained list.
    # A fresh interpreter catches eager imports that accidentally restore aliases.
    script = """
import importlib
import json
from pathlib import Path

previous = json.loads(Path("api/python-public-api-v0.5.0.json").read_text())
current = json.loads(Path("api/python-public-api-v0.8.0.json").read_text())
for old_module, new_module in zip(previous["modules"][:3], current["modules"][:3], strict=True):
    assert old_module["module"] == new_module["module"]
    namespace = importlib.import_module(new_module["module"])
    old_names = {item["name"] for item in old_module["exports"]}
    new_names = {item["name"] for item in new_module["exports"]}
    for name in old_names - new_names:
        retired = f"{namespace.__name__}.{name}"
        assert not hasattr(namespace, name), f"retired alias still present: {retired}"
    assert not hasattr(namespace, "_DEPRECATED_EXPORTS")
    assert not hasattr(namespace, "_DEPRECATED_EXPORT_MODULES")
    assert not hasattr(namespace, "__getattr__")
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_module_owned_imports_work_without_deprecation_warnings() -> None:
    script = """
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)
    from phasesmith.automation import WorkflowSpec
    from phasesmith.io.topas import convert_rowles_topas_bundle
    from phasesmith.refinement.lebail import LeBailInput
    from phasesmith.refinement.rietveld import RietveldInput

assert WorkflowSpec.__name__ == "WorkflowSpec"
assert callable(convert_rowles_topas_bundle)
assert LeBailInput.__name__ == "LeBailInput"
assert RietveldInput.__name__ == "RietveldInput"
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)
