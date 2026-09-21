from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "public_api_snapshot.py"
SNAPSHOT = ROOT / "api" / "python-public-api-v0.7.0.json"
PREVIOUS_SNAPSHOT = ROOT / "api" / "python-public-api-v0.5.0.json"


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
    assert expected["package_version"] == "0.7.0"
    assert [item["module"] for item in expected["modules"]] == [
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
    module.verify_snapshot(
        ROOT / "api" / "python-pawley-api-v0.7.0.json",
        modules=(
            ("phasesmith.refinement.pawley", "workflow"),
            ("phasesmith.project_bundle", "workflow"),
            ("phasesmith.refinement.tof_pawley", "workflow"),
        ),
    )


def test_moved_exports_are_warning_backed_compatibility_aliases() -> None:
    script = """
import warnings
import phasesmith
from phasesmith.automation import WorkflowSpec
from phasesmith.io.topas import convert_rowles_topas_bundle
from phasesmith.refinement.lebail import LeBailInput

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert phasesmith.WorkflowSpec is WorkflowSpec
    assert phasesmith.io.convert_rowles_topas_bundle is convert_rowles_topas_bundle
    assert phasesmith.refinement.LeBailInput is LeBailInput

assert len(caught) == 3
assert all(item.category is DeprecationWarning for item in caught)
assert all("will be removed in 1.0" in str(item.message) for item in caught)
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_every_removed_0_5_export_has_a_0_6_compatibility_alias() -> None:
    import phasesmith

    previous = json.loads(PREVIOUS_SNAPSHOT.read_text(encoding="utf-8"))
    current = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    compatibility_names = (
        set(phasesmith._DEPRECATED_EXPORTS),
        set(phasesmith.io._DEPRECATED_EXPORTS),
        set(phasesmith.refinement._DEPRECATED_EXPORT_MODULES),
    )
    for old_module, new_module, aliases in zip(
        previous["modules"][:3], current["modules"][:3], compatibility_names, strict=True
    ):
        old_names = {item["name"] for item in old_module["exports"]}
        new_names = {item["name"] for item in new_module["exports"]}
        assert old_names - new_names == aliases
