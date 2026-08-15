from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "public_api_snapshot.py"
SNAPSHOT = ROOT / "api" / "python-public-api-v0.5.0.json"


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
    assert expected["package_version"] == "0.5.0"
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
