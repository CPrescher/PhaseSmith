from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/release_version.py"


def test_release_version_matches_every_checked_surface() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT, "--tag", "v0.2.0"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.2.0"


def test_release_version_rejects_a_mismatched_tag() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT, "--tag", "v9.9.9"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "does not match release version v0.2.0" in result.stderr
