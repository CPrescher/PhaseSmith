"""The frozen comparison cannot silently change controls or grade partial runs."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import solution_holdout_frozen as frozen


def test_frozen_controls_are_the_accepted_development_configuration():
    freeze = frozen.load(frozen.FREEZE)
    dev = frozen.load(ROOT / freeze["development_report"])
    for key, value in freeze["controls"].items():
        if key in dev and key not in {"split", "collection_schedule"}:
            assert value == dev[key], key
    assert freeze["controls"]["split"] == "holdout"
    assert freeze["controls"]["seconds"] == dev["seconds_per_job"] == 15
    assert not freeze["controls"]["legacy_counts"] and dev["adaptive_counts"]
    assert not freeze["controls"]["legacy_quality_normalization"] and dev["calculated_shell_normalization"]
    assert freeze["solver_options"] == {
        "cycles": 800, "trials": 4, "seed": 0, "max_seconds": 13.0,
        "refinement": "anisotropic", "completion": False,
    }
    manifest = frozen.load(ROOT / freeze["controls"]["manifest"])
    assert set(freeze["holdout_ids"]) == {
        c["id"] for c in manifest["cases"] if c["split"] == "holdout"
    }
    assert len(freeze["holdout_ids"]) == 65
    frozen.verify_freeze(freeze)


def test_source_mutation_is_rejected_before_any_reference_access(monkeypatch):
    freeze = frozen.load(frozen.FREEZE)
    original = frozen.sha
    name = next(iter(freeze["source_sha256"]))
    monkeypatch.setattr(frozen, "sha", lambda p: "changed" if Path(p) == ROOT / name else original(p))
    with pytest.raises(ValueError, match="frozen source changed"):
        frozen.verify_freeze(freeze)


@pytest.mark.parametrize("corruption", [None, "missing", "duplicate", "foreign", "graded"])
def test_reference_barrier_requires_all_ungraded_predictions(tmp_path, corruption):
    rows = [dict(id=i, method=m) for i in ["a", "b"] for m in ["phasesmith", "shelxt"]]
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = dict(rows[0])
    elif corruption == "foreign":
        rows[-1]["id"] = "unknown"
    elif corruption == "graded":
        rows[-1]["correct"] = False
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"runs": rows}))
    if corruption is None:
        assert frozen.validate_prediction_barrier(path, ["a", "b"]) == frozen.sha(path)
    else:
        with pytest.raises(ValueError):
            frozen.validate_prediction_barrier(path, ["a", "b"])


def test_opening_requires_explicit_flag_and_does_not_start_jobs(tmp_path):
    p = tmp_path / "predictions"
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/solution_holdout_frozen.py"), "--output", str(p)],
        capture_output=True,
    )
    assert result.returncode == 2 and b"--open-holdout" in result.stderr
    assert not p.exists()
