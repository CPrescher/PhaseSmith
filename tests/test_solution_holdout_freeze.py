"""The frozen comparison cannot silently change controls or grade partial runs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import solution_holdout_frozen as frozen  # noqa: E402 - repository-only benchmark helper


def test_frozen_controls_are_the_accepted_development_configuration():
    freeze = frozen.load(frozen.FREEZE)
    dev = frozen.load(ROOT / freeze["development_report"])
    for key, value in freeze["controls"].items():
        if key in dev and key not in {"split", "collection_schedule"}:
            assert value == dev[key], key
    assert freeze["controls"]["split"] == "holdout"
    assert freeze["controls"]["seconds"] == dev["seconds_per_job"] == 15
    assert not freeze["controls"]["legacy_counts"] and dev["adaptive_counts"]
    assert (
        not freeze["controls"]["legacy_quality_normalization"]
        and dev["calculated_shell_normalization"]
    )
    assert freeze["solver_options"] == {
        "cycles": 800,
        "trials": 4,
        "seed": 0,
        "max_seconds": 13.0,
        "refinement": "anisotropic",
        "completion": False,
    }
    manifest = frozen.load(ROOT / freeze["controls"]["manifest"])
    assert set(freeze["holdout_ids"]) == {
        c["id"] for c in manifest["cases"] if c["split"] == "holdout"
    }
    assert len(freeze["holdout_ids"]) == 65
    # The historical freeze describes its committed snapshot, not later merged
    # checkouts or newly compiled extensions. Keep the execution guard strict.
    result = frozen.load(ROOT / "benchmarks/structure-solution-holdout-frozen-summary-macos.json")
    assert frozen.sha(frozen.FREEZE) == result["freeze_sha256"]
    assert freeze["native_sha256"] == result["native_sha256"]
    available = subprocess.run(
        ["git", "cat-file", "-e", result["protocol_commit"] + "^{commit}"],
        cwd=ROOT,
        capture_output=True,
    )
    if available.returncode:
        pytest.skip("historical source proof requires the recorded commit in Git history")
    names = list(freeze["source_sha256"])
    objects = "".join(f"{result['protocol_commit']}:{name}\n" for name in names)
    stream = subprocess.check_output(
        ["git", "cat-file", "--batch"], input=objects.encode(), cwd=ROOT
    )
    offset = 0
    for name in names:
        end = stream.index(b"\n", offset)
        _, kind, size = stream[offset:end].split()
        assert kind == b"blob", name
        offset = end + 1
        length = int(size)
        content = stream[offset : offset + length]
        assert hashlib.sha256(content).hexdigest() == freeze["source_sha256"][name], name
        offset += length
        assert stream[offset : offset + 1] == b"\n"
        offset += 1
    assert offset == len(stream)


def test_source_mutation_is_rejected_before_any_reference_access(monkeypatch):
    freeze = frozen.load(frozen.FREEZE)
    original = frozen.sha
    name = next(iter(freeze["source_sha256"]))
    monkeypatch.setattr(
        frozen, "sha", lambda p: "changed" if Path(p) == ROOT / name else original(p)
    )
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
