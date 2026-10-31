"""Run the precommitted holdout protocol without changing solver controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from phasesmith import _core
from solution_corpus.data import sha
from solution_corpus.run import collect_jobs

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "validation/solution-holdout-freeze-v1.json"


def load(path):
    return json.loads(Path(path).read_text())


def verify_freeze(freeze):
    """Check source, executables and the immutable input registry, never references."""
    for name, expected in freeze["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen source changed: " + name)
    if sha(_core.__file__) != freeze["native_sha256"]:
        raise ValueError("frozen numerical binary changed")
    if sha(freeze["controls"]["shelxt"]) != freeze["comparator_sha256"]:
        raise ValueError("external test comparator changed")
    if sha(ROOT / freeze["controls"]["manifest"]) != freeze["manifest_sha256"]:
        raise ValueError("frozen registry changed")
    lock = ROOT / freeze["corpus_lock"]
    if sha(lock) != freeze["corpus_lock_sha256"]:
        raise ValueError("frozen corpus lock changed")


def validate_prediction_barrier(path, ids):
    """Reject missing, duplicated, foreign or already graded predictions."""
    report = load(path)
    rows = report["runs"]
    expected = {(i, method) for i in ids for method in ("phasesmith", "shelxt")}
    pairs = [(r["id"], r["method"]) for r in rows]
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("all holdout predictions must precede reference grading")
    if any("reference_audit" in r or "correct" in r for r in rows):
        raise ValueError("reference outcomes appeared before the prediction barrier")
    return sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args()
    if not args.open_holdout:
        parser.error("--open-holdout explicitly records the authorized holdout opening")
    if args.output.exists():
        parser.error("output must be new; no reruns or overwritten predictions")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        parser.error("commit the freeze protocol and all changes before opening the holdout")
    freeze = load(FREEZE)
    verify_freeze(freeze)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    controls = dict(freeze["controls"])
    controls.update(
        manifest=ROOT / controls["manifest"],
        shelxt=Path(controls["shelxt"]),
        output=args.output.resolve(),
    )
    events = []
    barrier = None
    started = datetime.now(timezone.utc).isoformat()
    for event in collect_jobs(SimpleNamespace(**controls)):
        verify_freeze(freeze)
        events.append(event)
        (args.output / "holdout-schedule.json").write_text(
            json.dumps(events, indent=2) + "\n"
        )
        if event.get("reference_audit_barrier"):
            if len(events) != 131:
                raise ValueError("unexpected prediction/reference barrier order")
            snapshot = args.output / "predictions-at-barrier.json"
            with snapshot.open("xb") as stream:
                stream.write((args.output / "before-reference-audit.json").read_bytes())
            barrier = {
                "predictions": 130,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_sha256": validate_prediction_barrier(
                    snapshot, freeze["holdout_ids"]
                ),
                "freeze_sha256": sha(FREEZE),
                "commit": commit,
                "source_and_binary_unchanged": True,
            }
            (args.output / "prediction-barrier.json").write_text(
                json.dumps(barrier, indent=2) + "\n"
            )
    if barrier is None:
        raise ValueError("missing prediction/reference barrier")
    verify_freeze(freeze)
    provenance = {
        "schema": "phasesmith.frozen-holdout-run.v1",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "solver_commit": freeze["solver_commit"],
        "protocol_commit": commit,
        "freeze_sha256": sha(FREEZE),
        "native_sha256": sha(_core.__file__),
        "holdout_opened": True,
        "development_options_unchanged": True,
        "source_and_binary_checks": len(events) + 2,
        "barrier": barrier,
    }
    (args.output / "frozen-run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
