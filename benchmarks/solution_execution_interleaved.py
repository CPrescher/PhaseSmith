"""Balanced serial four-way execution ablation, restricted to the development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from solution_corpus.run import collect_jobs

ROOT = Path(__file__).resolve().parents[1]
MODES = ("control", "score", "workspace", "combined")


def collect_balanced(streams, notify):
    """Rotate mode position by case, pausing every collector before any reference audit."""
    labels = tuple(streams)
    round_index = 0
    while True:
        offset = (round_index // 2) % len(labels)
        order = labels[offset:] + labels[:offset]
        round_rows = []
        for label in order:
            try:
                event = next(streams[label])
            except StopIteration as error:
                raise ValueError("collector ended before the reference barrier") from error
            round_rows.append(event)
            notify(label, event)
        if all(r.get("reference_audit_barrier") for r in round_rows):
            break
        if any(r.get("reference_audit_barrier") for r in round_rows):
            raise ValueError("collectors reached inconsistent reference barriers")
        if len({(r["id"], r["method"]) for r in round_rows}) != 1:
            raise ValueError("interleaved collectors used different case/method order")
        round_index += 1
    # All modes' complete chosen predictions now exist before the first reference.
    for stream in streams.values():
        if list(stream):
            raise ValueError("unexpected work after the reference barrier")
    return round_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "validation/solution-corpus-v5.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--lock-revision", type=int, default=14)
    parser.add_argument(
        "--shelxt",
        type=Path,
        default=Path("/Users/clemens/.local/opt/shelx/versions/2026-10-28/bin/shelxt"),
    )
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.seconds <= 120:
        parser.error("a new directory and seconds in [5,120] are required")
    args.output.mkdir(parents=True)
    streams = {}
    for mode in MODES:
        controls = SimpleNamespace(
            manifest=args.manifest.resolve(),
            output=(args.output / mode).resolve(),
            seconds=args.seconds,
            shelxt=args.shelxt.resolve(),
            split="development",
            resume=False,
            lock_revision=args.lock_revision,
            legacy_counts=False,
            wilson_normalization=True,
            phase_quality_selection=True,
            phase_origin_selection=True,
            chemical_assignment=True,
            iterative_refinement=True,
            missing_reflections=False,
            resolution_extension=1.0,
            adaptive_checkpoints=False,
            independent_origin_sampling=False,
            phase_first_selection=True,
            legacy_quality_normalization=False,
            score_only_evaluation=mode in ("score", "combined"),
            reusable_fourier_workspaces=mode in ("workspace", "combined"),
            collection_schedule="balanced serial case-interleaved four-way execution ablation",
        )
        streams[mode] = collect_jobs(controls)
    events = []

    def record(mode, event):
        events.append({"mode": mode, **event})
        (args.output / "schedule.json").write_text(json.dumps(events, indent=2) + "\n")
        print(json.dumps({"mode": mode, **event}), flush=True)

    count = collect_balanced(streams, record)
    expected = 2 * sum(
        c["split"] == "development" for c in json.loads(args.manifest.read_text())["cases"]
    )
    if count != expected:
        raise ValueError("incomplete development collection")
    print(json.dumps({"complete_rounds": count, "modes": MODES, "holdout_opened": False}))


if __name__ == "__main__":
    main()
