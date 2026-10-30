"""Balanced serial cached-normal refinement comparison with optimized kernels; development only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from solution_corpus.run import collect_jobs
from solution_execution_interleaved import collect_balanced

ROOT = Path(__file__).resolve().parents[1]
MODES = ("control", "cached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "validation/solution-corpus-v5.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--lock-revision", type=int, default=25)
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
            isotropic_assignment_refinement=True,
            filtered_peak_search=True,
            batched_fft=True,
            real_fft=False,
            fused_phase_workspace=False,
            cached_normal_refinement=mode == "cached",
            multiplicity_weighted_refinement=False,
            ambiguous_assignments=False,
            cached_assignment_refinement=False,
            work_sliced_phasing=True,
            independent_origin_sampling=False,
            phase_first_selection=True,
            legacy_quality_normalization=False,
            score_only_evaluation=True,
            reusable_fourier_workspaces=True,
            collection_schedule="balanced serial case-interleaved normal-cache comparison",
            collection_scheduler_source="benchmarks/solution_normal_interleaved.py",
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
