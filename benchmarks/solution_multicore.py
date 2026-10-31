"""Bounded latency and cold batch throughput; all development predictions precede audits.

Latency: serial independent solves, PhaseSmith workers=N and SHELXT -tN.
Batch: at most N independent processes, each solver internally one worker.
Parsing/import/startup and output serialization belong to batch elapsed time.
Input/resource preparation and reference grading are outside every timing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from solution_corpus.audit import shelxt_candidate
from solution_corpus.data import sha
from solution_corpus.run import ROOT, collect_jobs, launch

COUNTS = (1, 2, 4, 8)
ENABLED = [
    "wilson_normalization",
    "phase_quality_selection",
    "phase_origin_selection",
    "chemical_assignment",
    "iterative_refinement",
    "isotropic_assignment_refinement",
    "filtered_peak_search",
    "batched_fft",
    "real_fft",
    "fused_phase_workspace",
    "cached_normal_refinement",
    "multiplicity_weighted_refinement",
    "ambiguous_assignments",
    "cached_assignment_refinement",
    "specialized_isotropic_refinement",
    "streamed_peak_search",
    "reuse_normal_matrices",
    "short_phase_starts",
    "blocked_normal_matrices",
    "phase_plateau_stopping",
    "assignment_convergence_stopping",
    "late_phase_plateau_stopping",
    "model_consensus_stopping",
    "work_sliced_phasing",
    "phase_first_selection",
    "score_only_evaluation",
    "reusable_fourier_workspaces",
]


def controls(args, n):
    return SimpleNamespace(
        **dict(
            {key: True for key in ENABLED},
            manifest=args.manifest.resolve(),
            output=(args.output / f"latency-w{n}").resolve(),
            seconds=args.seconds,
            shelxt=args.shelxt.resolve(),
            split="development",
            resume=False,
            lock_revision=args.lock_revision,
            legacy_counts=False,
            missing_reflections=False,
            resolution_extension=1.0,
            adaptive_checkpoints=False,
            independent_origin_sampling=False,
            legacy_quality_normalization=False,
            workers=n,
            shelxt_threads=n,
            collection_schedule=(
                "balanced serial latency followed by bounded independent-process batches"
            ),
            collection_scheduler_source="benchmarks/solution_multicore.py",
        )
    )


def batch_panel(args, n, base):
    panel = args.output / f"batch-w{n}"
    panel.mkdir()
    header = {k: v for k, v in base.items() if k != "runs"}
    header.update(
        workers=1,
        shelxt_threads=1,
        batch_workers=n,
        collection_schedule="bounded independent-process batch",
        thread_policy=f"<= {n} independent processes; each solver has one numerical worker",
        source_snapshot="../latency-w1/" + base["source_snapshot"],
    )
    (panel / "protocol.json").write_text(json.dumps(header, indent=2) + "\n")
    plans = {method: [] for method in ["phasesmith", "shelxt"]}
    for old in base["runs"]:
        method = old["method"]
        source = args.output / "latency-w1" / old["directory"]
        work = panel / old["directory"]
        work.mkdir(parents=True)
        names = ["inputs.json", "sample.hkl"] + (["sample.ins"] if method == "shelxt" else [])
        hashes = {}
        for name in names:
            shutil.copyfile(source / name, work / name)
            hashes[name] = sha(work / name)
            assert hashes[name] == sha(source / name)
        command = [v.replace(str(source), str(work)) for v in old["command"]]
        record = {
            k: old[k]
            for k in [
                "id",
                "method",
                "directory",
                "input_only_resource_atom_cap",
                "expanded_measured_rows_for_resource_cap",
            ]
        }
        record.update(input_sha256=hashes, collection_stage=1)
        plans[method].append((command, work, record))
    records = []
    batches = {}

    def run(plan):
        command, work, record = plan
        record.update(
            host_load_average_before_job=list(os.getloadavg()),
            batch_start_offset_seconds=time.perf_counter() - started,
        )
        record.update(launch(command, work, args.seconds))
        try:
            if record["method"] == "phasesmith":
                record.update(json.loads((work / "solution/worker.json").read_text()))
            else:
                record["model"] = shelxt_candidate(work / "sample_a.res")
                log = (work / "stdout.txt").read_text(errors="replace")
                if "1 threads running in parallel" not in log:
                    raise ValueError("one-thread execution not confirmed")
                record["outputs_sha256"] = {p.name: sha(p) for p in work.iterdir() if p.is_file()}
            record["status"] = "model_returned" if record.get("model") else "no_model"
        except Exception as error:
            record.update(
                status="no_model", model=None, diagnostic=f"{type(error).__name__}: {error}"
            )
        record["batch_end_offset_seconds"] = time.perf_counter() - started
        return record

    methods = ["phasesmith", "shelxt"] if COUNTS.index(n) % 2 == 0 else ["shelxt", "phasesmith"]
    for method in methods:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(run, plans[method]))
        elapsed = time.perf_counter() - started
        batches[method] = {
            "elapsed_seconds": elapsed,
            "cases": len(results),
            "cases_per_second": len(results) / elapsed,
            "outer_workers": n,
            "inner_workers": 1,
            "timing": (
                "cold independent processes; includes import, startup, parsing "
                "and serialization; excludes input preparation and grading"
            ),
        }
        records.extend(results)
        (panel / "before-reference-audit.json").write_text(
            json.dumps({**header, "batches": batches, "runs": records}, indent=2) + "\n"
        )
        print(json.dumps({"batch_workers": n, "method": method, **batches[method]}), flush=True)
    assert len(records) == 70
    (panel / "comparison.json").write_text(
        json.dumps({**header, "batches": batches, "runs": records}, indent=2) + "\n"
    )
    return panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "validation/solution-corpus-v5.json")
    p.add_argument(
        "--shelxt",
        type=Path,
        default=Path("/Users/clemens/.local/opt/shelx/versions/2026-10-28/bin/shelxt"),
    )
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--lock-revision", type=int, default=34)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or not 5 <= args.seconds <= 120:
        p.error("new output directory and seconds in [5,120] required")
    args.output.mkdir(parents=True)
    streams = {n: collect_jobs(controls(args, n)) for n in COUNTS}
    events = []
    round_index = 0
    while True:
        offset = (round_index // 2) % len(COUNTS)
        order = COUNTS[offset:] + COUNTS[:offset]
        rows = []
        for n in order:
            event = next(streams[n])
            rows.append(event)
            events.append({"workers": n, **event})
            (args.output / "schedule.json").write_text(json.dumps(events, indent=2) + "\n")
        if all(e.get("reference_audit_barrier") for e in rows):
            break
        if any(e.get("reference_audit_barrier") for e in rows):
            raise ValueError("inconsistent barrier")
        if len({(r["id"], r["method"]) for r in rows}) != 1:
            raise ValueError("inconsistent job order")
        round_index += 1
    assert round_index == 70
    base = json.loads((args.output / "latency-w1/before-reference-audit.json").read_text())
    panels = [batch_panel(args, n, base) for n in COUNTS]
    seal = {"predictions": 560, "holdout_opened": False, "panels": {}}
    for panel in [args.output / f"latency-w{n}" for n in COUNTS] + panels:
        prediction = panel / "before-reference-audit.json"
        assert len(json.loads(prediction.read_text())["runs"]) == 70
        snapshot = panel / "all-predictions-snapshot.json"
        shutil.copyfile(prediction, snapshot)
        seal["panels"][panel.name] = {
            "path": str(snapshot.relative_to(args.output)),
            "sha256": sha(snapshot),
        }
    (args.output / "all-predictions-before-reference.json").write_text(
        json.dumps(seal, indent=2) + "\n"
    )
    # No generator crosses its reference boundary until every latency/batch outcome is saved.
    for stream in streams.values():
        assert not list(stream)
    for panel in [args.output / f"latency-w{n}" for n in COUNTS] + panels:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/solution_corpus_reaudit.py"),
                "--manifest",
                str(args.manifest.resolve()),
                "--run",
                str(panel),
                "--output",
                str(panel / "comparison-verified.json"),
            ],
            check=True,
        )
    print(json.dumps({"complete_predictions": 560, "holdout_opened": False}), flush=True)


if __name__ == "__main__":
    main()
