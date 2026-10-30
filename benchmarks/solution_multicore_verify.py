"""Verify bounded execution provenance and report latency, throughput and recovery."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
import sys
from pathlib import Path
import shutil
import subprocess
import numpy as np
from phasesmith import _core
from solution_multicore import COUNTS, ENABLED
from solution_corpus.run import INPUT_KEYS, instructions
from solution_shelxt_effort import compare as shelxt_effort

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def counts(row):
    return dict(Counter(row["model"]["species"])) if row.get("model") else {}


def success(rows, key):
    if key == "structures":
        return sorted(k for k, r in rows.items() if r["correct"])
    if key == "total_counts":
        return sorted(k for k, r in rows.items() if r["total_non_h_count_correct"])
    field = "counts_match" if key == "element_counts" else "strict_002_angstrom"
    return sorted(k for k, r in rows.items() if (r.get("reference_audit") or {}).get(field, False))


def machine_info():
    info = {"logical": os.cpu_count()}
    if sys.platform == "darwin":
        for name, key in [
            ("physical", "hw.physicalcpu"),
            ("performance", "hw.perflevel0.physicalcpu"),
            ("efficiency", "hw.perflevel1.physicalcpu"),
        ]:
            value = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=False
            )
            info[name] = int(value.stdout.strip()) if value.returncode == 0 else None
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--prefix", default="structure-solution-multicore")
    a = p.parse_args()
    a.base = a.base.resolve()
    manifest = ROOT / "validation/solution-corpus-v5.json"
    ids = {c["id"] for c in load(manifest)["cases"] if c["split"] == "development"}
    assert len(ids) == 35
    inputs = {c["id"]: c["files"] for c in load(manifest)["cases"] if c["split"] == "development"}
    baseline_path = ROOT / "benchmarks/structure-solution-cycle-effort-cache-macos.json"
    previous = {r["id"]: r for r in load(baseline_path)["runs"] if r["method"] == "phasesmith"}
    assert len(success(previous, "structures")) == 19
    events = load(a.base / "schedule.json")
    assert len(events) == 284
    for i in range(70):
        group = events[4 * i : 4 * i + 4]
        offset = (i // 2) % 4
        assert [e["workers"] for e in group] == list(COUNTS[offset:] + COUNTS[:offset])
        assert len({(e["id"], e["method"]) for e in group}) == 1
        assert not any(e.get("reference_audit_barrier") for e in group)
    assert all(e.get("reference_audit_barrier") for e in events[280:])
    seal = load(a.base / "all-predictions-before-reference.json")
    assert seal["predictions"] == 560 and not seal["holdout_opened"] and len(seal["panels"]) == 8
    for name, item in seal["panels"].items():
        assert sha(a.base / item["path"]) == item["sha256"]
        snapshot = load(a.base / item["path"])
        assert len(snapshot["runs"]) == 70
        assert not any("correct" in r or "reference_audit" in r for r in snapshot["runs"])
    panels = {}
    raw = {}
    efforts = {}
    snapshots = []
    for kind in ["latency", "batch"]:
        for n in COUNTS:
            label = f"{kind}-w{n}"
            base = a.base / label
            report = load(base / "comparison-verified.json")
            assert report["split"] == "development" and not report["holdout_opened"]
            assert report["manifest_sha256"] == sha(manifest)
            assert report["native_sha256"] == sha(_core.__file__)
            assert len(report["runs"]) == 70
            expected = n if kind == "latency" else 1
            assert report["workers"] == report["shelxt_threads"] == expected
            assert all(report[k] for k in ENABLED if k != "phase_first_selection")
            snapshots.append(report["implementation_sha256"])
            for name, h in report["implementation_sha256"].items():
                assert sha(base / report["source_snapshot"] / name) == h, name
                assert sha(ROOT / name) == h, name
            rows = {
                m: {r["id"]: r for r in report["runs"] if r["method"] == m}
                for m in ["phasesmith", "shelxt"]
            }
            assert all(set(v) == ids for v in rows.values())
            for method, cases in rows.items():
                for key, row in cases.items():
                    work_path = base / row["directory"]
                    packet = load(work_path / "inputs.json")
                    assert set(packet) == INPUT_KEYS
                    assert sha(work_path / "sample.hkl") == inputs[key]["common_hkl"]["sha256"]
                    assert packet == load(ROOT / inputs[key]["inputs"]["path"])
                    if method == "shelxt":
                        assert (work_path / "sample.ins").read_text() == instructions(packet)
            work = Counter()
            active = []
            maps = 0
            for r in rows["phasesmith"].values():
                e = r["effort"]
                assert e["requested_workers"] == expected
                assert 1 <= e["max_active_workers"] <= e["workers"] <= expected
                if expected == 1:
                    assert e["parallel_maps"] == e["parallel_tasks"] == 0
                assert (
                    r["options"]["cycles"] == 800
                    and r["options"]["trials"] == 4
                    and r["options"]["max_seconds"] == 13
                )
                active.append(e["max_active_workers"])
                maps += e["parallel_maps"]
                for sl in e["phase_slices"]:
                    quota = 200 if sl["trial"] < 2 else 800
                    assert sl["cycle_quota"] == quota and sl["completed_cycles"] <= quota
                    work["completed_cycles"] += sl["completed_cycles"]
                work["starts"] += e["phasing_starts"]
                work["parallel_maps"] += e["parallel_maps"]
                work["parallel_tasks"] += e["parallel_tasks"]
            for r in rows["shelxt"].values():
                assert f"-t{expected}" in r["command"]
                log = (base / r["directory"] / "stdout.txt").read_text(errors="replace")
                assert f"{expected} threads running in parallel" in log
            if kind == "batch":
                for method in ["phasesmith", "shelxt"]:
                    b = report["batches"][method]
                    assert b["outer_workers"] == n and b["inner_workers"] == 1
                    changes = []
                    # Sweep all interval endpoints, with finishes preceding ties at starts.
                    points = []
                    for r in rows[method].values():
                        points.extend(
                            [
                                (r["batch_start_offset_seconds"], 1),
                                (r["batch_end_offset_seconds"], -1),
                            ]
                        )
                    running = 0
                    peak = 0
                    for _, delta in sorted(points):
                        running += delta
                        peak = max(peak, running)
                    assert running == 0 and peak <= n
                    b["maximum_overlapping_job_intervals"] = peak
            retention = {
                key: {
                    "cases": success(rows["phasesmith"], key),
                    "lost_previous": sorted(
                        set(success(previous, key)) - set(success(rows["phasesmith"], key))
                    ),
                }
                for key in ["structures", "total_counts", "element_counts", "strict_002"]
            }
            changed = [
                dict(id=i, previous=counts(previous[i]), current=counts(rows["phasesmith"][i]))
                for i in sorted(ids)
                if counts(previous[i]) != counts(rows["phasesmith"][i])
            ]
            effort = shelxt_effort(report, base)
            efforts[label] = {r["id"]: r for r in effort["cases"]}
            path = ROOT / "benchmarks" / f"{a.prefix}-{label}-macos.json"
            shutil.copyfile(base / "comparison-verified.json", path)
            epath = ROOT / "benchmarks" / f"{a.prefix}-shelxt-{label}-macos.json"
            epath.write_text(json.dumps(effort, indent=2) + "\n")
            panel = dict(
                report_path=str(path.relative_to(ROOT)),
                report_sha256=sha(path),
                summary=report["summary"],
                shelxt_case_sets={
                    key: success(rows["shelxt"], key)
                    for key in ["structures", "total_counts", "element_counts", "strict_002"]
                },
                retention=retention,
                changed_count_vectors=changed,
                retains_all_previous_success_sets=not any(
                    v["lost_previous"] for v in retention.values()
                ),
                max_active_workers=max(active),
                cases_with_parallel_maps=sum(
                    r["effort"]["parallel_maps"] > 0 for r in rows["phasesmith"].values()
                ),
                work=dict(work),
                phasesmith_summed_solver_seconds=sum(
                    r["kernel_seconds"] for r in rows["phasesmith"].values()
                ),
                phasesmith_summed_wall_seconds=sum(
                    r["wall_seconds_including_startup"] for r in rows["phasesmith"].values()
                ),
                shelxt_summed_wall_seconds=sum(
                    r["wall_seconds_including_startup"] for r in rows["shelxt"].values()
                ),
                median_paired_ps_over_shelxt_wall=float(
                    np.median(
                        [
                            rows["phasesmith"][i]["wall_seconds_including_startup"]
                            / rows["shelxt"][i]["wall_seconds_including_startup"]
                            for i in sorted(ids)
                        ]
                    )
                ),
                shelxt_effort=effort["summary"],
                batches=report.get("batches"),
            )
            panel["per_element_count_correct"] = {
                m: len(success(v, "element_counts")) for m, v in rows.items()
            }
            panels[label] = panel
            raw[label] = rows
    assert all(s == snapshots[0] for s in snapshots)
    for kind in ["latency", "batch"]:
        control = raw[f"{kind}-w1"]["phasesmith"]
        for n in COUNTS:
            label = f"{kind}-w{n}"
            r = raw[label]["phasesmith"]
            out = panels[label]
            out["median_paired_solver_speedup_over_w1"] = float(
                np.median(
                    [control[i]["kernel_seconds"] / r[i]["kernel_seconds"] for i in sorted(ids)]
                )
            )
            out["median_paired_wall_speedup_over_w1"] = float(
                np.median(
                    [
                        control[i]["wall_seconds_including_startup"]
                        / r[i]["wall_seconds_including_startup"]
                        for i in sorted(ids)
                    ]
                )
            )
            xt_control = raw[f"{kind}-w1"]["shelxt"]
            xt_rows = raw[label]["shelxt"]
            out["shelxt_median_paired_wall_speedup_over_w1"] = float(
                np.median(
                    [
                        xt_control[i]["wall_seconds_including_startup"]
                        / xt_rows[i]["wall_seconds_including_startup"]
                        for i in sorted(ids)
                    ]
                )
            )
            out["shelxt_summed_wall_speedup_over_w1"] = (
                panels[f"{kind}-w1"]["shelxt_summed_wall_seconds"]
                / out["shelxt_summed_wall_seconds"]
            )
            out["summed_solver_speedup_over_w1"] = (
                panels[f"{kind}-w1"]["phasesmith_summed_solver_seconds"]
                / out["phasesmith_summed_solver_seconds"]
            )
            if kind == "batch":
                for method in ["phasesmith", "shelxt"]:
                    b = out["batches"][method]
                    b["speedup_over_w1"] = (
                        panels[f"{kind}-w1"]["batches"][method]["elapsed_seconds"]
                        / b["elapsed_seconds"]
                    )
    common = sorted(
        set.intersection(
            *[
                {
                    i
                    for i, row in efforts[f"latency-w{n}"].items()
                    if not row["shelxt_hard_timeout"]
                    and (row["shelxt"]["stages_seconds"]["total"] or 0) > 0
                }
                for n in COUNTS
            ]
        )
    )
    for n in COUNTS:
        label = f"latency-w{n}"
        out = panels[label]
        out["internal_common_completed_case_ids"] = common
        out["common_completed_summed_ps_solver_seconds"] = sum(
            efforts[label][i]["ps_solver_seconds"] for i in common
        )
        out["common_completed_summed_shelxt_reported_seconds"] = sum(
            efforts[label][i]["shelxt"]["stages_seconds"]["total"] for i in common
        )
        out["common_completed_median_paired_ps_over_shelxt_reported_time"] = float(
            np.median(
                [
                    efforts[label][i]["ps_solver_seconds"]
                    / efforts[label][i]["shelxt"]["stages_seconds"]["total"]
                    for i in common
                ]
            )
        )
        out["common_completed_shelxt_median_paired_reported_speedup_over_w1"] = float(
            np.median(
                [
                    efforts["latency-w1"][i]["shelxt"]["stages_seconds"]["total"]
                    / efforts[label][i]["shelxt"]["stages_seconds"]["total"]
                    for i in common
                ]
            )
        )
    result = dict(
        schema="phasesmith.bounded-multicore.v1",
        development_cases=35,
        holdout_opened=False,
        machine=machine_info(),
        qualification="One 1/2/4/8 collection on a shared host. Serial case-interleaved latency; cold independent-process batch throughput. Fixed 13-second native/15-second process safeguards. Pool budget is an upper bound; small kernels stay serial. PhaseSmith ARM native vs supplied SHELXT 2018/2 x86 compatibility. Internal endpoints differ. All 560 predictions saved before development grading. No holdout files opened.",
        baseline_sha256=sha(baseline_path),
        source_files_verified=len(snapshots[0]),
        native_sha256=sha(_core.__file__),
        pre_reference_seal_sha256=sha(a.base / "all-predictions-before-reference.json"),
        panels=panels,
        all19_retained=all(
            not p["retention"]["structures"]["lost_previous"] for p in panels.values()
        ),
        all_count_vectors_unchanged=all(not p["changed_count_vectors"] for p in panels.values()),
        verifier_sha256=sha(__file__),
    )
    output = ROOT / "benchmarks" / f"{a.prefix}-comparison-macos.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
