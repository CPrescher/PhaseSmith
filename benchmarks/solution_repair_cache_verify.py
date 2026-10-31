"""Verify exact repair reuse, retained development cases and matched overhead timings."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from phasesmith import _core
from solution_phase_ablation import compare
from solution_joint_verify import counts, sets

ROOT = Path(__file__).resolve().parents[1]
MODES = ("off", "control", "cache")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--prefix", default="structure-solution-repair-evaluation")
    a = p.parse_args()
    a.base = a.base.resolve()
    manifest = ROOT / "validation/solution-corpus-v5.json"
    ids = {c["id"] for c in load(manifest)["cases"] if c["split"] == "development"}
    assert len(ids) == 35
    events = load(a.base / "schedule.json")
    assert len(events) == 213
    for index in range(70):
        group = events[3 * index : 3 * index + 3]
        offset = (index // 2) % 3
        assert [e["mode"] for e in group] == list(MODES[offset:] + MODES[:offset])
        assert len({(e["id"], e["method"]) for e in group}) == 1
        assert not any(e.get("reference_audit_barrier") for e in group)
    assert all(e.get("reference_audit_barrier") for e in events[210:])
    previous_path = ROOT / "benchmarks/structure-solution-joint-assignment-repair-macos.json"
    previous = {r["id"]: r for r in load(previous_path)["runs"] if r["method"] == "phasesmith"}
    prior_off = {
        r["id"]: r
        for r in load(ROOT / "benchmarks/structure-solution-joint-assignment-control-macos.json")[
            "runs"
        ]
        if r["method"] == "phasesmith"
    }
    assert len(sets(previous)["structures"]) == 19 and len(sets(previous)["total_counts"]) == 27
    reports = {}
    records = {}
    paths = {}
    summaries = {}
    for mode in MODES:
        base = a.base / mode
        report = load(base / "comparison-verified.json")
        reports[mode] = report
        assert (
            report["split"] == "development"
            and not report["holdout_opened"]
            and len(report["runs"]) == 70
        )
        assert report["manifest_sha256"] == sha(manifest) and report["native_sha256"] == sha(
            _core.__file__
        )
        assert report["joint_assignment_refinement"] == (mode != "off") and report[
            "cached_repair_evaluation"
        ] == (mode == "cache")
        assert report["workers"] == report["shelxt_threads"] == 1
        assert report["before_reference_audit_sha256"] == sha(base / "before-reference-audit.json")
        for name, h in report["implementation_sha256"].items():
            assert sha(base / report["source_snapshot"] / name) == h, name
            assert sha(ROOT / name) == h, name
        for method in ["phasesmith", "shelxt"]:
            assert {r["id"] for r in report["runs"] if r["method"] == method} == ids
        rows = {r["id"]: r for r in report["runs"] if r["method"] == "phasesmith"}
        records[mode] = rows
        for r in rows.values():
            e = r["effort"]
            assert e["cached_repair_evaluation"] == (mode == "cache") and e[
                "joint_assignment_refinement"
            ] == (mode != "off")
            assert (
                r["options"]["cycles"] == 800
                and r["options"]["trials"] == 4
                and r["options"]["seed"] == 0
                and r["options"]["max_seconds"] == 13
            )
            assert r["options"]["refinement"] == "anisotropic" and not r["options"]["completion"]
            assert (
                e["workers"] == e["requested_workers"] == e["max_active_workers"] == 1
                and e["parallel_maps"] == 0
            )
            assert e["joint_assignment_fits"] <= 6 and e["joint_assignment_accepted"] <= 3
            assert e["joint_assignment_workspace_bytes"] <= 4 * 1024 * 1024 + 2048 * 104
            if mode != "cache":
                assert all(
                    e[k] == 0
                    for k in [
                        "joint_assignment_field_builds",
                        "joint_assignment_field_reuses",
                        "joint_assignment_score_only",
                        "joint_assignment_workspace_bytes",
                    ]
                )
        assert sum(r["reference_exceeds_native_resource_cap"] for r in rows.values()) == 7
        path = ROOT / "benchmarks" / f"{a.prefix}-{mode}-macos.json"
        shutil.copyfile(base / "comparison-verified.json", path)
        paths[mode] = path
        quality = sets(rows)
        prior = prior_off if mode == "off" else previous
        lost = {key: sorted(set(sets(prior)[key]) - set(quality[key])) for key in quality}
        changes = [
            dict(id=i, previous=counts(prior[i]), current=counts(rows[i]))
            for i in sorted(ids)
            if counts(prior[i]) != counts(rows[i])
        ]
        work_keys = [
            "joint_assignment_scores",
            "joint_assignment_depths",
            "joint_assignment_fits",
            "joint_assignment_accepted",
            "joint_assignment_field_builds",
            "joint_assignment_field_reuses",
            "joint_assignment_score_only",
            "joint_assignment_seconds",
        ]
        summaries[mode] = dict(
            summary=report["summary"],
            case_sets=quality,
            per_element_counts_correct=len(quality["element_counts"]),
            lost_previous=lost,
            changed_count_vectors=changes,
            summed_solver_seconds=sum(r["kernel_seconds"] for r in rows.values()),
            summed_wall_seconds=sum(r["wall_seconds_including_startup"] for r in rows.values()),
            work_totals={k: sum(r["effort"][k] for r in rows.values()) for k in work_keys},
            maximum_workspace_bytes=max(
                r["effort"]["joint_assignment_workspace_bytes"] for r in rows.values()
            ),
        )
    assert all(
        reports[m]["implementation_sha256"] == reports["control"]["implementation_sha256"]
        for m in MODES
    )
    comparison = compare(
        paths, manifest, [("control", "cache"), ("off", "control"), ("off", "cache")]
    )
    (ROOT / "benchmarks" / f"{a.prefix}-comparison-macos.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    baseline = records["control"]
    for mode in MODES:
        rows = records[mode]
        summaries[mode]["median_paired_solver_control_over_run"] = float(
            np.median(
                [baseline[i]["kernel_seconds"] / rows[i]["kernel_seconds"] for i in sorted(ids)]
            )
        )
        summaries[mode]["median_paired_wall_control_over_run"] = float(
            np.median(
                [
                    baseline[i]["wall_seconds_including_startup"]
                    / rows[i]["wall_seconds_including_startup"]
                    for i in sorted(ids)
                ]
            )
        )
    mismatches = []
    for i in sorted(ids):
        arow, brow = records["control"][i], records["cache"][i]
        for key in [
            "joint_assignment_scores",
            "joint_assignment_depths",
            "joint_assignment_fits",
            "joint_assignment_accepted",
        ]:
            if arow["effort"][key] != brow["effort"][key]:
                mismatches.append(
                    dict(id=i, field=key, control=arow["effort"][key], cache=brow["effort"][key])
                )
    repair_ratio = (
        summaries["cache"]["work_totals"]["joint_assignment_seconds"]
        / summaries["control"]["work_totals"]["joint_assignment_seconds"]
    )
    result = dict(
        schema="phasesmith.repair-evaluation-retention.v1",
        development_cases=35,
        holdout_opened=False,
        qualification="One balanced serial same-binary three-mode development comparison. Off is a no-repair overhead reference (26 correct total counts); control/cache both request the preceding repair (27). All 210 predictions precede all three reference barriers. Source snapshots, unchanged inputs/caps/seed/refinement/tolerances/budgets, actual workers and feature flags verified. No task-owned tests/builds/kernels/profiles overlap timed collection. Single-collection timings remain host sensitive.",
        previous_sha256=sha(previous_path),
        native_sha256=sha(_core.__file__),
        source_files_verified=len(reports["control"]["implementation_sha256"]),
        schedule_sha256=sha(a.base / "schedule.json"),
        report_sha256={m: sha(path) for m, path in paths.items()},
        modes=summaries,
        repair_time_ratio_cache_over_control=repair_ratio,
        logical_work_mismatches=mismatches,
        all19_and27_retained=not any(summaries["cache"]["lost_previous"].values()),
        all35_count_vectors_unchanged=not summaries["cache"]["changed_count_vectors"],
        verifier_sha256=sha(__file__),
    )
    output = ROOT / "benchmarks" / f"{a.prefix}-retention-macos.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all19_and27_retained"]:
        raise ValueError("Previous successes lost; preserve results and investigate")


if __name__ == "__main__":
    main()
