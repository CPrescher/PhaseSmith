"""Verify a complete matched development comparison and exact previous-case retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from phasesmith import _core
from solution_phase_ablation import compare

ROOT = Path(__file__).resolve().parents[1]
MODES = ("control", "repair")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def counts(r):
    return dict(Counter(r["model"]["species"])) if r.get("model") else {}


def sets(rows):
    return dict(
        structures=sorted(i for i, r in rows.items() if r["correct"]),
        total_counts=sorted(i for i, r in rows.items() if r["total_non_h_count_correct"]),
        element_counts=sorted(
            i for i, r in rows.items() if (r.get("reference_audit") or {}).get("counts_match")
        ),
        strict=sorted(
            i
            for i, r in rows.items()
            if (r.get("reference_audit") or {}).get("strict_002_angstrom")
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--prefix", default="structure-solution-joint-assignment")
    a = p.parse_args()
    a.base = a.base.resolve()
    manifest = ROOT / "validation/solution-corpus-v5.json"
    ids = {c["id"] for c in load(manifest)["cases"] if c["split"] == "development"}
    assert len(ids) == 35
    events = load(a.base / "schedule.json")
    assert len(events) == 142
    for i in range(70):
        group = events[2 * i : 2 * i + 2]
        offset = (i // 2) % 2
        assert [r["mode"] for r in group] == list(MODES[offset:] + MODES[:offset])
        assert len({(r["id"], r["method"]) for r in group}) == 1 and not any(
            r.get("reference_audit_barrier") for r in group
        )
    assert all(r.get("reference_audit_barrier") for r in events[140:])
    oldpath = ROOT / "benchmarks/structure-solution-multicore-latency-w1-macos.json"
    previous = {r["id"]: r for r in load(oldpath)["runs"] if r["method"] == "phasesmith"}
    assert len(sets(previous)["structures"]) == 19
    reports = {}
    rows = {}
    paths = {}
    for mode in MODES:
        base = a.base / mode
        report = load(base / "comparison-verified.json")
        assert report["split"] == "development" and not report["holdout_opened"]
        assert report["manifest_sha256"] == sha(manifest) and report["native_sha256"] == sha(
            _core.__file__
        )
        assert len(report["runs"]) == 70 and report["workers"] == report["shelxt_threads"] == 1
        assert report["joint_assignment_refinement"] == (mode == "repair")
        assert report["before_reference_audit_sha256"] == sha(base / "before-reference-audit.json")
        for name, h in report["implementation_sha256"].items():
            assert sha(base / report["source_snapshot"] / name) == h, name
            assert sha(ROOT / name) == h, name
        for method in ["phasesmith", "shelxt"]:
            assert {r["id"] for r in report["runs"] if r["method"] == method} == ids
        records = {r["id"]: r for r in report["runs"] if r["method"] == "phasesmith"}
        for r in records.values():
            e = r["effort"]
            assert e["joint_assignment_refinement"] == (mode == "repair")
            assert (
                r["options"]["seed"] == 0
                and r["options"]["cycles"] == 800
                and r["options"]["trials"] == 4
                and r["options"]["max_seconds"] == 13.0
            )
            assert r["options"]["refinement"] == "anisotropic" and not r["options"]["completion"]
            assert e["workers"] == e["requested_workers"] == e["max_active_workers"] == 1
            assert e["parallel_maps"] == e["parallel_tasks"] == 0
            assert 0 <= e["joint_assignment_fits"] <= 6 and 0 <= e["joint_assignment_accepted"] <= 3
            assert 0 <= e["joint_assignment_depths"] <= 48
            if mode == "control":
                assert all(
                    e[k] == 0
                    for k in [
                        "joint_assignment_scores",
                        "joint_assignment_depths",
                        "joint_assignment_fits",
                        "joint_assignment_accepted",
                        "joint_assignment_seconds",
                    ]
                )
            if e["joint_assignment_accepted"]:
                assert r["termination"] == "joint_assignment_repair"
        assert sum(r["reference_exceeds_native_resource_cap"] for r in records.values()) == 7
        reportpath = ROOT / "benchmarks" / f"{a.prefix}-{mode}-macos.json"
        shutil.copyfile(base / "comparison-verified.json", reportpath)
        reports[mode] = report
        rows[mode] = records
        paths[mode] = reportpath
    assert reports["control"]["implementation_sha256"] == reports["repair"]["implementation_sha256"]
    comparison = compare(paths, manifest, [("control", "repair")])
    (ROOT / "benchmarks" / f"{a.prefix}-comparison-macos.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    quality = {m: sets(r) for m, r in dict(previous=previous, **rows).items()}
    assert quality["control"] == quality["previous"]
    assert all(counts(rows["control"][i]) == counts(previous[i]) for i in ids)
    changes = []
    common = []
    for i in sorted(ids):
        old, new = rows["control"][i], rows["repair"][i]
        if old.get("model") and new.get("model"):
            common.append(i)
        if counts(old) != counts(new):
            ref = (new.get("reference_audit") or old.get("reference_audit") or {})[
                "reference_counts"
            ]

            def error(r, reference=ref):
                return sum(
                    abs(counts(r).get(e, 0) - reference.get(e, 0))
                    for e in set(counts(r)) | set(reference)
                )

            changes.append(
                dict(
                    id=i,
                    control=counts(old),
                    repair=counts(new),
                    reference=ref,
                    old_l1_count_error=error(old),
                    new_l1_count_error=error(new),
                    old_r1=(old.get("model") or {}).get("reported_r1"),
                    new_r1=(new.get("model") or {}).get("reported_r1"),
                )
            )
    mode_summary = {}
    for mode, records in rows.items():
        old = rows["control"]
        kernel = sum(r["kernel_seconds"] for r in records.values())
        wall = sum(r["wall_seconds_including_startup"] for r in records.values())
        speedup = float(
            np.median(
                [old[i]["kernel_seconds"] / records[i]["kernel_seconds"] for i in sorted(ids)]
            )
        )
        mode_summary[mode] = dict(
            case_sets=quality[mode],
            summary=reports[mode]["summary"],
            per_element_counts_correct=len(quality[mode]["element_counts"]),
            lost_previous={
                key: sorted(set(quality["previous"][key]) - set(quality[mode][key]))
                for key in quality[mode]
            },
            gained_structures=sorted(
                set(quality[mode]["structures"]) - set(quality["previous"]["structures"])
            ),
            summed_solver_seconds=kernel,
            summed_wall_seconds=wall,
            median_paired_solver_control_over_run=speedup,
            median_paired_wall_control_over_run=float(
                np.median(
                    [
                        old[i]["wall_seconds_including_startup"]
                        / records[i]["wall_seconds_including_startup"]
                        for i in sorted(ids)
                    ]
                )
            ),
            repair_work={
                key: sum(r["effort"][key] for r in records.values())
                for key in [
                    "joint_assignment_scores",
                    "joint_assignment_depths",
                    "joint_assignment_fits",
                    "joint_assignment_accepted",
                    "joint_assignment_seconds",
                ]
            },
            repaired_cases=[
                i for i, r in records.items() if r["effort"]["joint_assignment_accepted"]
            ],
        )
    count_errors = {}
    for mode in MODES:
        total = 0
        for i in common:
            audit = rows[mode][i]["reference_audit"]
            inferred, reference = audit["inferred_counts"], audit["reference_counts"]
            total += sum(
                abs(inferred.get(e, 0) - reference.get(e, 0))
                for e in set(inferred) | set(reference)
            )
        count_errors[mode] = total
    record = dict(
        schema="phasesmith.joint-assignment-retention.v1",
        development_cases=35,
        holdout_opened=False,
        qualification=(
            "One balanced serial, same-native, one-worker comparison. No "
            "task-owned test/build/kernel jobs overlap timed corpus "
            "collection. All 140 predictions precede both reference "
            "barriers. Original observations, count caps, physical "
            "objectives, coordinate tolerances, seeds and 13/15-second "
            "budgets retained. Pilot and oracle-assisted diagnostics are not "
            "blind benchmark outcomes."
        ),
        previous_report_sha256=sha(oldpath),
        native_sha256=sha(_core.__file__),
        source_files_verified=len(reports["control"]["implementation_sha256"]),
        schedule_sha256=sha(a.base / "schedule.json"),
        report_sha256={k: sha(v) for k, v in paths.items()},
        modes=mode_summary,
        count_vector_changes=changes,
        paired_models_for_count_error=len(common),
        paired_model_l1_count_error=count_errors,
        all19_retained=not mode_summary["repair"]["lost_previous"]["structures"],
        all_previous_success_sets_retained=not any(
            mode_summary["repair"]["lost_previous"].values()
        ),
        verifier_sha256=sha(__file__),
    )
    output = ROOT / "benchmarks" / f"{a.prefix}-retention-macos.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["all_previous_success_sets_retained"]:
        raise ValueError("previous successes lost; retain evidence and investigate")


if __name__ == "__main__":
    main()
