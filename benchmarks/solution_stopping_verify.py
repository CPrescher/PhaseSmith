"""Verify four-mode development provenance and report every recovery/count change."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from phasesmith import _core

MODES = ("control", "phase", "assignment", "cache")
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--prefix", default="structure-solution-stopping")
    a = p.parse_args()
    ids = {
        c["id"]
        for c in load(ROOT / "validation/solution-corpus-v5.json")["cases"]
        if c["split"] == "development"
    }
    events = load(a.base / "schedule.json")
    assert len(events) == 284 and len(ids) == 35
    assert not any(e.get("reference_audit_barrier") for e in events[:280])
    assert all(e.get("reference_audit_barrier") for e in events[280:])
    leads = Counter()
    for i in range(70):
        group = events[4 * i : 4 * i + 4]
        offset = (i // 2) % 4
        assert [e["mode"] for e in group] == list(MODES[offset:] + MODES[:offset])
        assert len({(e["id"], e["method"]) for e in group}) == 1
        if i % 2 == 0:
            leads[group[0]["mode"]] += 1
    reports = {}
    data = {}
    snapshots = []
    paths = {}
    post_collection_fixes = []
    for mode in MODES:
        path = ROOT / "benchmarks" / f"{a.prefix}-{mode}-macos.json"
        r = load(path)
        paths[mode] = path
        assert r["split"] == "development" and not r["holdout_opened"] and len(r["runs"]) == 70
        for method in ["phasesmith", "shelxt"]:
            assert {x["id"] for x in r["runs"] if x["method"] == method} == ids
        assert r["before_reference_audit_sha256"] == sha(
            a.base / mode / "before-reference-audit.json"
        )
        assert r["native_sha256"] == sha(_core.__file__)
        for name, h in r["implementation_sha256"].items():
            snapshot = a.base / mode / "source-snapshot-stage1" / name
            assert sha(snapshot) == h, name
            if sha(ROOT / name) != h:
                assert name == "benchmarks/solution_stopping_interleaved.py", name
                original = snapshot.read_text()
                assert "expected = len(MODES) * sum(" in original
                assert (
                    original.replace("expected = len(MODES) * sum(", "expected = 2 * sum(")
                    == (ROOT / name).read_text()
                )
                post_collection_fixes.append(
                    dict(
                        mode=mode,
                        source=name,
                        original_sha256=h,
                        current_sha256=sha(ROOT / name),
                        qualification=(
                            "Only the final completion assertion is corrected: 70 "
                            "case/method rounds, 280 predictions. All original sources are "
                            "retained in verified snapshots; no timed code or numerical "
                            "source changed."
                        ),
                    )
                )
        snapshots.append(r["implementation_sha256"])
        assert r["phase_plateau_stopping"] == (mode in ("phase", "cache"))
        assert r["assignment_convergence_stopping"] == (mode in ("assignment", "cache"))
        assert all(
            r[k]
            for k in [
                "blocked_normal_matrices",
                "reuse_normal_matrices",
                "short_phase_starts",
                "ambiguous_assignments",
                "multiplicity_weighted_refinement",
            ]
        )
        rows = {x["id"]: x for x in r["runs"] if x["method"] == "phasesmith"}
        for row in rows.values():
            assert (
                row["options"]["cycles"] == 800
                and row["options"]["trials"] == 4
                and row["options"]["refinement"] == "anisotropic"
            )
            e = row["effort"]
            assert e["phase_plateau_stopping"] == r["phase_plateau_stopping"]
            assert e["assignment_convergence_stopping"] == r["assignment_convergence_stopping"]
            for sl in e["phase_slices"]:
                quota = 200 if sl["trial"] < 2 else 800
                assert sl["cycle_quota"] == quota and 0 <= sl["completed_cycles"] <= quota
                if sl["stop_reason"] == "phase_plateau":
                    assert mode in ("phase", "cache") and sl["trial"] < 2
                    assert sl["completed_cycles"] >= 128 and sl["completed_cycles"] % 32 == 0
            if mode in ("control", "phase"):
                assert e["assignment_convergence_stops"] == 0
        assert sum(row["reference_exceeds_native_resource_cap"] for row in rows.values()) == 7
        data[mode] = rows
        reports[mode] = r
    assert all(s == snapshots[0] for s in snapshots)
    comparison = load(ROOT / "benchmarks" / f"{a.prefix}-comparison-macos.json")
    for mode in MODES:
        assert comparison["runs"][mode]["sha256"] == sha(paths[mode])
    previous_path = ROOT / "benchmarks/structure-solution-blocked-normal-cache-macos.json"
    previous = {x["id"]: x for x in load(previous_path)["runs"] if x["method"] == "phasesmith"}
    checks = {
        "structures": lambda x: bool(x["correct"]),
        "total_non_h_counts": lambda x: bool(x["total_non_h_count_correct"]),
        "per_element_counts": lambda x: bool((x.get("reference_audit") or {}).get("counts_match")),
        "strict_002_angstrom": lambda x: bool(
            (x.get("reference_audit") or {}).get("strict_002_angstrom")
        ),
    }
    sets = {
        key: {
            mode: sorted(i for i, row in rows.items() if f(row))
            for mode, rows in dict(previous=previous, **data).items()
        }
        for key, f in checks.items()
    }
    assert len(sets["structures"]["previous"]) == 19
    for key in sets:
        assert sets[key]["previous"] == sets[key]["control"], key
    modes = {}
    for mode, rows in data.items():
        ratios = [
            data["control"][i]["wall_seconds_including_startup"]
            / rows[i]["wall_seconds_including_startup"]
            for i in ids
        ]
        solver_ratios = [
            data["control"][i]["kernel_seconds"] / rows[i]["kernel_seconds"] for i in ids
        ]
        changes = []
        for i in sorted(ids):

            def count(r):
                return dict(Counter(r["model"]["species"])) if r.get("model") else {}

            if count(rows[i]) != count(data["control"][i]):
                changes.append(
                    dict(id=i, control=count(data["control"][i]), treatment=count(rows[i]))
                )
        stages = Counter()
        for case in comparison["runs"][mode]["cases"]:
            stages.update(case["stage_seconds"])
        lost = {
            key: sorted(set(groups["previous"]) - set(groups[mode])) for key, groups in sets.items()
        }
        modes[mode] = dict(
            lost_cases=lost,
            retains_previous_success_and_count_sets=not any(lost.values()),
            changed_count_vectors=changes,
            cumulative_wall_seconds=sum(x["wall_seconds_including_startup"] for x in rows.values()),
            cumulative_solver_seconds=sum(x["kernel_seconds"] for x in rows.values()),
            median_paired_wall_control_over_treatment=float(np.median(ratios)),
            median_paired_solver_control_over_treatment=float(np.median(solver_ratios)),
            faster_cases=sum(x > 1 for x in ratios),
            stage_seconds=dict(stages),
            work_totals=comparison["runs"][mode]["work_totals"],
        )
    recommended = min(
        (m for m in MODES if modes[m]["retains_previous_success_and_count_sets"]),
        key=lambda m: modes[m]["cumulative_solver_seconds"],
    )
    record = dict(
        schema="phasesmith.stopping-retention.v1",
        development_cases=35,
        holdout_opened=False,
        qualification=(
            "Four balanced serial development modes. Every prediction saved "
            "before any reference audit. Previous best case sets and all "
            "inferred count vectors checked. Timings from one collection "
            "remain host sensitive; early phase handoffs can alter later "
            "models."
        ),
        schedule_sha256=sha(a.base / "schedule.json"),
        leading_cases=dict(leads),
        report_sha256={m: sha(p) for m, p in paths.items()},
        native_sha256=sha(_core.__file__),
        source_files_verified=len(snapshots[0]),
        previous_sha256=sha(previous_path),
        case_sets=sets,
        modes=modes,
        recommended_mode=recommended,
        selection_rule=(
            "Minimum summed library solve time among development variants "
            "retaining every prior structure and count success set"
        ),
        post_collection_fixes=post_collection_fixes,
        verifier_sha256=sha(__file__),
        all_case_counts={
            i: {
                m: dict(Counter(rows[i]["model"]["species"])) if rows[i].get("model") else {}
                for m, rows in dict(previous=previous, **data).items()
            }
            for i in sorted(ids)
        },
    )
    (ROOT / "benchmarks" / f"{a.prefix}-retention-macos.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    for mode in MODES:
        x = modes[mode]
        print(mode, {k: v for k, v in x.items() if k not in ["work_totals", "stage_seconds"]})
        print("recovery", {k: len(v[mode]) for k, v in sets.items()})


if __name__ == "__main__":
    main()
