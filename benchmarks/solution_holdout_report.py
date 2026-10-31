"""Verify and summarize the complete frozen holdout comparison; never run a solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from solution_holdout_frozen import FREEZE, ROOT, load, validate_prediction_barrier, verify_freeze
from solution_corpus.data import sha


def summarize(base):
    freeze = load(FREEZE)
    verify_freeze(freeze)
    provenance = load(base / "frozen-run.json")
    report_path = base / "comparison-verified.json"
    report = load(report_path)
    ids = set(freeze["holdout_ids"])
    assert len(ids) == 65 and report["split"] == "holdout" and report["holdout_opened"]
    assert len(report["runs"]) == 130
    assert report["manifest_sha256"] == freeze["manifest_sha256"]
    assert report["native_sha256"] == freeze["native_sha256"]
    assert report["shelxt_sha256"] == freeze["comparator_sha256"]
    assert provenance["freeze_sha256"] == sha(FREEZE)
    assert provenance["barrier"]["prediction_sha256"] == validate_prediction_barrier(
        base / "predictions-at-barrier.json", freeze["holdout_ids"]
    )
    assert report["before_reference_audit_sha256"] == sha(base / "before-reference-audit.json")
    events = load(base / "holdout-schedule.json")
    assert len(events) == 131 and events[-1] == {"reference_audit_barrier": True}
    assert not any(e.get("reference_audit_barrier") for e in events[:-1])
    for i in range(65):
        pair = events[2 * i : 2 * i + 2]
        order = ["phasesmith", "shelxt"] if i % 2 == 0 else ["shelxt", "phasesmith"]
        assert [e["method"] for e in pair] == order and pair[0]["id"] == pair[1]["id"]
    for name, expected in report["implementation_sha256"].items():
        assert sha(ROOT / name) == sha(base / report["source_snapshot"] / name) == expected
    for key, value in freeze["controls"].items():
        if key in report:
            assert report[key] == value, key
    summaries = {}
    records = {}
    for method in ["phasesmith", "shelxt"]:
        rows = {r["id"]: r for r in report["runs"] if r["method"] == method}
        assert len(rows) == 65 and set(rows) == ids
        records[method] = rows
        summaries[method] = dict(report["summary"][method])
        summaries[method].update(
            per_element_counts_correct=sum(
                bool((r.get("reference_audit") or {}).get("counts_match")) for r in rows.values()
            ),
            summed_wall_seconds=sum(r["wall_seconds_including_startup"] for r in rows.values()),
            returned_models=sum(bool(r.get("model")) for r in rows.values()),
            audit_errors={i: r["audit_error"] for i, r in rows.items() if r.get("audit_error")},
        )
        if method == "phasesmith":
            summaries[method]["summed_solver_seconds_returned_jobs"] = sum(
                r.get("kernel_seconds", 0) for r in rows.values()
            )
            for r in rows.values():
                if "effort" not in r:
                    continue  # Process errors/timeouts remain in every primary denominator.
                e = r["effort"]
                assert e["workers"] == e["requested_workers"] == e["max_active_workers"] == 1
                assert e["joint_assignment_refinement"] and e["cached_repair_evaluation"]
                for key in ["cycles", "trials", "seed", "max_seconds", "refinement", "completion"]:
                    assert r["options"][key] == freeze["solver_options"][key], key
    ps, ext = records["phasesmith"], records["shelxt"]
    paired_ratios = [
        ps[i]["wall_seconds_including_startup"] / ext[i]["wall_seconds_including_startup"]
        for i in sorted(ids)
    ]
    both = sorted(i for i in ids if ps[i]["correct"] and ext[i]["correct"])
    common = {
        "cases": len(both),
        "ids": both,
        "qualification": "Secondary common-success subset; all 65 remain in primary timing/quality totals.",
    }
    if both:
        common["median_paired_wall_ratio_phasesmith_over_comparator"] = float(
            np.median(
                [ps[i]["wall_seconds_including_startup"] / ext[i]["wall_seconds_including_startup"] for i in both]
            )
        )
    result = {
        "schema": "phasesmith.frozen-holdout-summary.v1",
        "holdout_opened": True,
        "holdout_cases": 65,
        "solver_commit": freeze["solver_commit"],
        "protocol_commit": provenance["protocol_commit"],
        "freeze_sha256": sha(FREEZE),
        "native_sha256": freeze["native_sha256"],
        "comparison_sha256": sha(report_path),
        "prediction_barrier": provenance["barrier"],
        "no_tuning": True,
        "summary": summaries,
        "paired_outcomes": report["paired_outcomes"],
        "median_paired_wall_ratio_phasesmith_over_comparator_all65": float(np.median(paired_ratios)),
        "common_success_timing": common,
        "correct_cases": {m: sorted(i for i, r in rows.items() if r["correct"]) for m, rows in records.items()},
        "failure_cases": {m: {i: r["failure_class"] for i, r in rows.items() if not r["correct"]} for m, rows in records.items()},
        "algorithm_source": freeze["algorithm_sources"],
        "external_test_comparator": freeze["external_test_comparator"],
        "qualification": "One frozen seed-0 comparison, one worker per program, alternating serial program order, identical measured input HKLF4/cell/symmetry/element types and original input-only caps. All 130 predictions precede reference grading. Unmatched successes and failures are retained; timing is final-model-at-stop, not time to a correct structure. PhaseSmith is native arm64; the external comparator uses the host x86_64 compatibility runtime. This holdout is now exposed and cannot validate subsequent tuning as untouched data.",
    }
    return report, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report, result = summarize(args.run.resolve())
    for suffix, value in [("comparison", report), ("summary", result)]:
        p = ROOT / "benchmarks" / f"structure-solution-holdout-frozen-{suffix}-macos.json"
        if p.exists():
            raise ValueError("holdout report must be new; preserve existing evidence")
        p.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
