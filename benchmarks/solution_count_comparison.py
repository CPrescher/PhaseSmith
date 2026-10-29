"""Compare complete development-only count campaigns without running a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(baseline_path, updated_path, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    baseline, updated = (json.loads(p.read_text()) for p in (baseline_path, updated_path))
    cases = {c["id"]: c for c in manifest["cases"] if c["split"] == "development"}
    expected = {(id, method) for id in cases for method in ("phasesmith", "shelxt")}
    for report in (baseline, updated):
        actual = [(r["id"], r["method"]) for r in report["runs"]]
        if (
            report["split"] != "development"
            or report["holdout_opened"]
            or len(actual) != len(expected)
            or set(actual) != expected
            or report["reference_registry_sha256"] != digest(manifest_path)
        ):
            raise ValueError(
                "requires complete identical development panels and reference registry"
            )
    for key in (
        "score_policy",
        "seconds_per_job",
        "resource_policy",
        "input_contract",
        "thread_policy",
        "shelxt_sha256",
    ):
        if baseline[key] != updated[key]:
            raise ValueError("changed comparison policy: " + key)
    old = {r["id"]: r for r in baseline["runs"] if r["method"] == "phasesmith"}
    new = {r["id"]: r for r in updated["runs"] if r["method"] == "phasesmith"}
    same_build = baseline["native_sha256"] == updated["native_sha256"]
    if same_build and (
        any(r["effort"]["adaptive_counts"] for r in old.values())
        or not all(r["effort"]["adaptive_counts"] for r in new.values())
    ):
        raise ValueError("same-build ablation must disable/enable only the count policy")

    def failure_kind(record, expected_n):
        if not record.get("model"):
            return "no_model"
        found = len(record["model"]["species"])
        if found < expected_n:
            return "undercount"
        if found > expected_n:
            return "overcount"
        if not record.get("reference_audit", {}).get("counts_match", False):
            return "element_counts_wrong_at_correct_total"
        return "correct" if record["correct"] else "coordinates_wrong_at_correct_counts"

    rows = []
    for id, case in cases.items():
        a, b = old[id], new[id]
        if a["input_only_resource_atom_cap"] != b["input_only_resource_atom_cap"]:
            raise ValueError("resource cap changed")
        if a["options"] != b["options"]:
            raise ValueError("numerical solve options changed")
        for key in (
            "dual_checkpoints",
            "cached_refinement",
            "symmetry_recycling",
            "recycling_cycles",
            "tangent_refinement",
            "tangent_in_recycling",
            "integrated_peak_density",
            "peak_density_radius_angstrom",
            "staged_refinement",
            "screening_iterations",
            "refinement_shortlist",
            "maximum_atoms",
            "beam_width",
        ):
            if a["effort"][key] != b["effort"][key]:
                raise ValueError("shared-search configuration changed: " + key)
        expected_n = case["curation"]["reference_non_h_atoms"]
        rows.append(
            {
                "id": id,
                "reference_non_h_atoms": expected_n,
                "baseline_count_failure": failure_kind(a, expected_n),
                "updated_count_failure": failure_kind(b, expected_n),
                "reference_exceeds_resource_cap": a["reference_exceeds_native_resource_cap"],
                "baseline_atoms": len(a["model"]["species"]) if a.get("model") else None,
                "updated_atoms": len(b["model"]["species"]) if b.get("model") else None,
                "baseline_correct": a["correct"],
                "updated_correct": b["correct"],
                "baseline_counts_and_labels_correct": a.get("reference_audit", {}).get(
                    "counts_match", False
                ),
                "updated_counts_and_labels_correct": b.get("reference_audit", {}).get(
                    "counts_match", False
                ),
                "baseline_total_count_correct": a["total_non_h_count_correct"],
                "updated_total_count_correct": b["total_non_h_count_correct"],
                "baseline_wall_seconds": a["wall_seconds_including_startup"],
                "updated_wall_seconds": b["wall_seconds_including_startup"],
                "baseline_failure": a["failure_class"],
                "updated_failure": b["failure_class"],
            }
        )
    shared = [r for r in rows if r["baseline_correct"] and r["updated_correct"]]
    return {
        "schema": "phasesmith.solution-count-comparison.v1",
        "comparison_implementation_sha256": digest(Path(__file__)),
        "baseline_sha256": digest(baseline_path),
        "updated_sha256": digest(updated_path),
        "manifest_sha256": digest(manifest_path),
        "holdout_outcomes_opened": False,
        "cases": len(cases),
        "same_native_build": same_build,
        "baseline": baseline["summary"],
        "updated": updated["summary"],
        "gained_structures": [
            r["id"] for r in rows if r["updated_correct"] and not r["baseline_correct"]
        ],
        "lost_structures": [
            r["id"] for r in rows if r["baseline_correct"] and not r["updated_correct"]
        ],
        "baseline_count_failures": dict(Counter(r["baseline_count_failure"] for r in rows)),
        "updated_count_failures": dict(Counter(r["updated_count_failure"] for r in rows)),
        "baseline_exact_counts_and_labels": sum(
            r["baseline_counts_and_labels_correct"] for r in rows
        ),
        "updated_exact_counts_and_labels": sum(
            r["updated_counts_and_labels_correct"] for r in rows
        ),
        "common_recovered_cases": len(shared),
        "baseline_median_seconds_common_recovered": float(
            np.median([r["baseline_wall_seconds"] for r in shared])
        )
        if shared
        else None,
        "updated_median_seconds_common_recovered": float(
            np.median([r["updated_wall_seconds"] for r in shared])
        )
        if shared
        else None,
        "qualification": (
            (
                "Same native build, legacy/adaptive serial collections; "
                if same_build
                else "Previous frozen baseline versus new serial collection; "
            )
            + "not paired repeated timing. "
            "All 35 cases retained; identical input hashes and scoring policy. "
            "Native arm64 versus SHELXT x86_64 compatibility runtime. "
            "No holdout solver outcomes opened."
        ),
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    report = compare(args.baseline, args.updated, args.manifest)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
