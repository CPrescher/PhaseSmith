"""Verify complete development-only normalization/phase-quality comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(baseline, updated, manifest):
    a, b = (json.loads(p.read_text()) for p in (baseline, updated))
    m = json.loads(manifest.read_text())
    cases = {c["id"]: c for c in m["cases"] if c["split"] == "development"}
    expected = {(id, method) for id in cases for method in ("phasesmith", "shelxt")}
    for r in (a, b):
        pairs = [(x["id"], x["method"]) for x in r["runs"]]
        if (
            r["split"] != "development"
            or r["holdout_opened"]
            or len(pairs) != len(expected)
            or set(pairs) != expected
            or r["reference_registry_sha256"] != sha(manifest)
        ):
            raise ValueError("complete identical development panels/reference registry required")
    for key in (
        "score_policy",
        "seconds_per_job",
        "resource_policy",
        "thread_policy",
        "shelxt_sha256",
        "input_contract",
    ):
        if a[key] != b[key]:
            raise ValueError("comparison policy differs: " + key)
    old = {r["id"]: r for r in a["runs"] if r["method"] == "phasesmith"}
    new = {r["id"]: r for r in b["runs"] if r["method"] == "phasesmith"}
    controls = (
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
        "adaptive_counts",
    )
    records = []
    for id, case in cases.items():
        before, after = old[id], new[id]
        if (
            before["options"] != after["options"]
            or before["input_only_resource_atom_cap"] != after["input_only_resource_atom_cap"]
        ):
            raise ValueError("numerical budget/options/ceiling changed")
        for key in controls:
            if before["effort"][key] != after["effort"][key]:
                raise ValueError("other search controls changed: " + key)
        if any(
            before["effort"].get(k, False)
            for k in ("wilson_normalization", "phase_quality_selection")
        ) or not all(
            after["effort"][k] for k in ("wilson_normalization", "phase_quality_selection")
        ):
            raise ValueError("requires legacy normalization/SSE versus both new features")
        row = {"id": id, "reference_non_h_atoms": case["curation"]["reference_non_h_atoms"]}
        for label, r in (("baseline", before), ("updated", after)):
            row[label] = {
                "atoms": len(r["model"]["species"]) if r.get("model") else None,
                "counts_correct": r["total_non_h_count_correct"],
                "element_counts_correct": r.get("reference_audit", {}).get("counts_match", False),
                "structure_correct": r["correct"],
                "wall_seconds": r["wall_seconds_including_startup"],
                "failure_class": r["failure_class"],
                "phasing_starts": r["effort"]["phasing_starts"],
                "screening_refinements": r["effort"]["screening_refinements"],
                "final_refinements": r["effort"]["final_refinements"],
            }
        row["normalization_diagnostic"] = after["effort"].get("normalization_diagnostic")
        row["phase_model_trials"] = after["effort"].get("phase_model_trials", [])
        records.append(row)
    shared = [
        r
        for r in records
        if r["baseline"]["structure_correct"] and r["updated"]["structure_correct"]
    ]
    return {
        "schema": "phasesmith.phase-quality-comparison.v1",
        "implementation_sha256": sha(Path(__file__)),
        "baseline_sha256": sha(baseline),
        "updated_sha256": sha(updated),
        "manifest_sha256": sha(manifest),
        "holdout_outcomes_opened": False,
        "same_native_build": a["native_sha256"] == b["native_sha256"],
        "cases": len(records),
        "baseline": a["summary"],
        "updated": b["summary"],
        "gained_structures": [
            r["id"]
            for r in records
            if r["updated"]["structure_correct"] and not r["baseline"]["structure_correct"]
        ],
        "lost_structures": [
            r["id"]
            for r in records
            if r["baseline"]["structure_correct"] and not r["updated"]["structure_correct"]
        ],
        "baseline_exact_element_counts": sum(
            r["baseline"]["element_counts_correct"] for r in records
        ),
        "updated_exact_element_counts": sum(
            r["updated"]["element_counts_correct"] for r in records
        ),
        "normalization_fallbacks": [r["id"] for r in records if r["normalization_diagnostic"]],
        "common_recovered_cases": len(shared),
        "baseline_median_seconds_common_recovered": float(
            np.median([r["baseline"]["wall_seconds"] for r in shared])
        )
        if shared
        else None,
        "updated_median_seconds_common_recovered": float(
            np.median([r["updated"]["wall_seconds"] for r in shared])
        )
        if shared
        else None,
        "normalization_unit_mean_maximum_error": max(
            abs(r["effort"]["normalization"]["mean_e_squared"] - 1)
            for r in new.values()
            if r["effort"].get("normalization")
        ),
        "phase_model_attempts": dict(Counter(len(r["phase_model_trials"]) for r in records)),
        "qualification": (
            "Serial complete collections, not paired repeated timing. All 35 cases retained, "
            "identical source/derived inputs and scoring gates. Native arm64 versus "
            "SHELXT x86_64 compatibility runtime. No holdout outcomes opened."
        ),
        "runs": records,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "updated", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output must be new")
    report = compare(args.baseline, args.updated, args.manifest)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
