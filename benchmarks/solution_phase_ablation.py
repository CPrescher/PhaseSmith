"""Compare complete development panels without reading any atom reference/data file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(paths, manifest, pairs=()):
    registry = json.loads(manifest.read_text())
    ids = {c["id"] for c in registry["cases"] if c["split"] == "development"}
    expected = {(id, method) for id in ids for method in ("phasesmith", "shelxt")}
    reports = {label: json.loads(path.read_text()) for label, path in paths.items()}
    control = reports["control"]
    policy = (
        "score_policy",
        "seconds_per_job",
        "resource_policy",
        "thread_policy",
        "shelxt_sha256",
        "input_contract",
    )
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
    baseline = {r["id"]: r for r in control["runs"] if r["method"] == "phasesmith"}
    result = {}
    for label, report in reports.items():
        observed_pairs = [(r["id"], r["method"]) for r in report["runs"]]
        if (
            report["split"] != "development"
            or report["holdout_opened"]
            or report["reference_registry_sha256"] != sha(manifest)
            or len(observed_pairs) != len(expected)
            or set(observed_pairs) != expected
        ):
            raise ValueError("complete development-only predictions required: " + label)
        if any(report[k] != control[k] for k in policy):
            raise ValueError("benchmark policy changed: " + label)
        rows = {r["id"]: r for r in report["runs"] if r["method"] == "phasesmith"}
        for id, row in rows.items():
            a = baseline[id]
            if (
                row["options"] != a["options"]
                or row["input_only_resource_atom_cap"] != a["input_only_resource_atom_cap"]
                or any(row["effort"][k] != a["effort"][k] for k in controls)
            ):
                raise ValueError("other numerical controls changed: " + label + ":" + id)
            for flag in ("wilson_normalization", "phase_quality_selection"):
                if row["effort"][flag] != report[flag]:
                    raise ValueError("header/worker feature mismatch")
        common = [id for id in ids if rows[id]["correct"] and baseline[id]["correct"]]
        cases = []
        for id in sorted(ids):
            r = rows[id]
            cases.append(
                {
                    "id": id,
                    "correct": r["correct"],
                    "total_count_correct": r["total_non_h_count_correct"],
                    "element_counts_correct": r.get("reference_audit", {}).get(
                        "counts_match", False
                    ),
                    "atoms": len(r["model"]["species"]) if r.get("model") else None,
                    "seconds": r["wall_seconds_including_startup"],
                    "r1": r["model"]["reported_r1"] if r.get("model") else None,
                    "phasing_starts": r["effort"]["phasing_starts"],
                    "quality_alternatives": sum(
                        q.get("alternative_handoffs", 0)
                        for q in r["effort"].get("phase_quality", [])
                    ),
                }
            )
        result[label] = {
            "report": str(paths[label]),
            "sha256": sha(paths[label]),
            "native_sha256": report["native_sha256"],
            "same_native_as_control": report["native_sha256"] == control["native_sha256"],
            "wilson_normalization": report["wilson_normalization"],
            "phase_quality_selection": report["phase_quality_selection"],
            "phase_first_selection": report.get(
                "phase_first_selection", report["phase_quality_selection"]
            ),
            "calculated_shell_normalization": report.get("calculated_shell_normalization", False),
            "summary": report["summary"],
            "exact_element_counts": sum(c["element_counts_correct"] for c in cases),
            "gained": sorted(
                id for id in ids if rows[id]["correct"] and not baseline[id]["correct"]
            ),
            "lost": sorted(id for id in ids if baseline[id]["correct"] and not rows[id]["correct"]),
            "common_successes": len(common),
            "median_common_seconds_control": float(
                np.median([baseline[id]["wall_seconds_including_startup"] for id in common])
            )
            if common
            else None,
            "median_common_seconds_run": float(
                np.median([rows[id]["wall_seconds_including_startup"] for id in common])
            )
            if common
            else None,
            "cases": cases,
        }
    comparisons = []
    for before, after in pairs:
        a, b = result[before], result[after]
        old, new = ({c["id"]: c for c in r["cases"]} for r in (a, b))
        common = [id for id in ids if old[id]["correct"] and new[id]["correct"]]
        comparisons.append(
            {
                "before": before,
                "after": after,
                "same_native_build": a["native_sha256"] == b["native_sha256"],
                "gained": sorted(id for id in ids if new[id]["correct"] and not old[id]["correct"]),
                "lost": sorted(id for id in ids if old[id]["correct"] and not new[id]["correct"]),
                "common_successes": len(common),
                "before_median_seconds_common": float(
                    np.median([old[id]["seconds"] for id in common])
                )
                if common
                else None,
                "after_median_seconds_common": float(
                    np.median([new[id]["seconds"] for id in common])
                )
                if common
                else None,
            }
        )
    return {
        "schema": "phasesmith.phase-ablation.v1",
        "implementation_sha256": sha(Path(__file__)),
        "manifest_sha256": sha(manifest),
        "development_cases": len(ids),
        "holdout_outcomes_opened": False,
        "qualification": (
            "Complete serial collections, unchanged inputs/options/caps and primary scoring gates. "
            "Single-seed deadline-bounded outcomes; timing is not a paired repeated estimate. "
            "Old quality mode includes phase-first scheduling; improved mode changes that schedule."
        ),
        "runs": result,
        "comparisons": comparisons,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, help="unique label=report.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pair", action="append", default=[], help="before:after labels")
    args = p.parse_args()
    paths = dict(value.split("=", 1) for value in args.run)
    if len(paths) != len(args.run) or "control" not in paths or args.output.exists():
        p.error("unique labels, a control and a new output are required")
    pairs = [tuple(value.split(":")) for value in args.pair]
    if any(len(pair) != 2 or any(label not in paths for label in pair) for pair in pairs):
        p.error("pairs must name two supplied labels")
    report = compare({label: Path(path) for label, path in paths.items()}, args.manifest, pairs)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for label, row in report["runs"].items():
        print(label, row["summary"]["phasesmith"], "gained", row["gained"], "lost", row["lost"])


if __name__ == "__main__":
    main()
