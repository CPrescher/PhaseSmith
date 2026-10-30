"""Compare complete development panels without reading any atom reference/data file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_timing(baseline, rows, ids, field):
    """Compare matched cases; never silently drop a case with missing timing."""
    ids = sorted(ids)
    if not ids or any(field not in source[id] for id in ids for source in (baseline, rows)):
        return None
    old = np.array([baseline[id][field] for id in ids], dtype=float)
    new = np.array([rows[id][field] for id in ids], dtype=float)
    if not np.all(np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)):
        raise ValueError("paired timings must be positive finite values")
    ratio = old / new
    return {
        "cases": len(ids),
        "median_control_over_run": float(np.median(ratio)),
        "median_run_minus_control_seconds": float(np.median(new - old)),
        "faster_cases": int(np.count_nonzero(new < old)),
        "slower_cases": int(np.count_nonzero(new > old)),
    }


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
            for flag in (
                "wilson_normalization",
                "phase_quality_selection",
                "phase_origin_selection",
                "chemical_assignment",
                "iterative_refinement",
                "missing_reflections",
                "adaptive_checkpoints",
                "score_only_evaluation",
                "work_sliced_phasing",
                "reusable_fourier_workspaces",
                "independent_origin_sampling",
                "resolution_extension",
            ):
                default = 1.0 if flag == "resolution_extension" else False
                if row["effort"].get(flag, default) != report.get(flag, default):
                    raise ValueError("header/worker feature mismatch")
        common = [id for id in ids if rows[id]["correct"] and baseline[id]["correct"]]
        cases = []
        for id in sorted(ids):
            r = rows[id]
            origins = r["effort"].get("phase_origins", [])
            stage_seconds = (
                {
                    k: sum(a[k] for a in origins)
                    for k in (
                        "origin_seconds",
                        "recycling_seconds",
                        "completion_seconds",
                        "screening_seconds",
                        "refinement_seconds",
                    )
                }
                if report.get("phase_origin_selection", False)
                else None
            )
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
                    "phase_search_seconds": sum(
                        a["seconds"] for a in r["effort"].get("phase_slices", [])
                    ),
                    "short_phase_slices": sum(
                        a["cycle_quota"] < r["options"]["cycles"]
                        for a in r["effort"].get("phase_slices", [])
                    ),
                    "short_quota_completions": sum(
                        a["cycle_quota"] < r["options"]["cycles"]
                        and a["stop_reason"] == "cycle_limit"
                        for a in r["effort"].get("phase_slices", [])
                    ),
                    "score_only_evaluations": r["effort"].get("score_only_evaluations", 0),
                    "fourier_workspace_cycles": r["effort"].get("fourier_workspace_cycles", 0),
                    "adaptive_checkpoint_tests": len(r["effort"].get("phase_checkpoints", [])),
                    "stable_map_handoffs": sum(
                        c["reason"] == "stable_map"
                        for c in r["effort"].get("phase_checkpoints", [])
                    ),
                    "quality_scored_cycles": sum(
                        len(q.get("cfom", [])) for q in r["effort"].get("phase_quality", [])
                    ),
                    "screening_refinements": r["effort"].get("screening_refinements", 0),
                    "final_refinements": r["effort"].get("final_refinements", 0),
                    "completion_evaluations": sum(
                        s["evaluations"] for a in origins for s in a["settings"]
                    ),
                    "reassignment_evaluations": sum(
                        s.get("reassignment_evaluations", 0) for a in origins for s in a["settings"]
                    ),
                    "reassignment_changes": sum(
                        s.get("reassignment_changes", 0) for a in origins for s in a["settings"]
                    ),
                    "reassignment_screens": sum(
                        s.get("reassignment_screens", 0) for a in origins for s in a["settings"]
                    ),
                    "chemical_scale_proposals": sum(
                        len(s.get("chemical_scales", [])) for a in origins for s in a["settings"]
                    ),
                    "prediction": r["effort"].get("prediction"),
                    "prediction_steps": r["effort"].get("prediction_steps", 0),
                    "prediction_updates": r["effort"].get("prediction_updates", 0),
                    "prediction_clipped": r["effort"].get("prediction_clipped", 0),
                    "prediction_seconds": r["effort"].get("prediction_seconds", 0.0),
                    "stage_seconds": stage_seconds,
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
            "phase_origin_selection": report.get("phase_origin_selection", False),
            "chemical_assignment": report.get("chemical_assignment", False),
            "iterative_refinement": report.get("iterative_refinement", False),
            "missing_reflections": report.get("missing_reflections", False),
            "adaptive_checkpoints": report.get("adaptive_checkpoints", False),
            "score_only_evaluation": report.get("score_only_evaluation", False),
            "work_sliced_phasing": report.get("work_sliced_phasing", False),
            "reusable_fourier_workspaces": report.get("reusable_fourier_workspaces", False),
            "independent_origin_sampling": report.get("independent_origin_sampling", False),
            "resolution_extension": report.get("resolution_extension", 1.0),
            "phase_first_selection": report.get(
                "phase_first_selection", report["phase_quality_selection"]
            ),
            "calculated_shell_normalization": report.get("calculated_shell_normalization", False),
            "summary": report["summary"],
            "exact_element_counts": sum(c["element_counts_correct"] for c in cases),
            "work_totals": {
                k: sum(c[k] for c in cases)
                for k in (
                    "phasing_starts",
                    "quality_scored_cycles",
                    "adaptive_checkpoint_tests",
                    "score_only_evaluations",
                    "phase_search_seconds",
                    "short_phase_slices",
                    "short_quota_completions",
                    "fourier_workspace_cycles",
                    "stable_map_handoffs",
                    "screening_refinements",
                    "final_refinements",
                    "completion_evaluations",
                    "reassignment_evaluations",
                    "reassignment_changes",
                    "reassignment_screens",
                    "chemical_scale_proposals",
                    "prediction_steps",
                    "prediction_updates",
                    "prediction_clipped",
                    "prediction_seconds",
                )
            },
            "stage_seconds_totals": {
                k: sum(c["stage_seconds"][k] for c in cases) for k in cases[0]["stage_seconds"]
            }
            if cases[0]["stage_seconds"] is not None
            else None,
            "gained": sorted(
                id for id in ids if rows[id]["correct"] and not baseline[id]["correct"]
            ),
            "lost": sorted(id for id in ids if baseline[id]["correct"] and not rows[id]["correct"]),
            "paired_wall_timing_all_cases": paired_timing(
                baseline, rows, ids, "wall_seconds_including_startup"
            ),
            "paired_kernel_timing_all_cases": paired_timing(baseline, rows, ids, "kernel_seconds"),
            "paired_wall_timing_common_successes": paired_timing(
                baseline, rows, common, "wall_seconds_including_startup"
            ),
            "identical_selected_models_including_no_model": sum(
                rows[id].get("model") == baseline[id].get("model") for id in ids
            ),
            "identical_nonempty_selected_models": sum(
                baseline[id].get("model") is not None
                and rows[id].get("model") == baseline[id]["model"]
                for id in ids
            ),
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
            "Feature flags and actual work counters identify each requested mode."
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
