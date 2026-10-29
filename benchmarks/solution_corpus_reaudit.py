"""Re-audit immutable saved predictions after output-format adapter corrections.

No solver is executed and no prediction is selected using reference coordinates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from phasesmith.io.hkl import read_hkl
from solution_corpus.audit import audit_case, common_score, shelxt_candidate
from solution_corpus.data import sha
from solution_corpus.run import ROOT


def reaudit(args):
    report = json.loads((args.run / "comparison.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    if report["manifest_sha256"] != sha(args.manifest):
        parent = manifest.get("prediction_manifest_parent")
        if (
            not parent
            or parent["sha256"] != report["manifest_sha256"]
            or sha(ROOT / parent["path"]) != parent["sha256"]
        ):
            raise ValueError("reference registry has no verified collection parent")
        original = json.loads((ROOT / parent["path"]).read_text())
        before = {c["id"]: c for c in original["cases"]}
        for case in manifest["cases"]:
            old = before[case["id"]]
            if any(case[key] != old[key] for key in ["files", "family", "split"]):
                raise ValueError("reference revision changed prediction inputs/split/raw source")

    cases = [c for c in manifest["cases"] if c["split"] == report["split"]]
    expected = {(c["id"], method) for c in cases for method in ("phasesmith", "shelxt")}
    actual = [(r["id"], r["method"]) for r in report["runs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete prediction collection; do not grade a partial campaign")
    records = report["runs"]
    for record in records:
        for key in ("correct", "reference_audit", "common_fixed_uiso_score", "audit_error"):
            record.pop(key, None)
        if record["method"] != "shelxt":
            continue
        model_path = args.run / record["directory"] / "sample_a.res"
        if model_path.exists():
            try:
                model = shelxt_candidate(model_path)
                log = model_path.with_name("stdout.txt").read_text(errors="replace")
                if "1 threads running in parallel" not in log:
                    raise ValueError("one-thread external execution not confirmed")
                record["collection_output_adapter_status"] = record["status"]
                if "output_adapter_error" in record:
                    record["collection_output_adapter_error"] = record.pop("output_adapter_error")
                record["status"] = "model_returned"
                record["model"] = model
                record["selected_model_sha256"] = sha(model_path)
            except Exception as error:
                record["output_adapter_error"] = f"{type(error).__name__}: {error}"
    # Freeze repaired parsing of every prediction before reading a reference.
    normalized = args.output.with_name(args.output.stem + "-predictions-before-reference.json")
    if normalized.exists():
        raise ValueError("prediction audit snapshot must be new")
    normalized.write_text(json.dumps(records, indent=2) + "\n")
    for case in cases:
        reference = case["files"]["reference"]
        if sha(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError("reference changed")
        observations = read_hkl(ROOT / case["files"]["common_hkl"]["path"])
        for record in [r for r in records if r["id"] == case["id"]]:
            record["correct"] = False
            record["total_non_h_count_correct"] = (
                bool(record.get("model"))
                and len(record["model"]["species"]) == case["curation"]["reference_non_h_atoms"]
            )
            record["reference_exceeds_native_resource_cap"] = (
                case["curation"]["reference_non_h_atoms"] > record["input_only_resource_atom_cap"]
            )
            if not record.get("model"):
                record["failure_class"] = record["status"]
                continue
            try:
                result = audit_case(case, record["model"], ROOT)
                record["reference_audit"] = result
                record["correct"] = result["correct"]
                record["space_group_number_agrees"] = (
                    record["model"]["space_group_number"] == case["curation"]["space_group_number"]
                )
                record["failure_class"] = (
                    "correct"
                    if record["correct"]
                    else "wrong_non_h_counts_or_labels"
                    if not result["counts_match"]
                    else result.get("status", "coordinate_mismatch")
                )
                if len(record["model"]["species"]) <= 128 and np.allclose(
                    record["model"]["cell"], case["curation"]["cell"], rtol=1e-5, atol=1e-4
                ):
                    record["common_fixed_uiso_score"] = common_score(
                        record["model"], observations, case["curation"]["cell"]
                    )
            except Exception as error:
                record["audit_error"] = f"{type(error).__name__}: {error}"
                record["failure_class"] = "unassessed_audit_error"
    summary = {}
    for method in ["phasesmith", "shelxt"]:
        rows = [r for r in records if r["method"] == method]
        times = [r["wall_seconds_including_startup"] for r in rows]
        summary[method] = {
            "cases": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "total_non_h_count_correct": sum(r["total_non_h_count_correct"] for r in rows),
            "strict_002_angstrom": sum(
                r.get("reference_audit", {}).get("strict_002_angstrom", False) for r in rows
            ),
            "hard_timeouts": sum(r["hard_timeout"] for r in rows),
            "median_wall_seconds_all_outcomes": float(np.median(times)),
            "p95_wall_seconds_all_outcomes": float(np.percentile(times, 95)),
            "failure_classes": dict(Counter(r["failure_class"] for r in rows)),
            "cases_within_native_resource_cap": sum(
                not r["reference_exceeds_native_resource_cap"] for r in rows
            ),
            "correct_within_native_resource_cap": sum(
                r["correct"] and not r["reference_exceeds_native_resource_cap"] for r in rows
            ),
        }
    pairs = Counter()
    for case in cases:
        pair = {r["method"]: r["correct"] for r in records if r["id"] == case["id"]}
        key = (
            "both_correct"
            if all(pair.values())
            else "phasesmith_only"
            if pair["phasesmith"]
            else "shelxt_only"
            if pair["shelxt"]
            else "neither_correct"
        )
        pairs[key] += 1
    report.update(
        summary=summary,
        runs=records,
        paired_outcomes=dict(pairs),
        audit_correction=(
            "Normalize external group names, recover actual output operations/origin, "
            "and merge printed special-position image duplicates; "
            "raw selected models, source files, inputs and timings unchanged"
        ),
        reference_registry_sha256=sha(args.manifest),
        reference_registry_lock_sha256=sha(args.manifest.with_suffix(".benchmark4.lock.json")),
        reference_normalization=manifest.get("reference_normalization"),
        collection_report_sha256=sha(args.run / "comparison.json"),
        normalized_predictions_sha256=sha(normalized),
        normalized_predictions_path=str(normalized),
        audit_implementation_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__).resolve(),
                ROOT / "benchmarks/solution_corpus/audit.py",
                ROOT / "benchmarks/solution_corpus/data.py",
            ]
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "paired": dict(pairs)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new; no silent replacement of an audit")
    reaudit(args)


if __name__ == "__main__":
    main()
