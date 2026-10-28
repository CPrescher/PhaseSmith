"""Explicit comparison of two phasing strategies on reused, pinned development data.

Child processes run serially with identical seed/trial/cycle/time controls. Every
run saves the blind result before parsing reference coordinates. The output
folder must be new; frozen benchmark records are never regenerated implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--completion", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {"thiourea": {}, "BiSeI": {}, "K[BrF4]": {}}
    implementation = {}
    environment = {}
    for method in ("charge_flip", "dual_space"):
        for script, label in (
            ("structure_solution.py", "thiourea"),
            ("solution_adp_transfer.py", "inorganic"),
        ):
            output = args.output / f"{method}-{label}"
            command = [
                sys.executable,
                str(ROOT / "benchmarks" / script),
                "--phasing",
                method,
                "--output",
                str(output),
            ]
            if script == "structure_solution.py":
                command += ["--refinement", "anisotropic"]
            if args.completion:
                command += ["--completion"]
            if args.download_data:
                command += ["--download-data"]
            with (args.output / f"{method}-{label}.log").open("x") as stream:
                subprocess.run(
                    command, check=True, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                )
            report = json.loads((output / "benchmark.json").read_text())
            implementation.update(report["implementation_sha256"])
            environment.update(
                {
                    key: report[key]
                    for key in ("platform", "python", "numpy", "native_extension_sha256")
                }
            )
            if label == "thiourea":
                best = report["best_candidate"]
                cases[label][method] = {
                    "r1": None if best is None else best["r1"],
                    "target_met": report["amplitude_target_met"],
                    "termination": report["native_termination"],
                    "elapsed_seconds": report["wall_seconds"],
                    "phasing_trials": report["phasing_trials"],
                    "post_solve_reference": report.get("post_solve_reference_audit"),
                    "symmetry_refined": bool(best and best["refinement"]),
                    "completion": None if best is None else best["completion"],
                    "options": report["options"],
                }
            else:
                for record in report["cases"]:
                    case = "BiSeI" if record["cod"] == "2010577" else "K[BrF4]"
                    cases[case][method] = {**record, "options": report["options"]}
            print(f"{method} {label}: completed", flush=True)
    implementation["benchmarks/solution_dual_comparison.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifests = [
        "validation/structure-solution-thiourea.json",
        "validation/structure-solution-adp-transfer.json",
    ]
    result = {
        "schema": "phasesmith.experimental-dual-comparison.v1",
        "qualification": "Reused development cases, not new validation holdouts. "
        "Reference coordinates are excluded from all searches. "
        "Unweighted non-resonant amplitude fits; no equal-objective SHELX comparison.",
        **environment,
        "implementation_sha256": implementation,
        "source_manifest_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in manifests
        },
        "cases": cases,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for name, methods in cases.items():
        print(
            name,
            {
                m: {"r1": r["r1"], "target_met": r["target_met"], "seconds": r["elapsed_seconds"]}
                for m, r in methods.items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
