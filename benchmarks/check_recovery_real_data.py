#!/usr/bin/env python3
"""Compare unchanged real-data recipes across installed baseline/current builds.

This checks scientific results, not timings. Known holdout failures remain in
both records; they are not converted to passing gates.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import phasesmith as ps
from phasesmith import validation as validation

ROOT = Path(__file__).resolve().parents[1]


def reports():
    root = ROOT / "validation/data"
    cases = {
        "qarr_original": lambda: validation.run_qarr_1g_validation(
            root / "iucr-qarr-1g", execution=ps.ExecutionPolicy(threads=8)
        ),
        "pbso4_xray": lambda: validation.run_pbso4_cw_validation(
            root / "gsasii-pbso4-cw",
            ps.RadiationProbe.X_RAY,
            execution=ps.ExecutionPolicy(threads=1),
        ),
        "pbso4_neutron": lambda: validation.run_pbso4_cw_validation(
            root / "gsasii-pbso4-cw",
            ps.RadiationProbe.NEUTRON,
            execution=ps.ExecutionPolicy(threads=1),
        ),
        "qarr_holdout": lambda: validation.run_qarr_1h_validation(root / "iucr-qarr-1h"),
    }
    results = {}
    for name, run in cases.items():
        record = run().to_record()
        record.pop("elapsed_seconds")
        results[name] = record
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(reports(), allow_nan=False))
        return
    if args.baseline_package is None or args.json_output is None:
        parser.error("--baseline-package and --json-output required")
    output = {}
    for name, package in (("baseline", args.baseline_package.resolve()), ("current", None)):
        env = dict(os.environ)
        env.update(
            {
                key: "1"
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            }
        )
        if package is None:
            env.pop("PYTHONPATH", None)
        else:
            env["PYTHONPATH"] = str(package)
        run = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        output[name] = json.loads(run.stdout)
    args.json_output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for name, before in output["baseline"].items():
        after = output["current"][name]
        if before["status"] != after["status"] or before["checks"] != after["checks"]:
            raise RuntimeError(f"scientific checks changed: inspect retained record for {name}")


if __name__ == "__main__":
    main()
