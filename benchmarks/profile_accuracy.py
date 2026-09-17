#!/usr/bin/env python3
"""Measure explicit CW accuracy policies on existing measured-data workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import phasesmith as ps
from investigate_qarr_performance import run_case
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_pbso4_cw_validation, verify_validation_dataset

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "reference": ps.ProfileAccuracy(),
    "fast_fcj": ps.ProfileAccuracy(fast_fcj=True),
    "tail_1_percent": ps.ProfileAccuracy(tail_area_tolerance=0.01),
    "combined_1_percent": ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if ps._core.BUILD_MODE != "release" or args.repetitions <= 0:
        parser.error("release build and positive repetitions required")
    root = ROOT / "validation/data/iucr-qarr-1g"
    files = verify_validation_dataset("iucr-qarr-1g", root)
    cases = {}
    configs = [(name, threads) for threads in (1, 2, 8) for name in POLICIES]
    for repeat in range(-1, args.repetitions):
        for name, threads in configs if repeat % 2 else reversed(configs):
            accuracy = POLICIES[name]
            overrides = {i: {"profile_accuracy": accuracy} for i in (1, 2, 3)}
            run, _ = run_case(root, threads, overrides)
            if run["scientific_result"]["status"] != "passed":
                raise RuntimeError(f"real-data quality gate failed: {name}")
            key = f"{name}_{threads}"
            if repeat == -1:
                cases[key] = {
                    "accuracy": asdict(accuracy),
                    "threads": threads,
                    "seconds": [],
                    "scientific": run["scientific_result"],
                    "stage_seconds": [],
                }
            else:
                case = cases[key]
                if run["scientific_result"] != case["scientific"]:
                    raise RuntimeError(f"nonrepeatable scientific result: {key}")
                case["seconds"].append(run["seconds"])
                case["stage_seconds"].append([s["seconds"] for s in run["stages"]])
            print(f"round={repeat} {key}: {run['seconds']:.6f}s", flush=True)
    for name in POLICIES:
        scientific = cases[f"{name}_1"]["scientific"]
        for threads in (1, 2, 8):
            case = cases[f"{name}_{threads}"]
            if case["scientific"] != scientific:
                raise RuntimeError(f"worker count changed result: {name}")
            case["median_seconds"] = statistics.median(case["seconds"])
    pbso4 = {}
    original = rv.refine
    for name, accuracy in POLICIES.items():

        def refine(request, options, _accuracy=accuracy, **kwargs):
            return original(request, replace(options, profile_accuracy=_accuracy), **kwargs)

        with patch.object(rv, "refine", refine):
            for probe in (ps.RadiationProbe.X_RAY, ps.RadiationProbe.NEUTRON):
                report = run_pbso4_cw_validation(
                    ROOT / "validation/data/gsasii-pbso4-cw",
                    probe,
                    execution=ps.ExecutionPolicy(threads=1),
                )
                if report.status != "passed":
                    raise RuntimeError(f"PbSO4 quality gate failed: {name}/{probe}")
                pbso4[f"{name}_{probe.value}"] = report.to_record()
    record = {
        "powder_intensity_convention": "friedel_pair_average",
        "scope": "explicit profile approximations on existing real data; defaults unchanged",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "build": ps._core.BUILD_MODE,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "warmups": 1,
        "repetitions": args.repetitions,
        "qarr": cases,
        "pbso4": pbso4,
        "quality_gates_passed": True,
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
