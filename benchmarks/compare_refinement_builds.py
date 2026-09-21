#!/usr/bin/env python3
"""Interleave previous/current installed release builds on the existing real QARR fit."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = r"""
import json, sys, time
from pathlib import Path
from unittest.mock import patch
import phasesmith as ps
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset
root = Path(sys.argv[1])
verify_validation_dataset("iucr-qarr-1g", root)
assert ps._core.BUILD_MODE == "release"
for line in sys.stdin:
    threads = int(line)
    stages = []
    original = rv.refine
    def measured(request, options, **kwargs):
        start = time.perf_counter()
        result = original(request, options, **kwargs)
        stages.append(time.perf_counter() - start)
        return result
    start = time.perf_counter()
    with patch.object(rv, "refine", measured):
        report = run_qarr_1g_validation(root, execution=ps.ExecutionPolicy(threads=threads))
    elapsed = time.perf_counter() - start
    record = report.to_record()
    record.pop("elapsed_seconds")
    print(json.dumps({"seconds": elapsed, "stage_seconds": stages, "scientific": record,
                      "package": ps.__file__}, allow_nan=False), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", required=True, type=Path)
    parser.add_argument("--repetitions", default=5, type=int)
    parser.add_argument("--threads", type=int, action="append")
    parser.add_argument("--json-output", required=True, type=Path)
    args = parser.parse_args()
    if args.repetitions <= 0 or not (args.baseline_package / "phasesmith").is_dir():
        parser.error("positive repetitions and a baseline package directory are required")
    workers = args.threads or (1, 8)
    if any(value <= 0 for value in workers) or len(set(workers)) != len(workers):
        parser.error("worker counts must be positive and unique")
    children = {}
    results = {name: {} for name in ("previous", "current")}
    try:
        for name in results:
            env = dict(os.environ)
            env.update(
                {
                    key: "1"
                    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
                }
            )
            if name == "previous":
                env["PYTHONPATH"] = str(args.baseline_package.resolve())
            else:
                env.pop("PYTHONPATH", None)
            children[name] = subprocess.Popen(
                [sys.executable, "-u", "-c", WORKER, str(ROOT / "validation/data/iucr-qarr-1g")],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
        for threads in workers:
            for repeat in range(-1, args.repetitions):
                order = ("previous", "current") if repeat % 2 else ("current", "previous")
                for name in order:
                    child = children[name]
                    child.stdin.write(f"{threads}\n")
                    child.stdin.flush()
                    line = child.stdout.readline()
                    if not line:
                        raise RuntimeError(f"{name} worker exited unexpectedly")
                    run = json.loads(line)
                    if run["scientific"]["status"] != "passed":
                        raise RuntimeError(f"{name} quality gate failed")
                    if repeat == -1:
                        results[name][threads] = {
                            "scientific": run["scientific"],
                            "package": run["package"],
                            "seconds": [],
                            "stage_seconds": [],
                        }
                    else:
                        case = results[name][threads]
                        if run["scientific"] != case["scientific"]:
                            raise RuntimeError(f"{name} repeat changed scientific result")
                        case["seconds"].append(run["seconds"])
                        case["stage_seconds"].append(run["stage_seconds"])
            print(f"threads={threads} completed", flush=True)
        for cases in results.values():
            expected = next(iter(cases.values()))["scientific"]
            if any(case["scientific"] != expected for case in cases.values()):
                raise RuntimeError("worker count changed scientific result")
            for case in cases.values():
                case["median_seconds"] = statistics.median(case["seconds"])
                case["stage_median_seconds"] = [
                    statistics.median(row[i] for row in case["stage_seconds"]) for i in range(3)
                ]
        record = {
            "scope": "interleaved release builds; unchanged real QARR recipe",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "warmups": 1,
            "repetitions": args.repetitions,
            "results": results,
        }
        args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    finally:
        for child in children.values():
            child.stdin.close()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
