#!/usr/bin/env python3
"""Measure worker scaling at fixed converged QARR fit quality.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
in the existing pinned benchmark environment. Reuses the empirical Gaussian
benchmark's extended recipe and default profiles. No production defaults change.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import empirical_gaussian_qarr as bench
import phasesmith as ps
from phasesmith.refinement import rietveld as rv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            parser.error(f"launch with {name}=1")
    if ps._core.BUILD_MODE != "release":
        parser.error("requires a release build")
    original = rv.calculate
    root = Path(__file__).resolve().parents[1] / "validation/data/iucr-qarr-1g"
    cases = {}
    for repetition in range(-1, 3):
        for workers in [1, 2, 4, 8] if repetition % 2 else [8, 4, 2, 1]:

            def controlled(*call_args, workers=workers, **kwargs):
                kwargs.setdefault("execution", ps.ExecutionPolicy(threads=workers))
                return original(*call_args, **kwargs)

            with patch.object(rv, "calculate", controlled):
                started = time.perf_counter()
                result = bench.run_case(root, workers, empirical=True, extended=True, fast=False)
                elapsed = time.perf_counter() - started
            for stage in result["stages"]:
                stage.pop("seconds")
            if repetition == -1:
                cases[workers] = {"result": result, "seconds": []}
            else:
                if result != cases[workers]["result"]:
                    raise RuntimeError("scientific results changed between repetitions")
                cases[workers]["seconds"].append(elapsed)
            print(repetition, workers, elapsed, flush=True)
    for case in cases.values():
        case["median_seconds"] = statistics.median(case["seconds"])
        if case["result"] != cases[1]["result"]:
            raise RuntimeError("worker count changed scientific results")
    record = {
        "scope": "Current empirical extended default-profile QARR quality "
        "held fixed across worker budgets",
        "warmups": 1,
        "repetitions": 3,
        "shared_preparation": "same worker budget as fit",
        "blas_threads": 1,
        "native_sha256": hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
