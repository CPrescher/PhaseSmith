#!/usr/bin/env python3
"""Benchmark explicit empirical Gaussian conventions on measured IUCr QARR 1g.

Run in the pinned rietx 1.4.0 benchmark environment with BLAS threads set to 1.
All candidates, including failed scientific gates, are retained. Default
validation recipes and solver settings are not changed by this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import phasesmith as ps
from compare_rietx_qarr import run_rietx
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset
from rietx._about import COMPILED_THREADS_ENV
from rietx.model import compiled

ROOT = Path(__file__).resolve().parents[1]


def run_case(root, threads, *, empirical, extended, fast):
    original = rv.refine
    stages = []
    convention = ps.EmpiricalGaussianConvention("ZnO", 2e-4)
    accuracy = (
        ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)
        if fast
        else ps.ProfileAccuracy()
    )

    def refine(request, options, **kwargs):
        if empirical:
            # The legacy acceptance helper constructs independent requests per
            # stage; attach the same convention to each. Public recipe workflows
            # preserve this automatically through dataclass replacement.
            request = (
                convention.apply(request)
                if not stages
                else replace(request, empirical_gaussian=convention)
            )
        options = replace(options, profile_accuracy=accuracy)
        if extended:
            options = replace(
                options,
                limits=replace(
                    options.limits,
                    max_iterations=100,
                    max_evaluations=1500,
                    max_consecutive_rejections=200,
                ),
            )
        started = time.perf_counter()
        result = original(request, options, **kwargs)
        stages.append(
            {
                "seconds": time.perf_counter() - started,
                "rwp": result.metrics.rwp,
                "evaluations": result.evaluations,
                "iterations": len(result.history),
                "termination": result.termination_reason.value,
                "backend": result.backend,
                "limits": asdict(options.limits),
                "empirical_gaussian": None
                if result.checkpoint.empirical_gaussian is None
                else result.checkpoint.empirical_gaussian.to_record(),
            }
        )
        return result

    with patch.object(rv, "refine", refine):
        report = run_qarr_1g_validation(root, execution=ps.ExecutionPolicy(threads=threads))
    scientific = report.to_record()
    scientific.pop("elapsed_seconds")
    return {"scientific": scientific, "stages": stages, "profile_accuracy": asdict(accuracy)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0 or args.repetitions <= 0 or ps._core.BUILD_MODE != "release":
        parser.error("positive workers/repetitions and a release build required")
    if importlib.metadata.version("rietx") != "1.4.0":
        parser.error("requires rietx 1.4.0")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"launch with {key}=1")
    os.environ[COMPILED_THREADS_ENV] = str(args.threads)
    compiled.warm(block=True)
    if not compiled.enabled():
        raise RuntimeError("rietx compiled kernels unavailable")
    root = ROOT / "validation/data/iucr-qarr-1g"
    paths = verify_validation_dataset("iucr-qarr-1g", root)
    configs = {
        f"{name}_{'fast' if fast else 'default'}": dict(
            empirical=empirical, extended=extended, fast=fast
        )
        for name, empirical, extended in (
            ("baseline", False, False),
            ("empirical", True, False),
            ("empirical_extended", True, True),
        )
        for fast in (False, True)
    }
    cases = {}
    original_calculate = rv.calculate
    execution = ps.ExecutionPolicy(threads=args.threads)

    def controlled_calculate(*args, **kwargs):
        kwargs.setdefault("execution", execution)
        return original_calculate(*args, **kwargs)

    with patch.object(rv, "calculate", controlled_calculate):
        names = [*configs, "rietx"]
        for round_index in range(-1, args.repetitions):
            for name in names if round_index % 2 else reversed(names):
                started = time.perf_counter()
                record = (
                    run_rietx(root)
                    if name == "rietx"
                    else run_case(root, args.threads, **configs[name])
                )
                seconds = time.perf_counter() - started
                stage_seconds = (
                    [s.pop("seconds") for s in record.get("stages", [])] if name != "rietx" else []
                )
                if round_index == -1:
                    cases[name] = {
                        "config": configs.get(name),
                        "result": record,
                        "seconds": [],
                        "stage_seconds": [],
                    }
                else:
                    if record != cases[name]["result"]:
                        raise RuntimeError(f"nondeterministic scientific results: {name}")
                    cases[name]["seconds"].append(seconds)
                    cases[name]["stage_seconds"].append(stage_seconds)
                print(f"round={round_index} {name} {seconds:.4f}s", flush=True)
    for case in cases.values():
        case["median_seconds"] = statistics.median(case["seconds"])
    record = {
        "schema": "phasesmith.empirical-gaussian-qarr.v1",
        "threads": args.threads,
        "warmups": 1,
        "repetitions": args.repetitions,
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("phasesmith", "rietx", "numpy", "scipy", "numba")
            },
            "native_sha256": hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
        },
        "cases": cases,
        "notes": [
            "Full workflow timings include model/scale/background preparation, empirical "
            "transfer preflight and final QPA.",
            "Shared preparation respects the requested worker budget for both libraries; "
            "worker count is not measured CPU usage.",
            "Empirical convention fixes ZnO RMS strain to 0.0002; this is not a measured "
            "or calibrated width decomposition.",
            "Extended candidates explicitly allow 100 iterations, 1500 evaluations and 200 "
            "consecutive rejections per stage.",
            "Background freezes after stage one as in the existing acceptance recipe. "
            "Global defaults remain unchanged.",
            "Different refinement constraints, stopping rules and profile policies: these "
            "times do not establish equal-model speed superiority.",
            "Only deterministic repeatability and existing dataset checks are asserted; "
            "reference-phase sensitivity requires separate assessment.",
        ],
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
