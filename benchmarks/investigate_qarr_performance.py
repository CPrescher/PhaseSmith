#!/usr/bin/env python3
"""Investigate optimized QARR scheduling and solver controls without changing defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import phasesmith as ps
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset

ROOT = Path(__file__).resolve().parents[1]


def run_case(root, threads, overrides):
    stages = []
    captured = []
    original = rv.refine

    def record(request, options, **kwargs):
        stage = len(stages) + 1
        options = replace(options, **overrides.get(stage, {}))
        diagnostic_times = []
        original_calculate = rv.calculate

        def timed_calculate(*args, **kwargs):
            before = time.perf_counter()
            value = original_calculate(*args, **kwargs)
            diagnostic_times.append(time.perf_counter() - before)
            return value

        started = time.perf_counter()
        with patch.object(rv, "calculate", timed_calculate):
            result = original(request, options, **kwargs)
        stages.append(
            {
                "stage": stage,
                "seconds": time.perf_counter() - started,
                "diagnostic_calculation_seconds": sum(diagnostic_times),
                "backend": result.backend,
                "evaluations": result.evaluations,
                "iterations": len(result.history),
                "rwp": result.metrics.rwp,
                "termination": result.termination_reason.value,
                "history": [
                    {
                        "backtracks": h.backtracks,
                        "step_norm": h.scaled_step_norm,
                        "damping": h.damping,
                        "cg_iterations": h.cg_iterations,
                        "rwp": h.rwp,
                    }
                    for h in result.history
                ],
            }
        )
        captured.append((request, options, result))
        return result

    started = time.perf_counter()
    with patch.object(rv, "refine", record):
        report = run_qarr_1g_validation(root, execution=ps.ExecutionPolicy(threads=threads))
    seconds = time.perf_counter() - started
    scientific = report.to_record()
    scientific.pop("elapsed_seconds")
    return {"seconds": seconds, "stages": stages, "scientific_result": scientific}, captured


def linear_scale_probe(captured):
    request, options, native = captured[-1]
    started = time.perf_counter()
    calculation = rv.calculate(
        request.pattern,
        request.experiment,
        request.phases,
        background=request.background,
        support_fwhm=options.support_fwhm,
        execution=options.execution,
    )
    basis = np.column_stack(
        [
            c.profile_y / p.scale
            for c, p in zip(calculation.phase_calculations, request.phases, strict=True)
        ]
    )
    included = native.metrics.included
    sigma = request.pattern.uncertainty[included]
    matrix = basis[included] / sigma[:, None]
    observed = (request.pattern.observed_y[included] - calculation.background[included]) / sigma
    scales = np.linalg.lstsq(matrix, observed, rcond=None)[0]
    covariance = np.linalg.inv(matrix.T @ matrix)
    y = basis @ scales + calculation.background
    elapsed = time.perf_counter() - started
    weighted_residual = (y[included] - request.pattern.observed_y[included]) / sigma
    rwp = float(
        np.linalg.norm(weighted_residual)
        / np.linalg.norm(request.pattern.observed_y[included] / sigma)
    )
    if not np.all(scales > 0) or rwp > native.metrics.rwp + 1e-10:
        raise RuntimeError("linear scale proof did not recover the native fit")
    return {
        "seconds": elapsed,
        "scales": scales.tolist(),
        "rwp": rwp,
        "positive_scales": bool(np.all(scales > 0)),
        "finite_covariance": bool(np.isfinite(covariance).all()),
        "relative_y_delta_vs_native": float(
            np.linalg.norm(y - native.calculation.y) / np.linalg.norm(native.calculation.y)
        ),
        "relative_scale_delta_vs_native": float(
            np.max(np.abs(scales / np.array([p.scale for p in native.phases]) - 1))
        ),
        "scope": "One profile basis, unconstrained weighted scale solve and covariance; "
        "not a complete replacement including checkpoints, diagnostics and QPA.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions <= 0 or ps._core.BUILD_MODE != "release":
        parser.error("positive repetitions and release build required")
    root = ROOT / "validation/data/iucr-qarr-1g"
    paths = verify_validation_dataset("iucr-qarr-1g", root)
    configurations = [(f"threads_{n}", n, {}) for n in (1, 2, 4, 6, 8, 12, 14)]
    configurations += [
        (f"stage1_damping_{d:g}", 1, {1: {"initial_damping": d}}) for d in (1e-3, 1.0, 1e3)
    ]
    configurations += [
        (f"stage2_step_{s:g}", 1, {2: {"max_scaled_parameter_step": s}}) for s in (0.3, 0.6)
    ]
    cases = {}
    linear = []
    # One warmup per configuration, then reverse alternate rounds to reduce order bias.
    for repeat in range(-1, args.repetitions):
        order = configurations if repeat % 2 else list(reversed(configurations))
        for name, threads, overrides in order:
            run, captured = run_case(root, threads, overrides)
            if repeat == -1:
                cases[name] = {
                    "threads": threads,
                    "overrides": overrides,
                    "scientific_result": run["scientific_result"],
                    "stage_history": run["stages"],
                    "seconds": [],
                    "stage_seconds": [],
                }
            else:
                case = cases[name]
                if run["scientific_result"] != case["scientific_result"]:
                    raise RuntimeError(f"nondeterministic result for {name}")
                case["seconds"].append(run["seconds"])
                case["stage_seconds"].append([s["seconds"] for s in run["stages"]])
                if name == "threads_1":
                    linear.append(linear_scale_probe(captured))
            print(
                f"round={repeat} {name}: {run['seconds']:.4f}s "
                f"{run['scientific_result']['status']}",
                flush=True,
            )
    baseline = cases["threads_1"]["scientific_result"]
    for name, case in cases.items():
        case["median_seconds"] = statistics.median(case["seconds"])
        if name.startswith("threads_") and case["scientific_result"] != baseline:
            raise RuntimeError("thread count changed scientific result")
    record = {
        "scope": "QARR performance investigation; solver variants are experiments only",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "build": ps._core.BUILD_MODE,
            "thread_environment": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "warmups": 1,
        "repetitions": args.repetitions,
        "cases": cases,
        "linear_scale_probe": linear,
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
