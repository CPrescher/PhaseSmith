#!/usr/bin/env python3
"""Ablate rietx's QARR support/axial policies without changing either package.

These are diagnostic interventions in version-pinned benchmark internals, not
equivalent-model benchmarks or proposed PhaseSmith numerical defaults.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import phasesmith as ps
import rietx as rx
from compare_rietx_qarr import prepare, run_rietx
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset
from rietx._about import COMPILED_THREADS_ENV
from rietx.model import compiled, forward
from rietx.model.profiles import fcj

ROOT = Path(__file__).resolve().parents[1]
RX_REFINE = importlib.import_module("rietx.refine")
RX_LSQ = importlib.import_module("rietx.optimize.least_squares")
CASES = ("default", "retain_small_axial", "30_fwhm_windows", "both")


@contextmanager
def policy(name):
    with ExitStack() as stack:
        if name in ("retain_small_axial", "both"):
            stack.enter_context(patch.object(fcj, "SKIP_EXTENT_FWHM_RATIO", 0.0))
        if name in ("30_fwhm_windows", "both"):
            # Retain rietx's frozen windows, movement slack and FCJ extent.
            # This does NOT reproduce PhaseSmith's per-node support semantics.
            stack.enter_context(
                patch.object(forward, "window_fwhm_mult", lambda eta: np.full_like(eta, 30.0))
            )
        yield


def measured_rietx(root, name):
    with policy(name):
        start = time.perf_counter()
        result = run_rietx(root)
        return time.perf_counter() - start, result


def diagnostic_rietx(root, name):
    stages = []
    original_solve = RX_REFINE.run_least_squares
    original_scipy = RX_LSQ.least_squares

    def scipy(*args, **kwargs):
        result = original_scipy(*args, **kwargs)
        stages[-1].update(nfev=int(result.nfev), njev=int(result.njev))
        return result

    def solve(model, table, **kwargs):
        row = {"free_parameters": len(table.free_paths), "phases": []}
        for phase in model.phases:
            layout = phase.batch
            nodes = np.where(layout.fcj == 0, 1, 2 * np.maximum(layout.fcj // 2, 4))
            unique, counts = np.unique(layout.fcj, return_counts=True)
            row["phases"].append(
                {
                    "reflection_line_rows": len(layout.width),
                    "symmetric_rows": int(np.count_nonzero(layout.fcj == 0)),
                    "requested_node_histogram": dict(
                        zip(map(str, unique), map(int, counts), strict=True)
                    ),
                    "sample_visits": int(layout.width.sum()),
                    "sample_node_visits": int((layout.width * nodes).sum()),
                }
            )
        stages.append(row)
        return original_solve(model, table, **kwargs)

    with (
        policy(name),
        patch.object(RX_REFINE, "run_least_squares", solve),
        patch.object(RX_LSQ, "least_squares", scipy),
    ):
        result = run_rietx(root)
    return {"stages": stages, "scientific": result}


def phasesmith_run(root, *, symmetric=False):
    original = rv.refine

    def refine(request, options, **kwargs):
        if symmetric:
            request = replace(request, experiment=replace(request.experiment, axial_geometry=None))
        return original(request, options, **kwargs)

    start = time.perf_counter()
    with patch.object(rv, "refine", refine):
        report = run_qarr_1g_validation(root, execution=ps.ExecutionPolicy(threads=1))
    elapsed = time.perf_counter() - start
    if report.status != "passed":
        raise RuntimeError("PhaseSmith quality gate failed")
    record = report.to_record()
    record.pop("elapsed_seconds")
    return elapsed, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions <= 0 or rx.__version__ != "1.4.0" or ps._core.BUILD_MODE != "release":
        parser.error("positive repetitions, rietx 1.4.0 and release PhaseSmith required")
    root = ROOT / "validation/data/iucr-qarr-1g"
    verify_validation_dataset("iucr-qarr-1g", root)
    os.environ[COMPILED_THREADS_ENV] = "1"
    compiled.warm(block=True)
    if not compiled.enabled():
        raise RuntimeError("rietx compiled tier is unavailable")
    cases = {}
    for repeat in range(-1, args.repetitions):
        order = ["phasesmith", "phasesmith_symmetric", *CASES]
        if repeat % 2 == 0:
            order.reverse()
        for name in order:
            elapsed, result = (
                phasesmith_run(root, symmetric=name == "phasesmith_symmetric")
                if name.startswith("phasesmith")
                else measured_rietx(root, name)
            )
            if repeat == -1:
                cases[name] = {"seconds": [], "scientific": result}
            else:
                if result != cases[name]["scientific"]:
                    raise RuntimeError(f"nonrepeatable scientific result: {name}")
                cases[name]["seconds"].append(elapsed)
            print(f"round={repeat} {name}: {elapsed:.6f}s", flush=True)
    pattern, experiment, phases, structure, instrument = prepare(root)
    py = rv.calculate(pattern, experiment, phases, support_fwhm=30).profile_y
    for name, case in cases.items():
        case["median_seconds"] = statistics.median(case["seconds"])
        if not name.startswith("phasesmith"):
            diagnostic = diagnostic_rietx(root, name)
            if diagnostic["scientific"] != case["scientific"]:
                raise RuntimeError("instrumentation changed the result")
            case["workload"] = diagnostic["stages"]
            with policy(name):
                ry = rx.Refinement(structure, instrument, history=False).predict(pattern.x)
            case["initial_forward_relative_l2_vs_phasesmith"] = float(
                np.linalg.norm(py - (ry - pattern.background)) / np.linalg.norm(py)
            )
        elif name == "phasesmith_symmetric":
            sy = rv.calculate(
                pattern, replace(experiment, axial_geometry=None), phases, support_fwhm=30
            ).profile_y
            case["initial_forward_relative_l2_vs_phasesmith"] = float(
                np.linalg.norm(py - sy) / np.linalg.norm(py)
            )
    record = {
        "scope": "causal rietx support/axial ablations on real QARR; not identical models",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rietx": rx.__version__,
        "phasesmith_build": ps._core.BUILD_MODE,
        "threads": 1,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "warmups": 1,
        "repetitions": args.repetitions,
        "quality_gates_passed": True,
        "cases": cases,
        "limitations": [
            "30-FWHM intervention retains rietx's frozen windows and movement slack.",
            "Retaining small axial terms retains rietx's quadrature count rule.",
            "Timing changes include changes in solver trajectory and evaluation counts.",
            "Workload counts describe each frozen stage, not total optimizer kernel calls.",
            "Node visits include zero-weight nodes in rietx's equal-height split rule.",
            "phasesmith is the unchanged native benchmark; no production policies were modified.",
            "phasesmith_symmetric explicitly removes axial geometry for all fit stages only; "
            "initial scale preparation remains unchanged and 30-FWHM support is retained.",
        ],
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
