#!/usr/bin/env python3
"""Audit tighter continuation from the best standard original-width solution.

Retain stage-one background and all physical model choices. Restore the full
stage-two selection after scale polish, then repeat full/scale polish at most
five times. Diagnostic only, not a timed benchmark or production recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import phasesmith as ps
import qarr_convergence as benchmark
from investigate_qarr_holdout import request_after
from phasesmith.refinement import RefinementLimits
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import verify_validation_dataset


def run(sample, workers, feasible_width_steps=False):
    saved = []
    finish = benchmark.finish

    def capture(result, pattern, sample):
        saved.append((result, pattern))
        return finish(result, pattern, sample)

    with patch.object(benchmark, "finish", capture):
        baseline = benchmark.run_case(sample, workers, "standard")
    result, pattern = saved[-1]
    execution = ps.ExecutionPolicy(threads=workers)
    full_selection = rv.RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=True,
        sample_physics=True,
        background=False,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2", "zero_shift_deg"),
    )
    scale_selection = replace(
        full_selection, u_iso=False, sample_physics=False, instrument_parameters=()
    )
    options = rv.RietveldOptions(
        limits=RefinementLimits(
            max_iterations=1000, max_evaluations=20000, max_consecutive_rejections=500
        ),
        min_iterations=3,
        max_scaled_parameter_step=0.15,
        support_fwhm=30,
        objective_tolerance=1e-12,
        parameter_tolerance=1e-9,
        estimate_covariance=False,
        execution=execution,
        feasible_width_steps=feasible_width_steps,
    )
    iterations = []
    stable = False
    for _ in range(5):
        before = result.metrics.chi_square
        request = request_after(result, pattern, full_selection, None)
        full = rv.refine(request, options)
        probe = benchmark.stationarity_probe(request, options, full)
        scale = request_after(full, pattern, scale_selection, None)
        result = rv.refine(
            scale,
            replace(
                options, min_iterations=1, max_scaled_parameter_step=1.0, estimate_covariance=True
            ),
        )
        if full.backend != "native" or result.backend != "native":
            raise RuntimeError("unexpected Python solver dispatch")
        improvement = (before - result.metrics.chi_square) / max(before, 1.0)
        if improvement < -1e-12:
            raise RuntimeError("continuation increased the objective")
        iterations.append(
            dict(
                full=benchmark.call_record(full),
                scale=benchmark.call_record(result),
                probe=probe,
                relative_chi_square_improvement=improvement,
                scientific=finish(result, pattern, sample),
            )
        )
        stable = improvement <= benchmark.RESTART_RELATIVE_OBJECTIVE_TOLERANCE
        if stable:
            break
    return dict(baseline=baseline, polish=iterations, stable=stable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--feasible-width-steps", action="store_true")
    parser.add_argument("--json-output", required=True, type=Path)
    args = parser.parse_args()
    if args.workers < 1 or ps._core.BUILD_MODE != "release":
        parser.error("positive worker count and release build required")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"launch with {key}=1")
    record = dict(
        schema="phasesmith.qarr-convergence-polish.v1",
        workers=args.workers,
        feasible_width_steps=args.feasible_width_steps,
        native_sha256=hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        convergence_driver_sha256=hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
        solver_source_sha256={
            path: hashlib.sha256((benchmark.ROOT / path).read_bytes()).hexdigest()
            for path in (
                "crates/phasesmith-workflows/src/rietveld_general_solver.rs",
                "crates/phasesmith-workflows/src/rietveld_feasible_step.rs",
                "crates/phasesmith-workflows/src/rietveld_general_objective.rs",
                "python/phasesmith/refinement/rietveld.py",
                "python/phasesmith/refinement/_feasible_step.py",
            )
        },
        dataset_sha256={
            str(path.relative_to(benchmark.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for sample in ("1g", "1h")
            for path in verify_validation_dataset(
                f"iucr-qarr-{sample}", benchmark.ROOT / f"validation/data/iucr-qarr-{sample}"
            )
        },
        cases={},
        notes=[
            "No new profile terms, empirical convention, or background freedom.",
            "Native canonical scales are rebuilt when preparing fresh requests.",
            "Coordinate descent probes do not modify accepted fits.",
            "Polish stability does not certify absence of feasible descent.",
        ],
    )
    for sample in ("1g", "1h"):
        record["cases"][sample] = run(sample, args.workers, args.feasible_width_steps)
        last = record["cases"][sample]["polish"][-1]["scientific"]
        print(sample, last, flush=True)
        args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
