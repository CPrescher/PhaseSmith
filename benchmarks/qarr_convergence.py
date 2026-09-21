#!/usr/bin/env python3
"""Time original-width QARR fits with large budgets and convergence/restart audits.

No empirical convention, profile approximation, or background-stage change.
Reports every stop and failed gate; a convergence flag is not global optimality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import phasesmith as ps
from investigate_qarr_holdout import ROOT, TARGETS, request_after, starting_request
from phasesmith.refinement import RefinementLimits
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import real_data as rd
from phasesmith.validation import verify_validation_dataset

CONFIGS = {
    "standard": dict(objective_tolerance=1e-10, parameter_tolerance=1e-7, restarts=0),
    "tight": dict(objective_tolerance=1e-12, parameter_tolerance=1e-9, restarts=0),
    "restart_audit": dict(objective_tolerance=1e-12, parameter_tolerance=1e-9, restarts=5),
}
RESTART_RELATIVE_OBJECTIVE_TOLERANCE = 1e-10


def call_record(result):
    return dict(
        termination=result.termination_reason.value,
        rwp=result.metrics.rwp,
        chi_square=result.metrics.chi_square,
        iterations=len(result.history),
        evaluations=result.evaluations,
        final_damping=result.checkpoint.damping,
        last_step=asdict(result.history[-1]) if result.history else None,
    )


def finish(result, pattern, sample):
    quantitative = tuple(
        ps.QuantitativePhase(
            p.phase_id,
            p.scale,
            *rd._QARR_QPA_METADATA[p.phase_id],
            p.structure.cell.geometry().volume_angstrom3,
        )
        for p in result.phases
    )
    fractions = {
        p.phase_id: p.weight_fraction for p in ps.quantitative_phase_analysis(quantitative)
    }
    error = max(abs(fractions[k] - v) for k, v in TARGETS[sample].items())
    residual = result.calculation.y - pattern.observed_y
    unit_rwp = float(np.linalg.norm(residual) / np.linalg.norm(pattern.observed_y))
    correlation = float(
        np.corrcoef(
            pattern.observed_y - result.calculation.background, result.calculation.profile_y
        )[0, 1]
    )
    return dict(
        rwp=result.metrics.rwp,
        unit_rwp=unit_rwp,
        correlation=correlation,
        fractions=fractions,
        max_fraction_error=error,
        checks=dict(
            poisson_rwp=result.metrics.rwp <= 0.20,
            unit_weight_rwp=unit_rwp <= 0.15,
            profile_correlation=correlation >= 0.98,
            qpa_weight_fraction=error <= 0.02,
        ),
        profile_sha256=hashlib.sha256(result.calculation.y.tobytes()).hexdigest(),
        covariance_sha256=(
            None
            if result.covariance is None
            else hashlib.sha256(result.covariance.tobytes()).hexdigest()
        ),
    )


def stationarity_probe(request, options, result):
    """Look for feasible descent witnesses; this is not a full KKT certificate."""
    from compare_solvers_qarr import Objective

    state = replace(
        request,
        experiment=result.experiment,
        phases=result.phases,
        parameters=result.parameters,
        background=result.background,
    )
    objective = Objective(state, options)
    residual, jacobian, *_ = objective.evaluate(objective.x0)
    chi_square = float(residual @ residual)
    np.testing.assert_allclose(chi_square, result.metrics.chi_square, rtol=5e-12, atol=0)
    gradient = jacobian.T @ residual
    diagonal = np.sum(jacobian * jacobian, axis=0)
    witnesses = []
    for i, key in enumerate(objective.transform.free_keys):
        if diagonal[i] == 0:
            continue
        step = float(np.clip(-gradient[i] / diagonal[i], -0.01, 0.01))
        for factor in (1.0, 0.1, 0.01):
            x = objective.x0.copy()
            x[i] = np.clip(x[i] + factor * step, objective.bounds[0][i], objective.bounds[1][i])
            trial = objective.evaluate(x)
            if trial is None:
                continue
            trial_chi_square = float(trial[0] @ trial[0])
            improvement = (chi_square - trial_chi_square) / max(chi_square, 1.0)
            if improvement > 1e-10:
                witnesses.append(
                    dict(
                        parameter=key.label,
                        scaled_step=float(x[i] - objective.x0[i]),
                        relative_chi_square_improvement=improvement,
                        trial_rwp=float(np.sqrt(trial_chi_square) / objective.denominator),
                    )
                )
    return dict(
        scaled_gradient_inf=float(np.max(np.abs(gradient))),
        relative_objective_difference=chi_square / result.metrics.chi_square - 1,
        feasible_descent_witnesses=sorted(
            witnesses, key=lambda w: -w["relative_chi_square_improvement"]
        )[:10],
    )


def run_case(sample, workers, name, probe=False):
    config = CONFIGS[name]
    started = time.perf_counter()
    execution = ps.ExecutionPolicy(threads=workers)
    initial = starting_request(ROOT / f"validation/data/iucr-qarr-{sample}", sample, execution)
    request = initial
    selections = (
        initial.selection,
        replace(initial.selection, u_iso=True, sample_physics=True, background=False),
        rv.RietveldParameterSelection(
            phase_scale=True, lattice=False, coordinates=False, occupancy=False, u_iso=False
        ),
    )
    stages = []
    result = None
    for i, selection in enumerate(selections):
        stage_started = time.perf_counter()
        if i:
            assert result is not None
            request = request_after(result, initial.pattern, selection, None)
        options = rv.RietveldOptions(
            limits=RefinementLimits(
                max_iterations=1000, max_evaluations=20000, max_consecutive_rejections=500
            ),
            min_iterations=(2, 3, 1)[i],
            max_scaled_parameter_step=(0.2, 0.15, 1.0)[i],
            support_fwhm=30,
            estimate_covariance=(i == 2),
            execution=execution,
            objective_tolerance=config["objective_tolerance"],
            parameter_tolerance=config["parameter_tolerance"],
        )
        result = rv.refine(request, options)
        if result.backend != "native":
            raise RuntimeError("unexpected Python solver dispatch")
        calls = [call_record(result)]
        stable = None
        for _ in range(config["restarts"]):
            before = result.metrics.chi_square
            # Restart from the same physical state and scientific settings.
            # Native preparation rebuilds canonical numerical scales; record any
            # changes rather than confusing this with checkpoint continuation.
            restart = replace(
                request,
                experiment=result.experiment,
                phases=result.phases,
                parameters=result.parameters,
                background=result.background,
            )
            candidate = rv.refine(restart, options)
            if candidate.backend != "native":
                raise RuntimeError("restart left the native solver")
            improvement = (before - candidate.metrics.chi_square) / max(before, 1.0)
            if improvement < -1e-12:
                raise RuntimeError("restart increased the objective")
            record = call_record(candidate)
            record["numerical_scale_changes"] = [
                dict(parameter=p.key.label, before=p.scale, after=c.scale)
                for p, c in zip(result.parameters.specs, candidate.parameters.specs, strict=True)
                if p.scale != c.scale
            ]
            record["relative_chi_square_improvement"] = improvement
            calls.append(record)
            result = candidate
            stable = improvement <= RESTART_RELATIVE_OBJECTIVE_TOLERANCE
            if stable:
                break
        stages.append(
            dict(
                seconds=time.perf_counter() - stage_started,
                calls=calls,
                restart_stable=stable,
                parameters=[(p.key.label, p.value, p.scale) for p in result.parameters.specs],
                profile_sha256=hashlib.sha256(result.calculation.y.tobytes()).hexdigest(),
                stationarity_probe=stationarity_probe(request, options, result) if probe else None,
            )
        )
    assert result is not None
    scientific = finish(result, initial.pattern, sample)
    scientific["checks"]["all_calls_converged"] = all(
        c["termination"] == "converged" for s in stages for c in s["calls"]
    )
    scientific["checks"]["restart_stable"] = (
        all(s["restart_stable"] for s in stages) if config["restarts"] else None
    )
    return dict(
        seconds=time.perf_counter() - started,
        sample=sample,
        workers=workers,
        config=config,
        scientific=scientific,
        stages=stages,
    )


def invariant(record):
    record = json.loads(json.dumps(record))
    record.pop("seconds")
    record.pop("workers")
    for stage in record["stages"]:
        stage.pop("seconds")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--case", choices=CONFIGS, action="append")
    parser.add_argument("--sample", choices=TARGETS, action="append")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timing-note", default="")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Include feasible coordinate descent probes; not a timing benchmark",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0 or any(w < 1 for w in args.workers):
        parser.error("positive workers/repetitions and nonnegative warmups required")
    if ps._core.BUILD_MODE != "release":
        parser.error("release build required")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"launch with {key}=1")
    samples = args.sample or list(TARGETS)
    names = args.case or list(CONFIGS)
    combinations = [(s, w, n) for s in samples for w in args.workers for n in names]
    paths = [
        p
        for s in samples
        for p in verify_validation_dataset(
            f"iucr-qarr-{s}", ROOT / f"validation/data/iucr-qarr-{s}"
        )
    ]
    output = dict(
        schema="phasesmith.qarr-convergence.v1",
        warmups=args.warmups,
        repetitions=args.repetitions,
        timing_note=args.timing_note,
        stationarity_probes_enabled=args.probe,
        environment=dict(platform=platform.platform(), python=platform.python_version()),
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        native_sha256=hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preparation_driver_sha256=hashlib.sha256(
            (ROOT / "benchmarks/investigate_qarr_holdout.py").read_bytes()
        ).hexdigest(),
        dataset_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        limits=dict(max_iterations=1000, max_evaluations=20000, max_consecutive_rejections=500),
        restart_relative_objective_tolerance=RESTART_RELATIVE_OBJECTIVE_TOLERANCE,
        results={},
        notes=[
            "Original width model and stage selections; default full profiles with support 30.",
            "Native complete solver; no empirical convention or faster quadrature/tail policy.",
            "Includes preparation, all stages, final covariance and reporting; "
            "excludes imports/data verification.",
            "Fresh native restarts preserve physical states but rebuild numerical scales; "
            "scale changes are recorded.",
            "All unsuccessful stops remain visible; "
            "a converged label does not prove global optimality.",
            "This diagnostic does not change production defaults or the frozen validation gate.",
        ],
    )
    references = {}
    for repeat in range(-args.warmups, args.repetitions):
        for sample, workers, name in combinations if repeat % 2 else list(reversed(combinations)):
            record = run_case(sample, workers, name, args.probe)
            key = f"{sample}/{name}/{workers}"
            common = f"{sample}/{name}"
            value = invariant(record)
            if common in references and value != references[common]:
                raise RuntimeError(f"non-deterministic result: {key}")
            references[common] = value
            entry = output["results"].setdefault(key, dict(example=record, seconds=[]))
            if repeat >= 0:
                entry["seconds"].append(record["seconds"])
                entry["median_seconds"] = statistics.median(entry["seconds"])
            print(
                key,
                repeat,
                round(record["seconds"], 4),
                record["scientific"]["rwp"],
                record["scientific"]["max_fraction_error"],
                record["scientific"]["checks"],
                flush=True,
            )
            args.json_output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    output["exact_numerical_repeatability_across_runs_and_workers"] = True
    args.json_output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
