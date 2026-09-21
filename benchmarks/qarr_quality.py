#!/usr/bin/env python3
"""Compare native QARR time to Rwp targets, convergence and final QPA quality.

Default profiles, explicit ZnO reference-strain convention, and fixed numerical
budgets. Optional joint background refinement is a separately named recipe.
Native-only event tracing never switches to the Python solver. Target times
record first accepted crossings; they do not imply convergence or QPA quality
at that intermediate state. Final gates are reported independently.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import phasesmith as ps
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (0.20, 0.195, 0.193, 0.1925, 0.19, 0.189)


def run_case(threads, joint_background=False, trace=True):
    stages, crossings = [], {}
    original_refine, original_request, original_calculate = (
        rv.refine,
        rv._native_request,
        rv.calculate,
    )
    convention = ps.EmpiricalGaussianConvention("ZnO", 2e-4)
    started = time.perf_counter()

    def controlled_calculate(*args, **kwargs):
        kwargs.setdefault("execution", ps.ExecutionPolicy(threads=threads))
        return original_calculate(*args, **kwargs)

    def refine(request, options, **kwargs):
        request = (
            convention.apply(request)
            if not stages
            else replace(request, empirical_gaussian=convention)
        )
        if len(stages) == 1 and joint_background:
            selected = replace(request.selection, background=True)
            request = replace(
                request,
                selection=selected,
                parameters=rv.build_parameter_set(
                    request.phases,
                    request.lattice_domains,
                    selected,
                    experiment=request.experiment,
                    background=request.background,
                ),
            )
        options = replace(
            options,
            limits=replace(
                options.limits,
                max_iterations=100,
                max_evaluations=1500,
                max_consecutive_rejections=200,
            ),
        )
        native_records = []
        weighted_observed = request.pattern.observed_y.copy()
        if options.use_uncertainty and request.pattern.uncertainty is not None:
            weighted_observed /= request.pattern.uncertainty
        if request.pattern.mask is not None:
            weighted_observed = weighted_observed[request.pattern.mask]
        denominator = np.sum(weighted_observed**2)

        class TracedRequest:
            def __init__(self, native):
                self.native = native

            def refine(self, cancellation, checkpoint):
                offset = time.perf_counter() - started
                result = self.native.refine(cancellation, checkpoint, trace=trace)
                for (
                    kind,
                    attempt,
                    accepted,
                    evaluations,
                    seconds,
                    objective,
                    message,
                ) in result.trace_records():
                    row = dict(
                        kind=kind,
                        attempt=attempt,
                        accepted=accepted,
                        evaluations=evaluations,
                        seconds=seconds,
                        objective=objective,
                        message=message,
                    )
                    native_records.append(row)
                    if kind == "step_accepted" and objective is not None:
                        rwp = float(np.sqrt(2 * objective / denominator))
                        for target in TARGETS:
                            if rwp <= target and str(target) not in crossings:
                                crossings[str(target)] = dict(
                                    seconds=offset + seconds,
                                    stage=len(stages) + 1,
                                    evaluations=evaluations,
                                    rwp=rwp,
                                )
                return result

        before = time.perf_counter()
        with patch.object(
            rv, "_native_request", lambda *a, **kw: TracedRequest(original_request(*a, **kw))
        ):
            result = original_refine(request, options, **kwargs)
        if result.backend != "native":
            raise RuntimeError("quality benchmark unexpectedly left the native solver")
        if trace:
            traced_rwps = [
                np.sqrt(2 * event["objective"] / denominator)
                for event in native_records
                if event["kind"] == "step_accepted"
            ]
            np.testing.assert_allclose(
                traced_rwps, [h.rwp for h in result.history], rtol=1e-14, atol=0
            )
        stages.append(
            dict(
                seconds=time.perf_counter() - before,
                rwp=result.metrics.rwp,
                profile_sha256=hashlib.sha256(result.calculation.y.tobytes()).hexdigest(),
                parameters=[(p.key.label, p.value) for p in result.parameters.specs],
                covariance_sha256=(
                    None
                    if result.covariance is None
                    else hashlib.sha256(result.covariance.tobytes()).hexdigest()
                ),
                evaluations=result.evaluations,
                termination=result.termination_reason.value,
                history=[asdict(h) for h in result.history],
                trace=native_records,
            )
        )
        return result

    with patch.object(rv, "refine", refine), patch.object(rv, "calculate", controlled_calculate):
        report = run_qarr_1g_validation(
            ROOT / "validation/data/iucr-qarr-1g", execution=ps.ExecutionPolicy(threads=threads)
        )
    elapsed = time.perf_counter() - started
    scientific = report.to_record()
    scientific.pop("elapsed_seconds")
    all_converged = all(s["termination"] == "converged" for s in stages)
    return dict(
        provenance=dict(
            package=ps.__file__,
            native_sha256=hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
            versions={name: importlib.metadata.version(name) for name in ("phasesmith", "numpy")},
        ),
        seconds=elapsed,
        scientific=scientific,
        stages=stages,
        all_stages_converged=all_converged,
        quality_passed=scientific["status"] == "passed" and all_converged,
        time_to_targets={str(t): crossings.get(str(t)) for t in TARGETS},
    )


def invariant(record):
    record = json.loads(json.dumps(record))
    record.pop("seconds")
    record.pop("time_to_targets")
    for stage in record["stages"]:
        stage.pop("seconds")
        for event in stage["trace"]:
            event.pop("seconds")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--joint-background", action="store_true")
    parser.add_argument("--timing-note", default="", help="Record known machine-load caveats")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        for line in sys.stdin:
            request = json.loads(line)
            record = run_case(**request)
            print(json.dumps(record, allow_nan=False), flush=True)
        return
    if args.threads < 1 or args.repetitions < 1 or not args.json_output:
        parser.error("positive threads/repetitions and --json-output required")
    if ps._core.BUILD_MODE != "release":
        parser.error("release build required")
    paths = verify_validation_dataset("iucr-qarr-1g", ROOT / "validation/data/iucr-qarr-1g")
    children, results = {}, {}
    variants = {"current": None}
    if args.baseline_package:
        variants = {"baseline": args.baseline_package.resolve(), **variants}
    config = dict(threads=args.threads, joint_background=args.joint_background)
    try:
        for name, package in variants.items():
            env = dict(os.environ)
            env.update(
                {
                    key: "1"
                    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
                }
            )
            if package is not None:
                env["PYTHONPATH"] = str(package)
            else:
                env.pop("PYTHONPATH", None)
            children[name] = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            results[name] = []
        for repeat in range(-1, args.repetitions):
            names = list(variants)
            for name in names if repeat % 2 else reversed(names):
                child = children[name]
                child.stdin.write(json.dumps(config) + "\n")
                child.stdin.flush()
                line = child.stdout.readline()
                if not line:
                    raise RuntimeError(f"{name} worker exited")
                record = json.loads(line)
                if repeat == -1:
                    results[name].append(record)
                else:
                    if invariant(record) != invariant(results[name][0]):
                        raise RuntimeError(f"{name} is not deterministic")
                    results[name].append(record)
                print(
                    name,
                    repeat,
                    record["seconds"],
                    record["stages"][-1]["rwp"],
                    record["quality_passed"],
                    flush=True,
                )
        output = dict(
            schema="phasesmith.qarr-quality.v1",
            config=config,
            targets=TARGETS,
            warmups=1,
            repetitions=args.repetitions,
            dataset_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            environment=dict(platform=platform.platform(), python=platform.python_version()),
            timing_note=args.timing_note,
            results={},
        )
        for name, runs in results.items():
            samples = runs[1:]
            output["results"][name] = dict(
                trace_example=runs[0],
                seconds=[r["seconds"] for r in samples],
                median_seconds=statistics.median(r["seconds"] for r in samples),
                median_time_to_targets={
                    str(t): (
                        None
                        if samples[0]["time_to_targets"][str(t)] is None
                        else statistics.median(
                            r["time_to_targets"][str(t)]["seconds"] for r in samples
                        )
                    )
                    for t in TARGETS
                },
            )
        if "baseline" in results:
            before, after = results["baseline"][0], results["current"][0]
            output["comparison"] = dict(
                both_pass_quality=before["quality_passed"] and after["quality_passed"],
                scientific_checks_equal=before["scientific"]["checks"]
                == after["scientific"]["checks"],
                final_profile_bitwise_equal=before["stages"][-1]["profile_sha256"]
                == after["stages"][-1]["profile_sha256"],
                final_parameter_records_equal=before["stages"][-1]["parameters"]
                == after["stages"][-1]["parameters"],
                final_covariance_bitwise_equal=before["stages"][-1]["covariance_sha256"]
                == after["stages"][-1]["covariance_sha256"],
            )
        output["notes"] = [
            "Same dataset, explicit convention, profile policy, parameter roles, "
            "starting state and budgets within each build comparison.",
            "Native event traces are optional diagnostics; no Python callback changes "
            "solver dispatch.",
            "Target times approximate full-workflow elapsed time at first accepted "
            "objective crossing; include preparation and earlier stages.",
            "An intermediate target crossing does not assess intermediate QPA or "
            "establish convergence.",
            "Final quality requires every stage converged and all existing real-data gates passed.",
            "No failed target or quality gate is removed from the record.",
        ]
        args.json_output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    finally:
        for child in children.values():
            child.stdin.close()
            child.wait()


if __name__ == "__main__":
    main()
