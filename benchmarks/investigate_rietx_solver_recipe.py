#!/usr/bin/env python3
"""Test PhaseSmith-only QARR recipe hypotheses from the rietx source audit.

No production policies change. These are diagnostic fits, not timing claims.
All cases, including failed acceptance checks, remain in the output record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import phasesmith as ps
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "baseline": {},
    "partition": {"strain": 0.0002},
    "partition_bg": {"strain": 0.0002, "bg": True},
    "partition_bg_anchor": {"strain": 0.0002, "bg": True, "anchor": True},
    "partition_bg_anchor_long": {
        "strain": 0.0002,
        "bg": True,
        "anchor": True,
        "long": True,
    },
    "rejection_recovery": {"rejections": 200, "long": True},
    "damping": {"damping": 1.0, "long": True},
    "partition_bg_anchor_recovery": {
        "strain": 0.0002,
        "bg": True,
        "anchor": True,
        "long": True,
        "rejections": 200,
    },
    "partition_bg_anchor_recovery_default": {
        "strain": 0.0002,
        "bg": True,
        "anchor": True,
        "long": True,
        "rejections": 200,
        "default_accuracy": True,
    },
}


def update_request(request, **changes):
    """Keep the typed request's parameter values and selection consistent."""
    phases = changes.get("phases", request.phases)
    experiment = changes.get("experiment", request.experiment)
    selection = changes.get("selection", request.selection)
    parameters = rv.build_parameter_set(
        phases,
        request.lattice_domains,
        selection,
        experiment=experiment,
        background=request.background,
    )
    return replace(request, parameters=parameters, **changes)


def run_case(root, config, threads):
    original = rv.refine
    accuracy = (
        ps.ProfileAccuracy()
        if config.get("default_accuracy")
        else ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)
    )
    stages, deltas = [], []

    def refine(request, options, **kwargs):
        stage = len(stages) + 1
        options = replace(options, profile_accuracy=accuracy)
        if config.get("long") and stage < 3:
            options = replace(
                options,
                limits=replace(
                    options.limits,
                    max_iterations=100,
                    max_evaluations=3000,
                ),
            )
        if config.get("damping") and stage < 3:
            options = replace(options, initial_damping=config["damping"])
        if config.get("rejections") and stage < 3:
            options = replace(
                options,
                limits=replace(
                    options.limits,
                    max_consecutive_rejections=config["rejections"],
                ),
            )
        if stage == 1 and "strain" in config:

            def calculate(current):
                return rv.calculate(
                    current.pattern,
                    current.experiment,
                    current.phases,
                    background=current.background,
                    support_fwhm=30,
                    profile_accuracy=accuracy,
                    execution=options.execution,
                ).y

            before = calculate(request)
            strain = config["strain"]
            # QARR starts all phases at the same RMS strain. Transfer variance
            # into instrument U, preserving total Gaussian width at every angle.
            old_strains = [
                provider.rms_microstrain
                for phase in request.phases
                for provider in phase.physics.providers
                if isinstance(provider, ps.IsotropicMicrostrainBroadening)
            ]
            if len(old_strains) != len(request.phases) or len(set(old_strains)) != 1:
                raise RuntimeError("diagnostic requires one shared initial strain per phase")
            delta = (2 * 180 / np.pi) ** 2 * (old_strains[0] ** 2 - strain**2)
            experiment = replace(
                request.experiment,
                instrument=replace(
                    request.experiment.instrument,
                    u_deg2=request.experiment.instrument.u_deg2 + delta,
                ),
            )
            phases = tuple(
                replace(
                    phase,
                    physics=replace(
                        phase.physics,
                        providers=tuple(
                            replace(provider, rms_microstrain=strain)
                            if isinstance(provider, ps.IsotropicMicrostrainBroadening)
                            else provider
                            for provider in phase.physics.providers
                        ),
                    ),
                )
                for phase in request.phases
            )
            request = update_request(request, experiment=experiment, phases=phases)
            after = calculate(request)
            delta = float(np.linalg.norm(after - before) / np.linalg.norm(before))
            if delta > 1e-12:
                raise RuntimeError("initial profile changed under variance transfer")
            deltas.append(delta)
        if stage == 2:
            selection = request.selection
            if config.get("bg"):
                selection = replace(selection, background=True)
            if config.get("anchor"):
                selection = replace(
                    selection,
                    instrument_parameters=tuple(
                        key for key in selection.instrument_parameters if key != "u_deg2"
                    ),
                )
            request = update_request(request, selection=selection)
        start = time.perf_counter()
        result = original(request, options, **kwargs)
        stages.append(
            {
                "rwp": result.metrics.rwp,
                "termination": result.termination_reason.value,
                "iterations": len(result.history),
                "evaluations": result.evaluations,
                "backend": result.backend,
                "seconds": time.perf_counter() - start,
                "free_parameters": sum(p.refine for p in result.parameters.specs),
                "damping_history": [h.damping for h in result.history],
                "last_backtracks": [h.backtracks for h in result.history][-5:],
            }
        )
        return result

    with patch.object(rv, "refine", refine):
        result = run_qarr_1g_validation(root, execution=ps.ExecutionPolicy(threads=threads))
    scientific = result.to_record()
    scientific.pop("elapsed_seconds")
    return {
        "config": config,
        "stages": stages,
        "initial_profile_relative_l2": deltas,
        "scientific": scientific,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--case", choices=CONFIGS, action="append")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0 or ps._core.BUILD_MODE != "release":
        parser.error("requires positive threads and a release PhaseSmith build")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"launch with {key}=1")
    root = ROOT / "validation/data/iucr-qarr-1g"
    paths = verify_validation_dataset("iucr-qarr-1g", root)
    cases = {}
    for name in args.case or CONFIGS:
        result = run_case(root, CONFIGS[name], args.threads)
        cases[name] = result
        print(name, result["stages"][-1]["rwp"], result["scientific"]["status"], flush=True)
    record = {
        "scope": "PhaseSmith-only recipe diagnostics; not a performance comparison",
        "threads": args.threads,
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "cases": cases,
        "notes": [
            "No rietx fitted parameters or calculations are used by these candidates.",
            "Variance transfer preserves the initial total profile, but changes which widths "
            "stage one can reach with sample strain held fixed.",
            "The stage-two U anchor fixes an empirical decomposition, not measured resolution.",
            "Free-background anchored stage two has 21 parameters instead of the original 19.",
            "Long cases use 100 iterations and 3000 evaluations per first/second stage.",
            "The 200-rejection allowance is a diagnostic, not a proposed production default.",
            "Times include no repetitions or controlled warmup and are not benchmark results.",
            "Shared input preparation retains the validation helper's default two workers.",
        ],
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
