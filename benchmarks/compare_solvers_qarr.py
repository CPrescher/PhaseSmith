#!/usr/bin/env python3
"""Isolate native versus SciPy TRF optimization on PhaseSmith's QARR model.

Experimental adapter only. SciPy consumes PhaseSmith's scaled physical variables,
residuals and Rust analytical profile derivatives. No rietx model is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import phasesmith as ps
import scipy
from investigate_qarr_performance import run_case
from phasesmith.refinement import rietveld as rv
from phasesmith.refinement.core import ConstraintTransform
from phasesmith.refinement.runtime import RefinementLimits, RefinementRuntime
from phasesmith.validation import verify_validation_dataset
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


class Objective:
    """Existing PhaseSmith preparation/mapping with a one-point SciPy cache."""

    def __init__(self, request, options):
        if request.constraints or any(d is not None for d in request.lattice_domains):
            raise ValueError("this diagnostic supports the fixed-cell unconstrained QARR case")
        self.request, self.options = request, options
        self.transform = ConstraintTransform(request.parameters, request.constraints)
        self.x0 = self.transform.pack()
        specs = [request.parameters.spec(k) for k in self.transform.free_keys]
        self.bounds = tuple(
            np.asarray(v)
            for v in zip(
                *[(s.bounds.lower / s.scale, s.bounds.upper / s.scale) for s in specs], strict=True
            )
        )
        self.cache = rv._RietveldPreparationCache.build(
            request,
            request.background,
            request.phases,
            request.lattice_domains,
            request.parameters,
            options,
        )
        self.runtime = RefinementRuntime(
            RefinementLimits(
                max_iterations=1_000_000,
                max_evaluations=1_000_000,
            )
        )
        self.last_x = None
        self.last = None
        self.invalid = Counter()
        self.trace = []
        self.denominator = float(
            np.linalg.norm(request.pattern.observed_y * self.cache.sample_weight)
        )

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.last
        q = self.request
        values = self.transform.unpack(x)
        experiment, background = rv._apply_profile_background_values(
            q.experiment,
            q.background,
            values,
        )
        phases, changes = rv._apply_parameter_values(
            q.phases,
            q.lattice_domains,
            q.parameters,
            values,
            coordinate_models=self.cache.coordinate_models,
        )
        if changes:
            raise RuntimeError("fixed-cell experiment unexpectedly changed topology")
        parameters = q.parameters.replace_values(values)
        try:
            linearization = rv._RietveldLinearization.prepare(
                q,
                experiment,
                background,
                phases,
                q.lattice_domains,
                parameters,
                self.options,
                self.runtime,
                preparation_cache=self.cache,
            )
            calculation = linearization.calculate()
        except ValueError as error:
            # Preserve PhaseSmith's numerical domain. SciPy TRF rejects a
            # nonfinite trial residual and shrinks its radius, without using a
            # Jacobian there. Do not replace invalid shapes with clipped ones.
            message = str(error)
            if not any(
                term in message
                for term in (
                    "Gaussian variance",
                    "Lorentzian",
                    "gaussian variance",
                )
            ):
                raise
            self.invalid[message] += 1
            return None
        if linearization.weighted_free_jacobian is None:
            raise RuntimeError("expected dense analytical PhaseSmith Jacobian")
        residual = (calculation.y - q.pattern.observed_y) * self.cache.sample_weight
        jacobian = np.ascontiguousarray(linearization.weighted_free_jacobian.T)
        self.last_x = np.array(x, copy=True)
        self.last = (residual, jacobian, experiment, background, phases, parameters, calculation)
        self.trace.append(
            {
                "evaluation": len(self.trace) + sum(self.invalid.values()) + 1,
                "rwp": float(np.linalg.norm(residual) / self.denominator),
            }
        )
        return self.last

    def residual(self, x):
        result = self.evaluate(x)
        return np.full(self.request.pattern.x.size, np.inf) if result is None else result[0]

    def jacobian(self, x):
        result = self.evaluate(x)
        if result is None:
            raise RuntimeError("SciPy requested a Jacobian at an invalid trial")
        return result[1]

    def summary(self, x):
        residual, jacobian, *_ = self.evaluate(x)
        gradient = jacobian.T @ residual
        singular = np.linalg.svd(jacobian, compute_uv=False)
        return {
            "rwp": float(np.linalg.norm(residual) / self.denominator),
            "weighted_sse": float(residual @ residual),
            "scaled_gradient_inf": float(np.linalg.norm(gradient, ord=np.inf)),
            "jacobian_singular_values": singular.tolist(),
            "scaled_values": x.tolist(),
        }

    def check_derivatives(self):
        initial = self.evaluate(self.x0)
        analytical = initial[1].copy()
        errors = []
        for column in range(len(self.x0)):
            # Local differences, no global tolerance relaxation for support jumps.
            h = 1e-7
            plus, minus = self.x0.copy(), self.x0.copy()
            plus[column] += h
            minus[column] -= h
            fp, fm = self.evaluate(plus), self.evaluate(minus)
            if fp is None or fm is None:
                raise RuntimeError("initial point not interior for finite-difference check")
            fd = (fp[0] - fm[0]) / (2 * h)
            relative = float(
                np.linalg.norm(fd - analytical[:, column])
                / max(np.linalg.norm(fd), np.linalg.norm(analytical[:, column]), 1e-12)
            )
            errors.append(relative)
        return errors


def compare_stage(request, options, baseline, budget):
    # Same parameter units/scales, masks, weights, profile policy and domain.
    options = replace(
        options,
        estimate_covariance=False,
        limits=RefinementLimits(
            max_iterations=budget,
            max_evaluations=budget,
            max_consecutive_rejections=budget,
        ),
    )
    checks = Objective(request, options)
    initial = checks.summary(checks.x0)
    fd_errors = checks.check_derivatives()
    if max(fd_errors) > 1e-5:
        raise RuntimeError("analytical Jacobian failed the local finite-difference gate")
    start = time.perf_counter()
    native = rv.refine(request, options)
    native_seconds = time.perf_counter() - start
    if set(native.parameters.keys) != set(request.parameters.keys):
        raise RuntimeError("native and SciPy parameter selections differ")
    for spec in request.parameters.specs:
        actual = native.parameters.spec(spec.key)
        if (spec.scale, spec.bounds, spec.refine) != (actual.scale, actual.bounds, actual.refine):
            raise RuntimeError("native and SciPy parameter scales or bounds differ")
    native_x = checks.transform.pack(native.parameters.values())
    native_summary = checks.summary(native_x)
    native_rwp_delta = abs(native_summary["rwp"] - native.metrics.rwp)
    profile_delta = float(
        np.linalg.norm(checks.last[-1].y - native.calculation.y)
        / np.linalg.norm(native.calculation.y)
    )
    if native_rwp_delta > 1e-10 or profile_delta > 1e-10:
        raise RuntimeError("SciPy adapter does not reproduce native final objective")
    objective = Objective(request, options)
    start = time.perf_counter()
    result = least_squares(
        objective.residual,
        objective.x0,
        jac=objective.jacobian,
        bounds=objective.bounds,
        method="trf",
        x_scale=1.0,
        ftol=options.objective_tolerance,
        xtol=options.parameter_tolerance,
        gtol=None,
        max_nfev=budget,
    )
    scipy_seconds = time.perf_counter() - start
    scipy_summary = objective.summary(result.x)
    return {
        "initial": initial,
        "parameter_keys": [k.label for k in checks.transform.free_keys],
        "scaled_bounds": [
            [None if not np.isfinite(v) else float(v) for v in row] for row in checks.bounds
        ],
        "initial_fd_column_relative_errors": fd_errors,
        "adapter_native_final_rwp_difference": native_rwp_delta,
        "adapter_native_final_profile_relative_l2": profile_delta,
        "original_recipe_native_rwp": baseline.metrics.rwp,
        "native": {
            **native_summary,
            "termination": native.termination_reason.value,
            "evaluations": native.evaluations,
            "accepted_iterations": len(native.history),
            "seconds": native_seconds,
            "accepted_rwp": [h.rwp for h in native.history],
        },
        "scipy_trf": {
            **scipy_summary,
            "status": result.status,
            "message": result.message,
            "nfev": result.nfev,
            "njev": result.njev,
            "seconds": scipy_seconds,
            "invalid_trials": dict(objective.invalid),
            "rwp_trace": objective.trace,
        },
    }


def scipy_control(request, options, budget, **overrides):
    objective = Objective(request, options)
    settings = {
        "x_scale": 1.0,
        "ftol": options.objective_tolerance,
        "xtol": options.parameter_tolerance,
        "gtol": None,
    }
    settings.update(overrides)
    result = least_squares(
        objective.residual,
        objective.x0,
        jac=objective.jacobian,
        bounds=objective.bounds,
        method="trf",
        max_nfev=budget,
        **settings,
    )
    return {
        **objective.summary(result.x),
        "nfev": result.nfev,
        "njev": result.njev,
        "termination": result.message,
        "invalid_trials": dict(objective.invalid),
        "settings": settings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--suite", choices=("primary", "controls", "scaling"), default="primary")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 2 or ps._core.BUILD_MODE != "release":
        parser.error("requires release build and budget >= 2")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"launch with {key}=1")
    root = ROOT / "validation/data/iucr-qarr-1g"
    paths = verify_validation_dataset("iucr-qarr-1g", root)
    cases = {}
    for policy, accuracy in (
        ("default", ps.ProfileAccuracy()),
        ("combined", ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)),
    ):
        _, captured = run_case(root, 1, {i: {"profile_accuracy": accuracy} for i in (1, 2, 3)})
        for stage, (request, options, baseline) in enumerate(captured, 1):
            if args.suite != "primary":
                if stage == 3:
                    continue
                if args.suite == "controls":
                    key = f"{policy}_stage{stage}_tight_trf"
                    cases[key] = scipy_control(
                        request,
                        options,
                        args.budget,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12,
                    )
                    if stage == 2:
                        selection = replace(request.selection, instrument_parameters=())
                        parameters = rv.build_parameter_set(
                            request.phases,
                            request.lattice_domains,
                            selection,
                            experiment=request.experiment,
                            background=request.background,
                        )
                        fixed = replace(request, selection=selection, parameters=parameters)
                        cases[f"{policy}_stage2_fixed_instrument"] = compare_stage(
                            fixed,
                            options,
                            baseline,
                            args.budget,
                        )
                else:
                    for name, settings in (
                        ("jac_scaling", {"x_scale": "jac"}),
                        ("lsmr", {"tr_solver": "lsmr"}),
                    ):
                        key = f"{policy}_stage{stage}_{name}"
                        cases[key] = scipy_control(request, options, args.budget, **settings)
                print(policy, stage, args.suite, "complete", flush=True)
                continue
            key = f"{policy}_stage{stage}"
            record = compare_stage(request, options, baseline, args.budget)
            cases[key] = record
            print(
                key,
                record["native"]["rwp"],
                record["scipy_trf"]["rwp"],
                "FD",
                max(record["initial_fd_column_relative_errors"]),
                flush=True,
            )
    record = {
        "scope": "isolated fixed-start QARR stages, same PhaseSmith physical objective",
        "suite": args.suite,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "native_extension_sha256": hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
        "budget": args.budget,
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "cases": cases,
        "notes": [
            "Each stage starts from the original native recipe's captured input for that policy.",
            "No rietx calculations, transformed variables, changed backgrounds or seeds are used.",
            "Numerical-domain violations return infinite trial residuals to SciPy, which rejects "
            "them; the valid objective and Jacobian remain unchanged.",
            "Both use PhaseSmith's scaled physical coordinates and physical parameter bounds.",
            "Common evaluation ceilings and ftol/xtol numbers do not make solver-specific "
            "evaluation accounting or stopping formulas identical. SciPy gtol is disabled.",
            "Adapter builds derivatives for every valid SciPy trial; times are diagnostic only, "
            "not an optimized native-versus-SciPy speed comparison.",
            "Controls retain the primary physical problem except the explicitly labelled "
            "fixed-instrument stage, which removes the same four variables from both solvers.",
            "Scaling suite changes only SciPy internal linear algebra or automatic scaling; "
            "it is a sensitivity experiment, not the fixed-scaling primary comparison.",
        ],
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
