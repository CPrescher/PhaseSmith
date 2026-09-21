#!/usr/bin/env python3
"""Diagnose QARR Rwp differences; interventions do not change production defaults.

Requires the optional release PhaseSmith/rietx 1.4.0 benchmark environment.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1.
All timings are diagnostic only. The background and seed interventions alter
staging; they are not equal-workload performance comparisons.
"""

import hashlib
import json
import os
from pathlib import Path

import phasesmith as ps
from investigate_qarr_performance import run_case
from investigate_rietx_workload import diagnostic_rietx
from phasesmith.refinement.runtime import RefinementLimits
from phasesmith.validation import verify_validation_dataset
from rietx.model import compiled

ROOT = Path(__file__).resolve().parents[1]
root = ROOT / "validation/data/iucr-qarr-1g"
OUTPUT = ROOT / "validation/results/rietx-20260916-rwp-diagnostic.json"


def budget_cases():
    out = {
        "scope": "Corrected-physics fit-quality diagnostics; timings not a benchmark",
        "phasesmith": {},
        "rietx": {},
    }
    for policy, accuracy in [
        ("default", ps.ProfileAccuracy()),
        ("combined", ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)),
    ]:
        for extended in (False, True):
            overrides = {i: {"profile_accuracy": accuracy} for i in (1, 2, 3)}
            if extended:
                for i, count in [(1, 64), (2, 224)]:
                    overrides[i]["limits"] = RefinementLimits(
                        max_iterations=count, max_evaluations=10000
                    )
            record, captured = run_case(root, 8, overrides)
            for stage, (_, _options, result) in zip(record["stages"], captured, strict=True):
                stage["parameters"] = [
                    {
                        "key": p.key.label,
                        "value": p.value,
                        "scale": p.scale,
                        "lower": p.bounds.lower if abs(p.bounds.lower) != float("inf") else None,
                        "upper": p.bounds.upper if abs(p.bounds.upper) != float("inf") else None,
                    }
                    for p in result.parameters.specs
                ]
            key = policy + ("_extended" if extended else "_bounded")
            out["phasesmith"][key] = record
            print(
                key,
                [
                    (s["rwp"], s["termination"], s["iterations"], s["evaluations"])
                    for s in record["stages"]
                ],
                flush=True,
            )
    for name in ["default", "retain_small_axial", "30_fwhm_windows", "both"]:
        r = diagnostic_rietx(root, name)
        out["rietx"][name] = r
        print("rietx", name, r["scientific"]["poisson_rwp"], flush=True)
    OUTPUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


def width_audit():
    from unittest.mock import patch

    import numpy as np
    import rietx as rx
    from compare_rietx_qarr import run_rietx
    from rietx.model import forward

    fit = rx.Refinement.fit
    widths = forward.gaussian_fwhm
    stages = []

    def capture(self, *a, **kw):
        r = fit(self, *a, **kw)
        st = self.fitted_structure.model_copy(deep=True)
        ins = self.fitted_instrument.model_copy(deep=True)
        records = []

        def gaussian(theta, u, v, w, gauss_size=0.0, gauss_strain=0.0):
            t = np.tan(np.radians(theta))
            base = u * t * t + v * t + w
            total = base + gauss_strain * t * t + gauss_size / np.cos(np.radians(theta)) ** 2
            records.append(
                {
                    "rows": len(theta),
                    "gauss_strain": float(gauss_strain),
                    "base_min_deg2": float(base.min()),
                    "base_nonpositive_rows": int((base <= 0).sum()),
                    "total_min_deg2": float(total.min()),
                    "total_floored_rows": int((total < 1e-8).sum()),
                }
            )
            return widths(theta, u, v, w, gauss_size, gauss_strain)

        with patch.object(forward, "gaussian_fwhm", gaussian):
            rx.Refinement(st, ins, history=False).predict(np.asarray(a[0].two_theta))
        stages.append(
            {
                "profile": {k: getattr(ins.profile, k).value for k in ("u", "v", "w")},
                "background_coefficients": [p.value for p in ins.background.chebyshev.coefficients],
                "phase_strain": {p.name: p.gauss_strain.value for p in st.phases},
                "width_calls_during_fresh_prediction": records,
            }
        )
        return r

    with patch.object(rx.Refinement, "fit", capture):
        run_rietx(root)
    record = json.loads(OUTPUT.read_text())
    record["rietx_default_fitted_widths"] = stages
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    for i, s in enumerate(stages):
        print("stage", i + 1)
        for r in s["width_calls_during_fresh_prediction"]:
            print(r)


def background_cases():
    from dataclasses import replace
    from unittest.mock import patch

    from phasesmith.refinement import rietveld as rv

    original = rv.refine
    record = json.loads(OUTPUT.read_text())
    for name, accuracy in [
        ("default", ps.ProfileAccuracy()),
        ("combined", ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)),
    ]:
        calls = []

        def bgfree(request, options, _calls=calls, **kw):
            if len(_calls) == 1:
                selection = replace(request.selection, background=True)
                params = rv.build_parameter_set(
                    request.phases,
                    request.lattice_domains,
                    selection,
                    experiment=request.experiment,
                    background=request.background,
                )
                request = replace(request, selection=selection, parameters=params)
                options = replace(
                    options, limits=RefinementLimits(max_iterations=224, max_evaluations=10000)
                )
            _calls.append(True)
            return original(request, options, **kw)

        with patch.object(rv, "refine", bgfree):
            run, _captured = run_case(
                root, 8, {i: {"profile_accuracy": accuracy} for i in (1, 2, 3)}
            )
        record["phasesmith"][name + "_extended_free_background"] = run
        print(
            name, [(v["rwp"], v["termination"], v["iterations"]) for v in run["stages"]], flush=True
        )
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def seed_cases():
    from dataclasses import replace
    from unittest.mock import patch

    import numpy as np
    import rietx as rx
    from compare_rietx_qarr import run_rietx
    from phasesmith.refinement import rietveld as rv

    saved = []
    fit = rx.Refinement.fit

    def capture(self, *a, **kw):
        r = fit(self, *a, **kw)
        saved.append(
            (
                self.fitted_structure.model_copy(deep=True),
                self.fitted_instrument.model_copy(deep=True),
            )
        )
        return r

    with patch.object(rx.Refinement, "fit", capture):
        run_rietx(root)
    s, ins = saved[0]
    original = rv.refine
    record = json.loads(OUTPUT.read_text())
    for name, accuracy in [
        ("default", ps.ProfileAccuracy()),
        ("combined", ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)),
    ]:
        calls = []

        def seeded(request, options, _calls=calls, **kw):
            if len(_calls) == 1:
                instrument = replace(
                    request.experiment.instrument,
                    u_deg2=0.0,
                    v_deg2=ins.profile.v.value / (8 * np.log(2)),
                    w_deg2=ins.profile.w.value / (8 * np.log(2)),
                )
                exp = replace(
                    request.experiment, instrument=instrument, zero_shift_deg=ins.zero_shift.value
                )
                phases = tuple(
                    replace(
                        p,
                        scale=q.scale.value,
                        physics=replace(
                            p.physics,
                            providers=tuple(
                                replace(
                                    provider,
                                    rms_microstrain=np.sqrt(
                                        (q.gauss_strain.value + ins.profile.u.value)
                                        / (8 * np.log(2))
                                    )
                                    / (2 * 180 / np.pi),
                                )
                                if isinstance(provider, ps.IsotropicMicrostrainBroadening)
                                else provider
                                for provider in p.physics.providers
                            ),
                        ),
                    )
                    for p, q in zip(request.phases, s.phases, strict=True)
                )
                bg = request.background.replace_coefficients(
                    [p.value for p in ins.background.chebyshev.coefficients]
                )
                params = rv.build_parameter_set(
                    phases,
                    request.lattice_domains,
                    request.selection,
                    experiment=exp,
                    background=bg,
                )
                request = replace(
                    request, phases=phases, experiment=exp, background=bg, parameters=params
                )
            _calls.append(True)
            return original(request, options, **kw)

        with patch.object(rv, "refine", seeded):
            run, _captured = run_case(
                root, 8, {i: {"profile_accuracy": accuracy} for i in (1, 2, 3)}
            )
        key = name + "_rietx_stage1_reparameterized_seed"
        record["phasesmith"][key] = run
        print(key, [(v["rwp"], v["termination"]) for v in run["stages"]], flush=True)
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main():
    import rietx as rx

    if rx.__version__ != "1.4.0" or ps._core.BUILD_MODE != "release":
        raise RuntimeError("requires rietx 1.4.0 and release PhaseSmith")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"launch with {key}=1")
    os.environ["RIETX_COMPILED_THREADS"] = "1"
    compiled.warm(block=True)
    paths = verify_validation_dataset("iucr-qarr-1g", root)
    budget_cases()
    width_audit()
    background_cases()
    seed_cases()
    record = json.loads(OUTPUT.read_text())
    record["dataset_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    record["notes"] = [
        "Extended budgets are 64/224 accepted-iteration limits for stages one/two; "
        "other guards unchanged.",
        "Free-background cases retain the bounded first stage, then allow 224 "
        "iterations with three extra background parameters.",
        "Seed cases replace the start of stage two with rietx stage-one scales, "
        "background, widths and zero.",
        "Seed instrument U is set to zero and the same negative U is added to each "
        "sample Gaussian strain coefficient, preserving total Gaussian widths.",
        "Seed cases retain PhaseSmith stage-two and polish budgets. Their recorded "
        "first-stage result is the original unused PhaseSmith stage-one fit.",
        "Width calls include compilation and prediction, so each reflection line is "
        "recorded twice; do not sum them as unique rows.",
        "No production defaults or scientific acceptance criteria were changed. "
        "Inspect status and termination for each diagnostic.",
    ]
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
