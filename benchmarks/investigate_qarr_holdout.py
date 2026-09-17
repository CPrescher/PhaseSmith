#!/usr/bin/env python3
"""Fixed-protocol QARR 1g/1h accuracy diagnostics, without changing validation gates.

The weighed compositions are used only for final evaluation. These exploratory
fits do not reclassify the existing holdout or establish an unbiased new gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import phasesmith as ps
from phasesmith.refinement import RefinementLimits
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import real_data as rd
from phasesmith.validation import verify_validation_dataset

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "1g": {"Al2O3": 0.3137, "ZnO": 0.3421, "CaF2": 0.3442},
    "1h": {"Al2O3": 0.3512, "ZnO": 0.3019, "CaF2": 0.3469},
}
CONFIGS = {
    "bounded": dict(iterations=(8, 35, 10), rejections=20, empirical=False, joint=False),
    "extended": dict(iterations=(100, 100, 100), rejections=200, empirical=False, joint=False),
    "empirical": dict(iterations=(100, 100, 100), rejections=200, empirical=True, joint=False),
    "empirical_joint": dict(iterations=(100, 100, 100), rejections=200, empirical=True, joint=True),
}
CONFIGS["extended_500"] = {**CONFIGS["extended"], "iterations": (500, 500, 100)}
CONFIGS["empirical_tight"] = {
    **CONFIGS["empirical"],
    "iterations": (500, 500, 100),
    "objective_tolerance": 1e-12,
    "parameter_tolerance": 1e-9,
}
CONFIGS["empirical_reference_low"] = {**CONFIGS["empirical"], "reference_rms": 1e-4}
CONFIGS["empirical_reference_high"] = {**CONFIGS["empirical"], "reference_rms": 3e-4}
for _config in CONFIGS.values():
    _config.setdefault("objective_tolerance", 1e-10)
    _config.setdefault("parameter_tolerance", 1e-7)


def starting_request(root, sample, execution):
    """Use the existing QARR physical model and observed-data-only scale estimate."""
    data = rd.read_powder_data(root / f"cpd-{sample}.prn", format="columns")
    background = rd.SmoothBrucknerBackground(
        smooth_width=1.0, iterations=50, chebyshev_order=None
    ).estimate(data.x, data.observed_y)
    pattern = ps.PowderPattern(
        data.x,
        observed_y=data.observed_y,
        uncertainty=np.sqrt(np.maximum(data.observed_y, 1.0)),
        background=background,
    )
    v = rd._qarr_instrument_values(root / "cuka.instprm")
    instrument = ps.ConstantWavelengthInstrument(
        v["Lam1"],
        v["U"] * 1e-4,
        v["V"] * 1e-4,
        v["W"] * 1e-4,
        v["X"] * 1e-2,
        v["Y"] * 1e-2,
    )
    experiment = ps.ConstantWavelengthExperiment.x_ray_components(
        instrument,
        ps.WavelengthComponents.doublet(v["Lam1"], v["Lam2"], v["I(L2)/I(L1)"]),
        axial_geometry=ps.FcjGeometry(v["SH/L"] / 2, v["SH/L"] / 2),
    )
    fixed = rv.RietveldParameterSelection(
        phase_scale=False, lattice=False, coordinates=False, occupancy=False, u_iso=False
    )
    phases = []
    for name in TARGETS[sample]:
        loaded = rv.RietveldInput.from_cif(
            pattern,
            experiment,
            root / f"{name}.cif",
            phase_id=name,
            selection=fixed,
            scattering=ps.XrayFixedDispersion(rd.QARR_1G_CUKA_FIXED_DISPERSION),
            intensity_correction=ps.BraggBrentanoPolarizedLp(v["Lam1"], v["Polariz."]),
            coordinate_tolerance=rd._QARR_COORDINATE_TOLERANCE,
        )
        structure = rd._qarr_displacement_defaults(loaded.phases[0].structure)
        phases.append(
            replace(
                loaded.phases[0], structure=structure, physics=rd._qarr_physics(name, structure)
            )
        )
    calculation = rv.calculate(
        pattern, experiment, tuple(phases), support_fwhm=30, execution=execution
    )
    design = np.column_stack([p.profile_y for p in calculation.phase_calculations])
    scales = np.linalg.lstsq(
        design / pattern.uncertainty[:, None],
        (pattern.observed_y - background) / pattern.uncertainty,
        rcond=None,
    )[0]
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("nonpositive initial scales")
    phases = tuple(replace(p, scale=float(s)) for p, s in zip(phases, scales, strict=True))
    background_model = rv.ChebyshevBackground(
        "qarr_residual", (0.0, 0.0, 0.0), (float(data.x[0]), float(data.x[-1]))
    )
    selected = replace(
        fixed,
        phase_scale=True,
        background=True,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2", "zero_shift_deg"),
    )
    domains = (None,) * len(phases)
    return rv.RietveldInput(
        pattern,
        experiment,
        phases,
        domains,
        rv.build_parameter_set(
            phases, domains, selected, experiment=experiment, background=background_model
        ),
        selection=selected,
        background=background_model,
    )


def request_after(result, pattern, selected, convention):
    domains = (None,) * len(result.phases)
    return rv.RietveldInput(
        pattern,
        result.experiment,
        result.phases,
        domains,
        rv.build_parameter_set(
            result.phases,
            domains,
            selected,
            experiment=result.experiment,
            background=result.background,
        ),
        selection=selected,
        background=result.background,
        empirical_gaussian=convention,
    )


def run_case(initial, sample, config, execution):
    convention = (
        ps.EmpiricalGaussianConvention("ZnO", config.get("reference_rms", 2e-4))
        if config["empirical"]
        else None
    )
    request = convention.apply(initial) if convention else initial
    initial_profile_relative_l2 = 0.0
    if convention:

        def profile(q):
            return rv.calculate(
                q.pattern,
                q.experiment,
                q.phases,
                background=q.background,
                support_fwhm=30,
                execution=execution,
            ).y

        before, after = profile(initial), profile(request)
        initial_profile_relative_l2 = float(np.linalg.norm(after - before) / np.linalg.norm(before))
        if initial_profile_relative_l2 > 1e-12:
            raise RuntimeError("empirical convention changed the initial total profile")
    first = initial.selection
    second = replace(first, u_iso=True, sample_physics=True, background=config["joint"])
    polish = rv.RietveldParameterSelection(
        phase_scale=True, lattice=False, coordinates=False, occupancy=False, u_iso=False
    )
    stages = []
    result = None
    for i, selected in enumerate((first, second, polish)):
        if i:
            assert result is not None
            request = request_after(result, initial.pattern, selected, convention)
        options = rv.RietveldOptions(
            limits=RefinementLimits(
                max_iterations=config["iterations"][i],
                max_evaluations=3000,
                max_consecutive_rejections=config["rejections"],
            ),
            min_iterations=(2, 3, 1)[i],
            max_scaled_parameter_step=(0.2, 0.15, 1.0)[i],
            support_fwhm=30,
            objective_tolerance=config["objective_tolerance"],
            parameter_tolerance=config["parameter_tolerance"],
            estimate_covariance=True,
            execution=execution,
        )
        result = rv.refine(request, options)
        if result.backend != "native":
            raise RuntimeError("diagnostic unexpectedly left native solver")
        stages.append(
            dict(
                rwp=result.metrics.rwp,
                termination=result.termination_reason.value,
                iterations=len(result.history),
                evaluations=result.evaluations,
                rwp_history=[h.rwp for h in result.history],
                parameters=[
                    dict(
                        label=p.key.label,
                        value=p.value,
                        lower=p.bounds.lower if np.isfinite(p.bounds.lower) else None,
                        upper=p.bounds.upper if np.isfinite(p.bounds.upper) else None,
                    )
                    for p in result.parameters.specs
                ],
                jacobian_rank=result.jacobian_rank,
                covariance_available=result.covariance is not None,
            )
        )
    assert result is not None
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
    propagated = (
        ps.quantitative_phase_analysis_with_covariance(quantitative, result.covariance)
        if result.covariance is not None
        else None
    )
    max_fraction_su = (
        float(np.sqrt(np.max(np.diag(propagated.covariance))))
        if propagated is not None and np.all(np.diag(propagated.covariance) >= 0)
        else None
    )
    residual = result.calculation.y - initial.pattern.observed_y
    unit_rwp = float(np.linalg.norm(residual) / np.linalg.norm(initial.pattern.observed_y))
    correlation = float(
        np.corrcoef(
            initial.pattern.observed_y - result.calculation.background, result.calculation.profile_y
        )[0, 1]
    )
    checks = dict(
        poisson_rwp=result.metrics.rwp <= 0.20,
        unit_weight_rwp=unit_rwp <= 0.15,
        profile_correlation=correlation >= 0.98,
        qpa_weight_fraction=error <= 0.02,
        conditional_qpa_covariance=max_fraction_su is not None
        and bool(np.isfinite(max_fraction_su)),
        all_stages_converged=all(s["termination"] == "converged" for s in stages),
    )
    diagnostics = ps.diagnose_residuals(
        initial.pattern.x,
        residual,
        weighted_residual=result.metrics.weighted_residual,
        region_count=29,
    )
    return dict(
        config=config,
        initial_profile_relative_l2=initial_profile_relative_l2,
        stages=stages,
        rwp=result.metrics.rwp,
        unit_rwp=unit_rwp,
        correlation=correlation,
        fractions=fractions,
        max_fraction_error=error,
        maximum_conditional_fraction_su=max_fraction_su,
        checks=checks,
        profile_sha256=hashlib.sha256(result.calculation.y.tobytes()).hexdigest(),
        worst_residual_regions=[asdict(r) for r in diagnostics.worst_regions(5)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--case", choices=CONFIGS, action="append")
    parser.add_argument("--sample", choices=TARGETS, action="append")
    parser.add_argument("--json-output", required=True, type=Path)
    args = parser.parse_args()
    if args.threads < 1 or ps._core.BUILD_MODE != "release":
        parser.error("positive thread count and release build required")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"launch with {key}=1")
    execution = ps.ExecutionPolicy(threads=args.threads)
    output = dict(
        schema="phasesmith.qarr-holdout-diagnostic.v1",
        threads=args.threads,
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        native_sha256=hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_sha256={},
        results={},
        notes=[
            "Default full profiles, fixed physical bounds and convergence tolerances.",
            "Same diagnostic recipes on both samples; weighed fractions only score final fits.",
            "Python initial scale least squares can differ slightly from native preparation.",
            "Exploration does not change the frozen validation recipe or its holdout status.",
            "No timings are interpreted as benchmark measurements.",
        ],
    )
    for sample in args.sample or TARGETS:
        root = ROOT / f"validation/data/iucr-qarr-{sample}"
        paths = verify_validation_dataset(f"iucr-qarr-{sample}", root)
        output["dataset_sha256"][sample] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        }
        initial = starting_request(root, sample, execution)
        output["results"][sample] = {}
        for name in args.case or CONFIGS:
            result = run_case(initial, sample, CONFIGS[name], execution)
            output["results"][sample][name] = result
            print(
                sample,
                name,
                result["rwp"],
                result["max_fraction_error"],
                result["checks"],
                flush=True,
            )
            args.json_output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
