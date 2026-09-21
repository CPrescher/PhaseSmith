#!/usr/bin/env python3
"""Compare complete public CW fits on a controlled synthetic Gaussian pattern.

One P1 phase, eight Si sites, 423 reflections, 20,001 samples, four free
parameters: scale, Gaussian W and two Chebyshev coefficients. Crystal geometry
and atomic parameters are held. This does not benchmark general structural
refinement, FCJ, multiphase fitting or real-data robustness.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import phasesmith as ps
import rietx as rx
from compare_xrd_rust import measure_interleaved
from phasesmith.refinement import rietveld as rv
from rietx.schemas.instrument import BackgroundChebyshev


def run(repetitions: int, threads: int) -> dict:
    from rietx._about import COMPILED_THREADS_ENV
    from rietx.model import compiled

    os.environ[COMPILED_THREADS_ENV] = str(threads)
    compiled.warm(block=True)
    if not compiled.enabled() or ps._core.BUILD_MODE != "release":
        raise RuntimeError("both compiled kernels and a release PhaseSmith build are required")
    if rx.__version__ != "1.4.0":
        raise RuntimeError("the adapter is pinned to rietx 1.4.0")
    x = np.linspace(12.0, 120.0, 20_001)
    coordinates = [
        ((0.071 * j + 0.11) % 1, (0.137 * j + 0.23) % 1, (0.193 * j + 0.31) % 1) for j in range(8)
    ]
    p = rx.Parameter
    rs = rx.Structure(
        phases=[
            rx.Phase(
                name="alpha",
                space_group="P 1",
                cell=rx.Cell.cubic(5.2),
                atoms=[
                    rx.Atom(
                        label=f"Si{j}",
                        species="Si",
                        x=p(value=a),
                        y=p(value=b),
                        z=p(value=c),
                        biso=p(value=8 * np.pi**2 * 0.01),
                    )
                    for j, (a, b, c) in enumerate(coordinates)
                ],
            )
        ]
    )
    ri = rx.Instrument.debye_scherrer(1.5406, polarization=0.5)
    ri.geometry.goniometer_radius_mm = 217.5
    ri.geometry.kind = "bragg_brentano"
    ri.geometry.axial_sl.value = ri.geometry.axial_hl.value = 0
    ri.source.dispersion = None
    for key, value in {
        "u": 0.002 * 8 * np.log(2),
        "v": 0,
        "w": 0.003 * 8 * np.log(2),
        "x": 0,
        "y": 0,
    }.items():
        getattr(ri.profile, key).value = value
    ri.background = BackgroundChebyshev(coefficients=[p(value=0)])
    ry = rx.Refinement(rs, ri, history=False).predict(x)
    structure = ps.CrystalStructure(
        "alpha",
        "alpha",
        ps.UnitCell(5.2, 5.2, 5.2, 90, 90, 90),
        ps.SpaceGroup.p1(),
        tuple(
            ps.AtomSite(f"Si{j}", f"Si{j}", "Si", "Si", xyz, 1.0, 0.01)
            for j, xyz in enumerate(coordinates)
        ),
    )
    instrument = ps.ConstantWavelengthInstrument(1.5406, 0.002, 0, 0.003, 0, 0)
    experiment = ps.ConstantWavelengthExperiment.x_ray(instrument)
    generated = ps.PreparedReflectionGenerator(structure.space_group).generate(
        structure.cell, ps.CwTwoThetaRange(12, 120, 1.5406)
    )
    phase = ps.RietveldPhase(
        "alpha",
        "alpha",
        structure,
        ps.StructuralReflectionBatch.from_generated(generated),
        ps.XrayNonResonant(),
        ps.BraggBrentanoUnpolarizedLp(1.5406),
    )
    py = rv.calculate(ps.PowderPattern(x), experiment, (phase,)).y
    profile_relative_l2 = float(np.linalg.norm(py - ry) / np.linalg.norm(py))
    if profile_relative_l2 >= 1e-7:
        raise RuntimeError(f"forward model agreement gate failed: {profile_relative_l2}")
    observed = py + 500 + 100 * (2 * (x - 12) / 108 - 1)
    sigma = np.sqrt(observed)
    data = rx.PatternData(two_theta=x.tolist(), intensity=observed.tolist(), sigma=sigma.tolist())
    rs.phases[0].scale.value = 0.9
    ri.profile.w.value *= 1.15
    ri.background = BackgroundChebyshev(coefficients=[p(value=400), p(value=80)])
    plan = rx.RefinementPlan(
        [
            rx.Stage(
                "matched",
                ["phases.*.scale", "instrument.profile.w", "instrument.background.*"],
                max_iter=100,
                ftol=1e-10,
            )
        ]
    )
    selection = rv.RietveldParameterSelection(
        phase_scale=True, lattice=False, instrument_parameters=("w_deg2",), background=True
    )
    background = ps.refinement.ChebyshevBackground("b", (400.0, 80.0), (12.0, 120.0))
    experiment = replace(experiment, instrument=replace(instrument, w_deg2=0.003 * 1.15))
    phase = replace(phase, scale=0.9)
    parameters = rv.build_parameter_set(
        (phase,), (None,), selection, experiment=experiment, background=background
    )
    request = rv.RietveldInput(
        ps.PowderPattern(x, observed_y=observed, uncertainty=sigma),
        experiment,
        (phase,),
        (None,),
        parameters,
        selection=selection,
        background=background,
    )
    options = rv.RietveldOptions(execution=ps.ExecutionPolicy(threads=threads))

    def phase_operation():
        result = rv.refine(request, options)
        return result

    def rietx_operation():
        # Fresh session each repetition: no accepted fit carries into another.
        return rx.Refinement(rs, ri, history=False).fit(data, plan=plan)

    def scientific(result, name):
        if name == "phasesmith":
            values = [
                result.phases[0].scale,
                result.experiment.instrument.w_deg2,
                *result.background.coefficients,
            ]
            rwp, status = result.metrics.rwp, result.termination_reason.value
        else:
            values = [
                result.parameter("phases.0.scale").value,
                result.parameter("instrument.profile.w").value / (8 * np.log(2)),
                result.parameter("instrument.background.c0").value,
                result.parameter("instrument.background.c1").value,
            ]
            rwp, status = result.statistics.rwp, result.status
        np.testing.assert_allclose(values, [1.0, 0.003, 500.0, 100.0], rtol=2e-6, atol=2e-6)
        if status != "converged" or rwp >= 1e-6:
            raise RuntimeError(f"{name} recovery gate failed: {status}, {rwp}")
        return {"physical_parameters": values, "rwp": rwp, "termination": status}

    # Gate every operation outside its measured interval. measure_interleaved
    # returns only last results, so use a lightweight recording wrapper.
    results = {"phasesmith": [], "rietx": []}

    def recorded(name, operation):
        def call():
            result = operation()
            results[name].append(result)
            return result

        return call

    _, timings = measure_interleaved(
        {
            "phasesmith": recorded("phasesmith", phase_operation),
            "rietx": recorded("rietx", rietx_operation),
        },
        2,
        repetitions,
    )
    records = {
        name: [scientific(result, name) for result in items] for name, items in results.items()
    }
    return {
        "schema": "phasesmith.rietx-refinement-benchmark.v1",
        "scope": "complete_public_single_phase_scale_width_background_refinement",
        "samples": len(x),
        "reflections": len(generated.hkl),
        "sites": 8,
        "free_parameters": 4,
        "threads": threads,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("phasesmith", "rietx", "numpy", "scipy", "numba")
            },
        },
        "physical_parameter_order": [
            "scale",
            "gaussian_W_variance_deg2",
            "background_c0",
            "background_c1",
        ],
        "forward_relative_l2": profile_relative_l2,
        "gate": {
            "passed": True,
            "forward_relative_l2_limit": 1e-7,
            "parameter_rtol": 2e-6,
            "parameter_atol": 2e-6,
            "rwp_limit": 1e-6,
        },
        "notes": [
            "Common noiseless synthetic observations and explicit uncertainties.",
            "Gaussian U/V/W: rietx FWHM squared = 8 ln(2) times PhaseSmith variance.",
            "Default support policies retained; Gaussian tails agree under the declared gate.",
            "Both fit covariance enabled; rietx history and stage reports disabled.",
            "Input construction and imports excluded; native preparation, "
            "optimization and result assembly included.",
            "PhaseSmith owns pre-generated fixed HKLs; rietx compiles reflections within fit.",
            "All repeats start from identical perturbed inputs; every recovery is checked.",
            "Different solvers/stopping rules; same recovery requirements. "
            "Not a general refinement speed claim.",
        ],
        "scientific_results": {name: values[-1] for name, values in records.items()},
        "timings": timings,
        "rietx_over_phasesmith_median": timings["rietx"]["median_ms"]
        / timings["phasesmith"]["median_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.threads < 1:
        raise ValueError("repetitions and threads must be positive")
    report = run(args.repetitions, args.threads)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
