"""Benchmark realistic matrix-free monochromatic structural refinement."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace

import numpy as np
import rietveld
from rietveld.refinement import PolynomialBackground
from rietveld.refinement import rietveld as structural_refinement


def benchmark_request(samples: int) -> structural_refinement.RietveldInput:
    x = np.linspace(12.0, 120.0, samples)
    instrument = rietveld.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    experiment = rietveld.ConstantWavelengthExperiment.x_ray(instrument)
    sites = tuple(
        rietveld.AtomSite(
            f"site-{index}",
            f"X{index}",
            "Si" if index % 2 == 0 else "O",
            "Si" if index % 2 == 0 else "O",
            (
                (0.071 * index + 0.11) % 1.0,
                (0.137 * index + 0.23) % 1.0,
                (0.193 * index + 0.31) % 1.0,
            ),
            0.9 if index % 3 == 0 else 1.0,
            0.012 + 0.001 * index,
        )
        for index in range(8)
    )
    structure = rietveld.CrystalStructure(
        "benchmark",
        "Rietveld benchmark",
        rietveld.UnitCell(5.2, 5.2, 5.2, 90.0, 90.0, 90.0),
        rietveld.SpaceGroup.p1(),
        sites,
    )
    generated = rietveld.PreparedReflectionGenerator(structure.space_group).generate(
        structure.cell,
        rietveld.CwTwoThetaRange(12.0, 120.0, instrument.wavelength_angstrom),
    )
    truth_phase = rietveld.RietveldPhase(
        "alpha",
        "Benchmark alpha",
        structure,
        rietveld.StructuralReflectionBatch.from_generated(generated),
        rietveld.XrayNonResonant(),
        rietveld.NeutralIntegratedIntensityCorrection(),
    )
    truth_background = PolynomialBackground("main", (1.0, 0.1, -0.05))
    empty = rietveld.PowderPattern(x)
    truth = structural_refinement.calculate(
        empty,
        experiment,
        (truth_phase,),
        background=truth_background,
    )
    starting_sites = list(sites)
    first_xyz = list(starting_sites[0].fractional_xyz)
    first_xyz[0] += 5.0e-4
    starting_sites[0] = replace(starting_sites[0], fractional_xyz=tuple(first_xyz))
    starting_phase = replace(
        truth_phase,
        structure=replace(structure, sites=tuple(starting_sites)),
        scale=0.92,
    )
    starting_experiment = replace(
        experiment,
        instrument=replace(instrument, w_deg2=instrument.w_deg2 + 2.0e-5),
    )
    starting_background = PolynomialBackground("main", (0.8, 0.0, 0.0))
    selection = structural_refinement.RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
        coordinates=True,
        instrument_parameters=("w_deg2",),
        background=True,
    )
    parameters = structural_refinement.build_parameter_set(
        (starting_phase,),
        (None,),
        selection,
        experiment=starting_experiment,
        background=starting_background,
    )
    return structural_refinement.RietveldInput(
        rietveld.PowderPattern(x, observed_y=truth.y),
        starting_experiment,
        (starting_phase,),
        (None,),
        parameters,
        selection=selection,
        background=starting_background,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20_001)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.samples < 2 or arguments.repeats <= 0 or arguments.iterations <= 0:
        raise ValueError("benchmark counts must be positive and samples at least two")
    request = benchmark_request(arguments.samples)
    options = structural_refinement.RietveldOptions(
        limits=structural_refinement.RefinementLimits(
            max_iterations=arguments.iterations,
            max_evaluations=2_000,
        ),
        min_iterations=arguments.iterations,
        max_cg_iterations=12,
        estimate_covariance=False,
    )
    structural_refinement.refine(request, options)
    timings = []
    result = None
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        result = structural_refinement.refine(request, options)
        timings.append(1_000.0 * (time.perf_counter() - started))
    if result is None:  # pragma: no cover - repeats are validated positive
        raise RuntimeError("benchmark produced no result")
    print(
        json.dumps(
            {
                "scope": "matrix_free_monochromatic_rietveld",
                "samples": arguments.samples,
                "reflections": request.phases[0].reflections.reflection_count,
                "sites": len(request.phases[0].structure.sites),
                "free_parameters": len(
                    structural_refinement.ConstraintTransform(request.parameters).free_keys
                ),
                "iterations": arguments.iterations,
                "median_ms": statistics.median(timings),
                "minimum_ms": min(timings),
                "evaluations": result.evaluations,
                "final_rwp": result.metrics.rwp,
                "termination_reason": result.termination_reason.value,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
