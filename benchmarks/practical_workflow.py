"""Benchmark realistic X-ray and monochromatic-neutron pattern calculations."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial

import numpy as np
import phasesmith
from phasesmith.refinement import (
    AmorphousBackground,
    AmorphousPeak,
    ChebyshevBackground,
    CompositeBackground,
)
from phasesmith.refinement import rietveld as structural_refinement


def benchmark_case(
    probe: phasesmith.RadiationProbe,
    samples: int,
) -> tuple[
    phasesmith.PowderPattern,
    phasesmith.ConstantWavelengthExperiment,
    tuple[phasesmith.RietveldPhase, ...],
    CompositeBackground,
]:
    """Construct one deterministic multi-physics monochromatic workload."""

    x = np.linspace(12.0, 120.0, samples)
    wavelength = 1.5406 if probe is phasesmith.RadiationProbe.X_RAY else 1.8
    instrument = phasesmith.ConstantWavelengthInstrument(
        wavelength, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    experiment = (
        phasesmith.ConstantWavelengthExperiment(
            phasesmith.MonochromaticRadiation.x_ray(wavelength),
            instrument,
            zero_shift_deg=0.012,
            geometry=phasesmith.BraggBrentanoGeometry(240.0, 0.12),
        )
        if probe is phasesmith.RadiationProbe.X_RAY
        else phasesmith.ConstantWavelengthExperiment(
            phasesmith.MonochromaticRadiation.neutron(wavelength),
            instrument,
            zero_shift_deg=-0.008,
        )
    )
    sites = tuple(
        phasesmith.AtomSite(
            f"site-{index}",
            f"{'Si' if index % 2 == 0 else 'O'}{index}",
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
    structure = phasesmith.CrystalStructure(
        "benchmark",
        "Practical workflow benchmark",
        phasesmith.UnitCell(5.2, 5.2, 5.2, 90.0, 90.0, 90.0),
        phasesmith.SpaceGroup.p1(),
        sites,
    )
    generated = phasesmith.PreparedReflectionGenerator(structure.space_group).generate(
        structure.cell,
        phasesmith.CwTwoThetaRange(12.0, 120.0, wavelength),
    )
    metric = phasesmith.ReciprocalMetric(structure.cell.geometry().reciprocal_metric)
    physics = phasesmith.CompositePhysicsProvider(
        (
            phasesmith.IsotropicSizeBroadening(85.0),
            phasesmith.IsotropicMicrostrainBroadening(5.0e-4),
            phasesmith.MarchDollasePreferredOrientation(0.88, (0.0, 0.0, 1.0), metric),
        )
    )
    phase = phasesmith.RietveldPhase(
        "alpha",
        "Benchmark alpha",
        structure,
        phasesmith.StructuralReflectionBatch.from_generated(generated),
        (
            phasesmith.XrayNonResonant()
            if probe is phasesmith.RadiationProbe.X_RAY
            else phasesmith.NeutronNuclear()
        ),
        (
            phasesmith.BraggBrentanoUnpolarizedLp(wavelength)
            if probe is phasesmith.RadiationProbe.X_RAY
            else phasesmith.NeutralIntegratedIntensityCorrection()
        ),
        physics=physics,
    )
    background = CompositeBackground(
        "background",
        (
            ChebyshevBackground("chebyshev", (1.0, 0.1, -0.05), (12.0, 120.0)),
            AmorphousBackground("glass", (AmorphousPeak(25.0, 48.0, 15.0),)),
        ),
    )
    return phasesmith.PowderPattern(x), experiment, (phase,), background


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20_001)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--require-release", action="store_true")
    arguments = parser.parse_args()
    if arguments.samples < 2 or arguments.repeats <= 0:
        raise ValueError("samples must be at least two and repeats positive")
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")
    reports = {}
    for probe in (phasesmith.RadiationProbe.X_RAY, phasesmith.RadiationProbe.NEUTRON):
        pattern, experiment, phases, background = benchmark_case(probe, arguments.samples)

        operation = partial(
            structural_refinement.calculate,
            pattern,
            experiment,
            phases,
            background=background,
        )

        operation()
        timings = []
        result = None
        for _ in range(arguments.repeats):
            started = time.perf_counter_ns()
            result = operation()
            timings.append((time.perf_counter_ns() - started) / 1.0e6)
        if result is None:  # pragma: no cover - repeats validated positive
            raise RuntimeError("benchmark produced no result")
        reports[probe.value] = {
            "samples": pattern.x.size,
            "reflections": phases[0].reflections.reflection_count,
            "sites": len(phases[0].structure.sites),
            "median_ms": statistics.median(timings),
            "minimum_ms": min(timings),
            "finite": bool(np.isfinite(result.y).all()),
        }
    print(json.dumps({"scope": "practical_monochromatic_patterns", **reports}, indent=2))


if __name__ == "__main__":
    main()
