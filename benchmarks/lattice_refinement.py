"""Benchmark a realistic guarded-domain Le Bail lattice iteration."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import phasesmith
from phasesmith.refinement import LatticeParameterBounds, LatticeParameterization, lebail


def measure(operation: Callable[[], Any], warmups: int, repetitions: int) -> list[float]:
    for _ in range(warmups):
        operation()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
    return timings


def report(name: str, timings: list[float]) -> None:
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)]
    print(
        f"case={name} min_ms={min(timings):.3f} "
        f"median_ms={statistics.median(timings):.3f} p95_ms={p95:.3f}"
    )


def tetragonal_group() -> phasesmith.SpaceGroup:
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    operations = []
    current = np.eye(3, dtype=np.int64)
    for _ in range(4):
        operations.append(phasesmith.SymmetryOperation(current, (0, 0, 0)))
        current = rotation @ current
    return phasesmith.SpaceGroup(operations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_001)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--require-release", action="store_true")
    arguments = parser.parse_args()
    if arguments.samples < 2 or arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("samples/repetitions must be positive and warmups non-negative")
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")

    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 2.0e-4, 1.5e-3, 3.0e-3
    )
    cell = phasesmith.UnitCell(4.0, 4.0, 6.0, 90.0, 90.0, 90.0)
    structure = phasesmith.CrystalStructure(
        "benchmark", "Tetragonal benchmark", cell, tetragonal_group()
    )
    parameterization = LatticeParameterization(structure.space_group, cell)
    bounds = LatticeParameterBounds.around(parameterization, relative_length=0.04)
    phase = lebail.LeBailPhase.from_structure(
        structure,
        phase_id="alpha",
        wavelength_angstrom=instrument.wavelength_angstrom,
        two_theta_min_deg=10.0,
        two_theta_max_deg=130.0,
        lattice_bounds=bounds,
    )
    seeded = 5.0 + np.arange(phase.reflections.reflection_count) % 13
    phase = replace(
        phase,
        reflections=phasesmith.ReflectionBatch(
            phase.reflections.reflection_ids,
            phase.reflections.hkl,
            phase.reflections.d_spacing_angstrom,
            phase.reflections.two_theta_deg,
            seeded,
        ),
    )
    x = np.linspace(10.0, 130.0, arguments.samples)
    calculated = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument, (phase,))
    pattern = phasesmith.PowderPattern(x, observed_y=calculated.y)
    fixed_request = lebail.LeBailInput(pattern, instrument, (phase,))
    lattice_parameters = lebail.build_parameter_set(instrument, (phase,), lattice_parameters=True)
    lattice_request = lebail.LeBailInput(pattern, instrument, (phase,), lattice_parameters)
    domain = phase.reflection_domain
    assert domain is not None

    fixed_result = lebail.iterate_once(fixed_request)
    lattice_result = lebail.iterate_once(lattice_request)
    if not np.isfinite(fixed_result.metrics.rwp) or not np.isfinite(lattice_result.metrics.rwp):
        raise RuntimeError("benchmark setup produced invalid Le Bail metrics")

    regeneration = measure(
        lambda: domain.generate(phase.structure.cell, phase.reflections),
        arguments.warmups,
        arguments.repetitions,
    )
    fixed = measure(
        lambda: lebail.iterate_once(fixed_request),
        arguments.warmups,
        arguments.repetitions,
    )
    lattice = measure(
        lambda: lebail.iterate_once(lattice_request),
        arguments.warmups,
        arguments.repetitions,
    )
    print(
        f"python={platform.python_version()} platform={platform.platform()} "
        f"build_mode={phasesmith._core.BUILD_MODE} samples={x.size} "
        f"reflections={phase.reflections.reflection_count} "
        f"visible_reflections={int(np.count_nonzero(phase.visible_reflection_mask))} "
        f"lattice_parameters={len(lattice_parameters.specs)}"
    )
    report("guarded_domain_regeneration", regeneration)
    report("fixed_intensity_iteration", fixed)
    report("lattice_profile_iteration", lattice)


if __name__ == "__main__":
    main()
