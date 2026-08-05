"""Benchmark Le Bail kernel, redistribution, profile update, and total workflow."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import rietveld
from rietveld.refinement import lebail


def measure(
    operation: Callable[[], Any], warmups: int, repetitions: int
) -> tuple[Any, list[float]]:
    for _ in range(warmups):
        operation()
    timings = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = operation()
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, timings


def report(name: str, timings: list[float]) -> None:
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)]
    print(
        f"case={name} min_ms={min(timings):.3f} "
        f"median_ms={statistics.median(timings):.3f} p95_ms={p95:.3f}"
    )


def models() -> tuple[
    rietveld.PowderPattern,
    rietveld.ConstantWavelengthInstrument,
    tuple[rietveld.Phase, ...],
]:
    x = np.linspace(10.0, 110.0, 5_001)
    count = 200
    index = np.arange(count)
    positions = 10.2 + 0.497 * index
    truth_intensities = 30.0 + index % 23
    starting_intensities = np.full(count, 20.0)
    hkl = np.column_stack(
        (
            index + 1,
            np.ones(count, dtype=np.int64),
            index % 7,
        )
    )
    instrument = rietveld.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )
    d_spacing = instrument.wavelength_angstrom / (
        2.0 * np.sin(np.deg2rad(positions / 2.0))
    )

    def phase(intensities: np.ndarray) -> rietveld.Phase:
        return rietveld.Phase(
            "benchmark",
            "Benchmark phase",
            rietveld.ReflectionBatch(
                [f"reflection-{number}" for number in index],
                hkl,
                d_spacing,
                positions,
                intensities,
            ),
        )

    background = 0.2 + 0.0005 * (x - x[0])
    truth = rietveld.calculate_pattern(
        rietveld.PowderPattern(x, background=background),
        instrument,
        (phase(truth_intensities),),
    )
    pattern = rietveld.PowderPattern(
        x,
        observed_y=truth.y,
        background=background,
        uncertainty=np.sqrt(np.maximum(truth.y, 1.0)),
    )
    return pattern, instrument, (phase(starting_intensities),)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument(
        "--require-release",
        action="store_true",
        help="fail unless the imported native extension reports a release build",
    )
    arguments = parser.parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions positive")
    if arguments.require_release and rietveld._core.BUILD_MODE != "release":
        raise RuntimeError(
            f"release extension required, imported {rietveld._core.BUILD_MODE!r}"
        )
    pattern, instrument, phases = models()
    calculation = rietveld.calculate_pattern(pattern, instrument, phases)
    current = np.concatenate(
        tuple(phase.reflections.integrated_intensity for phase in phases)
    )
    input_data = lebail.LeBailInput(pattern, instrument, phases)
    parameters = lebail.build_parameter_set(
        instrument,
        phases,
        instrument_parameters=("w_deg2",),
    )
    profile_input = lebail.LeBailInput(pattern, instrument, phases, parameters)

    _, kernel = measure(
        lambda: rietveld.calculate_pattern(pattern, instrument, phases),
        arguments.warmups,
        arguments.repetitions,
    )
    _, extraction = measure(
        lambda: lebail.extract_intensities(pattern, calculation, current),
        arguments.warmups,
        arguments.repetitions,
    )
    intensity_result, intensity_iteration = measure(
        lambda: lebail.refine(
            input_data,
            lebail.LeBailOptions(max_iterations=1, min_iterations=1),
        ),
        arguments.warmups,
        arguments.repetitions,
    )
    profile_result, profile_iteration = measure(
        lambda: lebail.refine(
            profile_input,
            lebail.LeBailOptions(max_iterations=1, min_iterations=1),
        ),
        arguments.warmups,
        arguments.repetitions,
    )
    total_result, total = measure(
        lambda: lebail.refine(
            input_data,
            lebail.LeBailOptions(max_iterations=10, min_iterations=10),
        ),
        arguments.warmups,
        arguments.repetitions,
    )
    overhead = [
        profile - intensity
        for profile, intensity in zip(profile_iteration, intensity_iteration, strict=True)
    ]
    print(
        f"python={platform.python_version()} platform={platform.platform()} "
        f"build_mode={rietveld._core.BUILD_MODE} samples={pattern.x.size} "
        f"reflections={len(current)}"
    )
    report("profile_kernel", kernel)
    report("intensity_extraction", extraction)
    report("intensity_only_iteration", intensity_iteration)
    report("profile_optimizer_overhead_estimate", overhead)
    report("profile_update_iteration", profile_iteration)
    report("total_10_iterations", total)
    print(
        f"intensity_iteration_rwp={intensity_result.metrics.rwp:.6g} "
        f"profile_iteration_rwp={profile_result.metrics.rwp:.6g} "
        f"total_rwp={total_result.metrics.rwp:.6g}"
    )


if __name__ == "__main__":
    main()
