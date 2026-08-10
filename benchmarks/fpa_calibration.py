"""Reproducible benchmark for offline physical-target generation and compression."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import TypeVar

import phasesmith

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    return parser.parse_args()


def measure(operation: Callable[[], T], repetitions: int) -> tuple[T, list[float]]:
    timings_ms = []
    result = operation()
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = operation()
        timings_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, timings_ms


def main() -> None:
    arguments = parse_args()
    if arguments.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    first_wavelength = 1.5405929
    second_wavelength = 1.5444274
    ratio = second_wavelength / first_wavelength
    model = phasesmith.BraggBrentanoFundamentalProfile(
        radius_mm=217.5,
        source_width_mm=0.02,
        receiving_slit_width_mm=0.04,
        sample_half_length_mm=1.0875,
        detector_half_length_mm=1.0875,
        emission_lines=(
            phasesmith.FundamentalEmissionLine(
                first_wavelength, 2.0, 0.0002600, 0.0001200
            ),
            phasesmith.FundamentalEmissionLine(
                second_wavelength,
                1.0,
                0.0002600 * ratio,
                0.0001200 * ratio,
            ),
        ),
    )
    options = phasesmith.FundamentalProfileCalibrationOptions()
    target, target_ms = measure(
        lambda: phasesmith.simulate_fundamental_peaks(model, options),
        arguments.repetitions,
    )
    calibration, calibration_ms = measure(
        lambda: phasesmith.calibrate_fundamental_profile(model, options),
        arguments.repetitions,
    )
    print(
        f"case=physical_target samples={target.grid_deg.size} "
        f"median_ms={statistics.median(target_ms):.3f}"
    )
    print(
        f"case=profile_compression iterations={calibration.iterations} "
        f"accepted={calibration.accepted} "
        f"median_ms={statistics.median(calibration_ms):.3f}"
    )


if __name__ == "__main__":
    main()
