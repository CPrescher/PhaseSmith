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
    parser.add_argument("--soller-repetitions", type=int, default=3)
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
    if arguments.repetitions <= 0 or arguments.soller_repetitions <= 0:
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
            phasesmith.FundamentalEmissionLine(first_wavelength, 2.0, 0.0002600, 0.0001200),
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
    soller_model = phasesmith.BraggBrentanoFundamentalProfile(
        radius_mm=217.5,
        source_width_mm=0.04,
        receiving_slit_width_mm=0.2,
        sample_half_length_mm=7.5,
        detector_half_length_mm=2.5,
        emission_lines=model.emission_lines,
        soller_axial_geometry=phasesmith.SollerAxialGeometry(12.0, 15.0, 5.0, 6.776, 6.776),
    )
    soller_options = phasesmith.FundamentalProfileCalibrationOptions(
        peak_positions_deg=(20.0, 40.0, 60.0, 80.0, 100.0, 120.0),
        step_deg=0.004,
        aperture_quadrature_order=1,
    )
    soller_target, soller_ms = measure(
        lambda: phasesmith.simulate_fundamental_peaks(soller_model, soller_options),
        arguments.soller_repetitions,
    )
    print(
        f"case=soller_axial_target samples={soller_target.grid_deg.size} "
        f"axial_order={soller_options.axial_ray_quadrature_order} "
        f"median_ms={statistics.median(soller_ms):.3f}"
    )


if __name__ == "__main__":
    main()
