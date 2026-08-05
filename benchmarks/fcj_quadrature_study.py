"""Deterministic convergence study for the independent FCJ reference."""

from __future__ import annotations

import argparse
import time

import numpy as np
from rietveld.reference import profile_fcj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-order", type=int, default=256)
    parser.add_argument("--orders", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    cases = (
        ("low_equal_narrow", 12.0, 0.018, 0.006, 0.012, 0.012),
        ("low_unequal_broad", 18.0, 0.080, 0.035, 0.020, 0.006),
        ("middle_unequal", 70.0, 0.035, 0.012, 0.016, 0.009),
        ("high_equal", 138.0, 0.050, 0.020, 0.014, 0.014),
    )
    fields = (
        "value",
        "d_position",
        "d_gaussian_fwhm",
        "d_lorentzian_fwhm",
        "d_sample_over_radius",
        "d_detector_over_radius",
    )
    for name, position, gaussian, lorentzian, sample, detector in cases:
        x = np.linspace(position - 1.0, position + 1.0, 2_001)
        expected = profile_fcj(
            x,
            position,
            gaussian,
            lorentzian,
            sample,
            detector,
            quadrature_order=arguments.reference_order,
        )
        for order in arguments.orders:
            started = time.perf_counter_ns()
            actual = profile_fcj(
                x,
                position,
                gaussian,
                lorentzian,
                sample,
                detector,
                quadrature_order=order,
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            maximum = 0.0
            for field in fields:
                oracle = getattr(expected, field)
                scale = max(float(np.max(np.abs(oracle))), 1.0)
                maximum = max(
                    maximum,
                    float(np.max(np.abs(getattr(actual, field) - oracle))) / scale,
                )
            print(
                f"case={name} order={order} max_scaled_error={maximum:.6e} "
                f"elapsed_ms={elapsed_ms:.3f}"
            )


if __name__ == "__main__":
    main()
