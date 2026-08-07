"""Benchmark native Bruckner smoothing and the complete public estimator."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import rietveld


def measure(operation: Callable[[], Any], warmups: int, repetitions: int) -> list[float]:
    for _ in range(warmups):
        operation()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
    return timings


def summary(timings: list[float]) -> dict[str, float]:
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "minimum_ms": min(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": ordered[p95_index],
    }


def benchmark_signal(sample_count: int, peak_count: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(5.0, 105.0, sample_count)
    y = 10.0 + 0.03 * x + 0.8 * np.cos(x / 17.0)
    positions = np.linspace(5.3, 104.7, peak_count)
    for index, position in enumerate(positions):
        width = 0.025 + 0.003 * (index % 7)
        intensity = 20.0 + float(index % 31)
        y += intensity * np.exp(-0.5 * ((x - position) / width) ** 2)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_001)
    parser.add_argument("--peaks", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--smooth-width", type=float, default=0.2)
    parser.add_argument("--chebyshev-order", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--require-release", action="store_true")
    arguments = parser.parse_args()
    if (
        arguments.samples < 2
        or arguments.peaks < 0
        or arguments.iterations < 0
        or arguments.smooth_width < 0.0
        or arguments.chebyshev_order < 0
        or arguments.warmups < 0
        or arguments.repetitions <= 0
    ):
        raise ValueError("benchmark counts and widths are outside their valid ranges")
    if arguments.chebyshev_order >= arguments.samples:
        raise ValueError("chebyshev order must be smaller than the sample count")
    if arguments.require_release and rietveld._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {rietveld._core.BUILD_MODE!r}")

    x, y = benchmark_signal(arguments.samples, arguments.peaks)
    spacing = float(x[1] - x[0])
    smooth_points = int(arguments.smooth_width / spacing)
    model = rietveld.SmoothBrucknerBackground(
        arguments.smooth_width,
        arguments.iterations,
        arguments.chebyshev_order,
    )
    native = measure(
        lambda: rietveld.smooth_bruckner(y, smooth_points, arguments.iterations),
        arguments.warmups,
        arguments.repetitions,
    )
    complete = measure(
        lambda: model.subtract(x, y),
        arguments.warmups,
        arguments.repetitions,
    )
    print(
        json.dumps(
            {
                "build_mode": rietveld._core.BUILD_MODE,
                "samples": arguments.samples,
                "peaks": arguments.peaks,
                "iterations": arguments.iterations,
                "smooth_points": smooth_points,
                "chebyshev_order": arguments.chebyshev_order,
                "native_smoother": summary(native),
                "smoother_and_chebyshev": summary(complete),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
