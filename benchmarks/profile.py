"""Reproducible Python-to-Rust profile benchmark with allocation reporting."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from rietveld import _core, accumulate


def parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--require-release", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile from a non-empty sample."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def measure(
    operation: Callable[[], Any], *, warmups: int, repetitions: int
) -> tuple[Any, list[float]]:
    """Measure an operation after a fixed number of warmup calls."""

    for _ in range(warmups):
        operation()
    timings_ms = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = operation()
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    return result, timings_ms


def report_case(name: str, output_bytes: int, timings_ms: list[float]) -> None:
    """Print timing and allocation metrics for one derivative layout."""

    median_ms = statistics.median(timings_ms)
    print(
        f"case={name} output_mb={output_bytes / 1e6:.3f} "
        f"min_ms={min(timings_ms):.3f} median_ms={median_ms:.3f} "
        f"p95_ms={percentile(timings_ms, 0.95):.3f}"
    )


def main() -> None:
    """Run and report the end-to-end native accumulation benchmark."""

    arguments = parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    if arguments.require_release and _core.BUILD_MODE != "release":
        raise RuntimeError(
            "native extension is not optimized; run `maturin develop --release --uv` first"
        )

    x = np.linspace(10.0, 110.0, 5_001)
    count = 200
    index = np.arange(count)
    positions = 10.1 + index * 0.49
    intensities = 100.0 + index % 31
    fwhms = 0.03 + (index % 7) * 0.002
    etas = 0.2 + (index % 5) * 0.1

    support_fwhm = 20.0
    lower = np.searchsorted(x, positions - support_fwhm * fwhms, side="left")
    upper = np.searchsorted(x, positions + support_fwhm * fwhms, side="right")
    active_peak_samples = int(np.sum(upper - lower))

    call_arguments = (x, positions, intensities, fwhms, etas, support_fwhm)
    values, values_timings = measure(
        lambda: _core.accumulate_values(*call_arguments),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    support_result, support_timings = measure(
        lambda: accumulate(
            x,
            positions,
            intensities,
            fwhms,
            etas,
            support_fwhm=support_fwhm,
            jacobian_layout="support",
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    dense_result, dense_timings = measure(
        lambda: accumulate(
            x,
            positions,
            intensities,
            fwhms,
            etas,
            support_fwhm=support_fwhm,
            jacobian_layout="dense",
        ),
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )

    input_bytes = sum(
        array.nbytes for array in (x, positions, intensities, fwhms, etas)
    )
    print(f"python={platform.python_version()} numpy={np.__version__}")
    print(f"platform={platform.platform()}")
    print(f"native_build_mode={_core.BUILD_MODE} native_module={_core.__file__}")
    print(
        f"peaks={count} samples={x.size} active_peak_samples={active_peak_samples} "
        f"support_fwhm={support_fwhm:g}"
    )
    print(f"input_mb={input_bytes / 1e6:.3f} repetitions={arguments.repetitions}")
    report_case("values_only", values.nbytes, values_timings)
    report_case(
        "support_jacobian",
        support_result.y.nbytes + support_result.derivatives.local.nbytes,
        support_timings,
    )
    report_case(
        "dense_jacobian",
        dense_result.y.nbytes + dense_result.jacobian.nbytes,
        dense_timings,
    )


if __name__ == "__main__":
    main()
