"""Reproducible Python-to-Rust profile benchmark with allocation reporting."""

from __future__ import annotations

import argparse
import platform
import statistics
import time

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

    for _ in range(arguments.warmups):
        accumulate(
            x, positions, intensities, fwhms, etas, support_fwhm=support_fwhm
        )
    timings_ms = []
    for _ in range(arguments.repetitions):
        started = time.perf_counter_ns()
        result = accumulate(
            x, positions, intensities, fwhms, etas, support_fwhm=support_fwhm
        )
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)

    input_bytes = sum(
        array.nbytes for array in (x, positions, intensities, fwhms, etas)
    )
    median_ms = statistics.median(timings_ms)
    print(f"python={platform.python_version()} numpy={np.__version__}")
    print(f"platform={platform.platform()}")
    print(f"native_build_mode={_core.BUILD_MODE} native_module={_core.__file__}")
    print(
        f"peaks={count} samples={x.size} active_peak_samples={active_peak_samples} "
        f"support_fwhm={support_fwhm:g}"
    )
    print(
        f"input_mb={input_bytes / 1e6:.3f} y_mb={result.y.nbytes / 1e6:.3f} "
        f"jacobian_mb={result.jacobian.nbytes / 1e6:.3f}"
    )
    print(
        f"repetitions={arguments.repetitions} min_ms={min(timings_ms):.3f} "
        f"median_ms={median_ms:.3f} p95_ms={percentile(timings_ms, 0.95):.3f} "
        f"active_evaluations_per_s={active_peak_samples / (median_ms / 1e3):.0f}"
    )


if __name__ == "__main__":
    main()
