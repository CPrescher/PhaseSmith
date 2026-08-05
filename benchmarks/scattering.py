"""Benchmark prepared native scattering batches through the public Python API."""

from __future__ import annotations

import argparse
import platform
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


def report(name: str, timings: list[float]) -> None:
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)]
    print(
        f"case={name} min_ms={min(timings):.3f} "
        f"median_ms={statistics.median(timings):.3f} p95_ms={p95:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reflections", type=int, default=20_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument(
        "--require-release",
        action="store_true",
        help="fail unless the imported native extension reports a release build",
    )
    arguments = parser.parse_args()
    if arguments.reflections <= 0 or arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("reflections and repetitions must be positive; warmups non-negative")
    if arguments.require_release and rietveld._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {rietveld._core.BUILD_MODE!r}")

    s = np.linspace(0.0, 5.9, arguments.reflections)
    xray_species = (
        *(rietveld.ScatteringSpecies(key) for key in ("Si", "O", "O", "Na", "Al", "O", "O")),
        rietveld.ScatteringSpecies("Fe", charge=3),
    )
    neutron_species = tuple(
        rietveld.ScatteringSpecies(key) for key in ("Si", "O", "O", "Na", "Al", "O", "O", "Fe")
    )
    xray = rietveld.XrayNonResonant().prepare(xray_species)
    neutron = rietveld.NeutronNuclear().prepare(neutron_species)

    xray_timings = measure(lambda: xray.evaluate(s), arguments.warmups, arguments.repetitions)
    neutron_timings = measure(lambda: neutron.evaluate(s), arguments.warmups, arguments.repetitions)
    print(
        f"python={platform.python_version()} platform={platform.platform()} "
        f"build_mode={rietveld._core.BUILD_MODE} reflections={arguments.reflections} "
        f"sites={len(xray_species)} xray_unique={xray.unique_species_count} "
        f"neutron_unique={neutron.unique_species_count}"
    )
    report("xray_value_and_derivative", xray_timings)
    report("neutron_value_and_derivative", neutron_timings)


if __name__ == "__main__":
    main()
