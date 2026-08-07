#!/usr/bin/env python3
"""Benchmark the pinned GSAS-II CW profile kernel and Python accumulation.

This external-only worker imports no PhaseSmith module. It is launched by
``benchmarks/compare_gsasii.py`` with GSAS-II's own Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
GAUSSIAN_FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def import_gsasii(gsas_root: Path, binary_dir: Path | None) -> Any:
    revision = git_revision(gsas_root)
    if revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II benchmark requires revision {PINNED_REVISION}, detected {revision}"
        )
    sys.path.insert(0, str(gsas_root))
    from GSASII import GSASIIpath

    if binary_dir is not None:
        sys.path.insert(0, str(binary_dir))
        GSASIIpath.binaryPath = str(binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    from GSASII import GSASIIpwd

    if not hasattr(GSASIIpwd, "pyd"):
        raise RuntimeError("GSAS-II pypowder binary is unavailable")
    return GSASIIpwd


def width_terms(position_deg: float, instrument: np.ndarray) -> tuple[Any, ...]:
    u_deg2, v_deg2, w_deg2, x_deg, y_deg = instrument
    theta = np.deg2rad(position_deg / 2.0)
    tangent = np.tan(theta)
    secant = 1.0 / np.cos(theta)
    variance = u_deg2 * tangent**2 + v_deg2 * tangent + w_deg2
    sigma = np.sqrt(variance)
    gaussian = GAUSSIAN_FWHM_PER_SIGMA * sigma
    lorentzian = x_deg * secant + y_deg * tangent
    d_gaussian_d_variance = GAUSSIAN_FWHM_PER_SIGMA / (2.0 * sigma)
    d_gaussian_d_instrument = d_gaussian_d_variance * np.array(
        [tangent**2, tangent, 1.0, 0.0, 0.0], dtype=np.float64
    )
    d_lorentzian_d_instrument = np.array([0.0, 0.0, 0.0, secant, tangent], dtype=np.float64)
    radians_per_degree = np.pi / 360.0
    d_tangent = radians_per_degree * secant**2
    d_secant = radians_per_degree * secant * tangent
    d_variance_d_position = (2.0 * u_deg2 * tangent + v_deg2) * d_tangent
    d_gaussian_d_position = d_gaussian_d_variance * d_variance_d_position
    d_lorentzian_d_position = x_deg * d_secant + y_deg * d_tangent
    return (
        variance,
        gaussian,
        lorentzian,
        d_gaussian_d_instrument,
        d_lorentzian_d_instrument,
        d_gaussian_d_position,
        d_lorentzian_d_position,
    )


def tch_fwhm(gaussian: float, lorentzian: float) -> float:
    return float(
        (
            gaussian**5
            + 2.69269 * gaussian**4 * lorentzian
            + 2.42843 * gaussian**3 * lorentzian**2
            + 4.47163 * gaussian**2 * lorentzian**3
            + 0.07842 * gaussian * lorentzian**4
            + lorentzian**5
        )
        ** 0.2
    )


def accumulate(
    profile_module: Any,
    x: np.ndarray,
    positions: np.ndarray,
    intensities: np.ndarray,
    instrument: np.ndarray,
    support_fwhm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = np.empty(positions.size, dtype=np.int64)
    lengths = np.empty(positions.size, dtype=np.int64)
    widths: list[tuple[Any, ...]] = []
    for peak, position in enumerate(positions):
        terms = width_terms(float(position), instrument)
        widths.append(terms)
        radius = support_fwhm * tch_fwhm(float(terms[1]), float(terms[2]))
        start = int(np.searchsorted(x, position - radius, side="left"))
        stop = int(np.searchsorted(x, position + radius, side="right"))
        starts[peak] = start
        lengths[peak] = stop - start

    offsets = np.empty(positions.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    y = np.zeros_like(x)
    local = np.empty((int(offsets[-1]), 2), dtype=np.float64)
    global_jacobian = np.zeros((5, x.size), dtype=np.float64)

    for peak, (position, intensity) in enumerate(zip(positions, intensities, strict=True)):
        start = int(starts[peak])
        begin = int(offsets[peak])
        end = int(offsets[peak + 1])
        stop = start + end - begin
        (
            variance,
            gaussian,
            lorentzian,
            d_gaussian_d_instrument,
            d_lorentzian_d_instrument,
            d_gaussian_d_position,
            d_lorentzian_d_position,
        ) = widths[peak]
        native_profile, native_position, native_sigma2, native_gamma = profile_module.getdPsVoigt(
            float(position),
            10_000.0 * float(variance),
            100.0 * float(lorentzian),
            x[start:stop],
        )
        profile = 100.0 * np.asarray(native_profile, dtype=np.float64)
        d_gaussian = (
            100.0
            * np.asarray(native_sigma2, dtype=np.float64)
            * (20_000.0 * float(gaussian) / GAUSSIAN_FWHM_PER_SIGMA**2)
        )
        d_lorentzian = 10_000.0 * np.asarray(native_gamma, dtype=np.float64)
        d_position = (
            -100.0 * np.asarray(native_position, dtype=np.float64)
            + d_gaussian * float(d_gaussian_d_position)
            + d_lorentzian * float(d_lorentzian_d_position)
        )
        y[start:stop] += intensity * profile
        local[begin:end, 0] = profile
        local[begin:end, 1] = intensity * d_position
        global_jacobian[:, start:stop] += intensity * (
            d_gaussian[None, :] * d_gaussian_d_instrument[:, None]
            + d_lorentzian[None, :] * d_lorentzian_d_instrument[:, None]
        )
    return y, starts, offsets, local, global_jacobian


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)]


def main() -> None:
    arguments = parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    profile_module = import_gsasii(arguments.gsas_root.resolve(), arguments.binary_dir)
    with np.load(arguments.input, allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["x"], dtype=np.float64)
        positions = np.ascontiguousarray(archive["positions"], dtype=np.float64)
        intensities = np.ascontiguousarray(archive["intensities"], dtype=np.float64)
        instrument = np.ascontiguousarray(archive["instrument"], dtype=np.float64)
        support_fwhm = float(archive["support_fwhm"][0])

    def operation() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return accumulate(profile_module, x, positions, intensities, instrument, support_fwhm)

    for _ in range(arguments.warmups):
        operation()
    timings_ms = []
    result = None
    for _ in range(arguments.repetitions):
        started = time.perf_counter_ns()
        result = operation()
        timings_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    assert result is not None
    np.savez(
        arguments.output,
        y=result[0],
        starts=result[1],
        offsets=result[2],
        local=result[3],
        global_jacobian=result[4],
    )
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": PINNED_REVISION,
        "scope": "symmetric_cw_profile_values_and_analytical_derivatives",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "warmups": arguments.warmups,
        "repetitions": arguments.repetitions,
        "timings_ms": timings_ms,
        "min_ms": min(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": percentile(timings_ms, 0.95),
    }
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
