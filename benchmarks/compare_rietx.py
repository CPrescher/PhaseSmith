#!/usr/bin/env python3
"""Numerically gated symmetric multi-peak benchmark against pinned rietx 1.4.0.

This is a kernel comparison, not a complete-refinement benchmark. The rietx
adapter calls its installed compiled kernels; no numerical implementation is
copied. Both sides return a pattern and four local analytical derivatives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import phasesmith
from compare_xrd_rust import measure_interleaved


def benchmark(peaks: int, samples: int, repetitions: int, warmups: int) -> dict:
    import rietx.model.compiled as compiled
    from rietx.model.forward import _batch_layout

    x = np.linspace(10.0, 150.0, samples)
    rng = np.random.default_rng(20260916 + peaks)
    positions = np.sort(rng.uniform(12.0, 148.0, peaks))
    intensities = rng.uniform(10.0, 1000.0, peaks)
    widths = rng.uniform(0.025, 0.12, peaks)
    etas = rng.uniform(0.05, 0.95, peaks)
    support = 20.0
    # Closed support interval, exactly matching PhaseSmith. The rietx default
    # adaptive windows are intentionally replaced for this controlled test.
    starts = np.searchsorted(x, positions - support * widths, side="left")
    stops = np.searchsorted(x, positions + support * widths, side="right")
    windows = np.stack((starts, stops), axis=-1)[None, :, :]
    layout = _batch_layout(windows, np.zeros((1, peaks), dtype=np.int64), x)
    rows = np.arange(peaks, dtype=np.int64)

    def phase_operation():
        return phasesmith.accumulate(x, positions, intensities, widths, etas, support_fwhm=support)

    def rietx_operation():
        omega, d_pos, d_width, d_eta = (np.zeros_like(layout.x) for _ in range(4))
        if not compiled.bases_symmetric(
            omega,
            d_pos,
            d_width,
            d_eta,
            layout.x,
            rows,
            positions,
            widths,
            etas,
            layout.width,
        ):
            raise RuntimeError("rietx compiled derivative kernel declined")
        y = compiled.accumulate(samples, [(layout, [(intensities, omega)])])
        if y is None:
            raise RuntimeError("rietx compiled accumulation kernel declined")
        # Convert unit-area basis derivatives to the same physical intensity,
        # center, FWHM, eta derivatives returned by PhaseSmith.
        for derivative in (d_pos, d_width, d_eta):
            derivative *= intensities[:, None]
        return y, (omega, d_pos, d_width, d_eta)

    operations = {"phasesmith": phase_operation, "rietx": rietx_operation}
    checked = {name: operation() for name, operation in operations.items()}
    ps = checked["phasesmith"]
    ry, planes = checked["rietx"]
    np.testing.assert_allclose(ps.y, ry, rtol=2e-12, atol=2e-10)
    assert ps.derivatives.local_parameter_names == ("intensity", "position", "fwhm", "eta")
    jac = ps.derivatives.local
    np.testing.assert_array_equal(jac.starts, starts)
    np.testing.assert_array_equal(np.diff(jac.offsets), stops - starts)
    derivative_error = 0.0
    for peak in range(peaks):
        begin, end = jac.offsets[peak : peak + 2]
        expected = np.stack([plane[peak, : end - begin] for plane in planes], axis=-1)
        actual = jac.values[begin:end]
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-8)
        derivative_error = max(derivative_error, float(np.max(np.abs(actual - expected))))
    del checked, ps, ry, planes, jac
    _, timings = measure_interleaved(operations, warmups, repetitions)
    return {
        "peaks": peaks,
        "samples": samples,
        "active_peak_samples": int(np.sum(stops - starts)),
        "rietx_padded_peak_samples": int(layout.x.size),
        "support_fwhm": support,
        "gate": {
            "passed": True,
            "maximum_absolute_derivative_error": derivative_error,
            "value_rtol": 2e-12,
            "value_atol": 2e-10,
            "derivative_rtol": 2e-11,
            "derivative_atol": 2e-8,
        },
        "timings": timings,
        "rietx_over_phasesmith_median": (
            timings["rietx"]["median_ms"] / timings["phasesmith"]["median_ms"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rietx-threads", type=int, default=1)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0 or args.rietx_threads < 1:
        raise ValueError("repetitions/threads must be positive and warmups nonnegative")
    if phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError("build the PhaseSmith extension in release mode")
    if importlib.metadata.version("rietx") != "1.4.0":
        raise RuntimeError("this private adapter is pinned to rietx 1.4.0")
    from rietx._about import COMPILED_THREADS_ENV
    from rietx.model import compiled

    os.environ[COMPILED_THREADS_ENV] = str(args.rietx_threads)
    start = time.perf_counter()
    compiled.warm(block=True)
    startup_ms = 1000 * (time.perf_counter() - start)
    if not compiled.enabled():
        raise RuntimeError("rietx compiled kernels must be enabled")
    report = {
        "schema": "phasesmith.rietx-kernel-benchmark.v1",
        "scope": "symmetric_profile_accumulation_and_four_local_derivatives",
        "excludes": [
            "structure factors",
            "reflection generation",
            "optimization",
            "file I/O",
            "process/import startup",
            "rietx frozen layout preparation",
        ],
        "notes": [
            "PhaseSmith includes boundary validation and support discovery on each call.",
            "rietx uses precomputed frozen padded windows; "
            "allocations and derivative scaling are timed.",
            "No claim about default rietx support, FCJ, or complete refinement speed.",
        ],
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("phasesmith", "rietx", "numpy", "numba", "scipy")
            },
            "phasesmith_build": phasesmith._core.BUILD_MODE,
            "phasesmith_threads": 1,
            "rietx_threads": compiled.n_threads(),
            "rietx_kernel_source_sha256": hashlib.sha256(
                Path(compiled.__file__).with_name("_kernels_numba.py").read_bytes()
            ).hexdigest(),
        },
        "rietx_compile_or_cache_load_ms": startup_ms,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cases": [
            benchmark(peaks, samples, args.repetitions, args.warmups)
            for peaks, samples in ((400, 20_001), (1600, 20_001), (1600, 100_001))
        ],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
