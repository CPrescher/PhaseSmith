#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II CW profile performance."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import phasesmith

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_cw_profile.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "symmetric_cw_profile_values_and_analytical_derivatives"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument("--peaks", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5_001)
    parser.add_argument("--support-fwhm", type=float, default=20.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)]


def benchmark_inputs(
    peak_count: int, sample_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, phasesmith.ConstantWavelengthInstrument]:
    x = np.linspace(10.0, 110.0, sample_count, dtype=np.float64)
    index = np.arange(peak_count, dtype=np.float64)
    positions = 10.1 + index * (99.7 / peak_count)
    intensities = 100.0 + np.mod(index, 31.0)
    instrument = phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )
    return x, positions, intensities, instrument


def measure(operation: Any, warmups: int, repetitions: int) -> tuple[Any, list[float]]:
    for _ in range(warmups):
        operation()
    timings_ms = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = operation()
        timings_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, timings_ms


def normalized_maximum_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"benchmark output shape mismatch: engine {actual.shape}, GSAS-II {expected.shape}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError("benchmark outputs must be finite")
    if actual.size == 0:
        return 0.0
    scale = float(np.max(np.abs(expected)))
    difference = float(np.max(np.abs(actual - expected)))
    return difference if scale == 0.0 else difference / scale


def validate_outputs(
    engine_result: phasesmith.AccumulationResult, oracle_output: dict[str, np.ndarray]
) -> dict[str, float]:
    local = engine_result.derivatives.local
    np.testing.assert_array_equal(oracle_output["starts"], local.starts)
    np.testing.assert_array_equal(oracle_output["offsets"], local.offsets)
    comparisons = {
        "profile": (engine_result.y, oracle_output["y"]),
        "intensity_derivatives": (local.values[:, 0], oracle_output["local"][:, 0]),
        "position_derivatives": (local.values[:, 1], oracle_output["local"][:, 1]),
        "global_derivatives": (
            engine_result.derivatives.global_jacobian,
            oracle_output["global_jacobian"],
        ),
    }
    errors = {
        name: normalized_maximum_error(engine, oracle)
        for name, (engine, oracle) in comparisons.items()
    }
    failing = {name: value for name, value in errors.items() if value >= 6.0e-6}
    if failing:
        raise RuntimeError(f"GSAS-II numerical validation failed: {failing}")
    return errors


def timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "timings_ms": values,
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
    }


def main() -> None:
    arguments = parse_args()
    if (
        arguments.peaks <= 0
        or arguments.samples < 2
        or arguments.support_fwhm <= 0.0
        or arguments.warmups < 0
        or arguments.repetitions <= 0
    ):
        raise ValueError(
            "positive peaks/support/repetitions, at least two samples, "
            "and non-negative warmups required"
        )
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root (or GSASII_PYTHON and PHASESMITH_GSASII_ROOT)"
        )
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")

    x, positions, intensities, instrument = benchmark_inputs(arguments.peaks, arguments.samples)

    def operation() -> phasesmith.AccumulationResult:
        return phasesmith.accumulate_cw(
            x,
            positions,
            intensities,
            instrument,
            support_fwhm=arguments.support_fwhm,
        )

    engine_result, engine_timings = measure(operation, arguments.warmups, arguments.repetitions)
    assert engine_result is not None

    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-benchmark-") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.npz"
        output_path = directory / "output.npz"
        report_path = directory / "report.json"
        np.savez(
            input_path,
            x=x,
            positions=positions,
            intensities=intensities,
            instrument=np.array(
                [
                    instrument.u_deg2,
                    instrument.v_deg2,
                    instrument.w_deg2,
                    instrument.x_deg,
                    instrument.y_deg,
                ],
                dtype=np.float64,
            ),
            support_fwhm=np.array([arguments.support_fwhm], dtype=np.float64),
        )
        command = [
            str(arguments.gsas_python),
            str(WORKER),
            "--gsas-root",
            str(arguments.gsas_root),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--warmups",
            str(arguments.warmups),
            "--repetitions",
            str(arguments.repetitions),
        ]
        if arguments.binary_dir is not None:
            command.extend(["--binary-dir", str(arguments.binary_dir)])
        subprocess.run(command, check=True)
        gsas_report = json.loads(report_path.read_text())
        with np.load(output_path, allow_pickle=False) as archive:
            oracle_output = {name: archive[name].copy() for name in archive.files}

    if gsas_report["revision"] != PINNED_REVISION or gsas_report["scope"] != SCOPE:
        raise RuntimeError("external benchmark reported unexpected GSAS-II provenance or scope")
    gsas_timings = gsas_report.get("timings_ms")
    if (
        gsas_report.get("schema_version") != 1
        or gsas_report.get("implementation") != "GSAS-II"
        or not isinstance(gsas_timings, list)
        or len(gsas_timings) != arguments.repetitions
        or not all(np.isfinite(value) and value >= 0.0 for value in gsas_timings)
    ):
        raise RuntimeError("external benchmark returned an invalid timing report")
    errors = validate_outputs(engine_result, oracle_output)
    engine_summary = timing_summary(engine_timings)
    ratio = gsas_report["median_ms"] / engine_summary["median_ms"]
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "workload": {
            "peaks": arguments.peaks,
            "samples": arguments.samples,
            "support_fwhm": arguments.support_fwhm,
            "active_peak_samples": engine_result.derivatives.local.active_sample_count,
            "outputs": ["profile", "local_intensity_position_derivatives", "uvwxy_derivatives"],
        },
        "numerical_normalized_maximum_errors": errors,
        "phasesmith": {
            "build_mode": phasesmith._core.BUILD_MODE,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            **engine_summary,
        },
        "gsasii": gsas_report,
        "median_speed_ratio_gsasii_over_phasesmith": ratio,
    }
    print(
        f"scope={SCOPE} peaks={arguments.peaks} samples={arguments.samples} "
        f"active_peak_samples={engine_result.derivatives.local.active_sample_count}"
    )
    print(
        f"phasesmith build_mode={phasesmith._core.BUILD_MODE} "
        f"median_ms={engine_summary['median_ms']:.3f} p95_ms={engine_summary['p95_ms']:.3f}"
    )
    print(
        f"gsasii revision={PINNED_REVISION[:12]} median_ms={gsas_report['median_ms']:.3f} "
        f"p95_ms={gsas_report['p95_ms']:.3f}"
    )
    print(f"median_speed_ratio_gsasii_over_phasesmith={ratio:.3f}x")
    print(
        "normalized_maximum_errors="
        + ",".join(f"{name}:{value:.3e}" for name, value in errors.items())
    )
    if arguments.json_output is not None:
        arguments.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
