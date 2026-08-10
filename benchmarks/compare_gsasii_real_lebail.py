#!/usr/bin/env python3
"""Compare real PhaseSmith Le Bail validations with pinned external GSAS-II."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import phasesmith
from phasesmith.background import SmoothBrucknerBackground
from phasesmith.io.powder import read_powder_data
from phasesmith.validation import (
    run_sucrose_lebail_validation,
    verify_validation_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_real_lebail.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
CASES = {
    "aps-sucrose-11bmb": {
        "rwp_check": "smoke_rwp",
        "maximum_rwp_delta": 0.005,
        "maximum_correlation_delta": 0.02,
    },
}
RWP_NOTE = re.compile(r"First-cycle Rwp=[0-9.eE+-]+; final Rwp=([0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--gsas-cycles", type=int, default=20)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "timings_ms": values,
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


def measurement(report: Any, check_id: str) -> float:
    values = [check.measured for check in report.checks if check.check_id == check_id]
    if len(values) != 1 or values[0] is None or not np.isfinite(values[0]):
        raise RuntimeError(f"PhaseSmith report has no finite {check_id!r} measurement")
    return float(values[0])


def phase_result(case_id: str, report: Any) -> dict[str, Any]:
    if report.status != "passed":
        raise RuntimeError(f"PhaseSmith {case_id} validation returned {report.status!r}")
    rwp_check = CASES[case_id]["rwp_check"]
    if rwp_check is not None:
        rwp = measurement(report, rwp_check)
    else:
        match = next((RWP_NOTE.match(note) for note in report.notes if RWP_NOTE.match(note)), None)
        if match is None:
            raise RuntimeError("PhaseSmith Echidna report has no structured final Rwp")
        rwp = float(match.group(1))
    return {
        "sample_count": report.sample_count,
        "reflection_count": report.reflection_count,
        "poisson_rwp": rwp,
        "profile_correlation": measurement(report, "profile_correlation"),
        "validation_status": report.status,
    }


def run_phasesmith(
    case_id: str, data: Path, warmups: int, repetitions: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = run_sucrose_lebail_validation
    for _ in range(warmups):
        runner(data)
    reports = []
    elapsed_ms = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        reports.append(runner(data))
        elapsed_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    results = [phase_result(case_id, report) for report in reports]
    if any(result != results[0] for result in results[1:]):
        raise RuntimeError(f"PhaseSmith {case_id} repetitions are not deterministic")
    return results[-1], {"driver_wall": timing_summary(elapsed_ms)}


def run_gsas_once(
    arguments: argparse.Namespace,
    data: Path,
    report_path: Path,
    environment: dict[str, str],
    fixed_background: Path | None,
) -> tuple[dict[str, Any], float]:
    command = [
        str(arguments.gsas_python),
        str(WORKER),
        "--gsas-root",
        str(arguments.gsas_root),
        "--case",
        arguments.case,
        "--data-directory",
        str(data),
        "--report",
        str(report_path),
        "--cycles",
        str(arguments.gsas_cycles),
    ]
    if arguments.binary_dir is not None:
        command.extend(["--binary-dir", str(arguments.binary_dir)])
    if fixed_background is not None:
        command.extend(["--fixed-background", str(fixed_background)])
    started = time.perf_counter_ns()
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
    if process.returncode != 0:
        raise RuntimeError(
            "GSAS-II real Le Bail worker failed\n"
            f"stdout:\n{process.stdout[-4000:]}\n"
            f"stderr:\n{process.stderr[-4000:]}"
        )
    return json.loads(report_path.read_text(encoding="utf-8")), elapsed_ms


def run_gsas(
    arguments: argparse.Namespace, data: Path, temporary: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
    environment["PYTHONHASHSEED"] = "0"
    fixed_background = prepare_fixed_background(arguments.case, data, temporary)
    for index in range(arguments.warmups):
        run_gsas_once(
            arguments,
            data,
            temporary / f"warmup-{index}.json",
            environment,
            fixed_background,
        )
    reports = []
    elapsed_ms = []
    for index in range(arguments.repetitions):
        report, elapsed = run_gsas_once(
            arguments,
            data,
            temporary / f"result-{index}.json",
            environment,
            fixed_background,
        )
        reports.append(report)
        elapsed_ms.append(elapsed)
    scope = f"{arguments.case}_lebail_workflow"
    for report in reports:
        if (
            report.get("schema_version") != 1
            or report.get("implementation") != "GSAS-II"
            or report.get("revision") != PINNED_REVISION
            or report.get("scope") != scope
        ):
            raise RuntimeError("GSAS-II real Le Bail worker returned invalid provenance")
        if arguments.case == "aps-sucrose-11bmb":
            recipe = report["recipe"]
            if (
                recipe["background"]
                != "fixed Smooth Bruckner + 1-term refined Chebyshev residual"
                or recipe["background_start"] != [0.0]
                or recipe["instrument_start"]
                != {
                    "Lam": 0.413259,
                    "Zero": 0.0,
                    "U": 1.163,
                    "V": -0.126,
                    "W": 0.063,
                    "X": 0.173,
                    "Y": 0.0,
                    "Z": 0.0,
                    "SH/L": 0.0,
                }
                or recipe["instrument_parameters"] != ["U", "V", "W", "X", "Y"]
            ):
                raise RuntimeError("GSAS-II sucrose worker did not use the matched recipe")
    stable = ("recipe", "input_sha256", "result", "cycle_rwp_percent")
    for report in reports[1:]:
        if any(report[key] != reports[0][key] for key in stable):
            raise RuntimeError("GSAS-II real Le Bail repetitions are not deterministic")
    timing = {
        "external_process": timing_summary(elapsed_ms),
        "reported_total_workflow_ms": [report["timing_ms"]["total_workflow"] for report in reports],
    }
    metadata = {
        "recipe": reports[-1]["recipe"],
        "input_sha256": reports[-1]["input_sha256"],
        "cycle_rwp_percent": reports[-1]["cycle_rwp_percent"],
        "python_version": reports[-1]["python_version"],
        "numpy_version": reports[-1]["numpy_version"],
        "platform": reports[-1]["platform"],
    }
    return reports[-1]["result"], timing, metadata


def prepare_fixed_background(case_id: str, data: Path, temporary: Path) -> Path | None:
    """Write the exact PhaseSmith preprocessing array as plain oracle input."""

    if case_id != "aps-sucrose-11bmb":
        return None
    pattern = read_powder_data(data / "11bmb_8716.fxye", format="gsas_fxye")
    selected = (pattern.x >= 1.0) & (pattern.x <= 24.0)
    x = pattern.x[selected]
    observed = pattern.observed_y[selected]
    baseline = SmoothBrucknerBackground(
        smooth_width=0.1,
        iterations=50,
        chebyshev_order=None,
    ).estimate(x, observed)
    full_baseline = np.zeros_like(pattern.x)
    full_baseline[selected] = baseline
    path = temporary / "sucrose-fixed-background.xye"
    np.savetxt(
        path,
        np.column_stack((pattern.x, full_baseline, np.ones_like(pattern.x))),
        fmt="%.17g",
    )
    return path


def compare_scientific_results(
    case_id: str, phase: dict[str, Any], gsas: dict[str, Any]
) -> dict[str, Any]:
    if phase["sample_count"] != gsas["sample_count"]:
        raise RuntimeError("PhaseSmith and GSAS-II selected different sample counts")
    if phase["reflection_count"] != gsas["reflection_count"]:
        raise RuntimeError("PhaseSmith and GSAS-II generated different reflection counts")
    measurements = {
        "poisson_rwp_delta": abs(phase["poisson_rwp"] - gsas["poisson_rwp"]),
        "profile_correlation_delta": abs(
            phase["profile_correlation"] - gsas["profile_correlation"]
        ),
    }
    limits = {
        "poisson_rwp_delta": CASES[case_id]["maximum_rwp_delta"],
        "profile_correlation_delta": CASES[case_id]["maximum_correlation_delta"],
    }
    checks = {
        name: {"measured": value, "limit": limits[name], "passed": value <= limits[name]}
        for name, value in measurements.items()
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0 or arguments.gsas_cycles <= 0:
        raise ValueError("warmups must be non-negative and repetitions/cycles must be positive")
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")
    data = (
        REPOSITORY_ROOT / "validation" / "data" / arguments.case
        if arguments.data_directory is None
        else arguments.data_directory
    )
    verify_validation_dataset(arguments.case, data)
    phase_result_record, phase_timing = run_phasesmith(
        arguments.case, data, arguments.warmups, arguments.repetitions
    )
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-real-lebail-comparison-") as name:
        gsas_result, gsas_timing, gsas_metadata = run_gsas(arguments, data, Path(name))
    comparison = compare_scientific_results(arguments.case, phase_result_record, gsas_result)
    report = {
        "schema_version": 1,
        "scope": f"{arguments.case}_lebail_workflow",
        "comparison_kind": "independent_real_data_lebail_workflows",
        "workload": {
            "dataset_id": arguments.case,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
        },
        "phasesmith": {
            "build_mode": phasesmith._core.BUILD_MODE,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "result": phase_result_record,
            "timing": phase_timing,
        },
        "gsasii": {
            "revision": PINNED_REVISION,
            **gsas_metadata,
            "result": gsas_result,
            "timing": gsas_timing,
        },
        "cross_implementation_validation": comparison,
        "limitations": [
            (
                "The sucrose workflows share one fixed Smooth Bruckner array and refine one "
                "constant Chebyshev residual from zero. The GSAS-II worker receives that "
                "fixed array as plain prepared input."
            ),
            (
                "Both workflows start from the same explicit U/V/W/X/Y, wavelength, zero, "
                "and symmetric SH/L=0 instrument state; the legacy GSAS file import is "
                "overridden before refinement. Pinned GSAS-II reports its internal 0.0005 "
                "SH/L calculation floor, while PhaseSmith's zero is exactly symmetric."
            ),
        ],
    }
    print(
        f"case={arguments.case} phasesmith_rwp={phase_result_record['poisson_rwp']:.6f} "
        f"gsasii_rwp={gsas_result['poisson_rwp']:.6f} "
        f"phasesmith_corr={phase_result_record['profile_correlation']:.6f} "
        f"gsasii_corr={gsas_result['profile_correlation']:.6f}"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
