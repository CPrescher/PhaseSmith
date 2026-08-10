#!/usr/bin/env python3
"""Compare complete native QARR 1g/1h workflows with pinned external GSAS-II."""

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
from phasesmith.validation import (
    run_qarr_gsasii_parity_workflow,
    verify_validation_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_qarr.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "iucr_qarr_1g_native_workflow"
PHASE_NAMES = ("Al2O3", "ZnO", "CaF2")
CROSS_IMPLEMENTATION_LIMITS = {
    "maximum_phase_fraction_delta": 0.005,
    "poisson_rwp_delta": 0.005,
    "unit_weight_rwp_delta": 0.005,
    "profile_correlation_delta": 0.002,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
    )
    parser.add_argument("--sample", choices=("1g", "1h"), default="1g")
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="discarded complete runs; one warms GSAS-II's process-external font cache",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--phasesmith-threads",
        type=int,
        default=1,
        help="PhaseSmith worker threads; zero selects available logical CPUs",
    )
    parser.add_argument("--gsas-cycles", type=int, default=8)
    parser.add_argument(
        "--gsas-fcj",
        choices=("instrument", "below-minimum"),
        default="instrument",
    )
    parser.add_argument("--gsas-sample-broadening", choices=("refine", "fixed"), default="refine")
    parser.add_argument("--gsas-displacement", choices=("refine", "fixed"), default="refine")
    parser.add_argument(
        "--gsas-anisotropic",
        choices=("cif", "trace-mean-isotropic"),
        default="trace-mean-isotropic",
    )
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)]


def timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "timings_ms": values,
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
    }


def run_phasesmith(
    data_directory: Path,
    warmups: int,
    repetitions: int,
    execution: phasesmith.ExecutionPolicy,
    sample: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def run_once() -> Any:
        return run_qarr_gsasii_parity_workflow(
            data_directory,
            sample,
            execution=execution,
        )

    for _ in range(warmups):
        run_once()
    reports = []
    wall_ms = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        report = run_once()
        wall_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        reports.append(report)
    results = [report.to_record() for report in reports]
    stable = [
        {key: value for key, value in result.items() if key != "elapsed_seconds"}
        for result in results
    ]
    if any(result != stable[0] for result in stable[1:]):
        raise RuntimeError("PhaseSmith QARR repetitions are not deterministic")
    return results[-1], {
        "reported_workflow": timing_summary(
            [1_000.0 * report.elapsed_seconds for report in reports]
        ),
        "driver_wall": timing_summary(wall_ms),
        "stage_timing_available": False,
    }


def run_gsas_once(
    arguments: argparse.Namespace, report_path: Path, environment: dict[str, str]
) -> tuple[dict[str, Any], float]:
    command = [
        str(arguments.gsas_python),
        str(WORKER),
        "--gsas-root",
        str(arguments.gsas_root),
        "--data-directory",
        str(arguments.data_directory),
        "--report",
        str(report_path),
        "--cycles",
        str(arguments.gsas_cycles),
        "--sample",
        arguments.sample,
        "--fcj",
        arguments.gsas_fcj,
        "--sample-broadening",
        arguments.gsas_sample_broadening,
        "--displacement",
        arguments.gsas_displacement,
        "--anisotropic",
        arguments.gsas_anisotropic,
    ]
    if arguments.binary_dir is not None:
        command.extend(["--binary-dir", str(arguments.binary_dir)])
    started = time.perf_counter_ns()
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
    )
    process_ms = (time.perf_counter_ns() - started) / 1.0e6
    if process.returncode != 0:
        raise RuntimeError(
            "GSAS-II QARR worker failed\n"
            f"stdout:\n{process.stdout[-4000:]}\n"
            f"stderr:\n{process.stderr[-4000:]}"
        )
    return json.loads(report_path.read_text(encoding="utf-8")), process_ms


def validate_gsas_reports(reports: list[dict[str, Any]], sample: str = "1g") -> None:
    stable_keys = (
        "recipe",
        "input_sha256",
        "oracle_behavior",
        "result",
        "stage_rwp_percent",
    )
    for report in reports:
        if (
            report.get("schema_version") != 1
            or report.get("implementation") != "GSAS-II"
            or report.get("revision") != PINNED_REVISION
            or report.get("scope") != f"iucr_qarr_{sample}_native_workflow"
        ):
            raise RuntimeError("GSAS-II QARR worker returned invalid provenance")
    for report in reports[1:]:
        for key in stable_keys:
            if report[key] != reports[0][key]:
                raise RuntimeError(f"GSAS-II QARR repetitions disagree for {key}")


def compare_scientific_results(
    phasesmith_result: dict[str, Any], gsas_result: dict[str, Any]
) -> dict[str, Any]:
    """Numerically gate the two matched common-model native workflows."""

    if phasesmith_result["sample_count"] != gsas_result["sample_count"]:
        raise RuntimeError("PhaseSmith and GSAS-II used different QARR sample counts")
    if phasesmith_result["reflection_count"] != gsas_result["reflection_count"]:
        raise RuntimeError("PhaseSmith and GSAS-II generated different QARR reflection counts")
    phase_deltas = {
        name: abs(
            phasesmith_result["weight_fractions"][name] - gsas_result["weight_fractions"][name]
        )
        for name in PHASE_NAMES
    }
    measurements = {
        "maximum_phase_fraction_delta": max(phase_deltas.values()),
        "poisson_rwp_delta": abs(phasesmith_result["poisson_rwp"] - gsas_result["poisson_rwp"]),
        "unit_weight_rwp_delta": abs(
            phasesmith_result["unit_weight_rwp"] - gsas_result["unit_weight_rwp"]
        ),
        "profile_correlation_delta": abs(
            phasesmith_result["profile_correlation"] - gsas_result["profile_correlation"]
        ),
    }
    checks = {
        name: {
            "measured": measurement,
            "limit": CROSS_IMPLEMENTATION_LIMITS[name],
            "passed": measurement <= CROSS_IMPLEMENTATION_LIMITS[name],
        }
        for name, measurement in measurements.items()
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "failed_checks": failed,
        "phase_fraction_deltas": phase_deltas,
        "checks": checks,
    }


def run_gsas(
    arguments: argparse.Namespace, temporary: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
    for warmup in range(arguments.warmups):
        run_gsas_once(arguments, temporary / f"warmup-{warmup}.json", environment)
    reports = []
    process_ms = []
    for repetition in range(arguments.repetitions):
        report, elapsed = run_gsas_once(
            arguments, temporary / f"result-{repetition}.json", environment
        )
        reports.append(report)
        process_ms.append(elapsed)
    validate_gsas_reports(reports, arguments.sample)
    stage_names = tuple(reports[-1]["timing_ms"]["stages"])
    timing = {
        "import": timing_summary([report["import_ms"] for report in reports]),
        "setup": timing_summary([report["timing_ms"]["setup"] for report in reports]),
        "stages": {
            name: timing_summary([report["timing_ms"]["stages"][name] for report in reports])
            for name in stage_names
        },
        "finalization": timing_summary([report["timing_ms"]["finalization"] for report in reports]),
        "total_workflow": timing_summary(
            [report["timing_ms"]["total_workflow"] for report in reports]
        ),
        "external_process": timing_summary(process_ms),
    }
    return (
        reports[-1]["result"],
        timing,
        {
            "recipe": reports[-1]["recipe"],
            "input_sha256": reports[-1]["input_sha256"],
            "oracle_behavior": reports[-1]["oracle_behavior"],
            "stage_rwp_percent": reports[-1]["stage_rwp_percent"],
            "python_version": reports[-1]["python_version"],
            "numpy_version": reports[-1]["numpy_version"],
            "platform": reports[-1]["platform"],
        },
    )


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
    dataset_id = f"iucr-qarr-{arguments.sample}"
    data_directory = (
        REPOSITORY_ROOT / "validation" / "data" / dataset_id
        if arguments.data_directory is None
        else arguments.data_directory
    )
    arguments.data_directory = data_directory
    verify_validation_dataset(dataset_id, data_directory)

    if arguments.phasesmith_threads < 0:
        raise ValueError("--phasesmith-threads must be non-negative")
    execution = phasesmith.ExecutionPolicy(
        threads=None if arguments.phasesmith_threads == 0 else arguments.phasesmith_threads
    )
    phase_result_record, phase_timing = run_phasesmith(
        data_directory,
        arguments.warmups,
        arguments.repetitions,
        execution,
        arguments.sample,
    )
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-qarr-comparison-") as name:
        gsas_result, gsas_timing, gsas_metadata = run_gsas(arguments, Path(name))

    cross_validation = compare_scientific_results(phase_result_record, gsas_result)
    ratio = (
        gsas_timing["total_workflow"]["median_ms"] / phase_timing["reported_workflow"]["median_ms"]
    )
    report = {
        "schema_version": 1,
        "scope": f"iucr_qarr_{arguments.sample}_native_workflow",
        "comparison_kind": "native_workflows_matched_common_parameterizations",
        "workload": {
            "dataset_id": dataset_id,
            "samples": phase_result_record["sample_count"],
            "phases": list(PHASE_NAMES),
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "phasesmith_threads": execution.threads,
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
        "median_total_workflow_ratio_gsasii_over_phasesmith": ratio,
        "cross_implementation_validation": cross_validation,
        "limitations": [
            (
                "Both workflows use the explicit trace-mean isotropic ablation for CIF "
                "anisotropic displacement tensors."
            ),
            (
                "PhaseSmith uses the published continuous equal-height FCJ mapping; "
                "GSAS-II uses its pinned discretized one-parameter SH/L implementation."
            ),
            "The optimizers are native staged workflows, not a same-kernel benchmark.",
        ],
    }
    print(f"scope=iucr_qarr_{arguments.sample}_native_workflow repetitions={arguments.repetitions}")
    for name in PHASE_NAMES:
        print(
            f"phase={name} phasesmith={100 * phase_result_record['weight_fractions'][name]:.3f}% "
            f"gsasii={100 * gsas_result['weight_fractions'][name]:.3f}%"
        )
    print(
        f"phasesmith_rwp={100 * phase_result_record['poisson_rwp']:.3f}% "
        f"gsasii_rwp={100 * gsas_result['poisson_rwp']:.3f}%"
    )
    print(
        f"phasesmith_median_ms={phase_timing['reported_workflow']['median_ms']:.3f} "
        f"gsasii_median_ms={gsas_timing['total_workflow']['median_ms']:.3f} "
        f"ratio_gsasii_over_phasesmith={ratio:.3f}x"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
