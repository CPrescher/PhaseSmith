#!/usr/bin/env python3
"""Compare complete native QARR workflows with pinned external GSAS-II."""

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
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_qarr.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "iucr_qarr_1g_native_workflow"
PHASE_NAMES = ("Al2O3", "ZnO", "CaF2")
FRACTION_PATTERN = re.compile(r"(Al2O3|ZnO|CaF2)=([0-9]+(?:\.[0-9]+)?)%")


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
        default=REPOSITORY_ROOT / "validation" / "data" / "iucr-qarr-1g",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="discarded complete runs; one warms GSAS-II's process-external font cache",
    )
    parser.add_argument("--repetitions", type=int, default=3)
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
        default="cif",
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


def phase_fractions(report: Any) -> dict[str, float]:
    matches = dict(FRACTION_PATTERN.findall(" ".join(report.notes)))
    if set(matches) != set(PHASE_NAMES):
        raise RuntimeError("PhaseSmith QARR report does not contain structured phase fractions")
    return {name: float(matches[name]) / 100.0 for name in PHASE_NAMES}


def check_measurement(report: Any, check_id: str) -> float:
    matches = [check.measured for check in report.checks if check.check_id == check_id]
    if len(matches) != 1 or matches[0] is None or not np.isfinite(matches[0]):
        raise RuntimeError(f"PhaseSmith QARR report has no finite {check_id} measurement")
    return float(matches[0])


def phase_result(report: Any) -> dict[str, Any]:
    if report.status != "passed":
        raise RuntimeError(f"PhaseSmith QARR validation returned {report.status!r}")
    fractions = phase_fractions(report)
    return {
        "sample_count": report.sample_count,
        "reflection_count": report.reflection_count,
        "weight_fractions": fractions,
        "maximum_weight_fraction_error": check_measurement(report, "qpa_weight_fraction"),
        "poisson_rwp": check_measurement(report, "poisson_rwp"),
        "unit_weight_rwp": check_measurement(report, "unit_weight_rwp"),
        "profile_correlation": check_measurement(report, "profile_correlation"),
        "termination_notes": [note for note in report.notes if note.startswith("Stage ")],
        "approximations": list(report.notes[5:]),
    }


def run_phasesmith(
    data_directory: Path, warmups: int, repetitions: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _ in range(warmups):
        run_qarr_1g_validation(data_directory)
    reports = []
    wall_ms = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        report = run_qarr_1g_validation(data_directory)
        wall_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        reports.append(report)
    results = [phase_result(report) for report in reports]
    if any(result != results[0] for result in results[1:]):
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


def validate_gsas_reports(reports: list[dict[str, Any]]) -> None:
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
            or report.get("scope") != SCOPE
        ):
            raise RuntimeError("GSAS-II QARR worker returned invalid provenance")
    for report in reports[1:]:
        for key in stable_keys:
            if report[key] != reports[0][key]:
                raise RuntimeError(f"GSAS-II QARR repetitions disagree for {key}")


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
    validate_gsas_reports(reports)
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
    verify_validation_dataset("iucr-qarr-1g", arguments.data_directory)

    phase_result_record, phase_timing = run_phasesmith(
        arguments.data_directory, arguments.warmups, arguments.repetitions
    )
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-qarr-comparison-") as name:
        gsas_result, gsas_timing, gsas_metadata = run_gsas(arguments, Path(name))

    if phase_result_record["sample_count"] != gsas_result["sample_count"]:
        raise RuntimeError("PhaseSmith and GSAS-II used different QARR sample counts")
    ratio = (
        gsas_timing["total_workflow"]["median_ms"] / phase_timing["reported_workflow"]["median_ms"]
    )
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "comparison_kind": "native_workflows_not_matched_parameterizations",
        "workload": {
            "dataset_id": "iucr-qarr-1g",
            "samples": phase_result_record["sample_count"],
            "phases": list(PHASE_NAMES),
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
        "median_total_workflow_ratio_gsasii_over_phasesmith": ratio,
        "limitations": [
            (
                "PhaseSmith uses fixed Smooth Bruckner background; "
                "GSAS-II refines ten Chebyshev terms."
            ),
            "PhaseSmith approximates anisotropic displacement by trace-mean Uiso.",
            (
                "PhaseSmith uses the published continuous equal-height FCJ mapping; "
                "GSAS-II uses its pinned discretized one-parameter SH/L implementation."
            ),
            "This is a complete native-workflow comparison, not a same-kernel benchmark.",
        ],
    }
    print(f"scope={SCOPE} repetitions={arguments.repetitions}")
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
