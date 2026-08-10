#!/usr/bin/env python3
"""Compare a physical empirical SRM 660c subset with pinned GSAS-II."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import phasesmith
from phasesmith.validation import (
    run_nist_srm660c_parity_workflow,
    verify_validation_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_nist_srm660c.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
LIMITS = {
    "poisson_rwp_delta": 0.01,
    "unit_weight_rwp_delta": 0.02,
    "profile_correlation_delta": 0.005,
    "nist_reference_rwp_delta": 1.0e-12,
    "nist_reference_correlation_delta": 1.0e-12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--specimen", default="100a")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--phasesmith-threads", type=int, default=1)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def compare_scientific_results(
    phasesmith_result: dict[str, Any], gsas_result: dict[str, Any]
) -> dict[str, Any]:
    """Apply strict physical-common-subset gates without masking known drift."""

    for key in ("specimen", "sample_count", "reflection_count", "free_parameter_count"):
        if phasesmith_result[key] != gsas_result[key]:
            raise RuntimeError(f"PhaseSmith and GSAS-II disagree on {key}")
    measurements = {
        "poisson_rwp_delta": abs(phasesmith_result["poisson_rwp"] - gsas_result["poisson_rwp"]),
        "unit_weight_rwp_delta": abs(
            phasesmith_result["unit_weight_rwp"] - gsas_result["unit_weight_rwp"]
        ),
        "profile_correlation_delta": abs(
            phasesmith_result["profile_correlation"] - gsas_result["profile_correlation"]
        ),
        "nist_reference_rwp_delta": abs(
            phasesmith_result["nist_reference_rwp"] - gsas_result["nist_reference_rwp"]
        ),
        "nist_reference_correlation_delta": abs(
            phasesmith_result["nist_reference_correlation"]
            - gsas_result["nist_reference_correlation"]
        ),
    }
    checks = {
        name: {"measured": value, "limit": LIMITS[name], "passed": value <= LIMITS[name]}
        for name, value in measurements.items()
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "failed_checks": failed,
        "checks": checks,
    }


def run_phasesmith(
    data: Path, specimen: str, repetitions: int, execution: phasesmith.ExecutionPolicy
) -> tuple[dict[str, Any], list[float]]:
    records = []
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = run_nist_srm660c_parity_workflow(data, specimen, execution=execution)
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
        records.append(result.to_record())
    stable = [
        {key: value for key, value in record.items() if key != "elapsed_seconds"}
        for record in records
    ]
    if any(record != stable[0] for record in stable[1:]):
        raise RuntimeError("PhaseSmith SRM 660c repetitions are not deterministic")
    return records[-1], timings


def run_gsas(arguments: argparse.Namespace, temporary: Path) -> tuple[dict[str, Any], list[float]]:
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
    reports = []
    timings = []
    for repetition in range(arguments.repetitions):
        report_path = temporary / f"gsas-{repetition}.json"
        command = [
            str(arguments.gsas_python),
            str(WORKER),
            "--gsas-root",
            str(arguments.gsas_root),
            "--data-directory",
            str(arguments.data_directory),
            "--report",
            str(report_path),
            "--specimen",
            arguments.specimen,
            "--cycles",
            str(arguments.cycles),
        ]
        if arguments.binary_dir is not None:
            command.extend(("--binary-dir", str(arguments.binary_dir)))
        started = time.perf_counter_ns()
        process = subprocess.run(command, capture_output=True, text=True, env=environment)
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
        if process.returncode != 0:
            raise RuntimeError(
                "GSAS-II SRM 660c worker failed\n"
                f"stdout:\n{process.stdout[-4000:]}\nstderr:\n{process.stderr[-4000:]}"
            )
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    for report in reports:
        if (
            report.get("schema_version") != 1
            or report.get("implementation") != "GSAS-II"
            or report.get("revision") != PINNED_REVISION
            or report.get("scope") != "nist_srm660c_empirical_common_model"
        ):
            raise RuntimeError("GSAS-II SRM 660c worker returned invalid provenance")
    stable_keys = ("recipe", "input_sha256", "result", "stage_rwp_percent")
    for report in reports[1:]:
        if any(report[key] != reports[0][key] for key in stable_keys):
            raise RuntimeError("GSAS-II SRM 660c repetitions are not deterministic")
    return reports[-1]["result"], timings


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError("set --gsas-python and --gsas-root")
    if arguments.repetitions <= 0 or arguments.cycles <= 0:
        raise ValueError("repetitions and cycles must be positive")
    if arguments.phasesmith_threads < 0:
        raise ValueError("--phasesmith-threads must be non-negative")
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")
    arguments.data_directory = (
        REPOSITORY_ROOT / "validation" / "data" / "nist-srm660c-lab6-xray"
        if arguments.data_directory is None
        else arguments.data_directory
    )
    verify_validation_dataset("nist-srm660c-lab6-xray", arguments.data_directory)
    execution = phasesmith.ExecutionPolicy(
        threads=None if arguments.phasesmith_threads == 0 else arguments.phasesmith_threads
    )
    phase_result, phase_timings = run_phasesmith(
        arguments.data_directory, arguments.specimen, arguments.repetitions, execution
    )
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-nist660c-comparison-") as name:
        gsas_result, gsas_timings = run_gsas(arguments, Path(name))
    validation = compare_scientific_results(phase_result, gsas_result)
    report = {
        "schema_version": 1,
        "scope": "nist_srm660c_empirical_common_model",
        "comparison_kind": "physical_empirical_common_subset_expected_holdout",
        "workload": {
            "dataset_id": "nist-srm660c-lab6-xray",
            "specimen": arguments.specimen,
            "repetitions": arguments.repetitions,
            "phasesmith_threads": execution.threads,
        },
        "phasesmith": {
            "build_mode": phasesmith._core.BUILD_MODE,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "result": phase_result,
            "timings_ms": phase_timings,
        },
        "gsasii": {
            "revision": PINNED_REVISION,
            "result": gsas_result,
            "timings_ms": gsas_timings,
        },
        "cross_implementation_validation": validation,
        "limitations": [
            "The NIST fundamental-parameters Cu spectrum and optics model is deliberately omitted.",
            (
                "U/V/W remain fixed at a positive-variance state; the unconstrained "
                "GSAS-II optimum is nonphysical at high angle."
            ),
            "Microstrain is fixed at zero because the unconstrained GSAS-II optimum is negative.",
            (
                "The common model refines zero, isotropic size, two Uiso values, "
                "scale, and twelve Chebyshev terms."
            ),
        ],
    }
    print(
        f"scope={report['scope']} specimen={arguments.specimen} repetitions={arguments.repetitions}"
    )
    print(
        f"phasesmith_rwp={100 * phase_result['poisson_rwp']:.3f}% "
        f"gsasii_rwp={100 * gsas_result['poisson_rwp']:.3f}% "
        f"status={validation['status']}"
    )
    print(
        f"nist_reference_rwp={100 * phase_result['nist_reference_rwp']:.3f}% "
        f"failed_checks={','.join(validation['failed_checks']) or 'none'}"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
