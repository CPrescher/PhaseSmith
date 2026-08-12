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
    NIST_SRM660C_MATCHED_SH_OVER_L,
    NIST_SRM660C_STRESS_SH_OVER_L,
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
CASES = {
    "matched-small-fcj": {
        "sh_over_l": NIST_SRM660C_MATCHED_SH_OVER_L,
        "expected_cross_status": "passed",
        "expected_failed_checks": (),
        "comparison_kind": "matched_empirical_common_subset",
    },
    "large-fcj-stress": {
        "sh_over_l": NIST_SRM660C_STRESS_SH_OVER_L,
        "expected_cross_status": "failed",
        "expected_failed_checks": (
            "poisson_rwp_delta",
            "unit_weight_rwp_delta",
            "profile_correlation_delta",
        ),
        "comparison_kind": "large_asymmetry_expected_holdout",
    },
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
    parser.add_argument("--case", choices=(*CASES, "both"), default="both")
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

    for key in (
        "specimen",
        "sample_count",
        "reflection_count",
        "free_parameter_count",
        "sh_over_l",
    ):
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
    data: Path,
    specimen: str,
    repetitions: int,
    execution: phasesmith.ExecutionPolicy,
    sh_over_l: float,
) -> tuple[dict[str, Any], list[float]]:
    records = []
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = run_nist_srm660c_parity_workflow(
            data, specimen, execution=execution, sh_over_l=sh_over_l
        )
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
        records.append(result.to_record())
    stable = [
        {key: value for key, value in record.items() if key != "elapsed_seconds"}
        for record in records
    ]
    if any(record != stable[0] for record in stable[1:]):
        raise RuntimeError("PhaseSmith SRM 660c repetitions are not deterministic")
    return records[-1], timings


def run_gsas(
    arguments: argparse.Namespace, temporary: Path, sh_over_l: float
) -> tuple[dict[str, Any], list[float]]:
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
            "--sh-over-l",
            str(sh_over_l),
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
    selected_cases = tuple(CASES) if arguments.case == "both" else (arguments.case,)
    case_reports = {}
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-nist660c-comparison-") as name:
        temporary = Path(name)
        for case_name in selected_cases:
            case_temporary = temporary / case_name
            case_temporary.mkdir()
            case = CASES[case_name]
            sh_over_l = float(case["sh_over_l"])
            phase_result, phase_timings = run_phasesmith(
                arguments.data_directory,
                arguments.specimen,
                arguments.repetitions,
                execution,
                sh_over_l,
            )
            gsas_result, gsas_timings = run_gsas(arguments, case_temporary, sh_over_l)
            validation = compare_scientific_results(phase_result, gsas_result)
            expected_cross_status = str(case["expected_cross_status"])
            expected_failed_checks = list(case["expected_failed_checks"])
            expectation_met = (
                validation["status"] == expected_cross_status
                and validation["failed_checks"] == expected_failed_checks
            )
            case_reports[case_name] = {
                "status": "passed" if expectation_met else "failed",
                "comparison_kind": case["comparison_kind"],
                "expected_cross_status": expected_cross_status,
                "expected_failed_checks": expected_failed_checks,
                "phasesmith": {
                    "result": phase_result,
                    "timings_ms": phase_timings,
                },
                "gsasii": {
                    "result": gsas_result,
                    "timings_ms": gsas_timings,
                },
                "cross_implementation_validation": validation,
            }
    report = {
        "schema_version": 2,
        "scope": "nist_srm660c_fcj_parity_and_stress",
        "status": (
            "passed"
            if all(case_report["status"] == "passed" for case_report in case_reports.values())
            else "failed"
        ),
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
        },
        "gsasii": {
            "revision": PINNED_REVISION,
        },
        "cases": case_reports,
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
            (
                "The matched case checks SH/L=0.002 parity. SH/L=0.02 is an expected-failure "
                "stress test of the continuous and discretized FCJ implementations."
            ),
        ],
    }
    print(
        f"scope={report['scope']} specimen={arguments.specimen} repetitions={arguments.repetitions}"
    )
    for case_name, case_report in case_reports.items():
        phase_result = case_report["phasesmith"]["result"]
        gsas_result = case_report["gsasii"]["result"]
        validation = case_report["cross_implementation_validation"]
        print(
            f"case={case_name} phasesmith_rwp={100 * phase_result['poisson_rwp']:.3f}% "
            f"gsasii_rwp={100 * gsas_result['poisson_rwp']:.3f}% "
            f"cross_status={validation['status']} expectation={case_report['status']}"
        )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
