#!/usr/bin/env python3
"""Compare structural multi-bank TOF refinement on pinned LANL nickel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from phasesmith.validation import run_nickel_tof_validation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle/scripts/benchmark_nickel_tof_structural.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
BANKS = (2, 3, 4)
LIMITS = {
    "native_rwp": 0.04,
    "oracle_rwp": 0.04,
    "native_minimum_correlation": 0.995,
    "oracle_minimum_correlation": 0.995,
    "cell_delta_angstrom": 2.0e-3,
    "u_iso_delta_angstrom2": 5.0e-3,
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
        default=REPOSITORY_ROOT / "validation/data/lanl-nickel-tof",
    )
    parser.add_argument("--gsas-cycles", type=int, default=20)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def require_paths(arguments: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("gsas_python", "gsas_root", "binary_dir")
        if getattr(arguments, name) is None
    ]
    if missing:
        raise RuntimeError(f"missing required GSAS-II paths: {', '.join(missing)}")


def run_oracle(arguments: argparse.Namespace, directory: Path) -> dict[str, Any]:
    report_path = directory / "gsasii-structural.json"
    archive_path = directory / "gsasii-structural.npz"
    command = [
        str(arguments.gsas_python),
        str(WORKER),
        "--gsas-root",
        str(arguments.gsas_root),
        "--binary-dir",
        str(arguments.binary_dir),
        "--data-directory",
        str(arguments.data_directory),
        "--report",
        str(report_path),
        "--archive",
        str(archive_path),
        "--cycles",
        str(arguments.gsas_cycles),
    ]
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(directory / "matplotlib")
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    if process.returncode != 0:
        raise RuntimeError(
            "GSAS-II nickel structural worker failed\n"
            f"stdout:\n{process.stdout[-4000:]}\n"
            f"stderr:\n{process.stderr[-4000:]}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("revision") != PINNED_REVISION:
        raise RuntimeError("oracle report does not carry the pinned revision")
    with np.load(archive_path) as arrays:
        expected = {
            f"bank_{bank}_{name}"
            for bank in BANKS
            for name in (
                "x_us",
                "observed_y",
                "calculated_y",
                "background_y",
                "weight",
                "reflection_list",
            )
        }
        if set(arrays.files) != expected:
            raise RuntimeError("oracle archive does not match the structural plain-array schema")
        if any(arrays[f"bank_{bank}_reflection_list"].ndim != 2 for bank in BANKS):
            raise RuntimeError("oracle reflection lists must be two-dimensional plain arrays")
        if any(not np.isfinite(arrays[name]).all() for name in arrays.files):
            raise RuntimeError("oracle archive contains non-finite arrays")
    return report


def check_measurement(report: Any, check_id: str) -> float:
    values = [check.measured for check in report.checks if check.check_id == check_id]
    if len(values) != 1 or values[0] is None:
        raise RuntimeError(f"PhaseSmith report has no {check_id!r} measurement")
    return float(values[0])


def compare(report: dict[str, Any], data: Path) -> dict[str, Any]:
    oracle = report["result"]
    native_report = run_nickel_tof_validation(data)
    native = {
        "status": native_report.status,
        "sample_count": 3 * native_report.sample_count,
        "joint_poisson_rwp": check_measurement(native_report, "tof_nickel_structural_fit"),
        "minimum_profile_correlation": check_measurement(
            native_report, "tof_nickel_structural_correlation"
        ),
        "cell_angstrom": check_measurement(native_report, "tof_nickel_structural_lattice"),
        "u_iso_angstrom2": check_measurement(native_report, "tof_nickel_structural_u_iso"),
    }
    measurements = {
        "native_rwp": native["joint_poisson_rwp"],
        "oracle_rwp": oracle["joint_poisson_rwp"],
        "native_minimum_correlation": native["minimum_profile_correlation"],
        "oracle_minimum_correlation": oracle["minimum_profile_correlation"],
        "cell_delta_angstrom": abs(native["cell_angstrom"] - oracle["cell_angstrom"]),
        "u_iso_delta_angstrom2": abs(native["u_iso_angstrom2"] - oracle["u_iso_angstrom2"]),
    }
    checks = {}
    for name, limit in LIMITS.items():
        measured = measurements[name]
        passed = measured >= limit if name.endswith("minimum_correlation") else measured <= limit
        checks[name] = {"measured": measured, "limit": limit, "passed": passed}
    identity_checks = {
        "bank_count": oracle["bank_count"] == len(BANKS),
        "sample_count_conventions": oracle["sample_count"] == 13_290
        and native["sample_count"] == 13_293,
        "native_status": native["status"] == "passed",
        "oracle_parameter_count": oracle["free_parameter_count"] == 44,
    }
    return {
        "schema_version": 1,
        "status": "passed"
        if all(identity_checks.values()) and all(check["passed"] for check in checks.values())
        else "failed",
        "oracle_revision": report["revision"],
        "identity_checks": identity_checks,
        "checks": checks,
        "workflows": {"GSAS-II": oracle, "PhaseSmith": native},
        "workflow_context": {
            "rwp_delta": abs(native["joint_poisson_rwp"] - oracle["joint_poisson_rwp"]),
            "rwp_is_a_cross_gate": False,
            "reason": "different accepted solvers and background parameterizations",
        },
        "interpretation": (
            "Both implementations refine the same structural families and compare shared cell "
            "and Ni Uiso directly. Each profile must independently clear its quality gate; Rwp "
            "equality is not required across distinct background and optimizer contracts."
        ),
    }


def main() -> None:
    arguments = parse_args()
    require_paths(arguments)
    with tempfile.TemporaryDirectory(prefix="phasesmith-nickel-structural-compare-") as temporary:
        oracle = run_oracle(arguments, Path(temporary))
        result = compare(oracle, arguments.data_directory.resolve())
    if arguments.json_output is not None:
        arguments.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] != "passed":
        failed = [name for name, check in result["checks"].items() if not check["passed"]]
        failed.extend(name for name, passed in result["identity_checks"].items() if not passed)
        raise SystemExit(f"failed comparisons: {', '.join(failed)}")


if __name__ == "__main__":
    main()
