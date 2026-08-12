#!/usr/bin/env python3
"""Compare native multi-bank TOF geometry with pinned GSAS-II on LANL nickel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import phasesmith
from phasesmith.validation import run_nickel_tof_validation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle/scripts/benchmark_nickel_tof_multibank.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
BANKS = (2, 3, 4)
NICKEL_CELL_ANGSTROM = 3.5234
LIMITS = {
    "position_max_abs_us": 1.0e-9,
    "sigma2_max_abs_us2": 2.0e-10,
    "alpha_max_abs_per_us": 2.0e-14,
    "beta_max_abs_per_us": 2.0e-14,
    "reconstructed_pattern_relative_l2": 1.5e-2,
    "reconstructed_pattern_minimum_correlation": 0.9999,
    "native_oracle_cell_delta_angstrom": 5.0e-4,
    "native_reference_cell_delta_angstrom": 5.0e-4,
    "oracle_reference_cell_delta_angstrom": 7.0e-4,
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


def phase_instrument(values: dict[str, float]) -> phasesmith.TofInstrument:
    return phasesmith.TofInstrument(
        zero_us=values["Zero"],
        difc_us_per_angstrom=values["difC"],
        difa_us_per_angstrom2=values["difA"],
        difb_us_angstrom=values["difB"],
        alpha_coefficient=values["alpha"],
        beta0_per_us=values["beta-0"],
        beta1_angstrom4_per_us=values["beta-1"],
        betaq_angstrom2_per_us=0.0,
        sigma0_us2=values["sig-0"],
        sigma1_us2_per_angstrom2=values["sig-1"],
        sigma2_us2_per_angstrom4=values["sig-2"],
        sigmaq_us2_per_angstrom=0.0,
        x_us_per_angstrom=values["X"],
        y_us_per_angstrom2=values["Y"],
        z_us=values["Z"],
    )


def check_measurement(report: Any, check_id: str) -> float:
    values = [check.measured for check in report.checks if check.check_id == check_id]
    if len(values) != 1 or values[0] is None:
        raise RuntimeError(f"PhaseSmith report has no {check_id!r} measurement")
    return float(values[0])


def run_oracle(arguments: argparse.Namespace, directory: Path) -> tuple[dict[str, Any], Any]:
    report_path = directory / "gsasii.json"
    archive_path = directory / "gsasii.npz"
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
            "GSAS-II nickel multi-bank worker failed\n"
            f"stdout:\n{process.stdout[-4000:]}\n"
            f"stderr:\n{process.stderr[-4000:]}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("revision") != PINNED_REVISION:
        raise RuntimeError("oracle report does not carry the pinned revision")
    return report, np.load(archive_path)


def compare(report: dict[str, Any], arrays: Any, data: Path) -> dict[str, Any]:
    gsas = report["result"]
    parameter_errors: dict[str, float] = {
        "position_max_abs_us": 0.0,
        "sigma2_max_abs_us2": 0.0,
        "alpha_max_abs_per_us": 0.0,
        "beta_max_abs_per_us": 0.0,
    }
    reconstruction = {}
    for bank_result in gsas["banks"]:
        bank = int(bank_result["bank"])
        if bank not in BANKS:
            raise RuntimeError(f"unexpected oracle bank {bank}")
        reflections = arrays[f"bank_{bank}_reflection_list"]
        instrument = phase_instrument(bank_result["instrument"])
        parameters = phasesmith.tof_profile_parameters(reflections[:, 4], instrument)
        errors = {
            "position_max_abs_us": float(
                np.max(np.abs(parameters.position_us - reflections[:, 5]))
            ),
            "sigma2_max_abs_us2": float(
                np.max(np.abs(parameters.gaussian_variance_us2 - reflections[:, 6]))
            ),
            "alpha_max_abs_per_us": float(
                np.max(np.abs(parameters.alpha_per_us - reflections[:, 12]))
            ),
            "beta_max_abs_per_us": float(
                np.max(np.abs(parameters.beta_per_us - reflections[:, 13]))
            ),
        }
        parameter_errors = {
            name: max(parameter_errors[name], value) for name, value in errors.items()
        }
        x_us = arrays[f"bank_{bank}_x_us"]
        peak_only = arrays[f"bank_{bank}_calculated_y"] - arrays[f"bank_{bank}_background_y"]
        reconstructed = phasesmith.accumulate_tof(
            x_us,
            reflections[:, 4],
            reflections[:, 8] * reflections[:, 11],
            instrument,
            support_fwhm=20.0,
        ).y
        reconstruction[str(bank)] = {
            "relative_l2": float(
                np.linalg.norm(reconstructed - peak_only) / np.linalg.norm(peak_only)
            ),
            "correlation": float(np.corrcoef(reconstructed, peak_only)[0, 1]),
        }

    native = run_nickel_tof_validation(data)
    native_result = {
        "status": native.status,
        "sample_count": 3 * native.sample_count,
        "joint_rwp": check_measurement(native, "tof_nickel_multibank_fit"),
        "cell_angstrom": check_measurement(native, "tof_nickel_multibank_lattice"),
    }
    maximum_reconstruction_l2 = max(value["relative_l2"] for value in reconstruction.values())
    minimum_reconstruction_correlation = min(
        value["correlation"] for value in reconstruction.values()
    )
    measurements = parameter_errors | {
        "reconstructed_pattern_relative_l2": maximum_reconstruction_l2,
        "reconstructed_pattern_minimum_correlation": minimum_reconstruction_correlation,
        "native_oracle_cell_delta_angstrom": abs(
            native_result["cell_angstrom"] - gsas["cell_angstrom"]
        ),
        "native_reference_cell_delta_angstrom": abs(
            native_result["cell_angstrom"] - NICKEL_CELL_ANGSTROM
        ),
        "oracle_reference_cell_delta_angstrom": abs(gsas["cell_angstrom"] - NICKEL_CELL_ANGSTROM),
    }
    checks = {}
    for name, limit in LIMITS.items():
        value = measurements[name]
        passed = value >= limit if name.endswith("minimum_correlation") else value <= limit
        checks[name] = {"measured": value, "limit": limit, "passed": passed}
    identity_checks = {
        "bank_count": gsas["bank_count"] == len(BANKS),
        "sample_count_conventions": (
            gsas["sample_count"] == 13_290 and native_result["sample_count"] == 13_293
        ),
        "native_status": native_result["status"] == "passed",
    }
    return {
        "schema_version": 1,
        "status": "passed"
        if all(identity_checks.values()) and all(check["passed"] for check in checks.values())
        else "failed",
        "oracle_revision": report["revision"],
        "identity_checks": identity_checks,
        "checks": checks,
        "parameter_errors": parameter_errors,
        "same_extracted_intensity_patterns": reconstruction,
        "workflow_context": {
            "joint_rwp_delta": abs(native_result["joint_rwp"] - gsas["joint_poisson_rwp"]),
            "rwp_is_a_cross_gate": False,
            "reason": ("independent Le Bail intensity redistribution and background decomposition"),
        },
        "workflows": {"GSAS-II": gsas, "PhaseSmith": native_result},
        "interpretation": (
            "Reflection-parameter and same-extracted-intensity checks are like-for-like. "
            "Workflow deltas compare one shared cubic cell and three local Zero terms, but "
            "the programs retain independent optimizers and background decompositions."
        ),
    }


def main() -> None:
    arguments = parse_args()
    require_paths(arguments)
    with tempfile.TemporaryDirectory(prefix="phasesmith-nickel-compare-") as temporary:
        report, arrays = run_oracle(arguments, Path(temporary))
        result = compare(report, arrays, arguments.data_directory.resolve())
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
