#!/usr/bin/env python3
"""Compare PhaseSmith TOF behavior with pinned GSAS-II on real POWGEN data."""

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
from phasesmith.validation import run_powgen_tof_validation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_powgen_tof.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
LIMITS = {
    "position_max_abs_us": 1.0e-10,
    "sigma2_max_abs_us2": 2.0e-11,
    "alpha_max_abs_per_us": 2.0e-15,
    "beta_max_abs_per_us": 2.0e-15,
    "selected_profile_normalized_max_error": 2.0e-3,
    "reconstructed_pattern_relative_l2": 6.0e-3,
    "reconstructed_pattern_minimum_correlation": 0.99999,
    "native_workflow_rwp_delta": 0.03,
    "native_workflow_profile_correlation_delta": 0.02,
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
        default=REPOSITORY_ROOT / "validation/data/powgen-lab6-tof-calibration",
    )
    parser.add_argument("--gsas-cycles", type=int, default=8)
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
            "GSAS-II POWGEN worker failed\n"
            f"stdout:\n{process.stdout[-4000:]}\n"
            f"stderr:\n{process.stderr[-4000:]}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("revision") != PINNED_REVISION:
        raise RuntimeError("oracle report does not carry the pinned revision")
    return report, np.load(archive_path)


def compare(report: dict[str, Any], arrays: Any, data: Path) -> dict[str, Any]:
    gsas = report["result"]
    instrument = phase_instrument(gsas["instrument"])
    reflections = arrays["reflection_list"]
    parameters = phasesmith.tof_profile_parameters(reflections[:, 4], instrument)
    parameter_errors = {
        "position_max_abs_us": float(np.max(np.abs(parameters.position_us - reflections[:, 5]))),
        "sigma2_max_abs_us2": float(
            np.max(np.abs(parameters.gaussian_variance_us2 - reflections[:, 6]))
        ),
        "alpha_max_abs_per_us": float(np.max(np.abs(parameters.alpha_per_us - reflections[:, 12]))),
        "beta_max_abs_per_us": float(np.max(np.abs(parameters.beta_per_us - reflections[:, 13]))),
    }

    profile_errors = {}
    for d_spacing in (0.4, 1.0, 4.0):
        key = str(d_spacing).replace(".", "_")
        x = arrays[f"profile_{key}_x"]
        oracle = arrays[f"profile_{key}_y"]
        derived = phasesmith.tof_profile_parameters([d_spacing], instrument)
        native = phasesmith.profile_tof(
            x,
            derived.position_us[0],
            derived.alpha_per_us[0],
            derived.beta_per_us[0],
            derived.gaussian_fwhm_us[0],
            derived.lorentzian_fwhm_us[0],
        ).value
        profile_errors[f"d={d_spacing:g}"] = {
            "normalized_max_error": float(np.max(np.abs(native - oracle)) / np.max(np.abs(oracle))),
            "correlation": float(np.corrcoef(native, oracle)[0, 1]),
        }

    peak_only = arrays["calculated_y"] - arrays["background_y"]
    integrated_intensities = reflections[:, 8] * reflections[:, 11]
    reconstructed = phasesmith.accumulate_tof(
        arrays["x_us"],
        reflections[:, 4],
        integrated_intensities,
        instrument,
        support_fwhm=20.0,
    ).y
    pattern_comparison = {
        "relative_l2": float(np.linalg.norm(reconstructed - peak_only) / np.linalg.norm(peak_only)),
        "normalized_max_error": float(
            np.max(np.abs(reconstructed - peak_only)) / np.max(np.abs(peak_only))
        ),
        "correlation": float(np.corrcoef(reconstructed, peak_only)[0, 1]),
    }

    phasesmith_report = run_powgen_tof_validation(data)
    phasesmith_result = {
        "sample_count": phasesmith_report.sample_count,
        "reflection_count": phasesmith_report.reflection_count,
        "rwp": float(
            next(
                note.split("final Rwp=")[1].split(";")[0]
                for note in phasesmith_report.notes
                if "final Rwp=" in note
            )
        ),
        "profile_correlation": check_measurement(phasesmith_report, "tof_profile_correlation"),
        "status": phasesmith_report.status,
    }
    maximum_profile_error = max(value["normalized_max_error"] for value in profile_errors.values())
    measurements = parameter_errors | {
        "selected_profile_normalized_max_error": maximum_profile_error,
        "reconstructed_pattern_relative_l2": pattern_comparison["relative_l2"],
        "reconstructed_pattern_minimum_correlation": pattern_comparison["correlation"],
        "native_workflow_rwp_delta": abs(phasesmith_result["rwp"] - gsas["poisson_rwp"]),
        "native_workflow_profile_correlation_delta": abs(
            phasesmith_result["profile_correlation"] - gsas["profile_correlation"]
        ),
    }
    checks = {}
    for name, limit in LIMITS.items():
        value = measurements[name]
        passed = value >= limit if name.endswith("minimum_correlation") else value <= limit
        checks[name] = {"measured": value, "limit": limit, "passed": passed}
    return {
        "schema_version": 1,
        "oracle_revision": report["revision"],
        "checks": checks,
        "parameter_errors": parameter_errors,
        "profile_probes": profile_errors,
        "same_extracted_intensity_pattern": pattern_comparison,
        "native_workflows": {
            "GSAS-II": {
                "sample_count": gsas["sample_count"],
                "reflection_count": gsas["reflection_count"],
                "rwp": gsas["poisson_rwp"],
                "profile_correlation": gsas["profile_correlation"],
            },
            "PhaseSmith": phasesmith_result,
        },
        "interpretation": (
            "Kernel and same-intensity pattern checks are like-for-like. GSAS-II refines a "
            "16-term Chebyshev background; PhaseSmith refines a 16-term Chebyshev residual on "
            "a fixed Smooth Bruckner baseline, so their Rwp and correlation deltas are now "
            "gated. They are not expected to be identical because GSAS-II uses 6824 calculation "
            "centers and its own joint optimizer while PhaseSmith uses independently parsed "
            "centers and alternating nonnegative Le Bail/background updates."
        ),
    }


def main() -> None:
    arguments = parse_args()
    require_paths(arguments)
    with tempfile.TemporaryDirectory(prefix="phasesmith-powgen-compare-") as temporary:
        report, arrays = run_oracle(arguments, Path(temporary))
        result = compare(report, arrays, arguments.data_directory.resolve())
    if arguments.json_output is not None:
        arguments.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    failed = [name for name, check in result["checks"].items() if not check["passed"]]
    if failed:
        raise SystemExit(f"failed comparisons: {', '.join(failed)}")


if __name__ == "__main__":
    main()
