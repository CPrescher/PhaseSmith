#!/usr/bin/env python3
"""Test GSAS-II FPA compression of the deposited Rowles instrument geometry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from phasesmith.io.topas import convert_rowles_topas_bundle
from phasesmith.validation import verify_validation_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "calibrate_rowles_fpa.py"
QPA_WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_rowles_qpa.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"


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
        default=REPOSITORY_ROOT / "validation" / "data" / "curtin-rowles-qpa-topas",
    )
    parser.add_argument("--sample", choices=("1a", "1e", "all"), default="all")
    parser.add_argument("--gsas-cycles", type=int, default=8)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def run_command(command: list[str], environment: dict[str, str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            "external GSAS-II Rowles FPA worker failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def common_worker_arguments(arguments: argparse.Namespace, bundle: Path, report: Path) -> list[str]:
    command = [
        "--gsas-root",
        str(arguments.gsas_root),
        "--bundle-directory",
        str(bundle),
        "--report",
        str(report),
    ]
    if arguments.binary_dir is not None:
        command.extend(("--binary-dir", str(arguments.binary_dir)))
    return command


def load_report(path: Path, expected_scope: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("revision") != PINNED_REVISION or report.get("scope") != expected_scope:
        raise RuntimeError("GSAS-II Rowles FPA worker returned inconsistent provenance")
    return report


def run_qpa(
    arguments: argparse.Namespace,
    bundle: Path,
    sample: str,
    report_path: Path,
    environment: dict[str, str],
    profile_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        str(arguments.gsas_python),
        str(QPA_WORKER),
        *common_worker_arguments(arguments, bundle, report_path),
        "--sample",
        sample,
        "--cycles",
        str(arguments.gsas_cycles),
    ]
    if profile_path is not None:
        command.extend(("--instrument-profile", str(profile_path)))
    run_command(command, environment)
    report = load_report(report_path, "curtin_rowles_qpa_topas_common_subset")
    if report.get("sample") != sample:
        raise RuntimeError("GSAS-II Rowles FPA worker returned the wrong sample")
    return report


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    if arguments.gsas_cycles <= 0:
        raise ValueError("GSAS-II cycles must be positive")
    verify_validation_dataset("curtin-rowles-qpa-topas", arguments.data_directory)
    samples = ("1a", "1e") if arguments.sample == "all" else (arguments.sample,)
    with tempfile.TemporaryDirectory(prefix="phasesmith-rowles-fpa-") as name:
        temporary = Path(name)
        bundle = temporary / "converted"
        manifest_path = convert_rowles_topas_bundle(arguments.data_directory, bundle)
        calibration_path = temporary / "calibration.json"
        profile_path = temporary / "profile.json"
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
        calibration_command = [
            str(arguments.gsas_python),
            str(CALIBRATION_WORKER),
            *common_worker_arguments(arguments, bundle, calibration_path),
            "--profile-output",
            str(profile_path),
        ]
        run_command(calibration_command, environment)
        calibration = load_report(
            calibration_path, "curtin_rowles_qpa_topas_gsasii_fpa_compression"
        )
        comparisons = {}
        for sample in samples:
            empirical = run_qpa(
                arguments,
                bundle,
                sample,
                temporary / f"empirical-{sample}.json",
                environment,
            )
            physical = run_qpa(
                arguments,
                bundle,
                sample,
                temporary / f"fpa-{sample}.json",
                environment,
                profile_path,
            )
            empirical_result = empirical["result"]
            physical_result = physical["result"]
            delta = physical_result["poisson_rwp"] - empirical_result["poisson_rwp"]
            comparisons[sample] = {
                "empirical_profile": empirical,
                "fpa_compressed_profile": physical,
                "fpa_minus_empirical": {
                    "poisson_rwp": delta,
                    "unit_weight_rwp": (
                        physical_result["unit_weight_rwp"] - empirical_result["unit_weight_rwp"]
                    ),
                    "profile_correlation": (
                        physical_result["profile_correlation"]
                        - empirical_result["profile_correlation"]
                    ),
                },
                "fpa_improved_poisson_rwp": delta < 0.0,
            }
            print(
                f"sample={sample} empirical_rwp={100 * empirical_result['poisson_rwp']:.3f}% "
                f"fpa_rwp={100 * physical_result['poisson_rwp']:.3f}% "
                f"delta={100 * delta:+.3f}%"
            )
        conversion = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "scope": "curtin_rowles_qpa_topas_gsasii_fpa_diagnostic",
            "source_dataset_id": "curtin-rowles-qpa-topas",
            "conversion": conversion["translation"],
            "calibration": calibration,
            "comparisons": comparisons,
            "interpretation": [
                "Pinned GSAS-II generates a physical target from the deposited Rowles geometry "
                "and compresses it into its production U/V/W/X/Y/SH/L profile.",
                "The compression is not a TOPAS calculation and excludes the angle-dependent "
                "continuum, K-beta outside the K-alpha window, and specimen absorption.",
                "The empirical and FPA-compressed profiles use the same GSAS-II QPA workflow; "
                "only the instrument-profile calibration changes.",
                "A higher real-pattern Rwp for the FPA-compressed profile argues against adding "
                "more specialized Rowles optics to PhaseSmith at this stage.",
            ],
        }
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
