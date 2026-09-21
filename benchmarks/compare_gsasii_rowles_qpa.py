#!/usr/bin/env python3
"""Convert Rowles TOPAS inputs and compare PhaseSmith with pinned GSAS-II."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import phasesmith
from phasesmith.io.topas import convert_rowles_topas_bundle
from phasesmith.validation import run_rowles_qpa_workflow, verify_validation_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_rowles_qpa.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
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
        default=REPOSITORY_ROOT / "validation" / "data" / "curtin-rowles-qpa-topas",
    )
    parser.add_argument("--sample", choices=("1a", "1e", "all"), default="all")
    parser.add_argument("--gsas-cycles", type=int, default=8)
    parser.add_argument("--phasesmith-threads", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def compare_scientific_results(
    phasesmith_result: dict[str, Any], gsas_result: dict[str, Any]
) -> dict[str, Any]:
    if phasesmith_result["sample"] != gsas_result["sample"]:
        raise RuntimeError("cross-implementation Rowles sample mismatch")
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
            "measured": value,
            "limit": CROSS_IMPLEMENTATION_LIMITS[name],
            "passed": value <= CROSS_IMPLEMENTATION_LIMITS[name],
        }
        for name, value in measurements.items()
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "failed_checks": failed,
        "phase_fraction_deltas": phase_deltas,
        "checks": checks,
    }


def run_gsas(
    arguments: argparse.Namespace, bundle: Path, sample: str, report_path: Path
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(report_path.parent / "matplotlib")
    command = [
        str(arguments.gsas_python),
        str(WORKER),
        "--gsas-root",
        str(arguments.gsas_root),
        "--bundle-directory",
        str(bundle),
        "--sample",
        sample,
        "--cycles",
        str(arguments.gsas_cycles),
        "--report",
        str(report_path),
    ]
    if arguments.binary_dir is not None:
        command.extend(("--binary-dir", str(arguments.binary_dir)))
    subprocess.run(command, check=True, env=environment)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("revision") != PINNED_REVISION or report.get("sample") != sample:
        raise RuntimeError("GSAS-II Rowles worker returned inconsistent provenance")
    return report


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    if arguments.phasesmith_threads < 0 or arguments.gsas_cycles <= 0:
        raise ValueError("thread count must be nonnegative and cycles must be positive")
    verify_validation_dataset("curtin-rowles-qpa-topas", arguments.data_directory)
    samples = ("1a", "1e") if arguments.sample == "all" else (arguments.sample,)
    execution = phasesmith.ExecutionPolicy(
        threads=None if arguments.phasesmith_threads == 0 else arguments.phasesmith_threads
    )
    with tempfile.TemporaryDirectory(prefix="phasesmith-rowles-comparison-") as name:
        temporary = Path(name)
        bundle = temporary / "converted"
        manifest_path = convert_rowles_topas_bundle(arguments.data_directory, bundle)
        comparisons = {}
        for sample in samples:
            phase_result = run_rowles_qpa_workflow(bundle, sample, execution=execution).to_record()
            gsas_report = run_gsas(arguments, bundle, sample, temporary / f"gsasii-{sample}.json")
            cross = compare_scientific_results(phase_result, gsas_report["result"])
            comparisons[sample] = {
                "phasesmith": phase_result,
                "gsasii": gsas_report,
                "cross_implementation_validation": cross,
            }
            print(
                f"sample={sample} phasesmith_rwp={100 * phase_result['poisson_rwp']:.3f}% "
                f"gsasii_rwp={100 * gsas_report['result']['poisson_rwp']:.3f}% "
                f"cross_status={cross['status']}"
            )
            for phase in PHASE_NAMES:
                print(
                    f"  phase={phase} "
                    f"phasesmith={100 * phase_result['weight_fractions'][phase]:.3f}% "
                    f"gsasii={100 * gsas_report['result']['weight_fractions'][phase]:.3f}%"
                )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "scope": "curtin_rowles_qpa_topas_common_subset",
            "comparison_kind": "converted_shared_inputs_matched_common_parameterizations",
            "source_dataset_id": "curtin-rowles-qpa-topas",
            "conversion": manifest["translation"],
            "interpretation": [
                "Both programs consume the same converted patterns, CIFs, wavelength doublet, "
                "profile initializer, range, and weighed-fraction targets.",
                "The native refinement recipes are intentionally not identical: PhaseSmith uses "
                "a fixed Smooth-Bruckner baseline plus a Chebyshev residual, while GSAS-II uses "
                "its native Chebyshev background and staged public scripting API.",
                "Residual differences therefore diagnose the complete available workflows; they "
                "are not a pointwise comparison of identical profile kernels.",
                "The deposited TOPAS fundamental-parameters result is not reproduced by this "
                "common subset; omitted terms remain enumerated in conversion metadata.",
            ],
            "comparisons": comparisons,
        }
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
