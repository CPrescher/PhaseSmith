#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II on the trirubidium-citrate/Si holdout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from phasesmith.io.iucr_trirubidium_citrate_silicon import (
    IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES,
    convert_iucr_trirubidium_citrate_silicon_bundle,
)
from phasesmith.validation import (
    run_iucr_trirubidium_citrate_silicon_workflow,
    verify_validation_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_iucr_sodium_citrate_silicon.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
DATASET_ID = "iucr-trirubidium-citrate-si-standard"
PARITY_LIMITS = {
    "poisson_rwp_delta": 0.005,
    "profile_correlation_delta": 0.005,
    "phase_fraction_delta": 0.005,
    "silicon_calibration_rwp_delta": 0.005,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--gsas-cycles", type=int, default=12)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def comparison_checks(
    phasesmith: dict[str, object], gsasii: dict[str, object]
) -> dict[str, object]:
    native_fractions = phasesmith["weight_fractions"]
    oracle_fractions = gsasii["weight_fractions"]
    native_calibration = {
        "poisson_rwp": phasesmith["silicon_calibration_poisson_rwp"],
        "calibrated_sample_displacement_mm": phasesmith["calibrated_sample_displacement_mm"],
    }
    oracle_calibration = gsasii["silicon_calibration"]
    if not all(
        isinstance(value, dict)
        for value in (native_fractions, oracle_fractions, oracle_calibration)
    ):
        raise TypeError("comparison fractions and calibration must be mappings")
    measured = {
        "sample_count_equal": int(phasesmith["sample_count"]) == int(gsasii["sample_count"]),
        "poisson_rwp_delta": abs(float(phasesmith["poisson_rwp"]) - float(gsasii["poisson_rwp"])),
        "profile_correlation_delta": abs(
            float(phasesmith["profile_correlation"]) - float(gsasii["profile_correlation"])
        ),
        "phase_fraction_delta": max(
            abs(float(native_fractions[name]) - float(oracle_fractions[name]))
            for name in IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES
        ),
        "silicon_calibration_rwp_delta": abs(
            float(native_calibration["poisson_rwp"]) - float(oracle_calibration["poisson_rwp"])
        ),
        "sample_displacement_delta_mm": abs(
            float(native_calibration["calibrated_sample_displacement_mm"])
            - float(oracle_calibration["calibrated_sample_displacement_mm"])
        ),
    }
    checks = {
        "sample_count_equal": {
            "measured": measured["sample_count_equal"],
            "limit": True,
            "passed": measured["sample_count_equal"],
        }
    }
    for name, limit in PARITY_LIMITS.items():
        value = float(measured[name])
        checks[name] = {"measured": value, "limit": limit, "passed": value <= limit}
    displacement_delta = float(measured["sample_displacement_delta_mm"])
    checks["silicon_anchor_not_transferable"] = {
        "measured_displacement_delta_mm": displacement_delta,
        "minimum_expected_delta_mm": 0.02,
        "passed": displacement_delta >= 0.02,
    }
    return {
        "status": "qualified_pass"
        if all(check["passed"] for check in checks.values())
        else "failed",
        "checks": checks,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError("set --gsas-python and --gsas-root")
    verify_validation_dataset(DATASET_ID, arguments.data_directory)
    source = arguments.data_directory / "vn2123sup1.cif"
    with tempfile.TemporaryDirectory(prefix="phasesmith-iucr-rb-si-") as name:
        temporary = Path(name)
        bundle = temporary / "bundle"
        manifest_path = convert_iucr_trirubidium_citrate_silicon_bundle(source, bundle)
        phasesmith_result = run_iucr_trirubidium_citrate_silicon_workflow(bundle).to_record()
        gsasii_path = temporary / "gsasii.json"
        command = [
            str(arguments.gsas_python),
            str(WORKER),
            "--gsas-root",
            str(arguments.gsas_root),
            "--data-directory",
            str(bundle),
            "--cycles",
            str(arguments.gsas_cycles),
            "--report",
            str(gsasii_path),
        ]
        if arguments.binary_dir is not None:
            command.extend(("--binary-dir", str(arguments.binary_dir)))
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
        completed = subprocess.run(command, capture_output=True, text=True, env=environment)
        if completed.returncode != 0:
            raise RuntimeError(
                "external GSAS-II trirubidium-citrate/Si worker failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        gsasii_result = json.loads(gsasii_path.read_text(encoding="utf-8"))
        if gsasii_result.get("revision") != PINNED_REVISION:
            raise RuntimeError("GSAS-II worker returned inconsistent provenance")
        validation = comparison_checks(phasesmith_result, gsasii_result)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "scope": "iucr_trirubidium_citrate_silicon_transferability_comparison",
            "source_dataset_id": DATASET_ID,
            "pinned_gsasii_revision": PINNED_REVISION,
            "scientific_status": "qualified_source_translation_and_remaining_model_gap",
            "common_model": {
                "instrument": manifest["instrument"],
                "data_selection": manifest["data_selection"],
                "silicon_standard": manifest["silicon_standard"],
                "refined_parameters": ["two phase scales", "one constant residual background"],
                "fixed_parameters": [
                    "shared deposited base U/V/W/X/Y and matched SH/L=0.0194",
                    "instrument zero, source-translated specimen displacement, radius, "
                    "and Si lattice",
                    "all phase mixing, size/strain, and orientation parameters",
                ],
            },
            "legacy_gsas_reference": manifest["legacy_gsas_reference"],
            "phasesmith": phasesmith_result,
            "gsasii": gsasii_result,
            "cross_implementation_validation": validation,
            "interpretation": [
                "The fixed-geometry common subset agrees closely across implementations.",
                "The source-deposited radius removes the geometry assumption present in the "
                "potassium diagnostic.",
                "Legacy LX and shft now have explicit equation translations and documented "
                "sign conventions. Their corrected common subset materially lowers both "
                "implementations' Rwp while retaining cross-implementation parity.",
                "The 2.15 wt% silicon windows still do not transfer a displacement anchor; "
                "their estimates remain diagnostic and do not replace the source shft value.",
                "Both fits remain far from the phase-specific deposited GSAS model, so this "
                "does not justify adding transparency or another broadening term.",
            ],
        }
    print(
        f"status={validation['status']} "
        f"PhaseSmith Rwp={100 * phasesmith_result['poisson_rwp']:.3f}% "
        f"GSAS-II Rwp={100 * gsasii_result['poisson_rwp']:.3f}%"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if validation["status"] != "qualified_pass":
        raise RuntimeError("trirubidium-citrate/Si validation did not match expected outcome")


if __name__ == "__main__":
    main()
