#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II on the sodium-citrate/Si holdout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from phasesmith.io import (
    IUCR_SODIUM_CITRATE_SILICON_PHASES,
    convert_iucr_sodium_citrate_silicon_bundle,
)
from phasesmith.validation import (
    run_iucr_sodium_citrate_silicon_workflow,
    verify_validation_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_iucr_sodium_citrate_silicon.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
DATASET_ID = "iucr-sodium-dihydrogen-citrate-si-standard"
LIMITS = {
    "poisson_rwp_delta": 0.010,
    "profile_correlation_delta": 0.0075,
    "phase_fraction_delta": 0.010,
    "march_ratio_delta": 0.010,
    "silicon_calibration_rwp_delta": 0.005,
    "sample_displacement_delta_mm": 0.10,
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
    parser.add_argument("--phasesmith-cycles", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def comparison_checks(
    phasesmith: dict[str, object], gsasii: dict[str, object]
) -> dict[str, object]:
    native_fractions = phasesmith["weight_fractions"]
    oracle_fractions = gsasii["weight_fractions"]
    if not isinstance(native_fractions, dict) or not isinstance(oracle_fractions, dict):
        raise TypeError("comparison phase fractions must be mappings")
    gsas_hap = gsasii["refined_hap"]
    if not isinstance(gsas_hap, dict):
        raise TypeError("GSAS-II HAP result must be a mapping")
    gsas_march = float(gsas_hap["sodium_dihydrogen_citrate"]["Pref.Ori."][1])
    measured = {
        "sample_count_equal": int(phasesmith["sample_count"]) == int(gsasii["sample_count"]),
        "poisson_rwp_delta": abs(float(phasesmith["poisson_rwp"]) - float(gsasii["poisson_rwp"])),
        "profile_correlation_delta": abs(
            float(phasesmith["profile_correlation"]) - float(gsasii["profile_correlation"])
        ),
        "phase_fraction_delta": max(
            abs(float(native_fractions[name]) - float(oracle_fractions[name]))
            for name in IUCR_SODIUM_CITRATE_SILICON_PHASES
        ),
        "march_ratio_delta": abs(float(phasesmith["refined_march_ratio"]) - gsas_march),
        "silicon_calibration_rwp_delta": abs(
            float(phasesmith["silicon_calibration_poisson_rwp"])
            - float(gsasii["silicon_calibration"]["poisson_rwp"])
        ),
        "sample_displacement_delta_mm": abs(
            float(phasesmith["calibrated_sample_displacement_mm"])
            - float(gsasii["silicon_calibration"]["calibrated_sample_displacement_mm"])
        ),
    }
    checks = {
        "sample_count_equal": {
            "measured": measured["sample_count_equal"],
            "limit": True,
            "passed": measured["sample_count_equal"],
        }
    }
    for name, limit in LIMITS.items():
        value = float(measured[name])
        checks[name] = {"measured": value, "limit": limit, "passed": value <= limit}
    return {
        "status": "passed" if all(check["passed"] for check in checks.values()) else "failed",
        "checks": checks,
        "gsasii_march_ratio": gsas_march,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    verify_validation_dataset(DATASET_ID, arguments.data_directory)
    source = arguments.data_directory / "hb7585sup1.cif"
    with tempfile.TemporaryDirectory(prefix="phasesmith-iucr-na-si-") as name:
        temporary = Path(name)
        bundle = temporary / "bundle"
        manifest_path = convert_iucr_sodium_citrate_silicon_bundle(source, bundle)
        phasesmith_result = run_iucr_sodium_citrate_silicon_workflow(
            bundle, cycles=arguments.phasesmith_cycles
        ).to_record()
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
                "external GSAS-II sodium-citrate/Si worker failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        gsasii_result = json.loads(gsasii_path.read_text(encoding="utf-8"))
        if gsasii_result.get("revision") != PINNED_REVISION:
            raise RuntimeError("GSAS-II worker returned inconsistent provenance")
        validation = comparison_checks(phasesmith_result, gsasii_result)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "scope": "iucr_sodium_citrate_silicon_preferred_orientation_comparison",
            "source_dataset_id": DATASET_ID,
            "pinned_gsasii_revision": PINNED_REVISION,
            "common_model": {
                "instrument": manifest["instrument"],
                "data_selection": manifest["data_selection"],
                "silicon_standard": manifest["silicon_standard"],
                "translation_diagnostics": manifest["translation_diagnostics"],
                "refined_parameters": [
                    "two phase scales",
                    "one constant residual background",
                    "sodium-citrate March-Dollase (001) ratio",
                ],
                "fixed_parameters": [
                    "U/V/W/X/Y and matched SH/L=0.0187",
                    "instrument zero and Si lattice",
                    "Si-calibrated specimen displacement",
                    "all phase size/strain parameters",
                ],
            },
            "legacy_gsas_reference": manifest["legacy_gsas_reference"],
            "phasesmith": phasesmith_result,
            "gsasii": gsasii_result,
            "cross_implementation_validation": validation,
            "interpretation": [
                "Both programs receive the same observed counts, fixed legacy background, "
                "Cu doublet, silicon-derived profile, equal-height SH/L, structures, and "
                "March-Dollase (001) stress model.",
                "All isotropic sample-width refinement is excluded because trial runs made "
                "instrument/sample partitions non-identifiable and drove GSAS-II strain "
                "outside its physical domain.",
                "Agreement between the two common-subset fits is a preferred-orientation "
                "parity result, not equivalence to the deposited generalized spherical-"
                "harmonic, Stephens-anisotropy, and Suortti-roughness refinement.",
            ],
        }
    validation_status = validation["status"]
    print(
        f"status={validation_status} "
        f"PhaseSmith Rwp={100 * phasesmith_result['poisson_rwp']:.3f}% "
        f"GSAS-II Rwp={100 * gsasii_result['poisson_rwp']:.3f}% "
        f"Si={100 * phasesmith_result['weight_fractions']['silicon']:.2f}%/"
        f"{100 * gsasii_result['weight_fractions']['silicon']:.2f}%"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if validation_status != "passed":
        raise RuntimeError("sodium-citrate/Si cross-implementation validation failed")


if __name__ == "__main__":
    main()
