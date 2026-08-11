#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II on the IUCr mixed-Si example."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from phasesmith.io import IUCR_SILICON_PHASES, convert_iucr_silicon_standard_bundle
from phasesmith.validation import run_iucr_silicon_standard_workflow, verify_validation_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_iucr_silicon_standard.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
DATASET_ID = "iucr-dicesium-citrate-si-standard"


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


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    verify_validation_dataset(DATASET_ID, arguments.data_directory)
    source = arguments.data_directory / "wm5358sup1.cif"
    with tempfile.TemporaryDirectory(prefix="phasesmith-iucr-si-") as name:
        temporary = Path(name)
        bundle = temporary / "bundle"
        manifest_path = convert_iucr_silicon_standard_bundle(source, bundle)
        phasesmith_result = run_iucr_silicon_standard_workflow(
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
                "external GSAS-II IUCr silicon-standard worker failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        gsasii_result = json.loads(gsasii_path.read_text(encoding="utf-8"))
        if gsasii_result.get("revision") != PINNED_REVISION:
            raise RuntimeError("GSAS-II worker returned inconsistent provenance")
        fraction_differences = {
            phase_id: (
                phasesmith_result["weight_fractions"][phase_id]
                - gsasii_result["weight_fractions"][phase_id]
            )
            for phase_id in IUCR_SILICON_PHASES
        }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "scope": "iucr_dicesium_citrate_silicon_internal_standard_comparison",
            "source_dataset_id": DATASET_ID,
            "pinned_gsasii_revision": PINNED_REVISION,
            "common_model": {
                "instrument": manifest["instrument"],
                "data_selection": manifest["data_selection"],
                "silicon_calibration": manifest["silicon_calibration"],
                "translation": manifest["translation"],
            },
            "legacy_gsas_reference": manifest["legacy_gsas_reference"],
            "phasesmith": phasesmith_result,
            "gsasii": gsasii_result,
            "phase_fraction_differences": fraction_differences,
            "silicon_calibration_comparison": {
                "phasesmith_zero_shift_deg": phasesmith_result["calibrated_zero_shift_deg"],
                "gsasii_zero_shift_deg": gsasii_result["silicon_calibration"][
                    "calibrated_zero_shift_deg"
                ],
                "absolute_zero_shift_difference_deg": abs(
                    phasesmith_result["calibrated_zero_shift_deg"]
                    - gsasii_result["silicon_calibration"]["calibrated_zero_shift_deg"]
                ),
                "phasesmith_sample_displacement_mm": phasesmith_result[
                    "calibrated_sample_displacement_mm"
                ],
                "gsasii_sample_displacement_mm": gsasii_result["silicon_calibration"][
                    "calibrated_sample_displacement_mm"
                ],
                "absolute_sample_displacement_difference_mm": abs(
                    phasesmith_result["calibrated_sample_displacement_mm"]
                    - gsasii_result["silicon_calibration"]["calibrated_sample_displacement_mm"]
                ),
            },
            "interpretation": [
                "The fixed-cell silicon standard alone calibrates zero shift in isolated "
                "Si(220), Si(311), and Si(400) windows; that zero is frozen before the "
                "three-phase refinement.",
                "Both new workflows use the deposited background and the same Cu doublet, "
                "initial U/V/W/X/Y/zero, FCJ parameter, structures, and isotropic "
                "size/microstrain model.",
                "The deposited legacy-GSAS calculation remains the accuracy reference; the "
                "new GSAS-II run is a black-box parity comparator, not a regeneration of it.",
                "The legacy main-phase Stephens anisotropy and phase-specific profile "
                "functions are omitted deliberately and explain part of the residual floor.",
            ],
        }
    print(
        f"PhaseSmith Rwp={100 * phasesmith_result['poisson_rwp']:.3f}% "
        f"GSAS-II Rwp={100 * gsasii_result['poisson_rwp']:.3f}% "
        f"Si={100 * phasesmith_result['weight_fractions']['silicon']:.2f}%/"
        f"{100 * gsasii_result['weight_fractions']['silicon']:.2f}% "
        f"legacy={100 * manifest['legacy_gsas_reference']['weight_fractions']['silicon']:.2f}%"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
