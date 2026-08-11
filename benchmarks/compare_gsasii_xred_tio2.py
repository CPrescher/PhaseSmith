#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II on the XRED TiO2 example."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from phasesmith.io import convert_xred_tio2_bundle
from phasesmith.validation import run_xred_tio2_workflow, verify_validation_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_xred_tio2.py"
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
        default=REPOSITORY_ROOT / "validation" / "data" / "xred-tio2-anatase-rutile",
    )
    parser.add_argument("--gsas-cycles", type=int, default=12)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    if arguments.gsas_cycles <= 0:
        raise ValueError("GSAS-II cycles must be positive")
    verify_validation_dataset("xred-tio2-anatase-rutile", arguments.data_directory)
    with tempfile.TemporaryDirectory(prefix="phasesmith-xred-tio2-") as name:
        temporary = Path(name)
        bundle = temporary / "bundle"
        manifest_path = convert_xred_tio2_bundle(arguments.data_directory, bundle)
        phasesmith_result = run_xred_tio2_workflow(bundle).to_record()
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
                "external GSAS-II XRED worker failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        gsasii_result = json.loads(gsasii_path.read_text(encoding="utf-8"))
        if gsasii_result.get("revision") != PINNED_REVISION:
            raise RuntimeError("GSAS-II XRED worker returned inconsistent provenance")
        phase_fraction_differences = {
            phase: (
                phasesmith_result["weight_fractions"][phase]
                - gsasii_result["weight_fractions"][phase]
            )
            for phase in ("anatase", "rutile")
        }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "scope": "xred_tio2_anatase_rutile_comparison",
            "source_dataset_id": "xred-tio2-anatase-rutile",
            "pinned_gsasii_revision": PINNED_REVISION,
            "common_model": manifest["assumptions"],
            "setting_translation": manifest["setting_translation"],
            "phasesmith": phasesmith_result,
            "gsasii": gsasii_result,
            "phase_fraction_differences": phase_fraction_differences,
            "interpretation": [
                "The dataset has no instrument metadata or certified phase composition; the "
                "reported phase fractions are cross-program diagnostics, not an accuracy claim.",
                "The deposited anatase CIF requires a visible origin-choice translation for "
                "GSAS-II; PhaseSmith reads its explicit operation set directly.",
                "The programs agree closely on phase fractions after translation, while their "
                "different bounded refinement recipes reach different residual minima.",
                "This case exercises experimental two-phase identification and guarded lattice "
                "motion, but it is not suitable as a strict Rietveld parity acceptance test.",
            ],
        }
    print(
        f"PhaseSmith Rwp={100 * phasesmith_result['poisson_rwp']:.3f}% "
        f"GSAS-II Rwp={100 * gsasii_result['poisson_rwp']:.3f}% "
        f"anatase={100 * phasesmith_result['weight_fractions']['anatase']:.2f}%/"
        f"{100 * gsasii_result['weight_fractions']['anatase']:.2f}%"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
