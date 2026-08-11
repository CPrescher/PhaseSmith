#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II on converted Bath LTL profiles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from phasesmith.io import convert_bath_ltl_bundle
from phasesmith.validation import run_bath_ltl_workflow, verify_validation_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_bath_ltl.py"
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
        default=REPOSITORY_ROOT / "validation" / "data" / "bath-ltl-lab-xray",
    )
    parser.add_argument("--sample", choices=("K", "Li", "Cs", "all"), default="all")
    parser.add_argument("--gsas-cycles", type=int, default=12)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def run_worker(
    arguments: argparse.Namespace, bundle: Path, sample: str, report: Path
) -> dict[str, Any]:
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
        str(report),
    ]
    if arguments.binary_dir is not None:
        command.extend(("--binary-dir", str(arguments.binary_dir)))
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(report.parent / "matplotlib")
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            "external GSAS-II Bath worker failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(report.read_text(encoding="utf-8"))
    if result.get("revision") != PINNED_REVISION or result.get("sample") != sample:
        raise RuntimeError("GSAS-II Bath worker returned inconsistent provenance")
    return result


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    if arguments.gsas_cycles <= 0:
        raise ValueError("GSAS-II cycles must be positive")
    verify_validation_dataset("bath-ltl-lab-xray", arguments.data_directory)
    samples = ("K", "Li", "Cs") if arguments.sample == "all" else (arguments.sample,)
    with tempfile.TemporaryDirectory(prefix="phasesmith-bath-ltl-") as name:
        temporary = Path(name)
        bundle = temporary / "bundle"
        manifest_path = convert_bath_ltl_bundle(arguments.data_directory, bundle)
        comparisons: dict[str, Any] = {}
        for sample in samples:
            phasesmith_result = run_bath_ltl_workflow(bundle, sample).to_record()
            gsasii_result = run_worker(
                arguments, bundle, sample, temporary / f"gsasii-{sample}.json"
            )
            comparisons[sample] = {
                "phasesmith": phasesmith_result,
                "gsasii": gsasii_result,
                "refined_poisson_rwp_difference": (
                    phasesmith_result["refined_profile_poisson_rwp"]
                    - gsasii_result["refined_profile"]["poisson_rwp"]
                ),
            }
            print(
                f"sample={sample} released={100 * phasesmith_result['released_poisson_rwp']:.3f}% "
                f"PhaseSmith={100 * phasesmith_result['refined_profile_poisson_rwp']:.3f}% "
                f"GSAS-II={100 * gsasii_result['refined_profile']['poisson_rwp']:.3f}%"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 1,
            "scope": "bath_ltl_lab_xray_conversion_fidelity_comparison",
            "source_dataset_id": "bath-ltl-lab-xray",
            "pinned_gsasii_revision": PINNED_REVISION,
            "conversion": {
                "metadata_discrepancy": manifest["metadata_discrepancy"],
                "translation_notes": manifest["translation_notes"],
            },
            "comparisons": comparisons,
            "interpretation": [
                "Both programs use the same primary publication CIF, released observed curve, "
                "deposited fixed background and archived effective profile initializer.",
                "The released curve is a legacy GSAS result, not a calculation reconstructed "
                "from the publication CIF alone.",
                "Similar PhaseSmith and GSAS-II residuals identify conversion fidelity rather "
                "than a PhaseSmith-only profile-physics gap.",
                "The case remains a capability failure until the missing legacy semantics are "
                "identified or a cleaner accepted reference model is available.",
            ],
        }
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
