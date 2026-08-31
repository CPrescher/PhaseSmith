#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II on selected opXRD common models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith.validation.opxrd import (
    run_opxrd_robustness_campaign,
    run_opxrd_structural_common_model,
    write_opxrd_common_model_bundle,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_opxrd.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
DEFAULT_CASES = (
    "cnrs-nbs2-synchrotron-full",
    "cnrs-zrc-laboratory-full",
    "cnrs-li2tec2-doublet-full",
)


def profile_plausibility(
    profile: dict[str, float],
    two_theta_range_deg: tuple[float, float],
    common: dict[str, object],
    *,
    gsasii_units: bool,
) -> dict[str, object]:
    """Report effective Gaussian width and zero-offset warnings without hiding the fit."""

    if gsasii_units:
        u, v, w = (1.0e-4 * float(profile[name]) for name in ("U", "V", "W"))
        zero = float(profile["Zero"])
    else:
        u, v, w = (float(profile[name]) for name in ("u_deg2", "v_deg2", "w_deg2"))
        zero = float(profile["zero_shift_deg"])
    low, high = two_theta_range_deg
    tangent = np.tan(np.deg2rad(np.linspace(low, high, 1_001) / 2.0))
    variance = u * tangent**2 + v * tangent + w
    maximum_fwhm = float(np.sqrt(max(float(np.max(variance)), 0.0)))
    large_zero = abs(zero) > float(common["large_zero_shift_warning_deg"])
    large_width = maximum_fwhm > float(common["large_gaussian_fwhm_warning_deg"])
    return {
        "status": "warning" if large_zero or large_width else "plausible",
        "absolute_zero_shift_deg": abs(zero),
        "maximum_gaussian_fwhm_deg": maximum_fwhm,
        "large_zero_shift_warning": large_zero,
        "large_gaussian_width_warning": large_width,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=(REPOSITORY_ROOT / "validation" / "data" / "opxrd-robustness-v1" / "opxrd.zip"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "validation" / "opxrd-robustness-v1.json",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--phasesmith-cycles", type=int, default=2)
    parser.add_argument("--gsas-cycles", type=int, default=8)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    case_ids = DEFAULT_CASES if arguments.case_ids is None else tuple(arguments.case_ids)
    if not case_ids:
        raise ValueError("at least one --case is required")
    verified = run_opxrd_robustness_campaign(
        arguments.archive,
        arguments.selection,
        case_ids=case_ids,
    )
    selection_record = json.loads(arguments.selection.read_text(encoding="utf-8"))
    comparisons = []
    with tempfile.TemporaryDirectory(prefix="phasesmith-opxrd-comparison-") as name:
        temporary = Path(name)
        for case_id in case_ids:
            bundle = temporary / case_id / "bundle"
            write_opxrd_common_model_bundle(arguments.archive, arguments.selection, case_id, bundle)
            model = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
            phasesmith = asdict(
                run_opxrd_structural_common_model(bundle, cycles=arguments.phasesmith_cycles)
            )
            pattern = np.loadtxt(bundle / "pattern.xye")
            two_theta_range = (float(pattern[0, 0]), float(pattern[-1, 0]))
            initial_zero = float(phasesmith["position_alignment"]["selected_zero_shift_deg"])
            gsasii_path = temporary / case_id / "gsasii.json"
            command = [
                str(arguments.gsas_python),
                str(WORKER),
                "--gsas-root",
                str(arguments.gsas_root),
                "--bundle",
                str(bundle),
                "--cycles",
                str(arguments.gsas_cycles),
                "--initial-zero-shift-deg",
                str(initial_zero),
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
                    f"external GSAS-II opXRD worker failed for {case_id}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            gsasii = json.loads(gsasii_path.read_text(encoding="utf-8"))
            if gsasii.get("revision") != PINNED_REVISION:
                raise RuntimeError("GSAS-II opXRD worker returned inconsistent provenance")
            if phasesmith["sample_count"] != gsasii["sample_count"]:
                raise RuntimeError("cross-program opXRD sample counts differ")
            common_input_checks = {
                "same_sample_count": True,
                "same_member_sha256": (gsasii["member_sha256"] == model["member_sha256"]),
                "same_selection_sha256": (gsasii["selection_sha256"] == model["selection_sha256"]),
                "same_wavelength_components": (
                    tuple(phasesmith["wavelengths_angstrom"])
                    == tuple(gsasii["wavelengths_angstrom"])
                ),
                "same_starting_zero_shift": (
                    initial_zero == float(gsasii["initial_zero_shift_deg"])
                ),
            }
            if not all(common_input_checks.values()):
                raise RuntimeError("cross-program opXRD common inputs differ")
            metric_differences = {
                metric: (
                    phasesmith["residual_metrics"][metric] - gsasii["residual_metrics"][metric]
                )
                for metric in (
                    "relative_l2",
                    "normalized_l1",
                    "cosine_similarity",
                    "pearson_correlation",
                    "poisson_rwp",
                )
            }
            phasesmith_plausibility = profile_plausibility(
                phasesmith["profile_parameters"],
                two_theta_range,
                model["common_model"],
                gsasii_units=False,
            )
            if not np.isclose(
                phasesmith_plausibility["maximum_gaussian_fwhm_deg"],
                phasesmith["maximum_gaussian_fwhm_deg"],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError("PhaseSmith opXRD plausibility calculation is inconsistent")
            if phasesmith_plausibility["status"] != phasesmith["parameter_plausibility"]:
                raise RuntimeError("PhaseSmith opXRD plausibility status is inconsistent")
            comparisons.append(
                {
                    "case_id": case_id,
                    "phasesmith": phasesmith,
                    "gsasii": gsasii,
                    "metric_differences_phasesmith_minus_gsasii": metric_differences,
                    "common_model_assumptions": model["assumptions"],
                    "parameter_plausibility": {
                        "phasesmith": phasesmith_plausibility,
                        "gsasii": profile_plausibility(
                            gsasii["profile_parameters"],
                            two_theta_range,
                            model["common_model"],
                            gsasii_units=True,
                        ),
                    },
                    "checks": common_input_checks,
                }
            )
    report = {
        "schema_version": 1,
        "scope": "opxrd_structural_common_model_cross_program_diagnostic",
        "source_dataset_id": verified.dataset_id,
        "archive_sha256": verified.archive_sha256,
        "selection_sha256": verified.selection_sha256,
        "pinned_gsasii_revision": PINNED_REVISION,
        "common_model": selection_record["procedure"]["structural_common_model"],
        "recipe": {
            "phasesmith_cycles": arguments.phasesmith_cycles,
            "gsasii_cycles": arguments.gsas_cycles,
            "cell_and_atoms": "fixed expanded P1 structure from the opXRD label",
            "refined_blocks": [
                "phase scale plus eight-term Chebyshev background",
                "bounded global zero scan and zero-only local polish",
                "U, V, and W with the aligned zero fixed",
                "isotropic size and microstrain broadening",
            ],
        },
        "comparisons": comparisons,
        "interpretation": [
            "The common models use only the archived observations, wavelengths, cells, and "
            "expanded atoms; missing geometry and displacement metadata are disclosed assumptions.",
            "The five residual measures expose weighting and baseline sensitivity that a single "
            "Rwp value would hide.",
            "Large zero offsets and effective Gaussian widths are reported separately so a lower "
            "residual cannot hide compensating instrument parameters.",
            "The comparison tests same-input execution and optimization behavior; it is not a "
            "structural-accuracy or GSAS-II-equivalence acceptance test.",
        ],
    }
    for item in comparisons:
        phase = item["phasesmith"]["residual_metrics"]
        gsas = item["gsasii"]["residual_metrics"]
        print(
            f"{item['case_id']}: PhaseSmith/GSAS-II Poisson Rwp "
            f"{phase['poisson_rwp']:.4f}/{gsas['poisson_rwp']:.4f}, "
            f"relative L2 {phase['relative_l2']:.4f}/{gsas['relative_l2']:.4f}, "
            f"plausibility {item['parameter_plausibility']['phasesmith']['status']}/"
            f"{item['parameter_plausibility']['gsasii']['status']}"
        )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
