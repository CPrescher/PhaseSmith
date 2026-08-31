#!/usr/bin/env python3
"""Run one opXRD common-model bundle in pinned GSAS-II."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--initial-zero-shift-deg", required=True, type=float)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def refine(project: Any, histogram: Any) -> float:
    project.do_refinements([{}], outputnames=[None])
    value = histogram.get_wR()
    if value is None or not np.isfinite(value):
        raise RuntimeError("GSAS-II opXRD stage did not produce a finite Rwp")
    return float(value) / 100.0


def instrument_text(model: dict[str, Any]) -> str:
    wavelengths = tuple(float(value) for value in model["wavelengths_angstrom"])
    relative = tuple(float(value) for value in model["relative_intensities"])
    common = model["common_model"]
    if len(wavelengths) == 1:
        source = f"Lam:{wavelengths[0]:.12g}\n"
    elif len(wavelengths) == 2 and len(relative) == 2:
        source = (
            f"Lam1:{wavelengths[0]:.12g}\n"
            f"Lam2:{wavelengths[1]:.12g}\n"
            f"I(L2)/I(L1):{relative[1] / relative[0]:.12g}\n"
        )
    else:
        raise ValueError("GSAS-II comparator supports one wavelength or one doublet")
    u, v, w = (1.0e4 * float(value) for value in common["initial_profile_deg2"])
    sh_over_l = float(common["axial_sample_over_radius"]) + float(
        common["axial_detector_over_radius"]
    )
    return (
        "#GSAS-II instrument parameter file; do not add/delete items!\n"
        "Type:PXC\nBank:1.0\n"
        f"{source}"
        "Zero:0.0\n"
        f"Polariz.:{float(common['polarization_fraction']):.12g}\n"
        f"U:{u:.12g}\nV:{v:.12g}\nW:{w:.12g}\n"
        "X:0.0\nY:0.0\nZ:0.0\n"
        f"SH/L:{sh_over_l:.12g}\nAzimuth:0.0\n"
    )


def residual_metrics(observed: np.ndarray, calculated: np.ndarray) -> dict[str, Any]:
    residual = calculated - observed
    observed_norm = float(np.linalg.norm(observed))
    calculated_norm = float(np.linalg.norm(calculated))
    sigma = np.sqrt(np.maximum(observed, 1.0))
    poisson_denominator = float(np.sum((observed / sigma) ** 2))
    return {
        "relative_l2": float(np.linalg.norm(residual) / observed_norm),
        "normalized_l1": float(np.sum(np.abs(residual)) / np.sum(np.abs(observed))),
        "cosine_similarity": float(
            np.dot(observed, calculated) / (observed_norm * calculated_norm)
        ),
        "pearson_correlation": float(np.corrcoef(observed, calculated)[0, 1]),
        "poisson_rwp": float(np.sqrt(np.sum((residual / sigma) ** 2) / poisson_denominator)),
        "poisson_block_reason": None,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.cycles <= 0:
        raise ValueError("cycles must be positive")
    if not math.isfinite(arguments.initial_zero_shift_deg):
        raise ValueError("initial zero shift must be finite")
    actual_revision = revision(arguments.gsas_root)
    if actual_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II revision mismatch: expected {PINNED_REVISION}, got {actual_revision}"
        )
    model = json.loads((arguments.bundle / "model.json").read_text(encoding="utf-8"))
    source_data = np.loadtxt(arguments.bundle / "pattern.xye")
    if source_data.ndim != 2 or source_data.shape[1] != 3:
        raise ValueError("opXRD common-model pattern must have three XYE columns")
    x, source_observed, _source_sigma = source_data.T
    sys.path.insert(0, str(arguments.gsas_root))
    if arguments.binary_dir is not None:
        sys.path.insert(0, str(arguments.binary_dir))
    from GSASII import GSASIIpath

    if arguments.binary_dir is not None:
        GSASIIpath.binaryPath = str(arguments.binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable as G2sc

    common = model["common_model"]
    with tempfile.TemporaryDirectory(prefix="phasesmith-opxrd-gsasii-") as name:
        temporary = Path(name)
        instrument_path = temporary / "common.instprm"
        instrument_path.write_text(instrument_text(model), encoding="utf-8")
        project = G2sc.G2Project(newgpx=str(temporary / "opxrd.gpx"))
        histogram = project.add_powder_histogram(
            str(arguments.bundle / "pattern.xye"),
            str(instrument_path),
            fmthint="Topas",
        )
        histogram.set_refinements(
            {
                "Limits": [float(x[0]), float(x[-1])],
                "Background": {
                    "type": "chebyschev-1",
                    "refine": True,
                    "no. coeffs": int(common["background_terms"]),
                },
            }
        )
        instrument_values = histogram.data["Instrument Parameters"][0]
        instrument_values["Zero"][0] = arguments.initial_zero_shift_deg
        instrument_values["Zero"][1] = arguments.initial_zero_shift_deg
        instrument_values["Zero"][2] = False
        histogram.data["Sample Parameters"]["Scale"][1] = False
        phase = project.add_phase(
            str(arguments.bundle / "phase.cif"),
            phasename="opxrd",
            histograms=[histogram],
            fmthint="CIF",
        )
        phase.set_HAP_refinements({"Scale": True})
        project.set_Controls("cycles", arguments.cycles)
        stage_rwp = {"scale_background": refine(project, histogram)}
        histogram.set_refinements({"Instrument Parameters": ["Zero"]})
        stage_rwp["position"] = refine(project, histogram)
        histogram.clear_refinements({"Instrument Parameters": ["Zero"]})
        histogram.set_refinements({"Instrument Parameters": ["U", "V", "W"]})
        stage_rwp["instrument_widths"] = refine(project, histogram)
        phase.set_HAP_refinements(
            {
                "Size": {"type": "isotropic", "value": 0.25, "refine": True},
                "Mustrain": {"type": "isotropic", "value": 1000.0, "refine": True},
            }
        )
        stage_rwp["sample_broadening"] = refine(project, histogram)
        observed = np.asarray(histogram.getdata("Yobs"), dtype=np.float64)
        calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
        background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
        if observed.shape != source_observed.shape or not np.allclose(
            observed, source_observed, rtol=0.0, atol=1.0e-8
        ):
            raise RuntimeError("GSAS-II did not preserve the supplied observation vector")
        metrics = residual_metrics(observed, calculated)
        signal_correlation = float(
            np.corrcoef(observed - background, calculated - background)[0, 1]
        )
        instrument = histogram.data["Instrument Parameters"][0]
        result = {
            "schema_version": 1,
            "scope": "opxrd_structural_common_model",
            "revision": actual_revision,
            "case_id": model["case_id"],
            "member_sha256": model["member_sha256"],
            "selection_sha256": model["selection_sha256"],
            "sample_count": int(x.size),
            "reflection_count": int(
                sum(len(value["RefList"]) for value in histogram.reflections().values())
            ),
            "wavelengths_angstrom": model["wavelengths_angstrom"],
            "relative_intensities": model["relative_intensities"],
            "initial_zero_shift_deg": arguments.initial_zero_shift_deg,
            "residual_metrics": metrics,
            "profile_correlation": signal_correlation,
            "stage_poisson_rwp": stage_rwp,
            "profile_parameters": {
                name: float(instrument[name][1])
                for name in ("U", "V", "W", "X", "Y", "Zero", "SH/L")
            },
        }
    finite_values = (
        *metrics.values(),
        signal_correlation,
        *stage_rwp.values(),
        *result["profile_parameters"].values(),
    )
    if not all(
        value is None or isinstance(value, str) or math.isfinite(value) for value in finite_values
    ):
        raise RuntimeError("GSAS-II opXRD result contains non-finite values")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
