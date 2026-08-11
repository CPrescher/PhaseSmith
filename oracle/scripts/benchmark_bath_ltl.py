#!/usr/bin/env python3
"""Run the Bath primary-CIF common model in an exact pinned GSAS-II checkout."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "bath_ltl_lab_xray_conversion_fidelity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--bundle-directory", required=True, type=Path)
    parser.add_argument("--sample", required=True, choices=("K", "Li", "Cs"))
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def revision(root: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def metrics(histogram: object, released_background: np.ndarray) -> dict[str, float]:
    x = np.asarray(histogram.getdata("X"), dtype=np.float64)
    observed = np.asarray(histogram.getdata("Yobs"), dtype=np.float64)
    calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
    background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
    residual = calculated - observed
    weights = 1.0 / np.maximum(observed, 1.0)
    return {
        "poisson_rwp": float(
            np.sqrt(np.sum(weights * residual**2) / np.sum(weights * observed**2))
        ),
        "unit_weight_rwp": float(np.linalg.norm(residual) / np.linalg.norm(observed)),
        "profile_correlation": float(
            np.corrcoef(observed - background, calculated - background)[0, 1]
        ),
        "fixed_background_max_abs_difference": float(
            np.max(np.abs(background - released_background))
        ),
        "sample_count": int(x.size),
    }


def main() -> None:
    arguments = parse_args()
    if arguments.cycles <= 0:
        raise ValueError("cycles must be positive")
    actual_revision = revision(arguments.gsas_root)
    if actual_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II revision mismatch: expected {PINNED_REVISION}, got {actual_revision}"
        )
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

    root = arguments.bundle_directory
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("scope") != SCOPE:
        raise ValueError("unsupported Bath LTL bundle")
    record = manifest["samples"][arguments.sample]
    profile = np.loadtxt(root / record["released_profile"], delimiter=",", skiprows=1)
    x, observed, _released_calculated, released_background = profile.T
    legacy = record["legacy_gsas"]
    parameters = legacy["profile"]
    with tempfile.TemporaryDirectory(prefix="phasesmith-bath-ltl-gsasii-") as name:
        temporary = Path(name)
        data_path = temporary / "pattern.xye"
        np.savetxt(data_path, np.column_stack((x, observed, np.sqrt(np.maximum(observed, 1.0)))))
        instrument_path = temporary / "instrument.instprm"
        instrument_path.write_text(
            "#GSAS-II instrument parameter file; do not add/delete items!\n"
            "Type:PXC\nBank:1.0\n"
            f"Lam:{legacy['wavelength_angstrom']}\nZero:0.0\nPolariz.:0.7\n"
            f"U:{10000.0 * parameters['u_deg2']}\n"
            f"V:{10000.0 * parameters['v_deg2']}\n"
            f"W:{10000.0 * parameters['w_deg2']}\n"
            f"X:{100.0 * parameters['x_deg']}\n"
            f"Y:{100.0 * parameters['y_deg']}\nZ:0.0\n"
            f"SH/L:{parameters['sh_over_l']}\nAzimuth:0.0\n",
            encoding="utf-8",
        )
        project = G2sc.G2Project(newgpx=str(temporary / "bath.gpx"))
        histogram = project.add_powder_histogram(
            str(data_path), str(instrument_path), fmthint="Topas"
        )
        histogram.set_refinements({"Limits": [float(x[0]), float(x[-1])]})
        histogram.data["Sample Parameters"]["Scale"][1] = False
        phase = project.add_phase(
            str(root / record["phase"]),
            phasename="LTL",
            histograms=[histogram],
            fmthint="CIF",
        )
        phase.set_HAP_refinements({"Scale": True})
        background_record = histogram.data["Background"]
        background_record[0] = ["chebyschev-1", False, 1, 0.0]
        background_record[1]["fixback"] = released_background.copy()
        background_record[1]["background PWDR"] = ["", 1.0, False]
        project.set_Controls("cycles", arguments.cycles)
        project.do_refinements([{}], outputnames=[None])
        fixed = metrics(histogram, released_background)
        histogram.set_refinements(
            {"Instrument Parameters": ["U", "V", "W", "X", "Y", "SH/L", "Zero"]}
        )
        project.do_refinements([{}], outputnames=[None])
        refined = metrics(histogram, released_background)
        instrument = {
            key: float(value[1])
            for key, value in histogram.InstrumentParameters.items()
            if key in {"U", "V", "W", "X", "Y", "SH/L", "Zero"}
        }
    result = {
        "schema_version": 1,
        "scope": SCOPE,
        "revision": actual_revision,
        "sample": arguments.sample,
        "released_poisson_rwp": float(legacy["recomputed_poisson_rwp"]),
        "fixed_profile": fixed,
        "refined_profile": refined,
        "refined_instrument": instrument,
    }
    if not all(math.isfinite(value) for section in (fixed, refined) for value in section.values()):
        raise RuntimeError("GSAS-II Bath result contains non-finite values")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
