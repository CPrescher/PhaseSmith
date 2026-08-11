#!/usr/bin/env python3
"""Run the XRED anatase/rutile common model in pinned GSAS-II."""

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
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=12)
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
        raise RuntimeError("GSAS-II XRED stage did not produce a finite Rwp")
    return float(value) / 100.0


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

    data = np.loadtxt(arguments.data_directory / "data.csv", delimiter=",")
    x, observed = data.T
    with tempfile.TemporaryDirectory(prefix="phasesmith-xred-tio2-gsasii-") as name:
        temporary = Path(name)
        data_path = temporary / "pattern.xye"
        np.savetxt(data_path, np.column_stack((x, observed, np.sqrt(np.maximum(observed, 1.0)))))
        instrument_path = temporary / "instrument.instprm"
        instrument_path.write_text(
            "#GSAS-II instrument parameter file; do not add/delete items!\n"
            "Type:PXC\nBank:1.0\nLam:1.54051\nZero:0.0\nPolariz.:0.7\n"
            "U:2.0\nV:-2.0\nW:5.0\nX:0.0\nY:0.0\nZ:0.0\n"
            "SH/L:0.002\nAzimuth:0.0\n",
            encoding="utf-8",
        )
        project = G2sc.G2Project(newgpx=str(temporary / "xred.gpx"))
        histogram = project.add_powder_histogram(
            str(data_path), str(instrument_path), fmthint="Topas"
        )
        histogram.set_refinements(
            {
                "Limits": [float(x[0]), float(x[-1])],
                "Background": {"type": "chebyschev-1", "refine": True, "no. coeffs": 8},
            }
        )
        histogram.data["Sample Parameters"]["Scale"][1] = False
        phases = []
        for phase_id in ("anatase", "rutile"):
            phase_path = arguments.data_directory / f"{phase_id}.cif"
            normalized_path = arguments.data_directory / f"{phase_id}-gsas.cif"
            if normalized_path.is_file():
                phase_path = normalized_path
            phase = project.add_phase(
                str(phase_path),
                phasename=phase_id,
                histograms=[histogram],
                fmthint="CIF",
            )
            phase.set_HAP_refinements({"Scale": True})
            phases.append(phase)
        project.set_Controls("cycles", arguments.cycles)
        stage_rwp = {"scale_background": refine(project, histogram)}
        for phase in phases:
            phase.set_refinements({"Cell": True})
        stage_rwp["cell"] = refine(project, histogram)
        histogram.set_refinements({"Instrument Parameters": ["U", "V", "W", "Zero"]})
        stage_rwp["instrument"] = refine(project, histogram)
        for phase in phases:
            phase.set_HAP_refinements(
                {
                    "Size": {"type": "isotropic", "value": 1.0, "refine": True},
                    "Mustrain": {"type": "isotropic", "refine": True},
                }
            )
        stage_rwp["sample_broadening"] = refine(project, histogram)
        mass_fractions = histogram.ComputeMassFracs()
        fractions = {phase.name: float(mass_fractions[phase.name][0]) for phase in phases}
        calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
        background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
        residual = calculated - observed
        cells = {
            phase.name: (
                float(phase.get_cell()["length_a"]),
                float(phase.get_cell()["length_c"]),
            )
            for phase in phases
        }
        result = {
            "schema_version": 1,
            "scope": "xred_tio2_anatase_rutile_common_model",
            "revision": actual_revision,
            "sample_count": int(x.size),
            "weight_fractions": fractions,
            "cells_angstrom": cells,
            "poisson_rwp": float(histogram.get_wR()) / 100.0,
            "unit_weight_rwp": float(np.linalg.norm(residual) / np.linalg.norm(observed)),
            "profile_correlation": float(
                np.corrcoef(observed - background, calculated - background)[0, 1]
            ),
            "stage_poisson_rwp": stage_rwp,
        }
    if not all(
        math.isfinite(value)
        for value in (
            *fractions.values(),
            *(value for cell in cells.values() for value in cell),
            result["poisson_rwp"],
            result["unit_weight_rwp"],
            result["profile_correlation"],
        )
    ):
        raise RuntimeError("GSAS-II XRED result contains non-finite values")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
