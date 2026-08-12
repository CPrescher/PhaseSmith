#!/usr/bin/env python3
"""Run the LANL nickel multi-bank TOF case with pinned external GSAS-II.

This worker imports no PhaseSmith module. Public scripting APIs create one
phase linked to banks 2--4 and refine the shared cell plus bank-local Zero
terms. The version-gated ``newLeBail`` call is the only workflow probe; output
is restricted to plain JSON and NumPy arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
BANKS = (2, 3, 4)
FIT_LIMITS_US = (1_101.6, 8_189.6)
INITIAL_CELL_ANGSTROM = 3.523
REFLECTION_COLUMNS = (
    "h",
    "k",
    "l",
    "multiplicity",
    "d_spacing_angstrom",
    "position_us",
    "sigma2_us2",
    "gamma_us",
    "f_obs2",
    "f_calc2",
    "phase_deg",
    "intensity_correction",
    "alpha_per_us",
    "beta_per_us",
    "wavelength_angstrom",
    "preferred_orientation",
    "transmission",
    "extinction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=20)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_gsasii(root: Path, binary_dir: Path) -> tuple[Any, Any, Any]:
    sys.path[:0] = [str(root), str(binary_dir)]
    from GSASII import GSASIIpath

    GSASIIpath.binaryPath = str(binary_dir)
    GSASIIpath.BinaryPathLoaded = True
    GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable, GSASIIstrMain

    return GSASIIpath, GSASIIscriptable, GSASIIstrMain


def neutralize_sample_broadening(phase: Any) -> None:
    """Set every bank's HAP broadening to the instrument-only limit."""

    for key, configure in (
        ("Size", lambda current: ["isotropic", [1.0e12, current[1][1], 1.0], *current[2:]]),
        ("Mustrain", lambda current: ["isotropic", [0.0, current[1][1], 0.0], *current[2:]]),
        ("Pref.Ori.", lambda current: ["MD", 1.0, False, current[3], *current[4:]]),
    ):
        matches = phase.getHAPentryList(None, key)
        if len(matches) != len(BANKS):
            raise RuntimeError(f"expected one {key} HAP entry per bank")
        for path, _, current in matches:
            phase.setHAPentryValue(path, configure(current))


def instrument_values(histogram: Any) -> dict[str, float]:
    instrument = histogram.data["Instrument Parameters"][0]
    names = (
        "Zero",
        "difC",
        "difA",
        "difB",
        "alpha",
        "beta-0",
        "beta-1",
        "sig-0",
        "sig-1",
        "sig-2",
        "X",
        "Y",
        "Z",
    )
    return {name: float(instrument[name][1]) for name in names}


def selected_arrays(histogram: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays = {
        "x_us": np.asarray(histogram.getdata("X"), dtype=np.float64),
        "observed_y": np.asarray(histogram.getdata("Yobs"), dtype=np.float64),
        "calculated_y": np.asarray(histogram.getdata("Ycalc"), dtype=np.float64),
        "background_y": np.asarray(histogram.getdata("Background"), dtype=np.float64),
        "weight": np.asarray(histogram.getdata("Yweight"), dtype=np.float64),
    }
    selected = (arrays["x_us"] >= FIT_LIMITS_US[0]) & (arrays["x_us"] <= FIT_LIMITS_US[1])
    if int(np.count_nonzero(selected)) != 4_430:
        raise RuntimeError("unexpected LANL nickel selected sample count")
    return {name: np.ascontiguousarray(value[selected]) for name, value in arrays.items()}, selected


def run_workflow(
    scripting: Any,
    structure_main: Any,
    data: Path,
    cycles: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-nickel-") as temporary:
        project = scripting.G2Project(newgpx=str(Path(temporary) / "nickel.gpx"))
        histograms = [
            project.add_powder_histogram(
                str(data / "nickel.raw"),
                str(data / "inst_tof.prm"),
                fmthint="GSAS",
                databank=bank,
                instbank=bank,
            )
            for bank in BANKS
        ]
        phase = project.add_phase(
            phasename="Ni",
            spacegroup="F m -3 m",
            cell=[INITIAL_CELL_ANGSTROM] * 3 + [90.0, 90.0, 90.0],
            histograms=histograms,
        )
        phase.add_atom(0.0, 0.0, 0.0, element="Ni", lbl="Ni", occ=1.0, uiso=0.01)
        neutralize_sample_broadening(phase)
        for histogram in histograms:
            histogram.set_refinements(
                {
                    "Limits": list(FIT_LIMITS_US),
                    "Background": {
                        "type": "chebyschev-1",
                        "refine": True,
                        "no. coeffs": 12,
                    },
                }
            )
            histogram.data["Sample Parameters"]["Scale"][1] = False
        phase.set_HAP_refinements({"Scale": False})
        project.set_Controls("cycles", 1)
        project.do_refinements([{}], outputnames=[None])
        phase.set_refinements({"LeBail": True})
        project.index_ids()
        structure_main.Refine(project.filename, newLeBail=True)
        project.reload()

        phase = project.phase("Ni")
        phase.set_refinements({"Cell": True})
        histograms = [project.histogram(index) for index in range(len(BANKS))]
        for histogram in histograms:
            histogram.set_refinements({"Instrument Parameters": ["Zero"]})
        project.set_Controls("cycles", cycles)
        project.refine(makeBack=False)

        phase = project.phase("Ni")
        histograms = [project.histogram(index) for index in range(len(BANKS))]
        archive: dict[str, np.ndarray] = {}
        bank_results = []
        joint_numerator = 0.0
        joint_denominator = 0.0
        for bank, histogram in zip(BANKS, histograms, strict=True):
            selected_arrays_by_name, _ = selected_arrays(histogram)
            reflections = np.ascontiguousarray(
                histogram.reflections()["Ni"]["RefList"], dtype=np.float64
            )
            if reflections.ndim != 2 or reflections.shape[1] != len(REFLECTION_COLUMNS):
                raise RuntimeError(
                    f"unexpected bank {bank} reflection-list shape {reflections.shape}"
                )
            for name, values in selected_arrays_by_name.items():
                archive[f"bank_{bank}_{name}"] = values
            archive[f"bank_{bank}_reflection_list"] = reflections
            observed = selected_arrays_by_name["observed_y"]
            calculated = selected_arrays_by_name["calculated_y"]
            background = selected_arrays_by_name["background_y"]
            weight = selected_arrays_by_name["weight"]
            residual = calculated - observed
            joint_numerator += float(np.dot(weight, residual * residual))
            joint_denominator += float(np.dot(weight, observed * observed))
            bank_results.append(
                {
                    "bank": bank,
                    "sample_count": int(observed.size),
                    "reflection_count": int(reflections.shape[0]),
                    "poisson_rwp": float(histogram.get_wR()) / 100.0,
                    "profile_correlation": float(
                        np.corrcoef(observed - background, calculated - background)[0, 1]
                    ),
                    "instrument": instrument_values(histogram),
                }
            )
        cell = phase.get_cell()
        return (
            {
                "bank_count": len(bank_results),
                "sample_count": sum(item["sample_count"] for item in bank_results),
                "joint_poisson_rwp": float(np.sqrt(joint_numerator / joint_denominator)),
                "cell_angstrom": float(cell["length_a"]),
                "banks": bank_results,
                "maximum_refinement_cycles": cycles,
            },
            archive,
        )


def main() -> None:
    arguments = parse_args()
    root = arguments.gsas_root.resolve()
    revision = git_revision(root)
    if revision != PINNED_REVISION:
        raise RuntimeError("GSAS-II checkout does not match the recorded pin")
    gsasii_path, scripting, structure_main = configure_gsasii(root, arguments.binary_dir.resolve())
    result, arrays = run_workflow(
        scripting, structure_main, arguments.data_directory.resolve(), arguments.cycles
    )
    arguments.archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.archive, **arrays)
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "scope": "lanl_nickel_multibank_tof_lebail_geometry",
        "revision": revision,
        "tag": int(gsasii_path.GetVersionNumber()),
        "input_sha256": {
            name: sha256(arguments.data_directory / name) for name in ("nickel.raw", "inst_tof.prm")
        },
        "oracle_behavior": {
            "banks": BANKS,
            "fit_limits_us": FIT_LIMITS_US,
            "fit_endpoint_convention": (
                "GSAS-II selects 4430 samples per bank; PhaseSmith includes both explicit "
                "bin centers and selects 4431"
            ),
            "initial_cell_angstrom": INITIAL_CELL_ANGSTROM,
            "refined_parameters": ["shared cubic cell", "bank-local Zero"],
            "sample_broadening": "effectively neutral fixed HAP values",
            "background": "bank-local 12-term refined chebyschev-1",
            "private_probe": ["GSASIIstrMain.Refine(newLeBail=True)"],
            "reflection_columns": REFLECTION_COLUMNS,
        },
        "archive": {"file": arguments.archive.name, "sha256": sha256(arguments.archive)},
        "result": result,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
