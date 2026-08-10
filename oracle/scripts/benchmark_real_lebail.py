#!/usr/bin/env python3
"""Run a real constant-wavelength Le Bail case with pinned external GSAS-II.

This worker imports no PhaseSmith module. It creates a temporary GSAS-II
project, runs a pinned-revision Le Bail extraction/refinement, and writes only
plain finite arrays/records to its JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
CASES = {
    "aps-sucrose-11bmb": {
        "pattern": "11bmb_8716.fxye",
        "instrument": "11bmb_8716.prm",
        "format_hint": None,
        "limits": [1.00093351, 23.99988092],
        "space_group": "P 21",
        "cell": [
            7.715231369389035,
            8.663866877499101,
            10.809618877725404,
            90.0,
            102.98249193732556,
            90.0,
        ],
        "atom": [0.0, 0.0, 0.0, "C"],
        "instrument_parameters": ["U", "V", "W", "X", "Y"],
        "matched_instrument": {
            "Lam": 0.413259,
            "Zero": 0.0,
            "U": 1.163,
            "V": -0.126,
            "W": 0.063,
            "X": 0.173,
            "Y": 0.0,
            "Z": 0.0,
            "SH/L": 0.0,
        },
        "background_terms": 1,
        "background_start": [0.0],
        "fixed_background_required": True,
    },
    "ansto-echidna-lab6-cw-neutron": {
        "pattern": "ECH0034258_LaB6.xyd",
        "instrument": None,
        "format_hint": "Topas",
        "limits": [20.0, 125.5],
        "space_group": "P m -3 m",
        "cell": [4.156826, 4.156826, 4.156826, 90.0, 90.0, 90.0],
        "atom": [0.0, 0.0, 0.0, "La"],
        "instrument_parameters": ["U", "V", "W", "X", "Y"],
        "matched_instrument": {
            "Lam": 2.047,
            "Zero": 0.0,
            "U": 140.0,
            "V": -2530.0,
            "W": 5160.0,
            "X": 5.0,
            "Y": 0.0,
            "Z": 0.0,
            "SH/L": 0.0,
        },
        "background_terms": 10,
        "background_start": [1.0] + [0.0] * 9,
        "fixed_background_required": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixed-background", type=Path)
    parser.add_argument("--cycles", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def configure_gsasii(root: Path, binary_directory: Path | None) -> tuple[Any, Any, float]:
    started = time.perf_counter_ns()
    sys.path.insert(0, str(root))
    from GSASII import GSASIIpath

    if binary_directory is not None:
        binary_directory = binary_directory.resolve()
        sys.path.insert(0, str(binary_directory))
        GSASIIpath.binaryPath = str(binary_directory)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable, GSASIIstrMain

    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
    return GSASIIscriptable, GSASIIstrMain, elapsed_ms


def echidna_instrument(path: Path) -> None:
    path.write_text(
        "".join(
            [
                "# GSAS-II instrument parameters for the deposited Echidna pattern\n",
                "Type:PNC;Bank:1\n",
                "Lam:2.047;Zero:0;Polariz.:0;Azimuth:0\n",
                "U:140;V:-2530;W:5160;X:5;Y:0;Z:0;SH/L:0\n",
            ]
        ),
        encoding="utf-8",
    )


def finite_result(histogram: Any, phase_name: str) -> dict[str, Any]:
    x = np.asarray(histogram.getdata("X"), dtype=np.float64)
    observed = np.asarray(histogram.getdata("Yobs"), dtype=np.float64)
    calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
    background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
    low, high = histogram.data["Limits"][1]
    selected = (x >= low) & (x <= high)
    residual = calculated[selected] - observed[selected]
    unit_rwp = float(np.sqrt(np.sum(residual**2) / np.sum(observed[selected] ** 2)))
    correlation = float(
        np.corrcoef(
            observed[selected] - background[selected],
            calculated[selected] - background[selected],
        )[0, 1]
    )
    weighted_rwp = histogram.get_wR()
    if weighted_rwp is None:
        raise RuntimeError("GSAS-II Le Bail result does not contain Rwp")
    reflections = np.asarray(histogram.reflections()[phase_name]["RefList"], dtype=np.float64)
    background_record = histogram.data["Background"][0]
    instrument_record = histogram.data["Instrument Parameters"][0]
    result = {
        "sample_count": int(selected.sum()),
        "reflection_count": int(reflections.shape[0]),
        "poisson_rwp": float(weighted_rwp) / 100.0,
        "unit_weight_rwp": unit_rwp,
        "profile_correlation": correlation,
        "limits_deg": [float(low), float(high)],
        "background_coefficients": [
            float(value) for value in background_record[3 : 3 + int(background_record[2])]
        ],
        "instrument": {
            name: float(instrument_record[name][1])
            for name in ("Lam", "Zero", "U", "V", "W", "X", "Y", "Z", "SH/L")
        },
    }
    if not all(np.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise RuntimeError("GSAS-II Le Bail result contains a non-finite value")
    return result


def run_workflow(
    scripting: Any,
    structure_main: Any,
    case_id: str,
    data: Path,
    cycles: int,
    fixed_background: Path | None,
) -> dict[str, Any]:
    definition = CASES[case_id]
    total_started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-real-lebail-") as temporary:
        directory = Path(temporary)
        instrument = (
            data / definition["instrument"]
            if definition["instrument"] is not None
            else directory / "echidna.instprm"
        )
        if definition["instrument"] is None:
            echidna_instrument(instrument)
        setup_started = time.perf_counter_ns()
        project = scripting.G2Project(newgpx=str(directory / "real-lebail.gpx"))
        histogram_arguments: dict[str, Any] = {}
        if definition["format_hint"] is not None:
            histogram_arguments["fmthint"] = definition["format_hint"]
        histogram = project.add_powder_histogram(
            str(data / definition["pattern"]),
            str(instrument),
            **histogram_arguments,
        )
        if definition["fixed_background_required"]:
            if fixed_background is None or not fixed_background.is_file():
                raise FileNotFoundError("matched sucrose workflow requires a fixed background")
            fixed_histogram = project.add_powder_histogram(
                str(fixed_background),
                str(instrument),
                fmthint="Topas",
            )
            histogram.set_background("fixedHist", fixed_histogram)
            histogram.set_background("fixedFileMult", 1.0)
        instrument_record = histogram.data["Instrument Parameters"][0]
        for name, value in definition["matched_instrument"].items():
            instrument_record[name][:2] = [value, value]
        phase_name = f"oracle-{case_id}"
        phase = project.add_phase(
            phasename=phase_name,
            spacegroup=definition["space_group"],
            cell=definition["cell"],
            histograms=[histogram],
        )
        x, y, z, element = definition["atom"]
        phase.add_atom(x, y, z, element=element, lbl=f"{element}1", occ=1.0, uiso=0.005)
        histogram.set_refinements(
            {
                "Limits": definition["limits"],
                "Background": {
                    "type": "chebyschev-1",
                    "refine": True,
                    "no. coeffs": definition["background_terms"],
                },
            }
        )
        histogram.data["Background"][0][3:] = definition["background_start"]
        histogram.data["Sample Parameters"]["Scale"][1] = False
        phase.set_HAP_refinements({"Scale": False})
        project.set_Controls("cycles", 1)
        if definition["fixed_background_required"]:
            phase.set_refinements({"LeBail": True})
            histogram.set_refinements(
                {"Instrument Parameters": definition["instrument_parameters"]}
            )
            project.index_ids()
            project.save()
        else:
            project.do_refinements([{}], outputnames=[None])
            phase.set_refinements({"LeBail": True})
            project.index_ids()
        setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

        refinement_started = time.perf_counter_ns()
        structure_main.Refine(project.filename, newLeBail=True)
        project.reload()
        convergence = [float(project.histogram(0).get_wR())]
        if not definition["fixed_background_required"]:
            project.histogram(0).set_refinements(
                {"Instrument Parameters": definition["instrument_parameters"]}
            )
        for _ in range(1, cycles):
            project.refine(makeBack=False)
            convergence.append(float(project.histogram(0).get_wR()))
        refinement_ms = (time.perf_counter_ns() - refinement_started) / 1.0e6
        result = finite_result(project.histogram(0), phase_name)
        return {
            "result": result,
            "cycle_rwp_percent": convergence,
            "timing_ms": {
                "setup": setup_ms,
                "refinement": refinement_ms,
                "total_workflow": (time.perf_counter_ns() - total_started) / 1.0e6,
            },
        }


def main() -> None:
    arguments = parse_args()
    if arguments.cycles <= 0:
        raise ValueError("cycles must be positive")
    root = arguments.gsas_root.resolve()
    detected_revision = revision(root)
    if detected_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II real Le Bail oracle requires revision {PINNED_REVISION}, "
            f"detected {detected_revision}"
        )
    data = arguments.data_directory.resolve()
    definition = CASES[arguments.case]
    required = [definition["pattern"]]
    if definition["instrument"] is not None:
        required.append(definition["instrument"])
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError(f"real Le Bail data directory is missing {', '.join(missing)}")
    scripting, structure_main, import_ms = configure_gsasii(root, arguments.binary_dir)
    workflow = run_workflow(
        scripting,
        structure_main,
        arguments.case,
        data,
        arguments.cycles,
        arguments.fixed_background,
    )
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": detected_revision,
        "scope": f"{arguments.case}_lebail_workflow",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "recipe": {
            "cycles": arguments.cycles,
            "limits_deg": definition["limits"],
            "space_group": definition["space_group"],
            "starting_cell": definition["cell"],
            "background": (
                "fixed Smooth Bruckner + 1-term refined Chebyshev residual"
                if definition["fixed_background_required"]
                else "10-term refined Chebyshev"
            ),
            "background_start": definition["background_start"],
            "instrument_parameters": definition["instrument_parameters"],
            "instrument_start": definition["matched_instrument"],
            "staging": (
                "newLeBail extraction, then matched one-cycle background/profile updates"
                if definition["fixed_background_required"]
                else "background setup, newLeBail extraction, then one-cycle profile updates"
            ),
            "generated_instrument": (
                None
                if definition["instrument"] is not None
                else {
                    "Type": "PNC",
                    "Lam": 2.047,
                    "Zero": 0.0,
                    "U": 140.0,
                    "V": -2530.0,
                    "W": 5160.0,
                    "X": 5.0,
                    "Y": 0.0,
                }
            ),
            "le_bail": True,
            "private_probe": "GSASIIstrMain.Refine(newLeBail=True)",
        },
        "input_sha256": {
            **{name: sha256(data / name) for name in required},
            **(
                {"prepared_fixed_background": sha256(arguments.fixed_background)}
                if arguments.fixed_background is not None
                else {}
            ),
        },
        "import_ms": import_ms,
        **workflow,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
