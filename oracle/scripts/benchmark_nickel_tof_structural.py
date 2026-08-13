#!/usr/bin/env python3
"""Run structural LANL nickel TOF refinement with pinned external GSAS-II.

This isolated worker imports no PhaseSmith module. Public scripting APIs build
one Fm-3m Ni phase linked to banks 2--4, refine bank-local background, scale,
and Zero together with the shared cubic cell and Ni Uiso, then export only
plain JSON and NumPy arrays.
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
from gsas_bank_view import write_gsas_bank_view

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
BANKS = (2, 3, 4)
FIT_LIMITS_US = (1_101.6, 8_189.6)
INITIAL_CELL_ANGSTROM = 3.522
INTENSITY_UNIT_SCALE = 1.0e6
REFERENCE_HAP_SCALE = 1.0


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


def configure_gsasii(root: Path, binary_dir: Path) -> tuple[Any, Any]:
    sys.path[:0] = [str(root), str(binary_dir)]
    from GSASII import GSASIIpath

    GSASIIpath.binaryPath = str(binary_dir)
    GSASIIpath.BinaryPathLoaded = True
    GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable

    return GSASIIpath, GSASIIscriptable


def neutralize_sample_broadening(phase: Any) -> None:
    """Set each HAP broadening and preferred-orientation term to neutral."""

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


def selected_arrays(histogram: Any) -> dict[str, np.ndarray]:
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
    return {name: np.ascontiguousarray(values[selected]) for name, values in arrays.items()}


def instrument_values(histogram: Any) -> dict[str, float]:
    instrument = histogram.data["Instrument Parameters"][0]
    return {name: float(instrument[name][1]) for name in ("Zero", "difC", "difA", "difB")}


def rescale_intensity_units(histogram: Any) -> None:
    """Move density-valued observations into a stable GSAS-II scale range."""

    observed = histogram.data["data"][1][1]
    weight = histogram.data["data"][1][2]
    observed *= INTENSITY_UNIT_SCALE
    weight /= INTENSITY_UNIT_SCALE**2


def projected_hap_scale(histogram: Any) -> float:
    """Project the calculated structural profile onto one bank's observations."""

    arrays = selected_arrays(histogram)
    profile = arrays["calculated_y"] - arrays["background_y"]
    target = arrays["observed_y"] - arrays["background_y"]
    weight = arrays["weight"]
    denominator = float(np.dot(weight, profile * profile))
    numerator = float(np.dot(weight, profile * target))
    scale = REFERENCE_HAP_SCALE * numerator / denominator
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise RuntimeError(
            f"invalid projected HAP scale for {histogram.name}: {scale!r}; "
            f"numerator={numerator!r}, denominator={denominator!r}, "
            f"profile_range=({profile.min()!r}, {profile.max()!r}), "
            f"target_range=({target.min()!r}, {target.max()!r}), "
            f"profile_peak_us={arrays['x_us'][int(np.argmax(profile))]!r}, "
            f"target_peak_us={arrays['x_us'][int(np.argmax(target))]!r}"
        )
    return scale


def run_workflow(
    scripting: Any,
    data: Path,
    cycles: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-nickel-structural-") as temporary:
        work = Path(temporary)
        project = scripting.G2Project(newgpx=str(work / "nickel-structural.gpx"))
        bank_data = {}
        for bank in BANKS:
            bank_data[bank] = work / f"nickel-bank-{bank}.raw"
            write_gsas_bank_view(data / "nickel.raw", bank, bank_data[bank])
        histograms = [
            project.add_powder_histogram(
                str(bank_data[bank]),
                str(data / "inst_tof.prm"),
                fmthint="GSAS",
                databank=1,
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
            rescale_intensity_units(histogram)
            phase.data["Histograms"][histogram.name]["Scale"][0] = REFERENCE_HAP_SCALE
            histogram.set_refinements(
                {
                    "Limits": list(FIT_LIMITS_US),
                    "Background": {
                        "type": "chebyschev-1",
                        "refine": False,
                        "no. coeffs": 12,
                    },
                }
            )
            background = histogram.data["Background"][0]
            background[3:] = [0.0] * int(background[2])
            background[3] = float(np.median(histogram.getdata("Yobs")))
            histogram.data["Sample Parameters"]["Scale"][1] = False
        phase.set_HAP_refinements({"Scale": False})
        project.set_Controls("cycles", 1)
        project.do_refinements([{}], outputnames=[None])
        phase = project.phase("Ni")
        histograms = [project.histogram(index) for index in range(len(BANKS))]
        starting_scales = {
            histogram.name: projected_hap_scale(histogram) for histogram in histograms
        }
        for histogram in histograms:
            phase.data["Histograms"][histogram.name]["Scale"][0] = starting_scales[histogram.name]
        phase.set_HAP_refinements({"Scale": True})
        project.set_Controls("cycles", cycles)
        project.refine(makeBack=False)
        phase = project.phase("Ni")
        histograms = [project.histogram(index) for index in range(len(BANKS))]
        for histogram in histograms:
            histogram.set_refinements(
                {
                    "Background": {
                        "type": "chebyschev-1",
                        "refine": True,
                        "no. coeffs": 12,
                    }
                }
            )
        phase.set_HAP_refinements({"Scale": True})
        project.refine(makeBack=False)

        phase = project.phase("Ni")
        phase.set_refinements({"Cell": True, "Atoms": {"all": "U"}})
        histograms = [project.histogram(index) for index in range(len(BANKS))]
        for histogram in histograms:
            histogram.set_refinements({"Instrument Parameters": ["Zero"]})
        project.refine(makeBack=False)

        phase = project.phase("Ni")
        histograms = [project.histogram(index) for index in range(len(BANKS))]
        archive: dict[str, np.ndarray] = {}
        bank_results = []
        joint_numerator = 0.0
        joint_denominator = 0.0
        for bank, histogram in zip(BANKS, histograms, strict=True):
            arrays = selected_arrays(histogram)
            for name, values in arrays.items():
                archive[f"bank_{bank}_{name}"] = values
            archive[f"bank_{bank}_reflection_list"] = np.ascontiguousarray(
                histogram.reflections()["Ni"]["RefList"], dtype=np.float64
            )
            observed = arrays["observed_y"]
            calculated = arrays["calculated_y"]
            background = arrays["background_y"]
            weight = arrays["weight"]
            residual = calculated - observed
            joint_numerator += float(np.dot(weight, residual * residual))
            joint_denominator += float(np.dot(weight, observed * observed))
            bank_results.append(
                {
                    "bank": bank,
                    "sample_count": int(observed.size),
                    "reflection_count": len(histogram.reflections()["Ni"]["RefList"]),
                    "starting_hap_scale": starting_scales[histogram.name],
                    "hap_scale": float(phase.data["Histograms"][histogram.name]["Scale"][0]),
                    "poisson_rwp": float(
                        np.sqrt(
                            np.dot(weight, residual * residual)
                            / np.dot(weight, observed * observed)
                        )
                    ),
                    "profile_correlation": float(
                        np.corrcoef(observed - background, calculated - background)[0, 1]
                    ),
                    "instrument": instrument_values(histogram),
                }
            )
        if any(
            not np.isfinite(item[metric])
            for item in bank_results
            for metric in ("hap_scale", "poisson_rwp", "profile_correlation")
        ):
            raise RuntimeError(f"non-finite structural result: {bank_results!r}")
        cell = phase.get_cell()
        atoms = phase.atoms()
        if len(atoms) != 1:
            raise RuntimeError("expected one Ni asymmetric-unit site")
        covariance = project.data.get("Covariance", {}).get("data", {})
        return (
            {
                "bank_count": len(bank_results),
                "sample_count": sum(item["sample_count"] for item in bank_results),
                "joint_poisson_rwp": float(np.sqrt(joint_numerator / joint_denominator)),
                "minimum_profile_correlation": min(
                    item["profile_correlation"] for item in bank_results
                ),
                "cell_angstrom": float(cell["length_a"]),
                "u_iso_angstrom2": float(atoms[0].data[atoms[0].cia + 1]),
                "free_parameter_count": len(covariance.get("varyList", [])),
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
    gsasii_path, scripting = configure_gsasii(root, arguments.binary_dir.resolve())
    result, arrays = run_workflow(scripting, arguments.data_directory.resolve(), arguments.cycles)
    arguments.archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.archive, **arrays)
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "scope": "lanl_nickel_multibank_tof_structural",
        "revision": revision,
        "tag": int(gsasii_path.GetVersionNumber()),
        "input_sha256": {
            name: sha256(arguments.data_directory / name) for name in ("nickel.raw", "inst_tof.prm")
        },
        "oracle_behavior": {
            "banks": BANKS,
            "fit_limits_us": FIT_LIMITS_US,
            "fit_endpoint_convention": (
                "GSAS-II selects 4430 samples per bank; PhaseSmith includes 4431"
            ),
            "initial_cell_angstrom": INITIAL_CELL_ANGSTROM,
            "refined_parameters": [
                "shared cubic cell",
                "shared Ni Uiso",
                "bank-local HAP scale",
                "bank-local Zero",
                "bank-local 12-term background",
            ],
            "incident_spectrum": "type-4 bank calibration applied by GSAS-II powder import",
            "intensity_unit_scale": INTENSITY_UNIT_SCALE,
            "scale_initialization": "weighted projection from a unit HAP reference scale",
            "sample_broadening": "effectively neutral fixed HAP values",
            "private_probe": [],
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
