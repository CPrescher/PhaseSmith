#!/usr/bin/env python3
"""Run a citrate/Si common model in pinned GSAS-II."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SUPPORTED_PHASES = {
    "iucr_sodium_dihydrogen_citrate_silicon_holdout": (
        "sodium_dihydrogen_citrate",
        "silicon",
    ),
    "iucr_anhydrous_tripotassium_citrate_silicon_holdout": (
        "tripotassium_citrate",
        "silicon",
    ),
}


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
        raise RuntimeError("GSAS-II citrate/Si stage did not produce finite Rwp")
    return float(value) / 100.0


def update_hap_entry(phase: Any, key: str, transform: Callable[[list[Any]], list[Any]]) -> None:
    entries = phase.getHAPentryList(0, key)
    if len(entries) != 1:
        raise RuntimeError(f"expected one {key} HAP entry, found {len(entries)}")
    key_path = entries[0][0]
    phase.setHAPentryValue(key_path, transform(phase.getHAPentryValue(key_path)))


def configure_sample(phase: Any, *, sodium: bool) -> None:
    def size(current: list[Any]) -> list[Any]:
        current[0] = "isotropic"
        current[1][0] = 1.0e6
        current[1][2] = 1.0
        return current

    def microstrain(current: list[Any]) -> list[Any]:
        current[0] = "isotropic"
        current[1][0] = 0.0
        current[1][2] = 0.0
        return current

    update_hap_entry(phase, "Size", size)
    update_hap_entry(phase, "Mustrain", microstrain)
    if sodium:

        def orientation(current: list[Any]) -> list[Any]:
            current[0] = "MD"
            current[1] = 1.0
            current[2] = False
            current[3] = [0, 0, 1]
            return current

        update_hap_entry(phase, "Pref.Ori.", orientation)


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

    root = arguments.data_directory
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    scope = manifest.get("scope")
    if manifest.get("schema_version") != 1 or scope not in SUPPORTED_PHASES:
        raise ValueError("unsupported IUCr citrate/Si bundle")
    phases_in_scope = SUPPORTED_PHASES[scope]
    data = np.loadtxt(root / "pattern.csv", delimiter=",", skiprows=1)
    x, observed, legacy_calculated, legacy_background = data.T
    instrument = manifest["instrument"]
    profile = instrument.get("tripotassium_citrate_profile", instrument["silicon_profile"])
    wavelengths = instrument["wavelengths_angstrom"]
    with tempfile.TemporaryDirectory(prefix="phasesmith-iucr-na-si-gsasii-") as name:
        temporary = Path(name)
        data_path = temporary / "pattern.xye"
        np.savetxt(data_path, np.column_stack((x, observed, np.sqrt(np.maximum(observed, 1.0)))))

        def write_instrument(path: Path, profile_record: dict[str, float]) -> None:
            path.write_text(
                "#GSAS-II instrument parameter file; do not add/delete items!\n"
                "Type:PXC\nBank:1.0\n"
                f"Lam1:{wavelengths[0]}\nLam2:{wavelengths[1]}\n"
                f"I(L2)/I(L1):{instrument['k_alpha2_over_k_alpha1']}\n"
                f"Zero:{instrument['initial_zero_deg']}\n"
                f"Polariz.:{instrument['polarization_fraction']}\n"
                f"U:{profile_record['U']}\nV:{profile_record['V']}\nW:{profile_record['W']}\n"
                f"X:{profile_record['X']}\nY:{profile_record['Y']}\nZ:0.0\n"
                f"SH/L:{instrument['matched_sh_over_l']}\nAzimuth:0.0\nSource:CuKa\n",
                encoding="utf-8",
            )

        instrument_path = temporary / "instrument.instprm"
        write_instrument(instrument_path, profile)
        calibration_instrument_path = temporary / "silicon-instrument.instprm"
        write_instrument(calibration_instrument_path, instrument["silicon_profile"])
        standard = manifest["silicon_standard"]
        windows = standard.get(
            "calibration_windows_two_theta_deg",
            standard.get("candidate_calibration_windows_two_theta_deg"),
        )
        if windows is None:
            raise ValueError("citrate/Si bundle lacks calibration windows")
        calibration_mask = np.logical_or.reduce(
            [(x >= float(low)) & (x <= float(high)) for low, high in windows]
        )
        calibration_path = temporary / "silicon-calibration.xye"
        np.savetxt(
            calibration_path,
            np.column_stack(
                (
                    x[calibration_mask],
                    observed[calibration_mask],
                    np.sqrt(np.maximum(observed[calibration_mask], 1.0)),
                )
            ),
        )
        calibration_project = G2sc.G2Project(newgpx=str(temporary / "calibration.gpx"))
        calibration_histogram = calibration_project.add_powder_histogram(
            str(calibration_path), str(calibration_instrument_path), fmthint="Topas"
        )
        calibration_histogram.set_refinements(
            {"Limits": [float(x[calibration_mask][0]), float(x[calibration_mask][-1])]}
        )
        calibration_histogram.data["Sample Parameters"]["Scale"][1] = False
        calibration_background = calibration_histogram.data["Background"]
        calibration_background[0] = ["chebyschev-1", True, 1, 0.0]
        calibration_background[1]["fixback"] = legacy_background[calibration_mask].copy()
        calibration_background[1]["background PWDR"] = ["", 1.0, False]
        calibration_silicon = calibration_project.add_phase(
            str(root / "silicon.cif"),
            phasename="silicon",
            histograms=[calibration_histogram],
            fmthint="CIF",
        )
        configure_sample(calibration_silicon, sodium=False)
        calibration_silicon.set_HAP_refinements({"Scale": True})
        calibration_project.set_Controls("cycles", arguments.cycles)
        stage_rwp = {"silicon_scale_background": refine(calibration_project, calibration_histogram)}
        calibration_histogram.set_refinements({"Sample Parameters": ["Shift"]})
        stage_rwp["silicon_displacement_calibration"] = refine(
            calibration_project, calibration_histogram
        )
        calibrated_shift_micrometre = float(
            calibration_histogram.data["Sample Parameters"]["Shift"][0]
        )

        project = G2sc.G2Project(newgpx=str(temporary / "iucr-na-si.gpx"))
        histogram = project.add_powder_histogram(
            str(data_path), str(instrument_path), fmthint="Topas"
        )
        if scope == "iucr_sodium_dihydrogen_citrate_silicon_holdout":
            histogram.data["Sample Parameters"]["Shift"][0] = calibrated_shift_micrometre
        histogram.data["Sample Parameters"]["Shift"][1] = False
        histogram.set_refinements({"Limits": [float(x[0]), float(x[-1])]})
        histogram.data["Sample Parameters"]["Scale"][1] = False
        background_record = histogram.data["Background"]
        background_record[0] = ["chebyschev-1", True, 1, 0.0]
        background_record[1]["fixback"] = legacy_background.copy()
        background_record[1]["background PWDR"] = ["", 1.0, False]
        project.set_Controls("cycles", arguments.cycles)
        phases = []
        for phase_id in phases_in_scope:
            phase = project.add_phase(
                str(root / f"{phase_id}.cif"),
                phasename=phase_id,
                histograms=[histogram],
                fmthint="CIF",
            )
            configure_sample(phase, sodium=phase_id == "sodium_dihydrogen_citrate")
            phase.set_HAP_refinements({"Scale": True})
            phases.append(phase)
        stage_rwp["scale_background"] = refine(project, histogram)
        if scope == "iucr_sodium_dihydrogen_citrate_silicon_holdout":
            phases[0].set_HAP_refinements({"Pref.Ori.": True})
            stage_rwp["preferred_orientation"] = refine(project, histogram)

        mass_fractions = histogram.ComputeMassFracs()
        fractions = {phase.name: float(mass_fractions[phase.name][0]) for phase in phases}
        calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
        background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
        legacy_targets = manifest["legacy_gsas_reference"]["weight_fractions"]
        errors = {
            phase_id: fractions[phase_id] - float(legacy_targets[phase_id])
            for phase_id in phases_in_scope
        }
        hap_parameters = {}
        for phase in phases:
            hap = phase.getHAPvalues(histogram)
            hap_parameters[phase.name] = {
                key: hap[key] for key in ("Scale", "Size", "Mustrain", "Pref.Ori.") if key in hap
            }
        result = {
            "schema_version": 1,
            "scope": scope,
            "revision": actual_revision,
            "sample_count": int(x.size),
            "weight_fractions": fractions,
            "legacy_weight_fraction_errors": errors,
            "maximum_legacy_weight_fraction_error": max(abs(value) for value in errors.values()),
            "poisson_rwp": float(histogram.get_wR()) / 100.0,
            "profile_correlation": float(
                np.corrcoef(observed - background, calculated - background)[0, 1]
            ),
            "legacy_curve_poisson_rwp": float(manifest["legacy_gsas_reference"]["rwp"]),
            "legacy_curve_profile_correlation": float(
                np.corrcoef(
                    observed - legacy_background,
                    legacy_calculated - legacy_background,
                )[0, 1]
            ),
            "silicon_calibration": {
                "sample_count": int(np.count_nonzero(calibration_mask)),
                "poisson_rwp": stage_rwp["silicon_displacement_calibration"],
                "calibrated_sample_displacement_mm": calibrated_shift_micrometre * 1.0e-3,
                "windows_two_theta_deg": windows,
            },
            "stage_poisson_rwp": stage_rwp,
            "refined_instrument": {
                key: float(value[1])
                for key, value in histogram.InstrumentParameters.items()
                if key in {"U", "V", "W", "X", "Y", "Zero", "SH/L"}
            },
            "refined_hap": hap_parameters,
        }
    if not all(
        math.isfinite(value)
        for value in (
            *fractions.values(),
            *errors.values(),
            result["poisson_rwp"],
            result["profile_correlation"],
        )
    ):
        raise RuntimeError("GSAS-II citrate/Si result contains non-finite values")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
