#!/usr/bin/env python3
"""Compress the deposited Rowles optics with pinned GSAS-II's FPA workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
INPUT_SCOPE = "curtin_rowles_qpa_topas_common_subset"
OUTPUT_SCOPE = "curtin_rowles_qpa_topas_gsasii_fpa_compression"
PROFILE_NAMES = ("U", "V", "W", "X", "Y", "SH/L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--bundle-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def configure_gsasii(root: Path, binary_directory: Path | None) -> tuple[Any, Any, Any, Any]:
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

    # GSAS-II exposes FPA calibration only through its GUI module at this pinned
    # revision. Keep that private access contained in this external oracle worker.
    from GSASII import GSASIIfpaGUI, GSASIImath, GSASIIpwd, GSASIIscriptable

    return GSASIIfpaGUI, GSASIImath, GSASIIpwd, GSASIIscriptable


def add_peak(target: np.ndarray, center: int, peak: np.ndarray) -> None:
    start = center - peak.size // 2
    target_start = max(start, 0)
    target_end = min(start + peak.size, target.size)
    peak_start = target_start - start
    peak_end = peak_start + target_end - target_start
    target[target_start:target_end] += 10_000.0 * peak[peak_start:peak_end] / np.max(peak)


def fit_stage(
    powder: Any,
    peaks: list[list[Any]],
    background: list[Any],
    limits: list[float],
    instrument: dict[str, Any],
    instrument2: dict[str, Any],
    data: list[np.ndarray],
    fixed_background: np.ndarray,
) -> dict[str, Any]:
    result = powder.DoPeakFit(
        "LSQ",
        peaks,
        background,
        limits,
        instrument,
        instrument2,
        data,
        fixed_background,
        [],
        False,
        {"deriv type": "analytic", "min dM/M": 1.0e-6},
    )
    if result is None:
        raise RuntimeError("pinned GSAS-II FPA peak-profile fit failed")
    return {"rwp_percent": float(result[3]["Rwp"]), "varied": list(map(str, result[4]))}


def calibrate(
    fpa: Any,
    g2math: Any,
    powder: Any,
    scripting: Any,
    bundle: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    source_model = manifest["topas_source_model"]
    geometry = source_model["instrument_geometry"]
    axial = geometry["axial"]
    detector = geometry["linear_position_sensitive_detector"]
    tails = geometry["tube_tails"]
    source_radius = float(geometry["source_to_sample_radius_mm"])
    detector_radius = float(geometry["sample_to_detector_radius_mm"])
    if not math.isclose(source_radius, detector_radius, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("pinned GSAS-II FPA adapter requires equal Rowles radii")
    incident_soller = float(axial["incident_soller_full_width_deg"])
    diffracted_soller = float(axial["diffracted_soller_full_width_deg"])
    if not math.isclose(incident_soller, diffracted_soller, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("pinned GSAS-II FPA adapter requires equal Rowles Soller angles")

    edge = source_model["absorption_edge_filter"]
    edge_wavelength = float(edge["edge_angstrom"])
    sharpness = float(edge["sharpness_per_angstrom"])
    floor = float(edge["floor"])
    alpha_lines = [
        line
        for line in source_model["emission_lines"]
        if float(line["wavelength_angstrom"]) > edge_wavelength
    ]
    if len(alpha_lines) != 5:
        raise ValueError("Rowles FPA calibration expects the five deposited Cu K-alpha lines")
    transmitted_areas = [
        float(line["area"])
        * (
            floor
            + 0.5
            * (
                1.0
                + math.erf(
                    sharpness * (float(line["wavelength_angstrom"]) - edge_wavelength)
                )
            )
        )
        for line in alpha_lines
    ]
    fpa_input = {
        "wave": {
            index: float(line["wavelength_angstrom"])
            for index, line in enumerate(alpha_lines)
        },
        "int": {index: area for index, area in enumerate(transmitted_areas)},
        # TOPAS documents lh as Lorentzian HW; NIST FPA accepts FWHM.
        "lwidth": {
            index: 2.0 * float(line["lorentzian_hwhm_milliangstrom"])
            for index, line in enumerate(alpha_lines)
        },
        "divergence": float(detector["equatorial_divergence_deg"]),
        "soller_angle": incident_soller,
        "Rs": source_radius,
        "filament_length": float(axial["filament_full_length_mm"]),
        "sample_length": float(axial["illuminated_sample_full_length_mm"]),
        "receiving_slit_length": float(axial["receiving_slit_full_length_mm"]),
        "LAC_cm": 0.0,
        "sample_thickness": 1.0,
        "convolution_steps": 8,
        "source_width": float(tails["source_width_mm"]),
        "tube-tails_L-tail": float(tails["left_tail_mm"]),
        "tube-tails_R-tail": float(tails["right_tail_mm"]),
        "tube-tails_rel-I": float(tails["relative_intensity"]),
        "SiPSD_th2_angular_range": float(detector["two_theta_angular_range_deg"]),
    }
    fpa.DetMode = "BBPSD"
    fpa.IBmono = False
    fpa.NISTparms.clear()
    fpa.XferFPAsettings(fpa_input)
    physical_profile = fpa.setupFPAcalc()

    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-rowles-fpa-") as temporary:
        project = scripting.G2Project(newgpx=str(Path(temporary) / "fpa.gpx"))
        histogram = project.add_simulated_powder_histogram(
            "Rowles FPA target",
            str(bundle / "common.instprm"),
            18.0,
            153.0,
            Tstep=0.006,
        )
        data = histogram.data["data"][1]
        x = np.asarray(data[0], dtype=np.float64)
        step = float(x[1] - x[0])
        target = np.zeros_like(x)
        positions = np.linspace(21.0, 147.0, 13)
        maximum_half_height_points = 0
        for position in positions:
            center, peak_object = fpa.doFPAcalc(
                physical_profile, x, float(position), 3.0, step
            )
            add_peak(target, center, peak_object.peak)
            maximum_half_height_points = max(
                maximum_half_height_points,
                int(np.sum(peak_object.peak >= 0.5 * np.max(peak_object.peak))),
            )
        data[1][...] = target
        data[2][...] = 1.0
        for index in (3, 4, 5):
            data[index][...] = 0.0

        instrument = histogram.InstrumentParameters
        instrument2 = histogram.data["Instrument Parameters"][1]
        background = histogram.data["Background"]
        background[0][1] = False
        background[0][2] = 0
        background[0][3:] = [0.0] * len(background[0][3:])
        instrument["SH/L"][1] = (
            0.25
            * (
                float(axial["illuminated_sample_full_length_mm"])
                + float(axial["filament_full_length_mm"])
            )
            / source_radius
        )
        peaks = []
        for position in positions:
            index = int(np.searchsorted(x, position))
            area = float(
                np.sum(
                    target[
                        max(0, index - maximum_half_height_points) : min(
                            target.size, index + maximum_half_height_points
                        )
                    ]
                )
            )
            peaks.append(g2math.setPeakparms(instrument, instrument2, float(position), area))

        fixed_background = np.zeros_like(x)
        stages = {
            "positions": fit_stage(
                powder,
                peaks,
                background,
                [18.0, 153.0],
                instrument,
                instrument2,
                data,
                fixed_background,
            )
        }
        for peak in peaks:
            peak[1] = True
        stages["positions_areas"] = fit_stage(
            powder,
            peaks,
            background,
            [18.0, 153.0],
            instrument,
            instrument2,
            data,
            fixed_background,
        )
        for name in ("U", "V", "W", "X", "Y"):
            instrument[name][2] = True
        stages["profile"] = fit_stage(
            powder,
            peaks,
            background,
            [18.0, 153.0],
            instrument,
            instrument2,
            data,
            fixed_background,
        )
        instrument["SH/L"][2] = True
        stages["profile_asymmetry"] = fit_stage(
            powder,
            peaks,
            background,
            [18.0, 153.0],
            instrument,
            instrument2,
            data,
            fixed_background,
        )
        calculated = np.asarray(data[3], dtype=np.float64)
        profile = {name: float(instrument[name][1]) for name in PROFILE_NAMES}
        return {
            "instrument_profile": profile,
            "synthetic_diagnostics": {
                "sample_count": int(x.size),
                "peak_count": int(positions.size),
                "rwp": stages["profile_asymmetry"]["rwp_percent"] / 100.0,
                "relative_l2_error": float(
                    np.linalg.norm(calculated - target) / np.linalg.norm(target)
                ),
                "profile_correlation": float(np.corrcoef(target, calculated)[0, 1]),
                "stages": stages,
            },
            "target": {
                "included": [
                    "five deposited Cu K-alpha Lorentzian lines",
                    "deposited absorption-edge transmission sampled at each line center",
                    "full source/sample/receiver axial geometry and equal 2.5 degree Soller slits",
                    "LPSD angular range and equatorial divergence",
                    "tube tails",
                ],
                "excluded": [
                    "K-beta and white-continuum source contributions outside the "
                    "K-alpha calibration window",
                    "angle-dependent white continuum",
                    "specimen-dependent flat-plate absorption",
                ],
                "topas_lh_conversion": "documented Lorentzian HW multiplied by two for NIST FWHM",
                "positions_deg": positions.tolist(),
                "step_deg": step,
                "window_full_width_deg": 3.0,
            },
        }


def main() -> None:
    arguments = parse_args()
    root = arguments.gsas_root.resolve()
    detected_revision = revision(root)
    if detected_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II Rowles FPA benchmark requires revision {PINNED_REVISION}, "
            f"detected {detected_revision}"
        )
    bundle = arguments.bundle_directory.resolve()
    required = ("experiment.json", "common.instprm")
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"neutral Rowles bundle is missing {', '.join(missing)}")
    manifest = json.loads((bundle / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("scope") != INPUT_SCOPE:
        raise ValueError("unsupported neutral Rowles experiment manifest")
    modules = configure_gsasii(root, arguments.binary_dir)
    calibration = calibrate(*modules, bundle, manifest)
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": detected_revision,
        "scope": OUTPUT_SCOPE,
        "source_dataset_id": "curtin-rowles-qpa-topas",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "input_sha256": {name: sha256(bundle / name) for name in required},
        "oracle_boundary": (
            "version-gated private GSASIIfpaGUI adapter; output is plain JSON profile coefficients"
        ),
        **calibration,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if arguments.profile_output is not None:
        arguments.profile_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.profile_output.write_text(
            json.dumps(calibration["instrument_profile"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
