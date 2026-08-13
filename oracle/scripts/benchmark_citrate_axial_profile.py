#!/usr/bin/env python3
"""Generate source-specific pinned-GSAS-II axial-profile probes for Rb citrate.

This external-only worker imports no PhaseSmith module. GSAS-II exposes the
isolated FCJ evaluator only in ``GSASIIpwd``; that unavoidable private probe is
revision-gated here and emits only plain arrays and JSON metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SOURCE_PROFILE = {"W": 5.109, "X": 3.634, "S/L": 0.0097, "H/L": 0.0097}
GSASII_SH_OVER_L_CASES = (0.002, 0.0097, 0.0194)
SELECTED_HKL = ((1, 0, 2), (0, 1, 3), (2, 2, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_reflections(root: Path, manifest: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    record = manifest.get("source_reflections", {})
    columns = tuple(record.get("columns", ()))
    required = ("h", "k", "l", "phase_id", "wavelength_id", "d_spacing_angstrom")
    if not all(name in columns for name in required):
        raise ValueError("bundle does not contain the reviewed source-reflection columns")
    rows = np.loadtxt(root / str(record["file"]), delimiter=",", skiprows=1)
    selected = (rows[:, columns.index("phase_id")] == 1.0) & (
        rows[:, columns.index("wavelength_id")] == 1.0
    )
    result = {}
    for row in rows[selected]:
        hkl = tuple(int(row[columns.index(name)]) for name in ("h", "k", "l"))
        result[hkl] = float(row[columns.index("d_spacing_angstrom")])
    return result


def _profile(
    profile_module: Any,
    x: np.ndarray,
    position_deg: float,
    gaussian_variance_centideg2: float,
    lorentzian_fwhm_centideg: float,
    sh_over_l: float,
) -> np.ndarray:
    values, _ = profile_module.getFCJVoigt3(
        position_deg,
        gaussian_variance_centideg2,
        lorentzian_fwhm_centideg,
        sh_over_l,
        x,
    )
    result = 100.0 * np.asarray(values, dtype=np.float64)
    if result.shape != x.shape or not np.isfinite(result).all() or np.any(result < 0.0):
        raise RuntimeError("pinned GSAS-II returned an invalid isolated profile")
    return result


def main() -> None:
    arguments = parse_args()
    actual_revision = revision(arguments.gsas_root)
    if actual_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II revision mismatch: expected {PINNED_REVISION}, got {actual_revision}"
        )
    for output in (arguments.arrays, arguments.report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(arguments.gsas_root))
    if arguments.binary_dir is not None:
        sys.path.insert(0, str(arguments.binary_dir))
    from GSASII import GSASIIpath

    if arguments.binary_dir is not None:
        GSASIIpath.binaryPath = str(arguments.binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    from GSASII import GSASIIpwd

    if not hasattr(GSASIIpwd, "getFCJVoigt3"):
        raise RuntimeError("pinned GSAS-II FCJ profile probe is unavailable")
    root = arguments.data_directory
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != "iucr_anhydrous_trirubidium_citrate_silicon_holdout":
        raise ValueError("worker requires the reviewed anhydrous rubidium bundle")
    if manifest.get("source_dataset_id") != "iucr-trirubidium-citrate-si-standard":
        raise ValueError("worker requires the reviewed trirubidium source dataset")
    instrument = manifest.get("instrument", {})
    source_values = {
        "W": instrument.get("common_base_profile", {}).get("W"),
        "X": instrument.get("common_base_profile", {}).get("X"),
        "S/L": instrument.get("source_over_radius"),
        "H/L": instrument.get("detector_over_radius"),
    }
    if source_values != SOURCE_PROFILE:
        raise ValueError("bundle source-profile values do not match the reviewed contract")
    wavelength = float(instrument["wavelengths_angstrom"][0])
    source_d = _source_reflections(root, manifest)
    arrays = {}
    cases = []
    for index, hkl in enumerate(SELECTED_HKL):
        d_spacing = source_d[hkl]
        position = math.degrees(2.0 * math.asin(wavelength / (2.0 * d_spacing)))
        theta = math.radians(position / 2.0)
        gaussian_variance = SOURCE_PROFILE["W"]
        gaussian_fwhm_deg = math.sqrt(8.0 * math.log(2.0)) * math.sqrt(gaussian_variance) / 100.0
        lorentzian_centideg = SOURCE_PROFILE["X"] / math.cos(theta)
        x = np.linspace(position - 1.5, position + 1.5, 20_001)
        x_name = f"peak_{index}__x_deg"
        arrays[x_name] = x
        profile_arrays = {}
        for sh_over_l in GSASII_SH_OVER_L_CASES:
            suffix = str(sh_over_l).replace(".", "p")
            name = f"peak_{index}__gsasii_shl_{suffix}"
            arrays[name] = _profile(
                GSASIIpwd,
                x,
                position,
                gaussian_variance,
                lorentzian_centideg,
                sh_over_l,
            )
            profile_arrays[str(sh_over_l)] = name
        cases.append(
            {
                "hkl": list(hkl),
                "d_spacing_angstrom": d_spacing,
                "position_deg": position,
                "gaussian_variance_centideg2": gaussian_variance,
                "gaussian_fwhm_deg": gaussian_fwhm_deg,
                "lorentzian_fwhm_deg": lorentzian_centideg / 100.0,
                "x_array": x_name,
                "profile_arrays_by_sh_over_l": profile_arrays,
            }
        )
    np.savez_compressed(arguments.arrays, **arrays)
    report = {
        "schema_version": 1,
        "scope": "iucr_trirubidium_citrate_source_axial_profile_probe",
        "source_dataset_id": manifest["source_dataset_id"],
        "revision": actual_revision,
        "source_profile": SOURCE_PROFILE,
        "gsasii_sh_over_l_cases": list(GSASII_SH_OVER_L_CASES),
        "oracle_boundary": {
            "api": "GSASIIpwd.getFCJVoigt3",
            "private_probe": True,
            "reason": "GSAS-II scripting does not expose an isolated normalized FCJ profile",
            "native_units": {
                "gaussian_variance": "centidegree_squared",
                "lorentzian_fwhm": "centidegree",
                "profile": "inverse_centidegree converted to inverse_degree",
            },
        },
        "published_equation": {
            "citation": (
                "L. W. Finger, D. E. Cox and A. P. Jephcoat, J. Appl. Cryst. 27 (1994) 892-900"
            ),
            "source_geometry": (
                "legacy-GSAS full height/diameter ratios S/L=0.0097 and H/L=0.0097; "
                "these are numerically the FCJ half-height/radius ratios"
            ),
            "gsasii_compression": (
                "GSAS-II assumes equal sample and detector heights and documents its "
                "single SH/L field as the formal sum 0.0194"
            ),
        },
        "arrays_file": arguments.arrays.name,
        "arrays_sha256": _sha256_file(arguments.arrays),
        "cases": cases,
    }
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
