#!/usr/bin/env python3
"""Generate pinned GSAS-II size, microstrain, and March-Dollase fixtures.

The histogram and HAP values are configured through ``GSASIIscriptable`` and
plain public entry accessors. A pinned private profile call evaluates selected
reflection widths. This script imports no Rietveld Engine module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE_NAME = "data.npz"
MANIFEST_NAME = "manifest.json"
ADAPTER_VERSION = 1
GAUSSIAN_FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))
INSTRUMENT = {
    "wavelength_angstrom": 1.54056,
    "u_gsas_centideg2": 2.0,
    "v_gsas_centideg2": -1.0,
    "w_gsas_centideg2": 1.0,
    "x_gsas_centideg": 0.1,
    "y_gsas_centideg": 0.2,
}
PHASE = {
    "name": "sample-physics-cubic",
    "space_group": "P 1",
    "cell": [4.0, 4.0, 4.0, 90.0, 90.0, 90.0],
}
SAMPLE = {
    "size_micrometre": 0.05,
    "size_lorentzian_fraction": 1.0,
    "microstrain_ppm": 600.0,
    "microstrain_lorentzian_fraction": 0.0,
    "march_ratio": 0.72,
    "preferred_axis_hkl": [0, 0, 1],
}
REFLECTION_COLUMNS = [
    "h",
    "k",
    "l",
    "multiplicity",
    "d_spacing_angstrom",
    "position_deg",
    "sigma2_centideg2",
    "gamma_centideg",
    "f_obs2",
    "f_calc2",
    "phase_deg",
    "intensity_correction",
    "preferred_orientation",
    "transmission",
    "extinction",
]


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    """Return the exact Git revision for a checkout."""

    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def import_gsasii(gsas_root: Path, binary_dir: Path | None) -> tuple[Any, Any, Any]:
    """Import pinned scripting and profile modules with explicit binaries."""

    sys.path.insert(0, str(gsas_root))
    from GSASII import GSASIIpath

    if binary_dir is not None:
        sys.path.insert(0, str(binary_dir))
        GSASIIpath.binaryPath = str(binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIpwd, GSASIIscriptable

    if not hasattr(GSASIIpwd, "pyd"):
        raise RuntimeError("GSAS-II pypowder binary is unavailable")
    return GSASIIpath, GSASIIscriptable, GSASIIpwd


def instrument_text() -> str:
    """Return a deterministic GSAS-II instrument parameter file."""

    values = INSTRUMENT
    return "".join(
        [
            "#GSAS-II instrument parameter file for a sample-physics oracle\n",
            "Type:PXC;Bank:1\n",
            f"Lam:{values['wavelength_angstrom']};Zero:0;Polariz.:0.99;Azimuth:0\n",
            f"U:{values['u_gsas_centideg2']};V:{values['v_gsas_centideg2']};"
            f"W:{values['w_gsas_centideg2']};X:{values['x_gsas_centideg']};"
            f"Y:{values['y_gsas_centideg']};Z:0;SH/L:0\n",
        ]
    )


def update_hap_entry(phase: Any, key: str, transform: Callable[[list[Any]], list[Any]]) -> None:
    """Update one HAP value through the public generic entry API."""

    entries = phase.getHAPentryList(0, key)
    if len(entries) != 1:
        raise RuntimeError(f"expected one {key} HAP entry, found {len(entries)}")
    key_path = entries[0][0]
    current = phase.getHAPentryValue(key_path)
    phase.setHAPentryValue(key_path, transform(current))


def configure_sample(phase: Any) -> None:
    """Set pure-Lorentzian size, pure-Gaussian strain, and March-Dollase."""

    def size(current: list[Any]) -> list[Any]:
        current[0] = "isotropic"
        current[1][0] = SAMPLE["size_micrometre"]
        current[1][2] = SAMPLE["size_lorentzian_fraction"]
        return current

    def microstrain(current: list[Any]) -> list[Any]:
        current[0] = "isotropic"
        current[1][0] = SAMPLE["microstrain_ppm"]
        current[1][2] = SAMPLE["microstrain_lorentzian_fraction"]
        return current

    def orientation(current: list[Any]) -> list[Any]:
        current[0] = "MD"
        current[1] = SAMPLE["march_ratio"]
        current[2] = False
        current[3] = SAMPLE["preferred_axis_hkl"]
        return current

    update_hap_entry(phase, "Size", size)
    update_hap_entry(phase, "Mustrain", microstrain)
    update_hap_entry(phase, "Pref.Ori.", orientation)


def translated_public_parameters() -> dict[str, Any]:
    """Record the independently derived GSAS-to-public convention mapping."""

    return {
        "crystallite_size_nm": 1_000.0 * SAMPLE["size_micrometre"],
        "shape_factor": 1.0,
        "rms_microstrain": SAMPLE["microstrain_ppm"] * 1.0e-6 / (2.0 * GAUSSIAN_FWHM_PER_SIGMA),
        "march_ratio": SAMPLE["march_ratio"],
        "preferred_axis_hkl": SAMPLE["preferred_axis_hkl"],
        "reciprocal_metric_angstrom_minus2": [
            [1.0 / 16.0, 0.0, 0.0],
            [0.0, 1.0 / 16.0, 0.0],
            [0.0, 0.0, 1.0 / 16.0],
        ],
    }


def selected_profile_case(
    profile_module: Any,
    reflection: np.ndarray,
    index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Probe a normalized profile using public reflection-list widths."""

    position = float(reflection[5])
    sigma2 = float(reflection[6])
    gamma = float(reflection[7])
    preferred_orientation = float(reflection[12])
    half_span = max(1.5, 40.0 * np.sqrt(sigma2) / 100.0, 0.4 * gamma)
    x = np.linspace(position - half_span, position + half_span, 6_001)
    native, native_integral = profile_module.getPsVoigt(position, sigma2, gamma, x)
    profile = 100.0 * np.asarray(native, dtype=np.float64)
    corrected = preferred_orientation * profile
    area = float(np.trapezoid(corrected, x))
    centroid = float(np.trapezoid(x * corrected, x) / area)
    second = float(np.trapezoid((x - centroid) ** 2 * corrected, x) / area)
    case_id = f"sample_reflection_{index}"
    arrays = {
        f"{case_id}__x_deg": x,
        f"{case_id}__profile_per_deg": profile,
    }
    case = {
        "id": case_id,
        "case_kind": "sample_physics_reflection",
        "arrays": {
            "x": f"{case_id}__x_deg",
            "profile": f"{case_id}__profile_per_deg",
        },
        "parameters": {
            "hkl": [int(value) for value in reflection[:3]],
            "d_spacing_angstrom": float(reflection[4]),
            "position_deg": position,
            "sigma2_centideg2": sigma2,
            "gamma_centideg": gamma,
            "preferred_orientation": preferred_orientation,
            "public_translation": translated_public_parameters(),
        },
        "gsasii_reported_integral": float(native_integral),
        "sampled_corrected_moments": {
            "integral": area,
            "centroid_deg": centroid,
            "second_central_moment_deg2": second,
        },
    }
    return case, arrays


def generate_snapshot(
    scripting: Any, profile_module: Any
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Build one public scripting histogram and selected profile probes."""

    np.random.seed(20_260_805)
    with tempfile.TemporaryDirectory(prefix="rietveld-gsas-sample-") as temporary:
        work = Path(temporary)
        instrument_path = work / "sample.instprm"
        instrument_path.write_text(instrument_text(), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(work / "sample.gpx"))
        phase = project.add_phase(
            phasename=PHASE["name"],
            spacegroup=PHASE["space_group"],
            cell=PHASE["cell"],
        )
        phase.add_atom(0.0, 0.0, 0.0, element="Si", lbl="Si1", occ=1.0, uiso=0.01)
        histogram = project.add_simulated_powder_histogram(
            "sample-physics",
            str(instrument_path),
            10.0,
            100.0,
            Npoints=4_501,
            scale=100.0,
            phases=[phase],
        )
        configure_sample(phase)
        project.do_refinements([{}], outputnames=[None])
        reflection_list = np.ascontiguousarray(
            histogram.reflections()[PHASE["name"]]["RefList"], dtype=np.float64
        )
        arrays = {
            "x_deg": np.ascontiguousarray(histogram.getdata("X"), dtype=np.float64),
            "ycalc": np.ascontiguousarray(histogram.getdata("Ycalc"), dtype=np.float64),
            "background": np.ascontiguousarray(histogram.getdata("Background"), dtype=np.float64),
            "reflection_list": reflection_list,
        }
    indices = [0, reflection_list.shape[0] // 2, reflection_list.shape[0] - 1]
    cases = []
    for index in indices:
        case, case_arrays = selected_profile_case(profile_module, reflection_list[index], index)
        cases.append(case)
        arrays.update(case_arrays)
    return cases, arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    """Describe one plain archive member."""

    if name.endswith("x_deg"):
        unit, description = "degree_2theta", "Profile sampling coordinates"
    elif "profile_per_deg" in name:
        unit, description = "inverse_degree", "Normalized pinned GSAS-II profile"
    elif name == "reflection_list":
        unit, description = "mixed_by_column", "Public scripting reflection list"
    else:
        unit, description = "intensity", f"Public scripting {name} array"
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes(order="C")),
        "finite": bool(np.isfinite(array).all()),
        "unit": unit,
        "description": description,
    }


def write_fixture(
    output: Path,
    gsas_root: Path,
    gsasii_path: Any,
    scripting: Any,
    profile_module: Any,
    pinned: dict[str, Any],
    *,
    force: bool,
) -> None:
    """Generate, hash, and write the versioned fixture."""

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    existing = [path for path in (archive_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError("fixture exists; pass --force for explicit regeneration")
    cases, arrays = generate_snapshot(scripting, profile_module)
    np.savez_compressed(archive_path, **arrays)
    canonical_input = json.dumps(
        {"instrument": INSTRUMENT, "phase": PHASE, "sample": SAMPLE}, sort_keys=True
    ).encode()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_sample_physics_v1",
        "kind": "cw_sample_physics",
        "provenance": {
            "gsasii_repository": pinned["repository"],
            "gsasii_revision": git_revision(gsas_root),
            "gsasii_tag": int(gsasii_path.GetVersionNumber()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "source": {
            "api": [
                "GSASII.GSASIIscriptable.G2Phase.getHAPentryList/getHAPentryValue/setHAPentryValue",
                "GSASII.GSASIIscriptable.G2PwdrData.getdata/reflections",
                "GSASII.GSASIIpwd.getPsVoigt",
            ],
            "private_probe": True,
            "adapter_version": ADAPTER_VERSION,
            "reflection_columns": REFLECTION_COLUMNS,
            "native_units": {
                "size": "micrometre",
                "microstrain": "delta_Q_over_Q_times_1e6",
                "Gaussian_variance": "centidegree_squared",
                "Lorentzian_fwhm": "centidegree",
                "profile_density": "inverse_centidegree",
            },
        },
        "input_parameters": {"instrument": INSTRUMENT, "phase": PHASE, "sample": SAMPLE},
        "input": {"kind": "parameter_cases", "sha256": sha256_bytes(canonical_input)},
        "archive": {"file": ARCHIVE_NAME, "sha256": sha256_file(archive_path)},
        "arrays": {name: array_descriptor(name, array) for name, array in arrays.items()},
        "cases": cases,
    }
    if manifest["provenance"]["gsasii_revision"] != pinned["revision"]:
        raise RuntimeError("refusing to write a fixture from an unpinned GSAS-II revision")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse explicit oracle paths and overwrite control."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures" / "sample_physics_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate the pin, import GSAS-II, and write the fixture."""

    arguments = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    pinned = json.loads((repository_root / "oracle" / "PINNED_GSASII.json").read_text())
    gsas_root = arguments.gsas_root.resolve()
    if git_revision(gsas_root) != pinned["revision"]:
        raise RuntimeError("GSAS-II checkout does not match the recorded pin")
    gsasii_path, scripting, profile_module = import_gsasii(
        gsas_root, arguments.binary_dir.resolve() if arguments.binary_dir else None
    )
    write_fixture(
        arguments.output.resolve(),
        gsas_root,
        gsasii_path,
        scripting,
        profile_module,
        pinned,
        force=arguments.force,
    )
    print(f"wrote {arguments.output.resolve() / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
