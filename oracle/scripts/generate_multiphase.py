#!/usr/bin/env python3
"""Generate a pinned two-phase GSAS-II scripting and profile fixture.

Public scripting supplies X, Ycalc, background, HAP scales, and both reflection
lists. A pinned private profile call supplies one controlled two-reflection
composition. This script imports no Rietveld Engine module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE_NAME = "data.npz"
MANIFEST_NAME = "manifest.json"
ADAPTER_VERSION = 1
INSTRUMENT = {
    "wavelength_angstrom": 1.54056,
    "u_gsas_centideg2": 2.0,
    "v_gsas_centideg2": -1.0,
    "w_gsas_centideg2": 1.2,
    "x_gsas_centideg": 0.15,
    "y_gsas_centideg": 0.3,
}
PHASES = [
    {
        "phase_id": "alpha",
        "name": "oracle-alpha",
        "cell_angstrom": 4.0,
        "element": "Si",
        "scale": 1.25,
    },
    {
        "phase_id": "beta",
        "name": "oracle-beta",
        "cell_angstrom": 3.2,
        "element": "Ni",
        "scale": 0.65,
    },
]
COMPOSITION = {
    "base_integrated_intensities": [7.0, 4.0],
    "background": 0.37,
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
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def import_gsasii(gsas_root: Path, binary_dir: Path | None) -> tuple[Any, Any, Any]:
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
    values = INSTRUMENT
    return "".join(
        [
            "#GSAS-II instrument parameter file for a two-phase oracle\n",
            "Type:PXC;Bank:1\n",
            f"Lam:{values['wavelength_angstrom']};Zero:0;Polariz.:0.99;Azimuth:0\n",
            f"U:{values['u_gsas_centideg2']};V:{values['v_gsas_centideg2']};"
            f"W:{values['w_gsas_centideg2']};X:{values['x_gsas_centideg']};"
            f"Y:{values['y_gsas_centideg']};Z:0;SH/L:0\n",
        ]
    )


def phase_scale(phase: Any) -> float:
    entries = phase.getHAPentryList(0, "Scale")
    if len(entries) != 1:
        raise RuntimeError(f"expected one Scale HAP entry, found {len(entries)}")
    key_path = entries[0][0]
    stored = phase.getHAPentryValue(key_path)
    if bool(stored[1]):
        raise RuntimeError("phase scale refinement flag was not disabled")
    return float(stored[0])


def set_phase_scale(phase: Any, scale: float) -> None:
    entries = phase.getHAPentryList(0, "Scale")
    if len(entries) != 1:
        raise RuntimeError(f"expected one Scale HAP entry, found {len(entries)}")
    key_path = entries[0][0]
    current = phase.getHAPentryValue(key_path)
    current[0] = scale
    current[1] = False
    phase.setHAPentryValue(key_path, current)


def configure_neutral_sample(phase: Any) -> None:
    """Disable default sample broadening and orientation for composition tests."""

    for key, configure in (
        (
            "Size",
            lambda current: [
                "isotropic",
                [1.0e12, current[1][1], 1.0],
                *current[2:],
            ],
        ),
        (
            "Mustrain",
            lambda current: [
                "isotropic",
                [0.0, current[1][1], 0.0],
                *current[2:],
            ],
        ),
        (
            "Pref.Ori.",
            lambda current: ["MD", 1.0, False, current[3], *current[4:]],
        ),
    ):
        entries = phase.getHAPentryList(0, key)
        if len(entries) != 1:
            raise RuntimeError(f"expected one {key} HAP entry, found {len(entries)}")
        key_path = entries[0][0]
        phase.setHAPentryValue(key_path, configure(phase.getHAPentryValue(key_path)))


def selected_composition(
    profile_module: Any,
    reflection_lists: list[np.ndarray],
    scales: list[float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    positions_a = reflection_lists[0][:, 5]
    positions_b = reflection_lists[1][:, 5]
    distance = np.abs(positions_a[:, None] - positions_b[None, :])
    index_a, index_b = np.unravel_index(np.argmin(distance), distance.shape)
    selected = [reflection_lists[0][index_a], reflection_lists[1][index_b]]
    left = min(float(reflection[5]) for reflection in selected) - 1.5
    right = max(float(reflection[5]) for reflection in selected) + 1.5
    x = np.linspace(left, right, 8_001)
    profiles = []
    arrays: dict[str, np.ndarray] = {"composition_x_deg": x}
    reflection_parameters = []
    for phase_index, reflection in enumerate(selected):
        native, reported_integral = profile_module.getPsVoigt(
            float(reflection[5]),
            float(reflection[6]),
            float(reflection[7]),
            x,
        )
        profile = 100.0 * np.asarray(native, dtype=np.float64)
        key = f"composition_phase_{phase_index}_profile_per_deg"
        arrays[key] = profile
        profiles.append(profile)
        reflection_parameters.append(
            {
                "phase_id": PHASES[phase_index]["phase_id"],
                "reflection_index": int((index_a, index_b)[phase_index]),
                "hkl": [int(value) for value in reflection[:3]],
                "d_spacing_angstrom": float(reflection[4]),
                "position_deg": float(reflection[5]),
                "sigma2_centideg2": float(reflection[6]),
                "gamma_centideg": float(reflection[7]),
                "gsasii_reported_integral": float(reported_integral),
            }
        )
    intensities = COMPOSITION["base_integrated_intensities"]
    composed = (
        COMPOSITION["background"]
        + intensities[0] * scales[0] * profiles[0]
        + intensities[1] * scales[1] * profiles[1]
    )
    arrays["composition_ycalc"] = composed
    case = {
        "id": "controlled_two_phase_overlap",
        "case_kind": "multiphase_profile_composition",
        "arrays": {
            "x": "composition_x_deg",
            "ycalc": "composition_ycalc",
            "phase_0_profile": "composition_phase_0_profile_per_deg",
            "phase_1_profile": "composition_phase_1_profile_per_deg",
        },
        "parameters": {
            "phase_scales": scales,
            "base_integrated_intensities": intensities,
            "background": COMPOSITION["background"],
            "reflections": reflection_parameters,
        },
    }
    return case, arrays


def generate_snapshot(
    scripting: Any, profile_module: Any
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    np.random.seed(20_260_805)
    with tempfile.TemporaryDirectory(prefix="rietveld-gsas-multiphase-") as temporary:
        work = Path(temporary)
        instrument_path = work / "multiphase.instprm"
        instrument_path.write_text(instrument_text(), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(work / "multiphase.gpx"))
        phase_objects = []
        for phase_parameters in PHASES:
            phase = project.add_phase(
                phasename=phase_parameters["name"],
                spacegroup="P 1",
                cell=[phase_parameters["cell_angstrom"]] * 3 + [90.0, 90.0, 90.0],
            )
            phase.add_atom(
                0.0,
                0.0,
                0.0,
                element=phase_parameters["element"],
                lbl=f"{phase_parameters['element']}1",
                occ=1.0,
                uiso=0.01,
            )
            phase_objects.append(phase)
        histogram = project.add_simulated_powder_histogram(
            "two-phase",
            str(instrument_path),
            10.0,
            100.0,
            Npoints=4_501,
            scale=100.0,
            phases=phase_objects,
        )
        for phase, parameters in zip(phase_objects, PHASES, strict=True):
            set_phase_scale(phase, parameters["scale"])
        for phase in phase_objects:
            configure_neutral_sample(phase)
        project.do_refinements([{}], outputnames=[None])
        scales = [phase_scale(phase) for phase in phase_objects]
        public_reflections = histogram.reflections()
        reflection_lists = [
            np.ascontiguousarray(
                public_reflections[parameters["name"]]["RefList"], dtype=np.float64
            )
            for parameters in PHASES
        ]
        arrays = {
            "x_deg": np.ascontiguousarray(histogram.getdata("X"), dtype=np.float64),
            "ycalc": np.ascontiguousarray(histogram.getdata("Ycalc"), dtype=np.float64),
            "background": np.ascontiguousarray(histogram.getdata("Background"), dtype=np.float64),
            "reflection_list_alpha": reflection_lists[0],
            "reflection_list_beta": reflection_lists[1],
        }
    case, composition_arrays = selected_composition(profile_module, reflection_lists, scales)
    arrays.update(composition_arrays)
    return [case], arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    if name.endswith("x_deg"):
        unit, description = "degree_2theta", "Profile sampling coordinates"
    elif "profile_per_deg" in name:
        unit, description = "inverse_degree", "Normalized pinned GSAS-II profile"
    elif name.startswith("reflection_list"):
        unit, description = "mixed_by_column", "Public scripting reflection list"
    else:
        unit, description = "intensity", f"Calculated {name} array"
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
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    if any(path.exists() for path in (archive_path, manifest_path)) and not force:
        raise FileExistsError("fixture exists; pass --force for explicit regeneration")
    cases, arrays = generate_snapshot(scripting, profile_module)
    np.savez_compressed(archive_path, **arrays)
    canonical_input = json.dumps(
        {"instrument": INSTRUMENT, "phases": PHASES, "composition": COMPOSITION}, sort_keys=True
    ).encode()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_multiphase_v1",
        "kind": "cw_multiphase",
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
        },
        "input_parameters": {
            "instrument": INSTRUMENT,
            "phases": PHASES,
            "composition": COMPOSITION,
        },
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures" / "multiphase_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
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
