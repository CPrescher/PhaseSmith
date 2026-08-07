#!/usr/bin/env python3
"""Generate a pinned monochromatic-neutron CW fixture from GSAS-II.

The public scripting surface supplies X, Ycalc, background, and the reflection
list for a PNC histogram. Pinned private probes provide selected symmetric and
FCJ profile values. This script imports no PhaseSmith module.
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
    "probe": "neutron",
    "wavelength_angstrom": 1.909,
    "u_gsas_centideg2": 257.182710995,
    "v_gsas_centideg2": -640.525145369,
    "w_gsas_centideg2": 569.378664828,
    "x_gsas_centideg": 0.0,
    "y_gsas_centideg": 0.0,
    "sh_over_l": 0.024,
}
PHASE = {
    "name": "neutron-cw-cubic",
    "cell": [4.0, 4.0, 4.0, 90.0, 90.0, 90.0],
    "space_group": "P 1",
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
            "#GSAS-II instrument parameter file for a neutron CW oracle\n",
            "Type:PNC;Bank:1\n",
            f"Lam:{values['wavelength_angstrom']};Zero:0;Polariz.:0;Azimuth:0\n",
            f"U:{values['u_gsas_centideg2']};V:{values['v_gsas_centideg2']};"
            f"W:{values['w_gsas_centideg2']};X:{values['x_gsas_centideg']};"
            f"Y:{values['y_gsas_centideg']};Z:0;SH/L:{values['sh_over_l']}\n",
        ]
    )


def configure_neutral_sample(phase: Any) -> None:
    for key, configure in (
        (
            "Size",
            lambda current: ["isotropic", [1.0e12, current[1][1], 1.0], *current[2:]],
        ),
        (
            "Mustrain",
            lambda current: ["isotropic", [0.0, current[1][1], 0.0], *current[2:]],
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


def selected_case(
    profile_module: Any,
    reflection: np.ndarray,
    index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    position = float(reflection[5])
    sigma2 = float(reflection[6])
    gamma = float(reflection[7])
    x = np.linspace(position - 3.0, position + 3.0, 12_001)
    symmetric_native, symmetric_integral = profile_module.getPsVoigt(position, sigma2, gamma, x)
    fcj_native, fcj_integral = profile_module.getFCJVoigt3(
        position,
        sigma2,
        gamma,
        INSTRUMENT["sh_over_l"],
        x,
    )
    symmetric = 100.0 * np.asarray(symmetric_native, dtype=np.float64)
    fcj = 100.0 * np.asarray(fcj_native, dtype=np.float64)
    area = float(np.trapezoid(fcj, x))
    centroid = float(np.trapezoid(x * fcj, x) / area)
    third = float(np.trapezoid((x - centroid) ** 3 * fcj, x) / area)
    case_id = f"neutron_reflection_{index}"
    arrays = {
        f"{case_id}__x_deg": x,
        f"{case_id}__symmetric_per_deg": symmetric,
        f"{case_id}__fcj_per_deg": fcj,
    }
    case = {
        "id": case_id,
        "case_kind": "neutron_cw_reflection",
        "arrays": {
            "x": f"{case_id}__x_deg",
            "symmetric_profile": f"{case_id}__symmetric_per_deg",
            "fcj_profile": f"{case_id}__fcj_per_deg",
        },
        "parameters": {
            "hkl": [int(value) for value in reflection[:3]],
            "d_spacing_angstrom": float(reflection[4]),
            "position_deg": position,
            "sigma2_centideg2": sigma2,
            "gamma_centideg": gamma,
            "public_equal_height_mapping": {
                "sample_over_radius": INSTRUMENT["sh_over_l"] / 2.0,
                "detector_over_radius": INSTRUMENT["sh_over_l"] / 2.0,
            },
        },
        "gsasii_reported_integrals": {
            "symmetric": float(symmetric_integral),
            "fcj": float(fcj_integral),
        },
        "sampled_fcj_moments": {
            "integral": area,
            "centroid_deg": centroid,
            "third_central_moment_deg3": third,
        },
    }
    return case, arrays


def generate_snapshot(
    scripting: Any, profile_module: Any
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    np.random.seed(20_260_805)
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsas-neutron-") as temporary:
        work = Path(temporary)
        instrument_path = work / "neutron.instprm"
        instrument_path.write_text(instrument_text(), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(work / "neutron.gpx"))
        phase = project.add_phase(
            phasename=PHASE["name"],
            spacegroup=PHASE["space_group"],
            cell=PHASE["cell"],
        )
        phase.add_atom(0.0, 0.0, 0.0, element="Si", lbl="Si1", occ=1.0, uiso=0.01)
        histogram = project.add_simulated_powder_histogram(
            "neutron-cw",
            str(instrument_path),
            10.0,
            150.0,
            Npoints=7_001,
            scale=100.0,
            phases=[phase],
        )
        configure_neutral_sample(phase)
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
    cases = []
    for index in (0, reflection_list.shape[0] // 2, reflection_list.shape[0] - 1):
        case, case_arrays = selected_case(profile_module, reflection_list[index], index)
        cases.append(case)
        arrays.update(case_arrays)
    return cases, arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    if name.endswith("x_deg"):
        unit, description = "degree_2theta", "Profile sampling coordinates"
    elif "per_deg" in name:
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
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    if any(path.exists() for path in (archive_path, manifest_path)) and not force:
        raise FileExistsError("fixture exists; pass --force for explicit regeneration")
    cases, arrays = generate_snapshot(scripting, profile_module)
    np.savez_compressed(archive_path, **arrays)
    canonical_input = json.dumps(
        {"instrument": INSTRUMENT, "phase": PHASE}, sort_keys=True
    ).encode()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_neutron_cw_v1",
        "kind": "neutron_cw",
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
            "histogram_type": "PNC",
            "api": [
                "GSASII.GSASIIscriptable.G2PwdrData.getdata/reflections",
                "GSASII.GSASIIpwd.getPsVoigt",
                "GSASII.GSASIIpwd.getFCJVoigt3",
            ],
            "private_probe": True,
            "adapter_version": ADAPTER_VERSION,
            "reflection_columns": REFLECTION_COLUMNS,
        },
        "input_parameters": {"instrument": INSTRUMENT, "phase": PHASE},
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "neutron_cw_v1",
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
