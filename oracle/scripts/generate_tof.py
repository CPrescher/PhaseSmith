#!/usr/bin/env python3
"""Generate a pinned neutron-TOF fixture from GSAS-II.

The public scripting API supplies bin-center X, Ycalc, background, and the
reflection list for a PNT histogram. A small pinned private probe supplies
selected exponential-pseudo-Voigt values and derivatives. This script imports
no PhaseSmith module.
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
    "zero_us": -0.773346536757,
    "difc_us_per_angstrom": 5084.82763065,
    "difa_us_per_angstrom2": -2.6304177486,
    "difb_us_angstrom": 1.25,
    "alpha_coefficient": 5.0,
    "beta0_per_us": 0.028,
    "beta1_angstrom4_per_us": 0.0012,
    "betaq_angstrom2_per_us": 0.003,
    "sigma0_us2": 1.5,
    "sigma1_us2_per_angstrom2": 15.1402867268,
    "sigma2_us2_per_angstrom4": 0.08,
    "sigmaq_us2_per_angstrom": 0.7,
    "x_us_per_angstrom": 0.8,
    "y_us_per_angstrom2": 0.15,
    "z_us": 1.2,
    "flight_path_m": 10.32567,
    "two_theta_deg": 151.0,
}
PHASE = {
    "name": "tof-cubic",
    "cell": [4.0, 4.0, 4.0, 90.0, 90.0, 90.0],
    "space_group": "P 1",
}
REFLECTION_COLUMNS = [
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
    p = INSTRUMENT
    return "".join(
        [
            "#GSAS-II instrument parameter file for the neutron TOF oracle\n",
            f"Type:PNT;Bank:1;fltPath:{p['flight_path_m']};"
            f"2-theta:{p['two_theta_deg']};Azimuth:0\n",
            f"Zero:{p['zero_us']};difC:{p['difc_us_per_angstrom']};"
            f"difA:{p['difa_us_per_angstrom2']};difB:{p['difb_us_angstrom']}\n",
            f"alpha:{p['alpha_coefficient']}\n",
            f"beta-0:{p['beta0_per_us']};beta-1:{p['beta1_angstrom4_per_us']};"
            f"beta-q:{p['betaq_angstrom2_per_us']}\n",
            f"sig-0:{p['sigma0_us2']};sig-1:{p['sigma1_us2_per_angstrom2']};"
            f"sig-2:{p['sigma2_us2_per_angstrom4']};"
            f"sig-q:{p['sigmaq_us2_per_angstrom']}\n",
            f"X:{p['x_us_per_angstrom']};Y:{p['y_us_per_angstrom2']};Z:{p['z_us']}\n",
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


def derived_parameters(d_spacing: float) -> dict[str, float]:
    p = INSTRUMENT
    d = d_spacing
    return {
        "d_spacing_angstrom": d,
        "position_us": p["zero_us"]
        + p["difc_us_per_angstrom"] * d
        + p["difa_us_per_angstrom2"] * d**2
        + p["difb_us_angstrom"] / d,
        "alpha_per_us": p["alpha_coefficient"] / d,
        "beta_per_us": p["beta0_per_us"]
        + p["beta1_angstrom4_per_us"] / d**4
        + p["betaq_angstrom2_per_us"] / d**2,
        "sigma2_us2": p["sigma0_us2"]
        + p["sigma1_us2_per_angstrom2"] * d**2
        + p["sigma2_us2_per_angstrom4"] * d**4
        + p["sigmaq_us2_per_angstrom"] * d,
        "gamma_us": p["z_us"] + p["x_us_per_angstrom"] * d + p["y_us_per_angstrom2"] * d**2,
    }


def selected_case(
    profile_module: Any, d_spacing: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    parameters = derived_parameters(d_spacing)
    position = parameters["position_us"]
    x = np.linspace(position - 900.0, position + 900.0, 7_201)
    value, reported_integral = profile_module.getEpsVoigt(
        position,
        parameters["alpha_per_us"],
        parameters["beta_per_us"],
        parameters["sigma2_us2"],
        parameters["gamma_us"],
        x,
    )
    derivatives = profile_module.getdEpsVoigt(
        position,
        parameters["alpha_per_us"],
        parameters["beta_per_us"],
        parameters["sigma2_us2"],
        parameters["gamma_us"],
        x,
    )
    names = ("value", "d_position", "d_alpha", "d_beta", "d_sigma2", "d_gamma")
    case_id = f"tof_d_{str(d_spacing).replace('.', '_')}"
    arrays = {f"{case_id}__x_us": np.ascontiguousarray(x, dtype=np.float64)}
    array_refs = {"x": f"{case_id}__x_us"}
    for name, raw in zip(names, derivatives, strict=True):
        key = f"{case_id}__{name}"
        arrays[key] = np.ascontiguousarray(raw, dtype=np.float64)
        array_refs[name] = key
    if not np.array_equal(arrays[array_refs["value"]], np.asarray(value)):
        raise RuntimeError("GSAS-II value and derivative probes disagree")
    y = arrays[array_refs["value"]]
    area = float(np.trapezoid(y, x))
    centroid = float(np.trapezoid(x * y, x) / area)
    third = float(np.trapezoid((x - centroid) ** 3 * y, x) / area)
    case = {
        "id": case_id,
        "case_kind": "tof_private_profile",
        "arrays": array_refs,
        "parameters": parameters,
        "gsasii_reported_integral": float(reported_integral),
        "sampled_moments": {
            "integral": area,
            "centroid_us": centroid,
            "third_central_moment_us3": third,
        },
    }
    return case, arrays


def generate_snapshot(
    scripting: Any, profile_module: Any
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    np.random.seed(20_260_805)
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsas-tof-") as temporary:
        work = Path(temporary)
        instrument_path = work / "tof.instprm"
        instrument_path.write_text(instrument_text(), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(work / "tof.gpx"))
        phase = project.add_phase(
            phasename=PHASE["name"],
            spacegroup=PHASE["space_group"],
            cell=PHASE["cell"],
        )
        phase.add_atom(0.0, 0.0, 0.0, element="Si", lbl="Si1", occ=1.0, uiso=0.01)
        histogram = project.add_simulated_powder_histogram(
            "neutron-tof",
            str(instrument_path),
            2.0,
            20.0,
            Npoints=6_001,
            scale=100_000.0,
            phases=[phase],
        )
        configure_neutral_sample(phase)
        project.do_refinements([{}], outputnames=[None])
        reflection_list = np.ascontiguousarray(
            histogram.reflections()[PHASE["name"]]["RefList"], dtype=np.float64
        )
        if reflection_list.shape[1] != len(REFLECTION_COLUMNS):
            raise RuntimeError(f"unexpected TOF reflection-list width {reflection_list.shape[1]}")
        arrays = {
            "x_us": np.ascontiguousarray(histogram.getdata("X"), dtype=np.float64),
            "ycalc": np.ascontiguousarray(histogram.getdata("Ycalc"), dtype=np.float64),
            "background": np.ascontiguousarray(histogram.getdata("Background"), dtype=np.float64),
            "reflection_list": reflection_list,
        }
    cases = [
        {
            "id": "tof_public_histogram",
            "case_kind": "tof_public_histogram",
            "arrays": {
                "x": "x_us",
                "ycalc": "ycalc",
                "background": "background",
                "reflection_list": "reflection_list",
            },
            "parameters": {"coordinate_convention": "bin_center_microseconds"},
            "reflection_tables": [{"array": "reflection_list", "columns": REFLECTION_COLUMNS}],
        }
    ]
    for d_spacing in (0.8, 1.5, 2.5):
        case, case_arrays = selected_case(profile_module, d_spacing)
        cases.append(case)
        arrays.update(case_arrays)
    return cases, arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    if name.endswith("x_us") or name == "x_us":
        unit, description = "microsecond_bin_center", "TOF calculation coordinates"
    elif name == "reflection_list":
        unit, description = "mixed_by_column", "Public scripting reflection list"
    elif name in {"ycalc", "background"}:
        unit, description = "intensity", f"Public scripting {name} array"
    else:
        unit, description = "profile_dependent", "Pinned GSAS-II TOF profile or derivative"
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
        "fixture_id": "gsasii_tof_v1",
        "kind": "neutron_tof",
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
            "histogram_type": "PNT",
            "api": [
                "GSASII.GSASIIscriptable.G2PwdrData.getdata/reflections",
                "GSASII.GSASIIpwd.getEpsVoigt/getdEpsVoigt",
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "tof_v1",
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
