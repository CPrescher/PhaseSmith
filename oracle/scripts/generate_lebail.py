#!/usr/bin/env python3
"""Generate a pinned GSAS-II Le Bail fixture without importing Rietveld Engine.

Public scripting creates the phase/histogram, supplies the calculated pattern,
enables Le Bail mode, and extracts arrays and reflection records. One pinned,
version-gated call requests GSAS-II's documented new-Le-Bail initialization;
that internal call is recorded in the manifest and never enters the package.
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
PHASE = {
    "phase_id": "alpha",
    "name": "lebail-cubic",
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
    from GSASII import GSASIIscriptable, GSASIIstrMain

    return GSASIIpath, GSASIIscriptable, GSASIIstrMain


def instrument_text() -> str:
    values = INSTRUMENT
    return "".join(
        [
            "#GSAS-II instrument parameter file for a Le Bail oracle\n",
            "Type:PNC;Bank:1\n",
            f"Lam:{values['wavelength_angstrom']};Zero:0;Polariz.:0;Azimuth:0\n",
            f"U:{values['u_gsas_centideg2']};V:{values['v_gsas_centideg2']};"
            f"W:{values['w_gsas_centideg2']};X:{values['x_gsas_centideg']};"
            f"Y:{values['y_gsas_centideg']};Z:0;SH/L:0\n",
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
            raise RuntimeError(f"expected one {key} HAP entry")
        path = entries[0][0]
        phase.setHAPentryValue(path, configure(phase.getHAPentryValue(path)))
    scale_path = phase.getHAPentryList(0, "Scale")[0][0]
    scale = phase.getHAPentryValue(scale_path)
    scale[1] = False
    phase.setHAPentryValue(scale_path, scale)


def residual_row(histogram: Any, cycle: int) -> list[float]:
    residuals = histogram.residuals
    return [
        float(cycle),
        float(residuals["R"]),
        float(residuals["wR"]),
        float(residuals["Rb"]),
        float(residuals["wRb"]),
    ]


def generate_snapshot(
    scripting: Any, structure_main: Any
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    np.random.seed(20_260_805)
    with tempfile.TemporaryDirectory(prefix="rietveld-gsas-lebail-") as temporary:
        work = Path(temporary)
        instrument_path = work / "lebail.instprm"
        instrument_path.write_text(instrument_text(), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(work / "lebail.gpx"))
        phase = project.add_phase(
            phasename=PHASE["name"],
            spacegroup=PHASE["space_group"],
            cell=PHASE["cell"],
        )
        phase.add_atom(0.0, 0.0, 0.0, element="Si", lbl="Si1", occ=1.0, uiso=0.01)
        histogram = project.add_simulated_powder_histogram(
            "lebail",
            str(instrument_path),
            20.0,
            80.0,
            Npoints=3_001,
            scale=100.0,
            phases=[phase],
        )
        configure_neutral_sample(phase)
        histogram.clear_refinements({"Sample Parameters": ["Scale"]})
        project.do_refinements([{}], outputnames=[None])

        observed = np.ascontiguousarray(histogram.getdata("Ycalc"), dtype=np.float64)
        # Pinned internal probe: install deterministic synthetic observations and
        # unit weights because getdata intentionally returns copies.
        histogram.data["data"][1][1][:] = observed
        histogram.data["data"][1][2][:] = 1.0
        phase.set_refinements({"LeBail": True})
        project.set_Controls("cycles", 1)
        project.index_ids()

        trends = []
        structure_main.Refine(project.filename, newLeBail=True)
        project.reload()
        trends.append(residual_row(project.histogram(0), 0))
        for cycle in range(1, 8):
            project.refine(makeBack=False)
            trends.append(residual_row(project.histogram(0), cycle))

        histogram = project.histogram(0)
        reflections = np.ascontiguousarray(
            histogram.reflections()[PHASE["name"]]["RefList"], dtype=np.float64
        )
        arrays = {
            "x_deg": np.ascontiguousarray(histogram.getdata("X"), dtype=np.float64),
            "observed_y": observed,
            "ycalc": np.ascontiguousarray(histogram.getdata("Ycalc"), dtype=np.float64),
            "background": np.ascontiguousarray(
                histogram.getdata("Background"), dtype=np.float64
            ),
            "weight": np.ascontiguousarray(histogram.getdata("Yweight"), dtype=np.float64),
            "reflection_list": reflections,
            "convergence": np.ascontiguousarray(trends, dtype=np.float64),
        }
    case = {
        "id": "neutron_cw_exact_synthetic",
        "case_kind": "lebail_refinement",
        "arrays": {
            "x": "x_deg",
            "observed_y": "observed_y",
            "ycalc": "ycalc",
            "background": "background",
            "weight": "weight",
            "reflection_list": "reflection_list",
            "convergence": "convergence",
        },
        "parameters": {
            "integrated_intensity_convention": (
                "0.01_times_f_obs2_times_intensity_correction"
            ),
            "convergence_columns": [
                "cycle",
                "R_percent",
                "wR_percent",
                "Rb_percent",
                "wRb_percent",
            ],
        },
        "reflection_tables": [
            {"array": "reflection_list", "columns": REFLECTION_COLUMNS}
        ],
    }
    return [case], arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    if name == "x_deg":
        unit, description = "degree_2theta", "Profile sampling coordinates"
    elif name == "reflection_list":
        unit, description = "mixed_by_column", "Final public reflection list"
    elif name == "convergence":
        unit, description = "mixed_by_column", "Public residual trend"
    elif name == "weight":
        unit, description = "inverse_intensity_squared", "Observation weights"
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
    structure_main: Any,
    pinned: dict[str, Any],
    *,
    force: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    if any(path.exists() for path in (archive_path, manifest_path)) and not force:
        raise FileExistsError("fixture exists; pass --force for explicit regeneration")
    cases, arrays = generate_snapshot(scripting, structure_main)
    np.savez_compressed(archive_path, **arrays)
    canonical_input = json.dumps(
        {"instrument": INSTRUMENT, "phase": PHASE}, sort_keys=True
    ).encode()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_lebail_v1",
        "kind": "lebail",
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
                "GSASII.GSASIIscriptable.G2Phase.set_refinements",
                "GSASII.GSASIIscriptable.G2Project.refine",
                "GSASII.GSASIIscriptable.G2PwdrData.getdata/reflections",
                "GSASII.GSASIIstrMain.Refine(newLeBail=True)",
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
        raise RuntimeError("refusing to write a fixture from an unpinned revision")
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "lebail_v1",
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
    gsasii_path, scripting, structure_main = import_gsasii(
        gsas_root, arguments.binary_dir.resolve() if arguments.binary_dir else None
    )
    write_fixture(
        arguments.output.resolve(),
        gsas_root,
        gsasii_path,
        scripting,
        structure_main,
        pinned,
        force=arguments.force,
    )
    print(f"wrote {arguments.output.resolve() / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
