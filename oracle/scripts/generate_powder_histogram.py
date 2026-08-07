#!/usr/bin/env python3
"""Generate a minimal public-scripting GSAS-II powder-histogram fixture.

The project is constructed entirely inside a temporary directory. Only plain
arrays and provenance are written to the fixture; the GPX file and all GSAS-II
objects are discarded. Existing fixture files require ``--force`` to replace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
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

INSTRUMENT_PARAMETERS = {
    "wavelength_angstrom": 1.5406,
    "u": 2.0,
    "v": -1.0,
    "w": 1.0,
    "x": 0.1,
    "y": 0.2,
    "z": 0.0,
    "sh_over_l": 0.002,
}
PHASE_PARAMETERS = {
    "name": "synthetic-silicon",
    "space_group": "P 1",
    "cell": [4.5, 4.5, 4.5, 90.0, 90.0, 90.0],
    "atoms": [
        {"label": "Si1", "element": "Si", "fractional": [0.0, 0.0, 0.0]},
        {"label": "Si2", "element": "Si", "fractional": [0.25, 0.25, 0.25]},
    ],
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


def import_scripting(gsas_root: Path, binary_dir: Path | None) -> tuple[Any, Any]:
    sys.path.insert(0, str(gsas_root))
    from GSASII import GSASIIpath

    if binary_dir is not None:
        sys.path.insert(0, str(binary_dir))
        GSASIIpath.binaryPath = str(binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable

    return GSASIIpath, GSASIIscriptable


def instrument_text() -> str:
    values = INSTRUMENT_PARAMETERS
    return "".join(
        [
            "#GSAS-II instrument parameter file for a synthetic oracle case\n",
            "Type:PXC;Bank:1\n",
            f"Lam:{values['wavelength_angstrom']};Zero:0.0;Polariz.:0.99;Azimuth:0.0\n",
            f"U:{values['u']};V:{values['v']};W:{values['w']};X:{values['x']};"
            f"Y:{values['y']};Z:{values['z']};SH/L:{values['sh_over_l']}\n",
        ]
    )


def generate_snapshot(scripting: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    random.seed(20_260_805)
    np.random.seed(20_260_805)
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsas-oracle-") as temporary:
        work = Path(temporary)
        instrument_path = work / "synthetic.instprm"
        instrument_path.write_text(instrument_text(), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(work / "synthetic.gpx"))
        phase = project.add_phase(
            phasename=PHASE_PARAMETERS["name"],
            spacegroup=PHASE_PARAMETERS["space_group"],
            cell=PHASE_PARAMETERS["cell"],
        )
        for atom in PHASE_PARAMETERS["atoms"]:
            phase.add_atom(
                *atom["fractional"],
                element=atom["element"],
                lbl=atom["label"],
                occ=1.0,
                uiso=0.01,
            )
        histogram = project.add_simulated_powder_histogram(
            "synthetic-cw",
            str(instrument_path),
            10.0,
            80.0,
            Npoints=3_501,
            scale=100.0,
            phases=[phase],
        )
        project.do_refinements([{}], outputnames=[None])

        arrays: dict[str, np.ndarray] = {
            "x_deg": np.ascontiguousarray(histogram.getdata("X"), dtype=np.float64),
            "ycalc": np.ascontiguousarray(histogram.getdata("Ycalc"), dtype=np.float64),
            "background": np.ascontiguousarray(
                histogram.getdata("Background"), dtype=np.float64
            ),
        }
        reflection_metadata = []
        for index, (phase_name, payload) in enumerate(sorted(histogram.reflections().items())):
            key = f"reflection_list_{index}"
            reflection_list = np.ascontiguousarray(payload["RefList"], dtype=np.float64)
            if reflection_list.shape[1] != len(REFLECTION_COLUMNS):
                raise RuntimeError(
                    f"unexpected powder reflection width {reflection_list.shape[1]}; "
                    f"expected {len(REFLECTION_COLUMNS)}"
                )
            arrays[key] = reflection_list
            reflection_metadata.append(
                {
                    "phase": str(phase_name),
                    "array": key,
                    "histogram_type": str(payload.get("Type", "")),
                    "superspace": bool(payload.get("Super", False)),
                    "column_count": int(reflection_list.shape[1]),
                    "columns": REFLECTION_COLUMNS,
                }
            )

    case = {
        "id": "minimal_public_cw_histogram",
        "case_kind": "histogram",
        "arrays": {
            "x": "x_deg",
            "ycalc": "ycalc",
            "background": "background",
        },
        "parameters": {
            "histogram_name": "PWDR synthetic-cw",
            "coordinate_unit": "degree_2theta",
            "range_deg": [10.0, 80.0],
            "sample_count": 3_501,
            "instrument": INSTRUMENT_PARAMETERS,
            "phase": PHASE_PARAMETERS,
        },
        "reflection_tables": reflection_metadata,
    }
    return case, arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    unit = "dimensionless"
    description = "GSAS-II powder reflection table"
    if name == "x_deg":
        unit = "degree_2theta"
        description = "Public scripting X coordinates"
    elif name in {"ycalc", "background"}:
        unit = "intensity"
        description = f"Public scripting {name} array"
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": sha256_bytes(array.tobytes(order="C")),
        "finite": bool(np.isfinite(array).all()),
        "unit": unit,
        "description": description,
    }


def write_fixture(
    output: Path,
    gsas_root: Path,
    gsasii_path: Any,
    scripting: Any,
    pinned: dict[str, Any],
    *,
    force: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    existing = [path for path in (archive_path, manifest_path) if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite {joined}; pass --force explicitly")

    case, arrays = generate_snapshot(scripting)
    np.savez_compressed(archive_path, **arrays)
    canonical_input = json.dumps(
        {"instrument": INSTRUMENT_PARAMETERS, "phase": PHASE_PARAMETERS}, sort_keys=True
    ).encode()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_minimal_cw_histogram_v1",
        "kind": "powder_histogram",
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
            "api": "GSASII.GSASIIscriptable.G2PwdrData.getdata/reflections",
            "private_probe": False,
            "adapter_version": ADAPTER_VERSION,
            "native_units": {
                "coordinate": "degree_2theta",
                "profile": "intensity",
                "reflection_table": "GSAS-II powder reflection columns",
            },
        },
        "input": {
            "kind": "parameter_cases",
            "sha256": sha256_bytes(canonical_input),
        },
        "archive": {"file": ARCHIVE_NAME, "sha256": sha256_file(archive_path)},
        "arrays": {name: array_descriptor(name, array) for name, array in arrays.items()},
        "cases": [case],
    }
    if manifest["provenance"]["gsasii_revision"] != pinned["revision"]:
        raise RuntimeError("GSAS-II checkout changed while generating fixture")
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "minimal_cw_histogram_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    pinned = json.loads((repository_root / "oracle" / "PINNED_GSASII.json").read_text())
    gsas_root = arguments.gsas_root.resolve()
    revision = git_revision(gsas_root)
    if revision != pinned["revision"]:
        raise RuntimeError(f"GSAS-II revision {revision} does not match pin {pinned['revision']}")
    gsasii_path, scripting = import_scripting(
        gsas_root, arguments.binary_dir.resolve() if arguments.binary_dir else None
    )
    write_fixture(
        arguments.output.resolve(),
        gsas_root,
        gsasii_path,
        scripting,
        pinned,
        force=arguments.force,
    )
    print(f"wrote {arguments.output.resolve() / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
