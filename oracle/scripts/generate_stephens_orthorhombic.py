#!/usr/bin/env python3
"""Generate a pinned GSAS-II orthorhombic Stephens width fixture.

The histogram and generalized microstrain record are configured through the
public scripting API. The reflection list is public plain data; selected
normalized profiles use the small version-gated private profile probe already
declared by the sample-physics oracle boundary. This script imports no
PhaseSmith module.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from generate_sample_physics import (
    REFLECTION_COLUMNS,
    git_revision,
    import_gsasii,
    instrument_text,
    sha256_bytes,
    sha256_file,
    update_hap_entry,
)

ARCHIVE_NAME = "data.npz"
MANIFEST_NAME = "manifest.json"
ADAPTER_VERSION = 1
PHASE = {
    "name": "stephens-orthorhombic",
    "space_group": "P n m a",
    "cell": [8.0, 9.0, 10.0, 90.0, 90.0, 90.0],
}
SAMPLE = {
    "coefficient_order": ["S400", "S040", "S004", "S220", "S202", "S022"],
    "gsasii_coefficients": [0.46, 0.026, 0.049, 0.057, 0.012, 0.026],
    "lorentzian_fraction": 0.9,
}


def configure_stephens(phase: Any) -> None:
    """Configure generalized orthorhombic microstrain through public HAP access."""

    def disable_size(current: list[Any]) -> list[Any]:
        current[0] = "isotropic"
        current[1][0] = 1.0e20
        current[1][2] = 1.0
        return current

    def transform(current: list[Any]) -> list[Any]:
        current[0] = "generalized"
        current[1][2] = SAMPLE["lorentzian_fraction"]
        current[4] = list(SAMPLE["gsasii_coefficients"])
        current[5] = [False] * len(current[4])
        return current

    update_hap_entry(phase, "Size", disable_size)
    update_hap_entry(phase, "Mustrain", transform)


def public_translation() -> dict[str, Any]:
    """Return the independently derived physical coefficient conversion."""

    scale = 1.0e-12 / (8.0 * np.log(2.0))
    multipliers = np.asarray((1.0, 1.0, 1.0, 3.0, 3.0, 3.0))
    coefficients = np.asarray(SAMPLE["gsasii_coefficients"]) * scale * multipliers
    return {
        "coefficient_order": SAMPLE["coefficient_order"],
        "coefficients_angstrom_minus4": coefficients.tolist(),
        "lorentzian_fraction": SAMPLE["lorentzian_fraction"],
        "conversion": "pure*1e-12/(8ln2), mixed*3e-12/(8ln2)",
    }


def profile_case(
    profile_module: Any, reflection: np.ndarray, index: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Probe one normalized TCH profile from public reflection-list widths."""

    position = float(reflection[5])
    sigma2 = float(reflection[6])
    gamma = float(reflection[7])
    half_span = max(1.5, 40.0 * np.sqrt(sigma2) / 100.0, 0.4 * gamma)
    x = np.linspace(position - half_span, position + half_span, 6_001)
    native, native_integral = profile_module.getPsVoigt(position, sigma2, gamma, x)
    profile = 100.0 * np.asarray(native, dtype=np.float64)
    case_id = f"stephens_reflection_{index}"
    arrays = {
        f"{case_id}__x_deg": x,
        f"{case_id}__profile_per_deg": profile,
    }
    return (
        {
            "id": case_id,
            "case_kind": "orthorhombic_stephens_reflection",
            "arrays": {
                "x": f"{case_id}__x_deg",
                "profile": f"{case_id}__profile_per_deg",
                "reflection_list": "reflection_list",
            },
            "parameters": {
                "reflection_index": index,
                "hkl": [int(value) for value in reflection[:3]],
                "d_spacing_angstrom": float(reflection[4]),
                "position_deg": position,
                "sigma2_centideg2": sigma2,
                "gamma_centideg": gamma,
                "public_translation": public_translation(),
            },
            "gsasii_reported_integral": float(native_integral),
        },
        arrays,
    )


def generate_snapshot(
    scripting: Any, profile_module: Any
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Build one public orthorhombic histogram and selected profile probes."""

    import tempfile

    np.random.seed(20_260_813)
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsas-stephens-") as temporary:
        work = Path(temporary)
        instrument_path = work / "stephens.instprm"
        instrument_path.write_text(instrument_text(), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(work / "stephens.gpx"))
        phase = project.add_phase(
            phasename=PHASE["name"],
            spacegroup=PHASE["space_group"],
            cell=PHASE["cell"],
        )
        phase.add_atom(0.0, 0.0, 0.0, element="Si", lbl="Si1", occ=1.0, uiso=0.01)
        histogram = project.add_simulated_powder_histogram(
            "stephens",
            str(instrument_path),
            10.0,
            100.0,
            Npoints=4_501,
            scale=100.0,
            phases=[phase],
        )
        configure_stephens(phase)
        project.do_refinements([{}], outputnames=[None])
        reflection_list = np.ascontiguousarray(
            histogram.reflections()[PHASE["name"]]["RefList"], dtype=np.float64
        )
    arrays = {"reflection_list": reflection_list}
    cases = []
    for index in (0, reflection_list.shape[0] // 2, reflection_list.shape[0] - 1):
        case, profile_arrays = profile_case(profile_module, reflection_list[index], index)
        cases.append(case)
        arrays.update(profile_arrays)
    return cases, arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    """Describe and hash one fixture array."""

    unit = (
        "mixed_by_column"
        if name == "reflection_list"
        else "degree_2theta"
        if name.endswith("x_deg")
        else "inverse_degree"
    )
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes(order="C")),
        "finite": bool(np.isfinite(array).all()),
        "unit": unit,
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
    """Generate a reviewable, hash-locked oracle fixture."""

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    if not force and (archive_path.exists() or manifest_path.exists()):
        raise FileExistsError("fixture exists; pass --force for explicit regeneration")
    cases, arrays = generate_snapshot(scripting, profile_module)
    np.savez_compressed(archive_path, **arrays)
    generator_path = Path(__file__).resolve()
    helper_path = generator_path.with_name("generate_sample_physics.py")
    canonical_input = json.dumps({"phase": PHASE, "sample": SAMPLE}, sort_keys=True).encode()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_stephens_orthorhombic_v1",
        "kind": "cw_orthorhombic_stephens",
        "provenance": {
            "gsasii_repository": pinned["repository"],
            "gsasii_revision": git_revision(gsas_root),
            "gsasii_tag": int(gsasii_path.GetVersionNumber()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "generator_sha256": sha256_file(generator_path),
            "helper_sha256": sha256_file(helper_path),
        },
        "source": {
            "api": [
                "GSASII.GSASIIscriptable.G2Phase.getHAPentryList/getHAPentryValue/setHAPentryValue",
                "GSASII.GSASIIscriptable.G2PwdrData.reflections",
                "GSASII.GSASIIpwd.getPsVoigt",
            ],
            "private_probe": True,
            "adapter_version": ADAPTER_VERSION,
            "reflection_columns": REFLECTION_COLUMNS,
            "native_units": {
                "coefficients": "GSAS-II generalized microstrain stored values",
                "Gaussian_variance": "centidegree_squared",
                "Lorentzian_fwhm": "centidegree",
            },
        },
        "input_parameters": {"phase": PHASE, "sample": SAMPLE},
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "stephens_orthorhombic_v1",
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
