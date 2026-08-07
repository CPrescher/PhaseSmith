#!/usr/bin/env python3
"""Generate pinned GSAS-II FCJ plus wavelength-doublet fixtures.

Run only with the pinned external GSAS-II Python environment. This script
imports no PhaseSmith module and refuses overwrite without ``--force``.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from generate_fcj_profile import (
    git_revision,
    import_gsasii,
    sha256_bytes,
    sha256_file,
)

ARCHIVE_NAME = "data.npz"
MANIFEST_NAME = "manifest.json"
ADAPTER_VERSION = 1
GAUSSIAN_FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))
INSTRUMENT = {
    "wavelength_angstrom": 1.54056,
    "u_deg2": 2.0e-4,
    "v_deg2": -1.0e-4,
    "w_deg2": 1.2e-4,
    "x_deg": 1.5e-3,
    "y_deg": 3.0e-3,
}
WAVELENGTHS = np.array([1.54056, 1.54439], dtype=np.float64)
RELATIVE_INTENSITIES = np.array([1.0, 0.5], dtype=np.float64)
AXIAL_SUM_OVER_RADIUS = 0.024


def component_position(base_position_deg: float, wavelength_angstrom: float) -> float:
    base_theta = np.deg2rad(base_position_deg / 2.0)
    ratio = wavelength_angstrom / INSTRUMENT["wavelength_angstrom"]
    return float(np.rad2deg(2.0 * np.arcsin(ratio * np.sin(base_theta))))


def component_widths(position_deg: float) -> tuple[float, float]:
    theta = np.deg2rad(position_deg / 2.0)
    tangent = np.tan(theta)
    variance = (
        INSTRUMENT["u_deg2"] * tangent**2 + INSTRUMENT["v_deg2"] * tangent + INSTRUMENT["w_deg2"]
    )
    lorentzian = INSTRUMENT["x_deg"] / np.cos(theta) + INSTRUMENT["y_deg"] * tangent
    return float(variance), float(lorentzian)


def evaluate_case(
    profile_module: Any, base_position_deg: float, regime: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_id = f"{regime}_fcj_doublet"
    x = np.linspace(base_position_deg - 2.0, base_position_deg + 2.0, 10_001)
    positions = np.array(
        [component_position(base_position_deg, wavelength) for wavelength in WAVELENGTHS]
    )
    weights = RELATIVE_INTENSITIES / np.sum(RELATIVE_INTENSITIES)
    profiles = np.empty((WAVELENGTHS.size, x.size), dtype=np.float64)
    widths = []
    for component, position in enumerate(positions):
        variance, lorentzian = component_widths(float(position))
        native_profile, _native_integral = profile_module.getFCJVoigt3(
            float(position),
            10_000.0 * variance,
            100.0 * lorentzian,
            AXIAL_SUM_OVER_RADIUS,
            x,
        )
        profiles[component] = 100.0 * np.asarray(native_profile, dtype=np.float64)
        widths.append(
            {
                "gaussian_variance_deg2": variance,
                "gaussian_fwhm_deg": float(GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(variance)),
                "lorentzian_fwhm_deg": lorentzian,
            }
        )
    ycalc = np.sum(weights[:, None] * profiles, axis=0)
    area = float(np.trapezoid(ycalc, x))
    centroid = float(np.trapezoid(x * ycalc, x) / area)
    third_moment = float(np.trapezoid((x - centroid) ** 3 * ycalc, x) / area)
    arrays = {
        f"{case_id}__x_deg": x,
        f"{case_id}__component_positions_deg": positions,
        f"{case_id}__component_profiles_per_deg": profiles,
        f"{case_id}__ycalc": ycalc,
    }
    case = {
        "id": case_id,
        "case_kind": "fcj_wavelength_doublet",
        "arrays": {
            "x": f"{case_id}__x_deg",
            "component_positions": f"{case_id}__component_positions_deg",
            "component_profiles": f"{case_id}__component_profiles_per_deg",
            "ycalc": f"{case_id}__ycalc",
        },
        "parameters": {
            "instrument": INSTRUMENT,
            "base_position_deg": base_position_deg,
            "angular_regime": regime,
            "wavelengths_angstrom": WAVELENGTHS.tolist(),
            "relative_intensities": RELATIVE_INTENSITIES.tolist(),
            "normalized_intensities": weights.tolist(),
            "gsas_shl": AXIAL_SUM_OVER_RADIUS,
            "public_equal_height_mapping": {
                "sample_over_radius": AXIAL_SUM_OVER_RADIUS / 2.0,
                "detector_over_radius": AXIAL_SUM_OVER_RADIUS / 2.0,
            },
            "component_widths": widths,
        },
        "sampled_moments": {
            "integral": area,
            "centroid_deg": centroid,
            "third_central_moment_deg3": third_moment,
        },
    }
    return case, arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    if "__x_deg" in name:
        unit, description = "degree_2theta", "Profile sampling coordinates"
    elif "positions_deg" in name:
        unit, description = "degree_2theta", "Bragg-law component positions"
    elif "component_profiles" in name:
        unit, description = "inverse_degree", "Individual GSAS-II FCJ components"
    else:
        unit, description = "inverse_degree", "Normalized weighted doublet profile"
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
    profile_module: Any,
    pinned: dict[str, Any],
    *,
    force: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    existing = [path for path in (archive_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"refusing to overwrite {', '.join(map(str, existing))}; pass --force explicitly"
        )
    specifications = ((15.0, "low"), (70.0, "middle"), (140.0, "high"))
    cases: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for position, regime in specifications:
        case, case_arrays = evaluate_case(profile_module, position, regime)
        cases.append(case)
        arrays.update(case_arrays)
    np.savez_compressed(archive_path, **arrays)

    canonical_input = json.dumps(
        {
            "instrument": INSTRUMENT,
            "wavelengths": WAVELENGTHS.tolist(),
            "relative_intensities": RELATIVE_INTENSITIES.tolist(),
            "gsas_shl": AXIAL_SUM_OVER_RADIUS,
            "cases": specifications,
        },
        sort_keys=True,
    ).encode()
    helper = Path(__file__).with_name("generate_fcj_profile.py")
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_wavelength_components_v1",
        "kind": "fcj_wavelength_components",
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
            "api": ["GSASII.GSASIIpwd.getFCJVoigt3"],
            "private_probe": True,
            "adapter_version": ADAPTER_VERSION,
            "helper_sha256": sha256_file(helper),
            "native_units": {
                "coordinate": "degree_2theta",
                "Gaussian_variance": "centidegree_squared",
                "Lorentzian_fwhm": "centidegree",
                "profile_density": "inverse_centidegree",
                "axial_parameter": "SH_over_L",
            },
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "wavelength_components_v1",
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
    gsasii_path, profile_module = import_gsasii(
        gsas_root, arguments.binary_dir.resolve() if arguments.binary_dir else None
    )
    write_fixture(
        arguments.output.resolve(),
        gsas_root,
        gsasii_path,
        profile_module,
        pinned,
        force=arguments.force,
    )
    print(f"wrote {arguments.output.resolve() / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
