#!/usr/bin/env python3
"""Generate pinned GSAS-II FCJ profile and derivative fixtures.

Run only with the pinned external GSAS-II Python environment. This script
imports no Rietveld Engine module and refuses overwrite without ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE_NAME = "data.npz"
MANIFEST_NAME = "manifest.json"
ADAPTER_VERSION = 1
GAUSSIAN_FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))
INSTRUMENT = {
    "wavelength_angstrom": 1.5406,
    "u_deg2": 2.0e-4,
    "v_deg2": -1.0e-4,
    "w_deg2": 1.2e-4,
    "x_deg": 1.5e-3,
    "y_deg": 3.0e-3,
}


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


def import_gsasii(gsas_root: Path, binary_dir: Path | None) -> tuple[Any, Any]:
    sys.path.insert(0, str(gsas_root))
    from GSASII import GSASIIpath

    if binary_dir is not None:
        sys.path.insert(0, str(binary_dir))
        GSASIIpath.binaryPath = str(binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    from GSASII import GSASIIpwd

    if not hasattr(GSASIIpwd, "pyd"):
        raise RuntimeError("GSAS-II pypowder binary is unavailable")
    return GSASIIpath, GSASIIpwd


def component_widths(position_deg: float) -> tuple[float, float, float]:
    theta = np.deg2rad(position_deg / 2.0)
    tangent = np.tan(theta)
    variance = (
        INSTRUMENT["u_deg2"] * tangent**2 + INSTRUMENT["v_deg2"] * tangent + INSTRUMENT["w_deg2"]
    )
    gaussian = GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(variance)
    lorentzian = INSTRUMENT["x_deg"] / np.cos(theta) + INSTRUMENT["y_deg"] * tangent
    return float(variance), float(gaussian), float(lorentzian)


def evaluate_case(
    profile_module: Any,
    position_deg: float,
    axial_sum_over_radius: float,
    regime: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_id = f"{regime}_{'zero' if axial_sum_over_radius == 0.0 else 'fcj'}"
    variance, gaussian, lorentzian = component_widths(position_deg)
    x = np.linspace(position_deg - 2.0, position_deg + 2.0, 10_001)
    native_profile, _native_integral = profile_module.getFCJVoigt3(
        position_deg,
        10_000.0 * variance,
        100.0 * lorentzian,
        axial_sum_over_radius,
        x,
    )
    derivatives = profile_module.getdFCJVoigt3(
        position_deg,
        10_000.0 * variance,
        100.0 * lorentzian,
        axial_sum_over_radius,
        x,
    )
    profile = 100.0 * np.asarray(native_profile, dtype=np.float64)
    d_position = 100.0 * np.asarray(derivatives[1], dtype=np.float64)
    d_sigma2_d_gaussian = 20_000.0 * gaussian / GAUSSIAN_FWHM_PER_SIGMA**2
    d_gaussian = 100.0 * np.asarray(derivatives[2], dtype=np.float64) * d_sigma2_d_gaussian
    d_lorentzian = 10_000.0 * np.asarray(derivatives[3], dtype=np.float64)
    d_axial_sum = 100.0 * np.asarray(derivatives[4], dtype=np.float64)
    direct_jacobian = np.stack((d_position, d_gaussian, d_lorentzian, d_axial_sum), axis=0)
    area = float(np.trapezoid(profile, x))
    centroid = float(np.trapezoid(x * profile, x) / area)
    third_moment = float(np.trapezoid((x - centroid) ** 3 * profile, x) / area)
    arrays = {
        f"{case_id}__x_deg": x,
        f"{case_id}__profile_per_deg": profile,
        f"{case_id}__direct_jacobian": direct_jacobian,
    }
    case = {
        "id": case_id,
        "case_kind": "fcj_isolated_reflection",
        "arrays": {
            "x": f"{case_id}__x_deg",
            "profile": f"{case_id}__profile_per_deg",
            "direct_jacobian": f"{case_id}__direct_jacobian",
        },
        "parameters": {
            "instrument": INSTRUMENT,
            "position_deg": position_deg,
            "angular_regime": regime,
            "gaussian_variance_deg2": variance,
            "gaussian_fwhm_deg": gaussian,
            "lorentzian_fwhm_deg": lorentzian,
            "gsas_shl": axial_sum_over_radius,
            "public_equal_height_mapping": {
                "sample_over_radius": axial_sum_over_radius / 2.0,
                "detector_over_radius": axial_sum_over_radius / 2.0,
            },
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
    elif "profile_per_deg" in name:
        unit, description = "inverse_degree", "Normalized GSAS-II FCJ profile"
    else:
        unit = "mixed_by_row"
        description = "Position, Gaussian-FWHM, Lorentzian-FWHM, and SH/L derivatives"
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
    cases: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    specifications = (
        (15.0, 0.024, "low"),
        (70.0, 0.024, "middle"),
        (140.0, 0.024, "high"),
        (70.0, 0.0, "middle"),
    )
    for position, axial_sum, regime in specifications:
        case, case_arrays = evaluate_case(profile_module, position, axial_sum, regime)
        cases.append(case)
        arrays.update(case_arrays)
    np.savez_compressed(archive_path, **arrays)

    canonical_input = json.dumps(
        {"instrument": INSTRUMENT, "cases": specifications}, sort_keys=True
    ).encode()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_fcj_profile_v1",
        "kind": "fcj_profile",
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
                "GSASII.GSASIIpwd.getFCJVoigt3",
                "GSASII.GSASIIpwd.getdFCJVoigt3",
            ],
            "private_probe": True,
            "adapter_version": ADAPTER_VERSION,
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "fcj_profile_v1",
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
