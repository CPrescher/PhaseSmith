#!/usr/bin/env python3
"""Generate pinned GSAS-II U/V/W/X/Y profile and derivative fixtures.

Run this only with the pinned external GSAS-II Python environment. The script
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


def widths_and_derivatives(position_deg: float) -> dict[str, Any]:
    """Evaluate the documented public-unit U/V/W/X/Y equations."""

    theta = np.deg2rad(position_deg / 2.0)
    tangent = np.tan(theta)
    secant = 1.0 / np.cos(theta)
    variance = (
        INSTRUMENT["u_deg2"] * tangent**2 + INSTRUMENT["v_deg2"] * tangent + INSTRUMENT["w_deg2"]
    )
    sigma = np.sqrt(variance)
    gaussian = GAUSSIAN_FWHM_PER_SIGMA * sigma
    lorentzian = INSTRUMENT["x_deg"] * secant + INSTRUMENT["y_deg"] * tangent
    d_gaussian_d_variance = GAUSSIAN_FWHM_PER_SIGMA / (2.0 * sigma)
    d_gaussian = d_gaussian_d_variance * np.array(
        [tangent**2, tangent, 1.0, 0.0, 0.0], dtype=np.float64
    )
    d_lorentzian = np.array([0.0, 0.0, 0.0, secant, tangent], dtype=np.float64)
    radians_per_degree = np.pi / 360.0
    d_tangent = radians_per_degree * secant**2
    d_secant = radians_per_degree * secant * tangent
    d_variance = (2.0 * INSTRUMENT["u_deg2"] * tangent + INSTRUMENT["v_deg2"]) * d_tangent
    return {
        "gaussian_variance_deg2": float(variance),
        "gaussian_fwhm_deg": float(gaussian),
        "lorentzian_fwhm_deg": float(lorentzian),
        "d_gaussian_d_instrument": d_gaussian,
        "d_lorentzian_d_instrument": d_lorentzian,
        "d_gaussian_d_position": float(d_gaussian_d_variance * d_variance),
        "d_lorentzian_d_position": float(
            INSTRUMENT["x_deg"] * d_secant + INSTRUMENT["y_deg"] * d_tangent
        ),
    }


def evaluate_profile(
    profile_module: Any, position_deg: float, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return profile and public-unit position/Gaussian/Lorentzian derivatives."""

    widths = widths_and_derivatives(position_deg)
    native_profile, _integral = profile_module.getPsVoigt(
        position_deg,
        10_000.0 * widths["gaussian_variance_deg2"],
        100.0 * widths["lorentzian_fwhm_deg"],
        x,
    )
    _profile, native_position, native_sigma2, native_gamma = profile_module.getdPsVoigt(
        position_deg,
        10_000.0 * widths["gaussian_variance_deg2"],
        100.0 * widths["lorentzian_fwhm_deg"],
        x,
    )
    profile = 100.0 * np.asarray(native_profile, dtype=np.float64)
    translation_position = -100.0 * np.asarray(native_position, dtype=np.float64)
    d_sigma2_d_gaussian_fwhm = 20_000.0 * widths["gaussian_fwhm_deg"] / GAUSSIAN_FWHM_PER_SIGMA**2
    d_gaussian = 100.0 * np.asarray(native_sigma2, dtype=np.float64) * d_sigma2_d_gaussian_fwhm
    d_lorentzian = 10_000.0 * np.asarray(native_gamma, dtype=np.float64)
    full_position = (
        translation_position
        + d_gaussian * widths["d_gaussian_d_position"]
        + d_lorentzian * widths["d_lorentzian_d_position"]
    )
    global_derivatives = (
        d_gaussian[:, None] * widths["d_gaussian_d_instrument"][None, :]
        + d_lorentzian[:, None] * widths["d_lorentzian_d_instrument"][None, :]
    ).T
    return profile, full_position, d_gaussian, global_derivatives


def isolated_case(
    profile_module: Any, position_deg: float, angular_regime: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_id = f"isolated_{angular_regime}"
    widths = widths_and_derivatives(position_deg)
    half_span = max(0.75, 30.0 * widths["gaussian_fwhm_deg"])
    x = np.linspace(position_deg - half_span, position_deg + half_span, 5_001)
    profile, d_position, _d_gaussian, d_global = evaluate_profile(profile_module, position_deg, x)
    arrays = {
        f"{case_id}__x_deg": x,
        f"{case_id}__profile_per_deg": profile,
        f"{case_id}__d_position": d_position,
        f"{case_id}__d_instrument": d_global,
    }
    case = {
        "id": case_id,
        "case_kind": "cw_isolated_reflection",
        "arrays": {
            "x": f"{case_id}__x_deg",
            "profile": f"{case_id}__profile_per_deg",
            "d_position": f"{case_id}__d_position",
            "d_instrument": f"{case_id}__d_instrument",
        },
        "parameters": {
            "instrument": INSTRUMENT,
            "position_deg": position_deg,
            "integrated_intensity": 1.0,
            "angular_regime": angular_regime,
            "gaussian_variance_deg2": widths["gaussian_variance_deg2"],
            "gaussian_fwhm_deg": widths["gaussian_fwhm_deg"],
            "lorentzian_fwhm_deg": widths["lorentzian_fwhm_deg"],
        },
        "sampled_integral": float(np.trapezoid(profile, x)),
    }
    return case, arrays


def overlapping_case(profile_module: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_id = "overlapping_cw_reflections"
    x = np.linspace(48.0, 52.0, 8_001)
    positions = np.array([49.80, 50.15, 50.32], dtype=np.float64)
    intensities = np.array([125.0, 83.0, 47.0], dtype=np.float64)
    ycalc = np.zeros_like(x)
    local = np.zeros((positions.size, 2, x.size), dtype=np.float64)
    global_jacobian = np.zeros((5, x.size), dtype=np.float64)
    for reflection, (position, intensity) in enumerate(zip(positions, intensities, strict=True)):
        profile, d_position, _d_gaussian, d_global = evaluate_profile(
            profile_module, float(position), x
        )
        ycalc += intensity * profile
        local[reflection, 0] = profile
        local[reflection, 1] = intensity * d_position
        global_jacobian += intensity * d_global
    arrays = {
        f"{case_id}__x_deg": x,
        f"{case_id}__ycalc": ycalc,
        f"{case_id}__local_jacobian": local,
        f"{case_id}__global_jacobian": global_jacobian,
    }
    case = {
        "id": case_id,
        "case_kind": "cw_overlapping_reflections",
        "arrays": {
            "x": f"{case_id}__x_deg",
            "ycalc": f"{case_id}__ycalc",
            "local_jacobian": f"{case_id}__local_jacobian",
            "global_jacobian": f"{case_id}__global_jacobian",
        },
        "parameters": {
            "instrument": INSTRUMENT,
            "positions_deg": positions.tolist(),
            "integrated_intensities": intensities.tolist(),
        },
        "sampled_integral": float(np.trapezoid(ycalc, x)),
    }
    return case, arrays


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    if "__x_deg" in name:
        unit, description = "degree_2theta", "Profile sampling coordinates"
    elif "profile_per_deg" in name:
        unit, description = "inverse_degree", "Normalized GSAS-II profile"
    elif "d_position" in name:
        unit, description = "inverse_degree_squared", "Full reflection-position derivative"
    elif "local_jacobian" in name:
        unit, description = "mixed_by_column", "Intensity and position derivative rows"
    elif "global_jacobian" in name or "d_instrument" in name:
        unit, description = "mixed_by_row", "U/V/W/X/Y derivative rows"
    else:
        unit, description = "intensity_per_degree", "Calculated profile"
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
    case_specs = [(15.0, "low"), (70.0, "middle"), (145.0, "high")]
    for position, regime in case_specs:
        case, case_arrays = isolated_case(profile_module, position, regime)
        cases.append(case)
        arrays.update(case_arrays)
    case, case_arrays = overlapping_case(profile_module)
    cases.append(case)
    arrays.update(case_arrays)
    np.savez_compressed(archive_path, **arrays)

    canonical_input = json.dumps(
        {"instrument": INSTRUMENT, "cases": case_specs, "overlap": cases[-1]["parameters"]},
        sort_keys=True,
    ).encode()
    script_path = Path(__file__).resolve()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_cw_instrument_profile_v1",
        "kind": "cw_instrument_profile",
        "provenance": {
            "gsasii_repository": pinned["repository"],
            "gsasii_revision": git_revision(gsas_root),
            "gsasii_tag": int(gsasii_path.GetVersionNumber()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "generator_sha256": sha256_file(script_path),
        },
        "source": {
            "api": ["GSASII.GSASIIpwd.getPsVoigt", "GSASII.GSASIIpwd.getdPsVoigt"],
            "private_probe": True,
            "adapter_version": ADAPTER_VERSION,
            "native_units": {
                "coordinate": "degree_2theta",
                "U_V_W": "centidegree_squared_variance",
                "X_Y": "centidegree_fwhm",
                "profile_density": "inverse_centidegree",
                "parameter_order": ["U", "V", "W", "X", "Y"],
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
        default=Path(__file__).resolve().parents[1] / "fixtures" / "cw_instrument_profile_v1",
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
