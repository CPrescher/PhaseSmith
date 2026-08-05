#!/usr/bin/env python3
"""Generate the pinned GSAS-II symmetric-profile oracle fixture.

This script is run with GSAS-II's Python interpreter. It deliberately imports
no Rietveld Engine module. Existing fixture files are protected unless
``--force`` is supplied explicitly.
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
    """Return the exact revision checked out in a Git repository."""

    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def import_gsasii(gsas_root: Path, binary_dir: Path | None) -> tuple[Any, Any]:
    """Import the pinned GSAS-II profile module with optional external binaries."""

    sys.path.insert(0, str(gsas_root))
    from GSASII import GSASIIpath

    if binary_dir is not None:
        sys.path.insert(0, str(binary_dir))
        GSASIIpath.binaryPath = str(binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False

    from GSASII import GSASIIpwd

    if not hasattr(GSASIIpwd, "pyd"):
        raise RuntimeError("GSAS-II pypowder binary is unavailable in this environment")
    return GSASIIpath, GSASIIpwd


def isolated_case(
    profile_module: Any,
    *,
    case_id: str,
    position_deg: float,
    gaussian_sigma_deg: float,
    lorentzian_fwhm_deg: float,
    half_span_deg: float,
    sample_count: int,
    shape_class: str,
    width_scale: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one GSAS-II simple pseudo-Voigt case."""

    x = np.linspace(
        position_deg - half_span_deg,
        position_deg + half_span_deg,
        sample_count,
        dtype=np.float64,
    )
    sigma_centidegrees_squared = (100.0 * gaussian_sigma_deg) ** 2
    gamma_centidegrees = 100.0 * lorentzian_fwhm_deg
    native_profile, native_integral = profile_module.getPsVoigt(
        position_deg,
        sigma_centidegrees_squared,
        gamma_centidegrees,
        x,
    )
    _derivative_profile, derivative_position, derivative_sigma2, derivative_gamma = (
        profile_module.getdPsVoigt(
            position_deg,
            sigma_centidegrees_squared,
            gamma_centidegrees,
            x,
        )
    )
    profile_per_degree = 100.0 * np.asarray(native_profile, dtype=np.float64)
    gaussian_fwhm_deg = GAUSSIAN_FWHM_PER_SIGMA * gaussian_sigma_deg
    d_sigma2_d_gaussian_fwhm = (
        20_000.0 * gaussian_fwhm_deg / GAUSSIAN_FWHM_PER_SIGMA**2
    )
    d_position = -100.0 * np.asarray(derivative_position, dtype=np.float64)
    d_gaussian_fwhm = (
        100.0
        * np.asarray(derivative_sigma2, dtype=np.float64)
        * d_sigma2_d_gaussian_fwhm
    )
    d_lorentzian_fwhm = 10_000.0 * np.asarray(derivative_gamma, dtype=np.float64)
    x_key = f"{case_id}__x_deg"
    profile_key = f"{case_id}__profile_per_deg"
    position_key = f"{case_id}__d_position"
    gaussian_key = f"{case_id}__d_gaussian_fwhm"
    lorentzian_key = f"{case_id}__d_lorentzian_fwhm"
    case = {
        "id": case_id,
        "case_kind": "isolated_peak",
        "arrays": {
            "x": x_key,
            "profile": profile_key,
            "d_position": position_key,
            "d_gaussian_fwhm": gaussian_key,
            "d_lorentzian_fwhm": lorentzian_key,
        },
        "parameters": {
            "position_deg": position_deg,
            "gaussian_sigma_deg": gaussian_sigma_deg,
            "gaussian_fwhm_deg": gaussian_fwhm_deg,
            "lorentzian_fwhm_deg": lorentzian_fwhm_deg,
            "shape_class": shape_class,
            "width_scale": width_scale,
        },
        "gsasii_reported_integral": float(native_integral),
        "sampled_integral_per_degree": float(np.trapezoid(profile_per_degree, x)),
    }
    return case, {
        x_key: x,
        profile_key: profile_per_degree,
        position_key: d_position,
        gaussian_key: d_gaussian_fwhm,
        lorentzian_key: d_lorentzian_fwhm,
    }


def overlapping_case(profile_module: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate two overlapping GSAS-II profiles and their summed pattern."""

    case_id = "overlapping_mixed"
    x = np.linspace(40.0, 41.0, 5_001, dtype=np.float64)
    peaks = [
        {
            "position_deg": 40.43,
            "intensity": 125.0,
            "gaussian_sigma_deg": 0.021,
            "lorentzian_fwhm_deg": 0.017,
        },
        {
            "position_deg": 40.49,
            "intensity": 83.0,
            "gaussian_sigma_deg": 0.028,
            "lorentzian_fwhm_deg": 0.034,
        },
    ]
    ycalc = np.zeros_like(x)
    for peak in peaks:
        native_profile, _integral = profile_module.getPsVoigt(
            peak["position_deg"],
            (100.0 * peak["gaussian_sigma_deg"]) ** 2,
            100.0 * peak["lorentzian_fwhm_deg"],
            x,
        )
        ycalc += peak["intensity"] * 100.0 * np.asarray(native_profile, dtype=np.float64)

    x_key = f"{case_id}__x_deg"
    ycalc_key = f"{case_id}__ycalc"
    case = {
        "id": case_id,
        "case_kind": "overlapping_peaks",
        "arrays": {"x": x_key, "ycalc": ycalc_key},
        "parameters": {"peaks": peaks},
        "sampled_integral": float(np.trapezoid(ycalc, x)),
    }
    return case, {x_key: x, ycalc_key: ycalc}


def array_descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    """Build manifest metadata for an archive member."""

    if name.endswith("__x_deg"):
        unit = "degree_2theta"
        description = "Profile sampling coordinates"
    elif name.endswith("__profile_per_deg"):
        unit = "inverse_degree"
        description = "GSAS-II normalized profile converted from inverse centidegrees"
    elif name.endswith("__d_position"):
        unit = "inverse_degree_squared"
        description = "Profile derivative with respect to peak position in degrees"
    elif name.endswith("__d_gaussian_fwhm"):
        unit = "inverse_degree_squared"
        description = "Profile derivative with respect to Gaussian FWHM in degrees"
    elif name.endswith("__d_lorentzian_fwhm"):
        unit = "inverse_degree_squared"
        description = "Profile derivative with respect to Lorentzian FWHM in degrees"
    else:
        unit = "intensity_per_degree"
        description = "Summed calculated profile"
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
    """Generate archive and manifest after enforcing overwrite protection."""

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    manifest_path = output / MANIFEST_NAME
    existing = [path for path in (archive_path, manifest_path) if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite {joined}; pass --force explicitly")

    case_specs = [
        {
            "case_id": "gaussian_dominant_narrow",
            "position_deg": 20.0,
            "gaussian_sigma_deg": 0.006,
            "lorentzian_fwhm_deg": 0.0003,
            "half_span_deg": 0.25,
            "sample_count": 3_001,
            "shape_class": "gaussian_dominant",
            "width_scale": "narrow",
        },
        {
            "case_id": "gaussian_dominant_broad",
            "position_deg": 25.0,
            "gaussian_sigma_deg": 0.06,
            "lorentzian_fwhm_deg": 0.002,
            "half_span_deg": 2.0,
            "sample_count": 4_001,
            "shape_class": "gaussian_dominant",
            "width_scale": "broad",
        },
        {
            "case_id": "mixed_narrow",
            "position_deg": 32.1,
            "gaussian_sigma_deg": 0.008,
            "lorentzian_fwhm_deg": 0.012,
            "half_span_deg": 0.4,
            "sample_count": 3_001,
            "shape_class": "mixed",
            "width_scale": "narrow",
        },
        {
            "case_id": "mixed_broad",
            "position_deg": 45.2,
            "gaussian_sigma_deg": 0.07,
            "lorentzian_fwhm_deg": 0.12,
            "half_span_deg": 3.0,
            "sample_count": 5_001,
            "shape_class": "mixed",
            "width_scale": "broad",
        },
        {
            "case_id": "lorentzian_dominant_narrow",
            "position_deg": 75.0,
            "gaussian_sigma_deg": 0.001,
            "lorentzian_fwhm_deg": 0.018,
            "half_span_deg": 0.5,
            "sample_count": 3_001,
            "shape_class": "lorentzian_dominant",
            "width_scale": "narrow",
        },
        {
            "case_id": "lorentzian_dominant_broad",
            "position_deg": 82.0,
            "gaussian_sigma_deg": 0.01,
            "lorentzian_fwhm_deg": 0.15,
            "half_span_deg": 4.0,
            "sample_count": 5_001,
            "shape_class": "lorentzian_dominant",
            "width_scale": "broad",
        },
    ]
    cases: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for specification in case_specs:
        case, case_arrays = isolated_case(profile_module, **specification)
        cases.append(case)
        arrays.update(case_arrays)
    case, case_arrays = overlapping_case(profile_module)
    cases.append(case)
    arrays.update(case_arrays)

    np.savez_compressed(archive_path, **arrays)
    canonical_input = json.dumps(
        [*case_specs, cases[-1]["parameters"]], sort_keys=True
    ).encode()
    script_path = Path(__file__).resolve()
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_symmetric_pseudo_voigt_v1",
        "kind": "symmetric_profile",
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
                "gaussian_width": "centidegree_squared_variance",
                "lorentzian_width": "centidegree_fwhm",
                "profile_density": "inverse_centidegree",
                "position_derivative_sign": "opposite_peak_position_derivative",
            },
        },
        "input": {
            "kind": "parameter_cases",
            "sha256": sha256_bytes(canonical_input),
        },
        "archive": {
            "file": ARCHIVE_NAME,
            "sha256": sha256_file(archive_path),
        },
        "arrays": {name: array_descriptor(name, array) for name, array in arrays.items()},
        "cases": cases,
    }
    expected_revision = pinned["revision"]
    if manifest["provenance"]["gsasii_revision"] != expected_revision:
        raise RuntimeError(
            f"refusing fixture from {manifest['provenance']['gsasii_revision']}; "
            f"expected {expected_revision}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument(
        "--binary-dir",
        type=Path,
        help="directory containing pypowder for the selected Python/NumPy ABI",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures" / "symmetric_pseudo_voigt_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate the pin, import GSAS-II, and write the fixture."""

    arguments = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    pinned = json.loads((repository_root / "oracle" / "PINNED_GSASII.json").read_text())
    gsas_root = arguments.gsas_root.resolve()
    revision = git_revision(gsas_root)
    if revision != pinned["revision"]:
        raise RuntimeError(f"GSAS-II revision {revision} does not match pin {pinned['revision']}")
    gsasii_path, profile_module = import_gsasii(
        gsas_root,
        arguments.binary_dir.resolve() if arguments.binary_dir else None,
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
