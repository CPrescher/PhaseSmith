"""Offline high-precision FCJ equation audit; requires optional mpmath.

Direct angular integration of Finger, Cox & Jephcoat (1994), equations
1, 4-8, https://doi.org/10.1107/S0021889894004218. This deliberately does
not use the height substitution, Legendre nodes, or production profile code.
Only the comparison runner imports the native evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath
import numpy as np


def angular_profile(x, position, gaussian, lorentzian, sample, detector, *, digits=50):
    """Normalized per-degree profile for 0 < position < 90 and positive heights.

    Widths and observations are in degrees; sample and detector are half-height/L.
    This diagnostic intentionally covers unclipped low-angle geometry only.
    """
    mp = mpmath.mp.clone()
    mp.dps = digits
    b, hg, hl, s, h = map(mp.mpf, map(str, (position, gaussian, lorentzian, sample, detector)))
    if not (0 < b < 90 and hg > 0 and hl >= 0 and s > 0 and h > 0):
        raise ValueError("angular audit requires low-angle geometry and positive Gaussian width")
    b *= mp.pi / 180
    if mp.cos(b) * mp.sqrt(1 + (s + h) ** 2) >= 1:
        raise ValueError("angular audit excludes clipped geometry")
    lower = mp.acos(mp.cos(b) * mp.sqrt(1 + (s + h) ** 2))
    kink = b if s == h else mp.acos(mp.cos(b) * mp.sqrt(1 + (s - h) ** 2))
    bounds = sorted(set([lower, kink, b]))

    # FCJ eqs. (1), (6), (7). Constant 1/(2H) cancels in eq. (8).
    # The sine identity avoids subtracting two nearly equal cosines squared.
    def density(a):
        if a >= b:
            return mp.mpf(0)  # Ignore a rounded endpoint, never an interior sample.
        z = mp.sqrt(mp.sin(a + b) * mp.sin(b - a)) / mp.cos(b)
        if z == 0:
            return mp.mpf(0)  # Endpoint itself has measure zero; limit is singular.
        overlap = min(2 * min(s, h), s + h - z)
        return overlap / (z * mp.cos(a))

    # TCH intrinsic pseudo-Voigt, independently evaluated in arbitrary precision.
    coeff = list(map(mp.mpf, ["1", "2.69269", "2.42843", "4.47163", "0.07842", "1"]))
    width = sum(c * hg ** (5 - i) * hl**i for i, c in enumerate(coeff)) ** (mp.mpf(1) / 5)
    q = hl / width
    eta = mp.mpf("1.36603") * q - mp.mpf("0.47719") * q**2 + mp.mpf("0.11116") * q**3

    def intrinsic(delta):
        scaled = delta / width
        normal = 2 * mp.sqrt(mp.log(2) / mp.pi) * mp.exp(-4 * mp.log(2) * scaled**2)
        cauchy = 2 / (mp.pi * (1 + 4 * scaled**2))
        return ((1 - eta) * normal + eta * cauchy) / width

    norm = mp.quad(density, bounds, method="tanh-sinh")
    values = []
    for observation in x:
        obs = mp.mpf(str(observation))
        value = (
            mp.quad(
                lambda a, obs=obs: density(a) * intrinsic(obs - a * 180 / mp.pi),
                bounds,
                method="tanh-sinh",
            )
            / norm
        )
        values.append(float(value))
    return np.asarray(values)


def investigate(fixture_path, controls_path):
    from .. import _core
    from ..fcj import profile_fcj
    from ..instrument import FcjGeometry
    from ..oracle import load_fixture
    from .pawley_profile_diagnostic import sha

    fixture_path, controls_path = Path(fixture_path), Path(controls_path)
    fixture = load_fixture(fixture_path)
    manifest = json.loads((controls_path / "manifest.json").read_text())
    if manifest["original_manifest_sha256"] != sha(fixture_path / "manifest.json"):
        raise ValueError("controls refer to a different fixture")
    if manifest["archive_sha256"] != sha(controls_path / "profiles.npz"):
        raise ValueError("controls checksum mismatch")
    if manifest["gsasii_revision"] != fixture.manifest["provenance"]["gsasii_revision"]:
        raise ValueError("oracle revision mismatch")
    grid = fixture.arrays["fixed_cell_x_deg"]
    rows = fixture.arrays["fixed_cell_reflections"]
    results = []
    with np.load(controls_path / "profiles.npz", allow_pickle=False) as archive:
        for index in (0, 10, 30, 52):
            row = rows[index]
            hg, hl = np.sqrt(8 * np.log(2) * row[6]) / 100, row[7] / 100
            indices = np.unique(
                np.argmin(
                    abs(grid[:, None] - (row[5] + hg * np.array([-2, -1, -0.5, 0, 0.5, 1, 2]))),
                    axis=0,
                )
            )
            x = grid[indices]
            low = angular_profile(x, row[5], hg, hl, 0.001, 0.001, digits=35)
            high = angular_profile(x, row[5], hg, hl, 0.001, 0.001, digits=60)
            native = profile_fcj(x, row[5], hg, hl, FcjGeometry(0.001, 0.001)).value
            oracle = archive["axial"][index, indices]
            norm = np.linalg.norm(high)
            results.append(
                dict(
                    reflection=index,
                    position_deg=float(row[5]),
                    x_deg=x.tolist(),
                    angular_reference=high.tolist(),
                    native=native.tolist(),
                    oracle=oracle.tolist(),
                    precision_relative_l2=float(np.linalg.norm(low - high) / norm),
                    native_relative_l2=float(np.linalg.norm(native - high) / norm),
                    oracle_relative_l2=float(np.linalg.norm(oracle - high) / norm),
                )
            )
    checks = dict(
        precision_convergence=max(r["precision_relative_l2"] for r in results) < 1e-12,
        native_matches_published_integral=max(r["native_relative_l2"] for r in results) < 2e-10,
    )
    return dict(
        scope="Direct FCJ angular integral; seven samples at each of four fixture reflections",
        source_doi="10.1107/S0021889894004218",
        equations=[1, 4, 5, 6, 7, 8],
        precision_digits=[35, 60],
        mpmath_version=mpmath.__version__,
        runner_sha256=sha(Path(__file__)),
        native_binary_sha256=sha(Path(_core.__file__)),
        controls_manifest_sha256=sha(controls_path / "manifest.json"),
        gsasii_revision=manifest["gsasii_revision"],
        results=results,
        checks=checks,
        passed=all(checks.values()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = investigate(args.fixture, args.controls)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if not report["passed"]:
        raise SystemExit("angular-integral audit failed")


if __name__ == "__main__":
    main()
