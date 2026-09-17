"""Decompose the fixed Pawley oracle mismatch without changing production physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import _core, reference
from ..cw import cw_profile_parameters
from ..fcj import profile_fcj
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..oracle import load_fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def investigate(fixture_path, controls_path):
    fixture_path, controls_path = Path(fixture_path), Path(controls_path)
    fixture = load_fixture(fixture_path)
    manifest = json.loads((controls_path / "manifest.json").read_text())
    if manifest["original_manifest_sha256"] != sha(fixture_path / "manifest.json"):
        raise ValueError("controls refer to a different Pawley fixture")
    if manifest["archive_sha256"] != sha(controls_path / "profiles.npz"):
        raise ValueError("profile controls checksum mismatch")
    if manifest["gsasii_revision"] != fixture.manifest["provenance"]["gsasii_revision"]:
        raise ValueError("profile controls oracle revision mismatch")
    with np.load(controls_path / "profiles.npz", allow_pickle=False) as archive:
        controls = {key: archive[key] for key in archive.files}
    x = fixture.arrays["fixed_cell_x_deg"]
    rows = fixture.arrays["fixed_cell_reflections"]
    full = fixture.arrays["fixed_cell_ycalc"]
    observed = full - fixture.arrays["fixed_cell_background"]
    areas = 0.01 * rows[:, 8] * rows[:, 11]
    instrument = ConstantWavelengthInstrument(1.54056, 0.0002, -0.0001, 0.00012, 0.0015, 0.003)
    widths = cw_profile_parameters(rows[:, 5], instrument)
    native, symmetric, corrected, per_peak, quadrature = [], [], [], [], []
    for k, row in enumerate(rows):
        gaussian = np.sqrt(8 * np.log(2) * row[6]) / 100
        lorentzian = row[7] / 100
        p = profile_fcj(x, row[5], gaussian, lorentzian, FcjGeometry(0.001, 0.001))
        native.append(p.value)
        symmetric.append(profile_fcj(x, row[5], gaussian, lorentzian, FcjGeometry(0, 0)).value)
        # Diagnostic linearized translation/area fit; never a production correction.
        coefficients = np.linalg.lstsq(
            np.column_stack([p.value, p.d_position]), controls["axial"][k], rcond=None
        )[0]
        corrected.append(coefficients[0] * p.value + coefficients[1] * p.d_position)
        per_peak.append(
            dict(
                position_deg=float(row[5]),
                translation_deg=float(coefficients[1]),
                area_scale=float(coefficients[0]),
            )
        )
        if k in (0, 10, 30, 52):
            evaluations = [
                reference.profile_fcj(
                    x, row[5], gaussian, lorentzian, 0.001, 0.001, quadrature_order=order
                ).value
                for order in (8, 32, 128)
            ]
            norm = np.linalg.norm(evaluations[-1])
            quadrature.append(
                dict(
                    reflection=k,
                    native_vs_order128=float(np.linalg.norm(p.value - evaluations[-1]) / norm),
                    order32_vs_order128=float(
                        np.linalg.norm(evaluations[1] - evaluations[-1]) / norm
                    ),
                )
            )
    native, symmetric, corrected = map(np.asarray, (native, symmetric, corrected))
    support = controls["axial_support"]
    mask = (x[None, :] >= rows[:, 5, None] - support[:, 0, None]) & (
        x[None, :] < rows[:, 5, None] + support[:, 1, None]
    )
    external_full = areas @ controls["axial"]
    external_cut = areas @ (controls["axial"] * mask)
    native_full = areas @ native
    denominator = np.linalg.norm(full)

    def rel(delta):
        return float(np.linalg.norm(delta) / denominator)

    oracle_area_fit = np.linalg.lstsq((controls["axial"] * mask).T, observed, rcond=None)[0]
    native_area_fit = np.linalg.lstsq(native.T, observed, rcond=None)[0]
    metrics = dict(
        oracle_basis_area_max_relative_error=float(np.max(abs(oracle_area_fit / areas - 1))),
        native_basis_area_max_relative_error=float(np.max(abs(native_area_fit / areas - 1))),
        original_full_pattern_relative_l2=rel(native_full - observed),
        axial_kernel_component_relative_l2=rel(native_full - external_full),
        cutoff_component_relative_l2=rel(external_full - external_cut),
        histogram_reconstruction_relative_l2=rel(external_cut - observed),
        diagnostic_translation_area_and_cutoff_relative_l2=rel(
            areas @ (corrected * mask) - observed
        ),
        peak_signal_original_relative_l2=float(
            np.linalg.norm(native_full - observed) / np.linalg.norm(observed)
        ),
        symmetric_kernel_relative_l2=float(
            np.linalg.norm(areas @ (symmetric - controls["symmetric"]))
            / np.linalg.norm(areas @ controls["symmetric"])
        ),
        variance_conversion_max_absolute_error=float(
            np.max(abs(widths.gaussian_variance_deg2 - rows[:, 6] / 10000))
        ),
        lorentzian_conversion_max_absolute_error=float(
            np.max(abs(widths.lorentzian_fwhm_deg - rows[:, 7] / 100))
        ),
    )
    row = rows[0]
    gaussian, lorentzian = np.sqrt(8 * np.log(2) * row[6]) / 100, row[7] / 100
    grid_checks = []
    for count in (401, 4001, 40001):
        grid = controls[f"grid_{count}"]
        p = profile_fcj(grid, row[5], gaussian, lorentzian, FcjGeometry(0.001, 0.001)).value
        oracle = controls[f"profile_{count}"]
        grid_checks.append(
            dict(
                samples=count,
                relative_l2=float(np.linalg.norm(p - oracle) / np.linalg.norm(oracle)),
            )
        )
    sweep = []
    for case in manifest["sweep"]:
        sh, width = case["axial_sum"], case["width_scale"]
        grid = controls["sweep_grid"]
        p = reference.profile_fcj(
            grid, row[5], gaussian * width, lorentzian * width, sh / 2, sh / 2, quadrature_order=128
        ).value
        oracle = controls[case["key"]]
        sweep.append(
            dict(**case, relative_l2=float(np.linalg.norm(p - oracle) / np.linalg.norm(oracle)))
        )
    checks = dict(
        same_basis_area_recovery=metrics["oracle_basis_area_max_relative_error"] < 2e-7,
        width_translation=metrics["variance_conversion_max_absolute_error"] < 1e-15
        and metrics["lorentzian_conversion_max_absolute_error"] < 2e-14,
        independent_quadrature=max(q["native_vs_order128"] for q in quadrature) < 2e-11,
        oracle_histogram_reconstruction=metrics["histogram_reconstruction_relative_l2"] < 1e-8,
        diagnosed_components=metrics["diagnostic_translation_area_and_cutoff_relative_l2"] < 1e-6,
    )
    return dict(
        scope=(
            "Fixed-cell profile diagnosis; posthoc translation/area controls do not alter "
            "production or acceptance gates"
        ),
        native_binary_sha256=sha(Path(_core.__file__)),
        runner_sha256=sha(Path(__file__)),
        controls_manifest_sha256=sha(controls_path / "manifest.json"),
        gsasii_revision=manifest["gsasii_revision"],
        metrics=metrics,
        per_peak=per_peak,
        independent_quadrature=quadrature,
        grid_checks=grid_checks,
        axial_width_sweep=sweep,
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
        raise SystemExit("profile diagnosis control failed")


if __name__ == "__main__":
    main()
