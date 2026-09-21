"""Fixed-state and live-optimizer comparison using a pinned Pawley fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import _core
from ..crystallography import UnitCell
from ..cw import cw_profile_parameters
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..oracle import load_fixture
from ..pattern import PowderPattern
from ..refinement.lattice import (
    CwLatticeReflectionDomain,
    LatticeParameterBounds,
    LatticeParameterization,
)
from ..refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    build_parameter_set,
    calculate,
    refine,
)
from ..symmetry import SpaceGroup

CELL_NAMES = ("a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg")


def _relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _groups(positions, widths):
    order = np.argsort(positions, kind="stable")
    groups = [[int(order[0])]]
    for index in order[1:]:
        previous = groups[-1][-1]
        if positions[index] - positions[previous] <= 3 * max(widths[index], widths[previous]):
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return groups


def compare_fixture(path: str | Path) -> dict:
    """Compare profiles, cells, isolated areas and observable overlap-group sums.

    Agreement tolerances explicitly allow the documented distinct FCJ/profile
    implementations. A separate strict equivalence check remains visible.
    No GSAS-II import or mutable oracle state is used by this comparison.
    """
    fixture = load_fixture(path)
    if fixture.manifest["fixture_id"] != "gsasii_pawley_optimizer_v1":
        raise ValueError("expected the pinned Pawley optimizer fixture")
    results = []
    for case in fixture.cases:
        arrays = {key: fixture.arrays[name] for key, name in case["arrays"].items()}
        if not np.all(arrays["weight"] == 1.0):
            raise ValueError("comparison requires the declared unit-weight oracle")
        refs = arrays["reflections"]
        starting = arrays["initial_reflections"]
        # Correction already includes multiplicity and phase/histogram scale.
        areas = 0.01 * refs[:, 8] * refs[:, 11]
        initial = (
            0.01
            * case["parameters"]["initial_f_squared_fraction"]
            * starting[:, 9]
            * starting[:, 11]
        )
        settings = case["parameters"]["instrument"]
        instrument = ConstantWavelengthInstrument(
            settings["wavelength_angstrom"],
            settings["u_gsas_centideg2"] / 10000,
            settings["v_gsas_centideg2"] / 10000,
            settings["w_gsas_centideg2"] / 10000,
            settings["x_gsas_centideg"] / 100,
            settings["y_gsas_centideg"] / 100,
        )
        domain = None
        if case["id"] == "refined_cell":
            group = SpaceGroup.p1()
            par = LatticeParameterization(group, UnitCell(*case["parameters"]["starting_cell"]))
            domain = CwLatticeReflectionDomain(
                group,
                par,
                LatticeParameterBounds.around(par, relative_length=0.001, angle_delta_deg=0.01),
                instrument.wavelength_angstrom,
                *case["parameters"]["range_deg"],
                1.0,
            )
        phase = PawleyPhase(
            "oracle",
            tuple(str(i) for i in range(len(areas))),
            refs[:, 5],
            initial,
            refs[:, :3].astype(np.int64),
            domain,
        )
        axial = case["parameters"]["pinned_minimum_sh_over_l"] / 2
        request = PawleyInput(
            PowderPattern(
                arrays["x_deg"], observed_y=arrays["observed_y"], background=arrays["background"]
            ),
            instrument,
            (phase,),
            axial_geometry=FcjGeometry(axial, axial),
            signed_intensities=True,
        )
        if domain is not None:
            request = replace(request, parameters=build_parameter_set(request, lattice=True))
        options = PawleyOptions(solver="matrix_free", support_fwhm=10000.0)
        fixed_values = {
            spec.key: areas[int(spec.key.name)]
            for spec in request.parameters.specs
            if spec.key.module == "pawley_intensity"
        }
        if domain is not None:
            fixed_values.update(
                {
                    spec.key: arrays["cell"][CELL_NAMES.index(spec.key.name)]
                    for spec in request.parameters.specs
                    if spec.key.module == "pawley_lattice"
                }
            )
        fixed = calculate(
            replace(request, parameters=request.parameters.replace_values(fixed_values)), options
        )
        fit = refine(request, options, max_seconds=120)
        widths = cw_profile_parameters(refs[:, 5], instrument).total_fwhm_deg
        groups = _groups(refs[:, 5], widths)
        isolated = [g[0] for g in groups if len(g) == 1]
        isolated_error = float(np.max(np.abs(fit.intensities[isolated] / areas[isolated] - 1)))
        sum_error = float(
            max(abs(np.sum(fit.intensities[g]) / np.sum(areas[g]) - 1) for g in groups)
        )
        fitted_cell = (
            np.array(
                [
                    fit.parameters.spec(
                        next(
                            s.key
                            for s in fit.parameters.specs
                            if s.key.module == "pawley_lattice" and s.key.name == name
                        )
                    ).value
                    for name in CELL_NAMES
                ]
            )
            if domain is not None
            else np.array(case["parameters"]["cell"])
        )
        lengths_error = float(np.max(np.abs(fitted_cell[:3] / arrays["cell"][:3] - 1)))
        angles_error = float(np.max(np.abs(fitted_cell[3:] - arrays["cell"][3:])))
        fixed_error = _relative(fixed.calculated_y, arrays["ycalc"])
        oracle_error = _relative(arrays["ycalc"], arrays["observed_y"])
        checks = {
            "oracle_recovers_observations": oracle_error < 2e-5,
            "native_converged": fit.termination_reason == "converged",
            "fixed_profile_agreement": fixed_error < 1e-3,
            "fitted_profile_agreement": fit.calculation.rwp < 1e-3,
            "isolated_area_agreement": isolated_error < 2e-4,
            "overlap_sum_agreement": sum_error < 1e-3,
            "cell_length_agreement": lengths_error < 3e-6,
            "cell_angle_agreement": angles_error < 5e-5,
        }
        results.append(
            dict(
                case=case["id"],
                reflections=len(areas),
                samples=len(arrays["x_deg"]),
                oracle_relative_l2=oracle_error,
                fixed_profile_relative_l2=fixed_error,
                fitted_profile_relative_l2=fit.calculation.rwp,
                termination=fit.termination_reason,
                isolated_reflections=len(isolated),
                overlap_groups=[g for g in groups if len(g) > 1],
                isolated_area_max_relative_error=isolated_error,
                group_sum_max_relative_error=sum_error,
                all_individual_area_max_relative_error=float(
                    np.max(np.abs(fit.intensities / areas - 1))
                ),
                native_cell=fitted_cell.tolist(),
                oracle_cell=arrays["cell"].tolist(),
                cell_length_max_relative_error=lengths_error,
                cell_angle_max_absolute_error_deg=angles_error,
                strict_fixed_profile_equivalence=fixed_error < 1e-5,
                checks=checks,
                passed=all(checks.values()),
            )
        )
    return dict(
        scope=(
            "Pawley optimizer comparison with declared profile-model differences; "
            "not exact GSAS-II equivalence"
        ),
        area_convention=(
            "0.01 * F_obs_squared * intensity_correction; no second multiplicity factor"
        ),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_support_fwhm=10000.0,
        signed_areas=True,
        unit_weights=True,
        overlap_group_rule="connected adjacent gaps at most three times the larger TCH FWHM",
        fixture_manifest_sha256=hashlib.sha256(
            (Path(path) / "manifest.json").read_bytes()
        ).hexdigest(),
        native_binary_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        gsasii_revision=fixture.manifest["provenance"]["gsasii_revision"],
        results=results,
        passed=all(r["passed"] for r in results),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_fixture(args.fixture)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not report["passed"]:
        raise SystemExit("Pawley external-comparison agreement gate failed")


if __name__ == "__main__":
    main()
