"""Reproducible fixed-spectrum Pawley oracle and measured ceria comparison.

The ceria recipe is an exploratory measured regression, not a held-out promotion
claim. Component weights are fixed effective detected areas; no LP is reapplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from phasesmith import ConstantWavelengthInstrument, PowderPattern, _core
from phasesmith._numpy_compat import trapezoid
from phasesmith.instrument import FcjGeometry
from phasesmith.oracle import load_fixture
from phasesmith.radiation import WavelengthComponents
from phasesmith.refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    build_parameter_set,
    calculate,
    refine,
)
from phasesmith.validation.ceria_transferability import (
    REFERENCE_CELL_ANGSTROM,
    _ceria_cif,
    _prepare_pattern,
)
from phasesmith.validation.datasets import verify_validation_dataset


def oracle_comparison(root):
    fixture = load_fixture(root)
    rows = []
    for case in fixture.cases:
        p = case["parameters"]
        x, y = (fixture.arrays[case["arrays"][name]] for name in ("x", "ycalc"))
        r = PawleyInput(
            PowderPattern(x, observed_y=y),
            ConstantWavelengthInstrument(**dict(p["instrument"])),
            (PawleyPhase("p", ("family",), [p["base_position_deg"]], [0.5]),),
            axial_geometry=FcjGeometry(**dict(p["public_equal_height_mapping"])),
            fixed_spectrum=WavelengthComponents(
                p["wavelengths_angstrom"], p["relative_intensities"]
            ),
        )
        o = PawleyOptions(support_fwhm=100)
        fit = refine(r, o)
        unit = replace(r, phases=(replace(r.phases[0], intensities=np.ones(1)),), parameters=None)
        calculated = calculate(unit, o).calculated_y
        error = float(np.max(np.abs(calculated - y)) / np.max(np.abs(y)))
        # Existing pinned FCJ comparison contract, retained unchanged.
        assert error < 2.4e-2
        assert fit.termination_reason == "converged"
        area = float(trapezoid(calculated, x))
        assert abs(area / case["sampled_moments"]["integral"] - 1) < 4e-6
        rows.append(
            dict(
                case=case["id"],
                normalized_maximum_error=error,
                fitted_family_area=float(fit.intensities[0]),
                sampled_integral=area,
                rwp=fit.calculation.rwp,
                termination=fit.termination_reason,
            )
        )
    return rows


def measured_comparison(root):
    verify_validation_dataset("iucr-ceria-size-strain-round-robin", root)
    p = _prepare_pattern(
        root, ("langfsh1.xy", "langfsh2.xy", "langfsh3.xy"), smooth_width_deg=2.0
    ).pattern
    rows = []
    for ratio in (0.0, 0.016):
        spectrum = WavelengthComponents([1.5406, 1.5444], [1, ratio]) if ratio else None
        phase = PawleyPhase.from_cif(
            "ceria",
            _ceria_cif(REFERENCE_CELL_ANGSTROM),
            wavelength_angstrom=1.5406,
            two_theta_range=(float(p.x[0]) - 1, float(p.x[-1]) + 1),
            fixed_spectrum=spectrum,
            initial_intensity=100,
        )
        r = PawleyInput(
            p,
            ConstantWavelengthInstrument(1.5406, 0, 0, 0.002, 0.002, 0),
            (phase,),
            fixed_spectrum=spectrum,
        )
        r = replace(
            r,
            parameters=build_parameter_set(
                r, profile_parameters=("w_deg2", "x_deg", "y_deg"), lattice=True
            ),
        )
        o = PawleyOptions(solver="matrix_free", support_fwhm=10000)
        times = []
        previous = None
        for _ in range(2):
            start = perf_counter()
            fit = refine(r, o, max_iterations=100)
            times.append(perf_counter() - start)
            assert fit.termination_reason == "converged"
            if previous is not None:
                np.testing.assert_array_equal(fit.history, previous.history)
                np.testing.assert_array_equal(
                    fit.calculation.calculated_y, previous.calculation.calculated_y
                )
            previous = fit
        rows.append(
            dict(
                secondary_to_reference_area=ratio,
                samples=len(p.x),
                families=len(phase.reflection_ids),
                rwp=fit.calculation.rwp,
                termination=fit.termination_reason,
                history=fit.history.tolist(),
                times_seconds=times,
                parameters={str(s.key): s.value for s in fit.parameters.specs},
                profile_sha256=hashlib.sha256(fit.calculation.calculated_y.tobytes()).hexdigest(),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oracle-fixture",
        type=Path,
        default=Path("oracle/fixtures/wavelength_components_v1"),
        help="Pinned fixture directory (relative to the working directory, or an absolute path)",
    )
    args = parser.parse_args()
    if not args.oracle_fixture.is_dir():
        parser.error(
            f"oracle fixture not found: {args.oracle_fixture}; supply --oracle-fixture PATH"
        )
    record = dict(
        platform=platform.platform(),
        native_binary_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle=oracle_comparison(args.oracle_fixture),
        measured=measured_comparison(args.data_root),
        conventions=(
            "Ceria is an exploratory measured regression, not a holdout. Source 1.6% "
            "K-alpha2 is interpreted as a fixed detected-area ratio. No LP/scattering or "
            "phase-scale factor is applied. Original axes/counts retained. Existing fixed "
            "background preparation; W/X/Y and cubic cell fitted jointly. Support 10000 "
            "avoids moving cutoffs inside these observations. Support-100 probe stagnated "
            "at Rwp 0.209966; retained as a known diagnostic, not relabelled convergence. "
        ),
    )
    with args.output.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
