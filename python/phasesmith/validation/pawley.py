"""Checksum-pinned CW Pawley smoke gates; no GSAS-II dependency or implicit downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .. import _core
from ..background import SmoothBrucknerBackground, smooth_bruckner
from ..crystallography import UnitCell
from ..instrument import ConstantWavelengthInstrument
from ..io import read_cif, read_powder_data, space_group_by_number
from ..pattern import PowderPattern
from ..refinement.background import ChebyshevBackground
from ..refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    build_parameter_set,
    calculate,
    refine,
)
from ..symmetry import CwTwoThetaRange, PreparedReflectionGenerator
from .datasets import verify_validation_dataset
from .real_data import _SUCROSE_CIF


def run(dataset_id, directory, gate):
    """Run the frozen manifest contract twice and retain failed gates explicitly."""
    paths = verify_validation_dataset(dataset_id, directory)
    directory = Path(directory)
    if dataset_id == "aps-sucrose-11bmb":
        data = read_powder_data(directory / "11bmb_8716.fxye", format="gsas_fxye")
        structure = read_cif(_SUCROSE_CIF).structure
        cell, group = structure.cell, structure.space_group
        instrument = ConstantWavelengthInstrument(
            0.413259, 1.163e-4, -0.126e-4, 0.063e-4, 0.173e-2, 0.0
        )
        smoothing, support = 0.1, 30.0
    else:
        data = read_powder_data(directory / "ECH0034258_LaB6.xyd", format="columns")
        cell = UnitCell(4.156826, 4.156826, 4.156826, 90.0, 90.0, 90.0)
        group = space_group_by_number(221).space_group
        instrument = ConstantWavelengthInstrument(2.047, 0.014, -0.253, 0.516, 0.05, 0.0)
        smoothing, support = 0.8, 8.0
    lo, hi = gate["range_deg"]
    selected = (data.x >= lo) & (data.x <= hi)
    x, y, sigma = data.x[selected], data.observed_y[selected], data.uncertainty[selected]
    baseline = (
        smooth_bruckner(y, smooth_points=20, iterations=50)
        if dataset_id == "ansto-echidna-lab6-cw-neutron"
        else SmoothBrucknerBackground(
            smooth_width=smoothing, iterations=50, chebyshev_order=None
        ).estimate(x, y)
    )
    pattern = PowderPattern(x, observed_y=y, uncertainty=sigma, background=baseline)
    families = PreparedReflectionGenerator(group, merge_friedel=True).generate(
        cell, CwTwoThetaRange(float(x[0]), float(x[-1]), instrument.wavelength_angstrom)
    )
    positions = 2 * np.degrees(
        np.arcsin(instrument.wavelength_angstrom / (2 * families.d_spacing_angstrom))
    )
    phase = PawleyPhase(
        dataset_id, families.reflection_ids, positions, np.ones(len(positions)), families.hkl
    )
    request = PawleyInput(
        pattern,
        instrument,
        (phase,),
        ChebyshevBackground("residual", (0.0,), (float(x[0]), float(x[-1]))),
    )
    request = replace(
        request,
        parameters=build_parameter_set(
            request, profile_parameters=("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg")
        ),
    )
    options = PawleyOptions(support_fwhm=support, max_elements=150_000_000)
    initial = calculate(request, options)
    fits, times = [], []
    for _ in range(2):
        start = perf_counter()
        fit = refine(request, options, max_iterations=gate["max_iterations"], max_seconds=240.0)
        fits.append(fit)
        times.append(perf_counter() - start)
    fit, repeat = fits
    c = fit.calculation
    manual = np.sqrt(np.sum(((c.calculated_y - y) / sigma) ** 2) / np.sum((y / sigma) ** 2))
    correlation = float(np.corrcoef(y - c.background_y, c.calculated_y - c.background_y)[0, 1])
    checks = dict(
        rwp=c.rwp < gate["maximum_rwp"],
        correlation=correlation > gate["minimum_profile_correlation"],
        improvement=c.chi_square < 0.95 * initial.chi_square,
        manual_weighting=abs(manual - c.rwp) < 1e-12,
        nonnegative=bool(np.all(np.isfinite(fit.intensities)) and np.all(fit.intensities >= 0)),
        converged=fit.termination_reason == "converged",
        repeatable=bool(
            np.array_equal(fit.history, repeat.history)
            and np.array_equal(c.calculated_y, repeat.calculation.calculated_y)
        ),
    )
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(
        dataset=dataset_id,
        verified_files=[p.name for p in paths],
        samples=len(x),
        reflections=len(positions),
        times_seconds=times,
        rwp=c.rwp,
        initial_rwp=initial.rwp,
        profile_correlation=correlation,
        rank=fit.rank,
        termination=fit.termination_reason,
        accepted_steps=len(fit.history) - 1,
        active_width_bounds=fit.active_width_bounds,
        covariance_limitation=fit.covariance_limitation,
        profile_coefficients={
            s.key.name: s.value for s in fit.parameters.specs if s.key.module == "pawley_profile"
        },
        checks=checks,
        passed=all(checks.values()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("aps-sucrose-11bmb", "ansto-echidna-lab6-cw-neutron"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    provenance = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        native_binary_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    results = [
        run(name, args.data_root / name, gate)
        for name, gate in manifest["datasets"].items()
        if args.dataset is None or name == args.dataset
    ]
    report = dict(
        manifest=manifest,
        results=results,
        provenance=provenance,
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
