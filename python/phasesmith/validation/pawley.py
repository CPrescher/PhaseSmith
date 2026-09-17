"""Checksum-pinned, manifest-driven CW Pawley acceptance; no implicit downloads."""

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
from ..io import read_powder_data, space_group_by_number
from ..pattern import PowderPattern
from ..refinement.background import ChebyshevBackground
from ..refinement.core import ParameterSet
from ..refinement.lattice import LatticeParameterBounds, LatticeParameterization
from ..refinement.pawley import (
    PawleyInput,
    PawleyOptions,
    PawleyPhase,
    _input,
    build_parameter_set,
    calculate,
    refine,
)
from ..symmetry import CwTwoThetaRange, PreparedReflectionGenerator
from .datasets import verify_validation_dataset

_COMMON = set(
    [
        "profile_parameters",
        "signed_intensities",
        "use_uncertainty",
        "merge_friedel",
        "initial_intensity",
        "background_iterations",
        "max_evaluations",
        "max_rejections",
        "max_seconds",
        "max_elements",
        "rank_tolerance",
        "tolerance",
        "damping",
        "max_active_iterations",
        "minimum_relative_chi_square_improvement",
        "rwp_recomputation_absolute_tolerance",
        "accepted_terminations",
        "repeats",
    ]
)
_DATASET = set(
    [
        "range_deg",
        "maximum_rwp",
        "minimum_profile_correlation",
        "max_iterations",
        "instrument",
        "data_file",
        "data_format",
        "cell",
        "space_group_number",
        "support_fwhm",
        "background",
    ]
)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} requires exactly {sorted(expected)}")


def validate_manifest(manifest):
    """Reject unsupported contracts rather than silently ignoring controls."""
    _keys(manifest, {"schema_version", "scope", "common", "datasets"}, "manifest")
    if manifest["schema_version"] not in (2, 3):
        raise ValueError("use an explicit version-2 or version-3 Pawley acceptance manifest")
    common = manifest["common"]
    _keys(common, _COMMON, "common")
    for name in ("signed_intensities", "use_uncertainty", "merge_friedel"):
        if type(common[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name in (
        "background_iterations",
        "max_evaluations",
        "max_rejections",
        "max_elements",
        "max_active_iterations",
        "repeats",
    ):
        if type(common[name]) is not int or common[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if common["repeats"] < 2:
        raise ValueError("release validation requires at least two repetitions")
    for name in (
        "max_seconds",
        "rank_tolerance",
        "tolerance",
        "damping",
        "rwp_recomputation_absolute_tolerance",
    ):
        if isinstance(common[name], bool) or not np.isfinite(common[name]) or common[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < common["minimum_relative_chi_square_improvement"] < 1:
        raise ValueError("improvement threshold must be inside (0, 1)")
    if not np.isfinite(common["initial_intensity"]):
        raise ValueError("initial intensity must be finite")
    selected = common["profile_parameters"]
    if (
        not isinstance(selected, list)
        or len(set(selected)) != len(selected)
        or not set(selected) <= {"u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"}
    ):
        raise ValueError("invalid profile selection")
    if common["accepted_terminations"] != ["converged"]:
        raise ValueError("a release gate must require convergence")
    if not isinstance(manifest["datasets"], dict) or not manifest["datasets"]:
        raise ValueError("datasets must be a nonempty object")
    for dataset, gate in manifest["datasets"].items():
        _keys(gate, _DATASET | ({"lattice"} if manifest["schema_version"] == 3 else set()), dataset)
        if Path(gate["data_file"]).name != gate["data_file"]:
            raise ValueError("data_file must be a verified dataset basename")
        if type(gate["max_iterations"]) is not int or gate["max_iterations"] < 1:
            raise ValueError("max_iterations must be a positive integer")
        bounds = np.asarray(gate["range_deg"], dtype=float)
        if (
            bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or not 0 < bounds[0] < bounds[1] < 180
        ):
            raise ValueError("invalid two-theta range")
        if not 0 < gate["maximum_rwp"] < 1 or not 0 < gate["minimum_profile_correlation"] < 1:
            raise ValueError("invalid quality thresholds")
        bg = gate["background"]
        if bg.get("kind") == "physical_width":
            _keys(bg, {"kind", "width_deg"}, "background")
            if not np.isfinite(bg["width_deg"]) or bg["width_deg"] <= 0:
                raise ValueError("background width must be finite and positive")
        elif bg.get("kind") == "sample_window":
            _keys(bg, {"kind", "points"}, "background")
            if type(bg["points"]) is not int or bg["points"] < 1:
                raise ValueError("background points must be positive")
        else:
            raise ValueError("unknown background kind")
        ConstantWavelengthInstrument(*gate["instrument"])
        UnitCell(*gate["cell"])
        _options(gate, common)
        lattice = gate.get("lattice")
        if lattice is not None:
            _keys(
                lattice,
                {"refine", "relative_length", "angle_delta_deg", "accepted_ranges"},
                "lattice",
            )
            par = LatticeParameterization(
                space_group_by_number(gate["space_group_number"]).space_group,
                UnitCell(*gate["cell"]),
            )
            LatticeParameterBounds.around(
                par,
                relative_length=lattice["relative_length"],
                angle_delta_deg=lattice["angle_delta_deg"],
            )
            selected = lattice["refine"]
            if (
                not isinstance(selected, list)
                or not selected
                or len(set(selected)) != len(selected)
                or not set(selected) <= set(par.parameter_names)
            ):
                raise ValueError("invalid lattice selection")
            _keys(lattice["accepted_ranges"], set(selected), "lattice accepted_ranges")
            for interval in lattice["accepted_ranges"].values():
                a = np.asarray(interval, dtype=float)
                if a.shape != (2,) or not np.isfinite(a).all() or not a[0] < a[1]:
                    raise ValueError("invalid lattice acceptance interval")
    return manifest


def _options(gate, common):
    names = (
        "max_elements",
        "rank_tolerance",
        "tolerance",
        "damping",
        "max_active_iterations",
        "use_uncertainty",
    )
    return PawleyOptions(support_fwhm=gate["support_fwhm"], **{k: common[k] for k in names})


def build_request(dataset_id, directory, gate, common):
    """Resolve every scientific setting from the manifest and verified raw files."""
    paths = verify_validation_dataset(dataset_id, directory)
    path = Path(directory) / gate["data_file"]
    if path.name not in {p.name for p in paths}:
        raise ValueError("data_file is not in the checksum-pinned dataset")
    data = read_powder_data(path, format=gate["data_format"])
    cell = UnitCell(*gate["cell"])
    group = space_group_by_number(gate["space_group_number"]).space_group
    instrument = ConstantWavelengthInstrument(*gate["instrument"])
    lo, hi = gate["range_deg"]
    selected = (data.x >= lo) & (data.x <= hi)
    x, y, sigma = data.x[selected], data.observed_y[selected], data.uncertainty[selected]
    bg = gate["background"]
    baseline = (
        smooth_bruckner(y, smooth_points=bg["points"], iterations=common["background_iterations"])
        if bg["kind"] == "sample_window"
        else SmoothBrucknerBackground(
            smooth_width=bg["width_deg"],
            iterations=common["background_iterations"],
            chebyshev_order=None,
        ).estimate(x, y)
    )
    pattern = PowderPattern(x, observed_y=y, uncertainty=sigma, background=baseline)
    families = PreparedReflectionGenerator(group, merge_friedel=common["merge_friedel"]).generate(
        cell, CwTwoThetaRange(float(x[0]), float(x[-1]), instrument.wavelength_angstrom)
    )
    positions = 2 * np.degrees(
        np.arcsin(instrument.wavelength_angstrom / (2 * families.d_spacing_angstrom))
    )
    phase = PawleyPhase(
        dataset_id,
        families.reflection_ids,
        positions,
        np.full(len(positions), common["initial_intensity"]),
        families.hkl,
    )
    lattice = gate.get("lattice")
    if lattice is not None:
        par = LatticeParameterization(group, cell)
        bounds = LatticeParameterBounds.around(
            par,
            relative_length=lattice["relative_length"],
            angle_delta_deg=lattice["angle_delta_deg"],
        )
        phase = PawleyPhase.from_cell(
            dataset_id,
            cell,
            group,
            wavelength_angstrom=instrument.wavelength_angstrom,
            two_theta_range=(float(x[0]), float(x[-1])),
            bounds=bounds,
            initial_intensity=common["initial_intensity"],
        )
        if not common["merge_friedel"]:
            raise ValueError("lattice validation currently requires merged Friedel families")
    request = PawleyInput(
        pattern,
        instrument,
        (phase,),
        ChebyshevBackground("residual", (0.0,), (float(x[0]), float(x[-1]))),
        signed_intensities=common["signed_intensities"],
    )
    request = replace(
        request,
        parameters=build_parameter_set(
            request, profile_parameters=tuple(common["profile_parameters"])
        ),
    )
    if lattice is not None:
        request = replace(
            request,
            parameters=ParameterSet(
                [
                    replace(spec, refine=spec.key.name in lattice["refine"])
                    if spec.key.module == "pawley_lattice"
                    else spec
                    for spec in request.parameters.specs
                ]
            ),
        )
    return request, _options(gate, common), paths


def check_fit(fit, initial, request, repeats, gate, common):
    """Apply the manifest policy to actual accepted arrays and histories."""
    c = fit.calculation
    y = request.pattern.observed_y
    sigma = request.pattern.uncertainty if common["use_uncertainty"] else np.ones_like(y)
    use = c.included
    manual = np.sqrt(
        np.sum(((c.calculated_y[use] - y[use]) / sigma[use]) ** 2)
        / np.sum((y[use] / sigma[use]) ** 2)
    )
    correlation = float(
        np.corrcoef(y[use] - c.background_y[use], c.calculated_y[use] - c.background_y[use])[0, 1]
    )
    checks = dict(
        rwp=c.rwp < gate["maximum_rwp"],
        correlation=correlation > gate["minimum_profile_correlation"],
        improvement=c.chi_square
        < (1 - common["minimum_relative_chi_square_improvement"]) * initial.chi_square,
        manual_weighting=abs(manual - c.rwp) < common["rwp_recomputation_absolute_tolerance"],
        intensity_policy=bool(
            np.all(np.isfinite(fit.intensities))
            and (common["signed_intensities"] or np.all(fit.intensities >= 0))
        ),
        converged=all(
            f.termination_reason in common["accepted_terminations"] for f in (fit, *repeats)
        ),
        repeatable=all(
            np.array_equal(fit.history, f.history)
            and np.array_equal(c.calculated_y, f.calculation.calculated_y)
            for f in repeats
        ),
    )
    if gate.get("lattice") is not None:
        intervals = gate["lattice"]["accepted_ranges"]
        checks["lattice_interval"] = all(
            intervals[s.key.name][0] <= s.value <= intervals[s.key.name][1]
            for f in [fit, *repeats]
            for s in f.parameters.specs
            if s.key.module == "pawley_lattice" and s.key.name in intervals
        )
    return {k: bool(v) for k, v in checks.items()}, correlation


def run(dataset_id, directory, gate, common, checkpoint_directory=None):
    """Run the fully resolved contract repeatedly and retain failed gates."""
    request, options, paths = build_request(dataset_id, directory, gate, common)
    initial = calculate(request, options)
    fits, times = [], []
    budgets = {k: common[k] for k in ("max_evaluations", "max_rejections", "max_seconds")}
    for _ in range(common["repeats"]):
        start = perf_counter()
        fit = refine(request, options, max_iterations=gate["max_iterations"], **budgets)
        fits.append(fit)
        times.append(perf_counter() - start)
    fit = fits[0]
    checks, correlation = check_fit(fit, initial, request, fits[1:], gate, common)
    if checkpoint_directory is not None:
        path = Path(checkpoint_directory) / f"{dataset_id}.pawley.json"
        with path.open("x") as stream:
            stream.write(fit.checkpoint)
    return dict(
        dataset=dataset_id,
        verified_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        request_sha256=hashlib.sha256(
            json.dumps(_input(request), sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        samples=len(request.pattern.x),
        reflections=len(fit.intensities),
        times_seconds=times,
        rwp=fit.calculation.rwp,
        initial_rwp=initial.rwp,
        profile_correlation=correlation,
        rank=fit.rank,
        termination=fit.termination_reason,
        accepted_steps=len(fit.history) - 1,
        active_width_bounds=fit.active_width_bounds,
        covariance_limitation=fit.covariance_limitation,
        lattice_parameters={
            s.key.label: s.value for s in fit.parameters.specs if s.key.module == "pawley_lattice"
        },
        profile_coefficients={
            s.key.name: s.value for s in fit.parameters.specs if s.key.module == "pawley_profile"
        },
        diagnostics=[dict(f.diagnostics) for f in fits],
        accepted_history=fit.history.tolist(),
        checks=checks,
        passed=all(checks.values()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--dataset", choices=("aps-sucrose-11bmb", "ansto-echidna-lab6-cw-neutron"))
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    provenance = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        native_binary_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if args.checkpoint_directory is not None:
        args.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    results = [
        run(name, args.data_root / name, gate, manifest["common"], args.checkpoint_directory)
        for name, gate in manifest["datasets"].items()
        if args.dataset is None or name == args.dataset
    ]
    with args.output.open("x") as stream:
        json.dump(
            dict(manifest=manifest, results=results, provenance=provenance),
            stream,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps(results, indent=2))
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
