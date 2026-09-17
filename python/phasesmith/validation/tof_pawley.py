"""Executable measured POWGEN and joint LANL nickel Pawley acceptance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from phasesmith import UnitCell, _core, smooth_bruckner, space_group_by_number
from phasesmith.io import read_gsas_tof_instrument, read_tof_powder_data
from phasesmith.pattern import TofPowderPattern
from phasesmith.refinement import (
    LatticeParameterBounds,
    LatticeParameterization,
    TofSharedLatticePhase,
)
from phasesmith.refinement.core import Bounds, ParameterSet
from phasesmith.refinement.pawley import PawleyOptions, parameter_key
from phasesmith.refinement.tof_pawley import (
    TofPawleyBackground,
    TofPawleyBank,
    TofPawleyInput,
    TofPawleyPhase,
    build_parameter_set,
    calculate,
    refine,
)
from phasesmith.symmetry import DSpacingRange, PreparedReflectionGenerator, TofRange

from .datasets import verify_validation_dataset


def validate_manifest(manifest):
    """Reject ignored controls and malformed acceptance contracts before fitting."""

    def keys(record, expected):
        if not isinstance(record, dict) or set(record) != set(expected.split()):
            raise ValueError("unexpected or missing TOF acceptance manifest fields")

    def positive(value, integer=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a positive numerical TOF acceptance control")
        if not np.isfinite(value) or value <= 0 or (integer and not isinstance(value, int)):
            raise ValueError("expected a positive finite TOF acceptance control")

    keys(manifest, "version description options runtime repeats tail_log background cases")
    if manifest["version"] != 1:
        raise ValueError("unsupported TOF acceptance manifest version")
    positive(manifest["repeats"], True)
    if manifest["repeats"] < 2:
        raise ValueError("TOF repeatability gates require at least two repeats")
    positive(manifest["tail_log"])
    keys(manifest["options"], "solver support_fwhm max_elements tolerance")
    PawleyOptions(**manifest["options"])
    for name in ("support_fwhm", "tolerance"):
        positive(manifest["options"][name])
    positive(manifest["options"]["max_elements"], True)
    keys(manifest["runtime"], "max_iterations max_evaluations max_rejections max_seconds")
    for name, value in manifest["runtime"].items():
        positive(value, name != "max_seconds")
    keys(manifest["background"], "bruckner_radius bruckner_iterations")
    for value in manifest["background"].values():
        positive(value, True)
    if not isinstance(manifest["cases"], list) or not manifest["cases"]:
        raise ValueError("TOF acceptance requires cases")
    ids = set()
    for case in manifest["cases"]:
        expected = (
            "id dataset data_file instrument_file banks space_group cell_angstrom "
            "tof_range d_range background_terms refine_cell refine_zero "
            "minimum_correlation minimum_families maximum_rwp"
        )
        if case.get("refine_cell"):
            expected += " target_cell_angstrom maximum_cell_error_angstrom"
        keys(case, expected)
        if not case["id"] or case["id"] in ids:
            raise ValueError("TOF case IDs must be nonempty and unique")
        ids.add(case["id"])
        if not isinstance(case["refine_cell"], bool) or not isinstance(case["refine_zero"], bool):
            raise ValueError("TOF refinement selections must be booleans")
        if not case["banks"] or len(set(case["banks"])) != len(case["banks"]):
            raise ValueError("TOF bank selections must be nonempty and unique")
        for bank in case["banks"]:
            positive(bank, True)
        for name in ("space_group", "background_terms", "minimum_families"):
            positive(case[name], True)
        positive(case["cell_angstrom"])
        if not 0 <= case["minimum_correlation"] <= 1:
            raise ValueError("TOF correlation threshold must be in [0, 1]")
        if case["maximum_rwp"] is not None:
            positive(case["maximum_rwp"])
        for name in ("d_range", "tof_range"):
            interval = case[name]
            if interval is None and name == "tof_range":
                continue
            if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] >= interval[1]:
                raise ValueError("TOF acceptance ranges must be finite and increasing")
            if name == "d_range":
                positive(interval[0])
        if case["refine_cell"]:
            positive(case["target_cell_angstrom"])
            positive(case["maximum_cell_error_angstrom"])
    return manifest


def build_request(case, manifest, data_root):
    root = data_root / case["dataset"]
    sources = verify_validation_dataset(case["dataset"], root)
    a = case["cell_angstrom"]
    cell = UnitCell(a, a, a, 90, 90, 90)
    group = space_group_by_number(case["space_group"]).space_group
    generator = PreparedReflectionGenerator(group, merge_friedel=True, max_candidates=1000000)
    banks = []
    for number in case["banks"]:
        imported = read_tof_powder_data(root / case["data_file"], bank=number)
        calibration = read_gsas_tof_instrument(root / case["instrument_file"], bank=number)
        inst = calibration.instrument
        keep = np.ones(len(imported.tof_us), dtype=bool)
        if case["tof_range"] is not None:
            keep = (imported.tof_us >= case["tof_range"][0]) & (
                imported.tof_us <= case["tof_range"][1]
            )
        x, y = imported.tof_us[keep], imported.observed_y[keep]
        bg = manifest["background"]
        fixed = smooth_bruckner(
            y, smooth_points=bg["bruckner_radius"], iterations=bg["bruckner_iterations"]
        )
        pattern = TofPowderPattern(
            x,
            observed_y=y,
            uncertainty=None if imported.uncertainty is None else imported.uncertainty[keep],
            mask=None if imported.mask is None else imported.mask[keep],
            background=fixed,
        )
        if case["refine_cell"]:
            domain = DSpacingRange(*case["d_range"])
        else:
            domain = TofRange(
                float(x[0]),
                float(x[-1]),
                *case["d_range"],
                inst.zero_us,
                inst.difc_us_per_angstrom,
                inst.difa_us_per_angstrom2,
                inst.difb_us_angstrom,
            )
        families = generator.generate(cell, domain)
        phase = TofPawleyPhase(
            case["id"],
            families.reflection_ids,
            families.d_spacing_angstrom,
            np.zeros(len(families.reflection_ids)),
            families.hkl,
        )
        bank_id = f"bank{number}"
        banks.append(
            TofPawleyBank(
                bank_id,
                pattern,
                inst,
                (phase,),
                (
                    f"{imported.format}: source bin boundaries/counts converted by the native "
                    "importer to microsecond bin-center densities and matching sigmas; incident "
                    "spectrum is absorbed into independent bank-local family areas "
                    "and is not reapplied"
                ),
                TofPawleyBackground(
                    tuple(np.zeros(case["background_terms"])), (float(x[0]), float(x[-1]))
                ),
            )
        )
    cells = ()
    if case["refine_cell"]:
        par = LatticeParameterization(group, cell)
        cells = (
            TofSharedLatticePhase(
                case["id"],
                par,
                LatticeParameterBounds.around(par, relative_length=0.01, angle_delta_deg=1),
                cell,
            ),
        )
    request = TofPawleyInput(tuple(banks), cells, tail_log=manifest["tail_log"])
    parameters = build_parameter_set(
        request,
        profile_parameters={b.bank_id: ("zero",) for b in banks} if case["refine_zero"] else {},
        lattice=case["refine_cell"],
    )
    parameters = ParameterSet(
        tuple(
            replace(s, bounds=Bounds(s.value - 20, s.value + 20))
            if s.key.module == "pawley_profile" and s.key.name == "zero" and s.refine
            else s
            for s in parameters.specs
        )
    )
    return replace(request, parameters=parameters), {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }


def run_case(case, manifest, data_root):
    request, sources = build_request(case, manifest, data_root)
    options = PawleyOptions(**manifest["options"])
    initial = calculate(request, options)
    times = []
    previous = None
    for _ in range(manifest["repeats"]):
        start = perf_counter()
        fit = refine(request, options, **manifest["runtime"])
        times.append(perf_counter() - start)
        if previous is not None:
            np.testing.assert_array_equal(fit.history, previous.history)
            np.testing.assert_array_equal(
                fit.calculation.calculated_y, previous.calculation.calculated_y
            )
        previous = fit
    bank_records = []
    for bank, lo, hi in zip(
        request.banks, request.sample_offsets[:-1], request.sample_offsets[1:], strict=True
    ):
        included = fit.calculation.included[lo:hi]
        y = bank.pattern.observed_y[included]
        calc = fit.calculation.calculated_y[lo:hi][included]
        bg = fit.calculation.background_y[lo:hi][included]
        sigma = (
            bank.pattern.uncertainty[included]
            if bank.pattern.uncertainty is not None
            else np.ones(len(y))
        )
        bank_records.append(
            dict(
                id=bank.bank_id,
                samples=len(bank.pattern.tof_us),
                included=int(included.sum()),
                families=len(bank.phases[0].reflection_ids),
                rwp=float(np.sqrt(np.sum(((calc - y) / sigma) ** 2) / np.sum((y / sigma) ** 2))),
                correlation=float(np.corrcoef(y - bg, calc - bg)[0, 1]),
            )
        )
    checks = dict(
        converged=fit.termination_reason == "converged",
        improved=fit.calculation.rwp < initial.rwp,
        nonnegative=bool(np.all(fit.intensities >= 0)),
        finite=bool(np.isfinite(fit.intensities).all()),
        family_coverage=all(b["families"] >= case["minimum_families"] for b in bank_records),
        correlation=all(b["correlation"] >= case["minimum_correlation"] for b in bank_records),
    )
    if case["maximum_rwp"] is not None:
        checks["rwp"] = fit.calculation.rwp <= case["maximum_rwp"] and all(
            b["rwp"] <= case["maximum_rwp"] for b in bank_records
        )
    cell = None
    if case["refine_cell"]:
        cell = fit.parameters.values()[parameter_key("lattice", case["id"], "a_angstrom")]
        checks["cell"] = (
            abs(cell - case["target_cell_angstrom"]) <= case["maximum_cell_error_angstrom"]
        )
    return dict(
        id=case["id"],
        sources_sha256=sources,
        settings=case,
        checks=checks,
        passed=all(checks.values()),
        termination=fit.termination_reason,
        rwp=fit.calculation.rwp,
        initial_rwp=initial.rwp,
        cell_angstrom=cell,
        banks=bank_records,
        history=fit.history.tolist(),
        times_seconds=times,
        diagnostics=dict(fit.diagnostics),
        profile_sha256=hashlib.sha256(fit.calculation.calculated_y.tobytes()).hexdigest(),
        covariance_limitation=fit.covariance_limitation,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "validation/pawley-tof-acceptance-v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    cases = [run_case(c, manifest, args.data_root) for c in manifest["cases"]]
    record = dict(
        passed=all(c["passed"] for c in cases),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_binary_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),
        manifest=manifest,
        cases=cases,
    )
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    if not record["passed"]:
        raise SystemExit("TOF Pawley acceptance gate failed; see complete report")


if __name__ == "__main__":
    main()
