"""Development-only padding probe; save one prediction before its reference audit."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from solution_corpus.data import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("a new output directory is required")
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = "1"
    from phasesmith import _core
    from phasesmith.crystallography import UnitCell
    from phasesmith.io.hkl import read_hkl
    from phasesmith.io.space_groups import space_group_by_number
    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
    from solution_corpus.audit import audit_case

    manifest_path = ROOT / "validation/solution-corpus-v5.json"
    manifest = json.loads(manifest_path.read_text())
    case = next(c for c in manifest["cases"] if c["id"] == args.case)
    if case["split"] != "development":
        raise ValueError("grid probe is restricted to development cases")
    control = json.loads((args.control / "comparison-verified.json").read_text())
    extension = json.loads((args.extension / "comparison-verified.json").read_text())
    a = next(r for r in control["runs"] if r["id"] == args.case and r["method"] == "phasesmith")
    b = next(r for r in extension["runs"] if r["id"] == args.case and r["method"] == "phasesmith")
    if (
        control["holdout_opened"]
        or extension["holdout_opened"]
        or control["native_sha256"] != extension["native_sha256"]
    ):
        raise ValueError("compatible development controls required")
    if sha(Path(_core.__file__)) != control["native_sha256"]:
        raise ValueError("compiled native build changed")
    directory = args.control / a["directory"]
    packet_path = directory / "inputs.json"
    hkl_path = directory / "sample.hkl"
    packet = json.loads(packet_path.read_text())
    options = {**a["options"], "grid_shape": tuple(b["effort"]["prediction"]["shape"])}
    data = read_hkl(hkl_path)
    started = time.perf_counter()
    result = solve_structure_elements(
        data,
        cell=UnitCell(*packet["cell"]),
        space_group=space_group_by_number(packet["space_group_number"]).space_group,
        elements=packet["elements"],
        maximum_atoms=a["input_only_resource_atom_cap"],
        options=SolutionOptions(**options),
        symmetry_recycling=True,
        tangent_refinement=True,
        wilson_normalization=True,
        phase_quality_selection=True,
        phase_origin_selection=True,
        chemical_assignment=True,
        iterative_refinement=True,
        missing_reflections=False,
        resolution_extension=1.0,
    )
    elapsed = time.perf_counter() - started
    model = (
        None
        if result.best is None
        else {
            "species": list(result.best.species),
            "fractional_xyz": result.best.fractional_xyz.tolist(),
            "cell": list(result.cell.as_tuple()),
            "space_group": packet["space_group"],
            "space_group_number": packet["space_group_number"],
            "reported_r1": result.best.r1,
        }
    )
    prediction = {
        "schema": "phasesmith.prediction-grid-probe.v1",
        "id": args.case,
        "qualification": (
            "One development case selected after observing an extension gain. "
            "Padding-only geometry diagnostic, not independent success-rate or timing evidence."
        ),
        "holdout_outcomes_opened": False,
        "native_sha256": control["native_sha256"],
        "manifest_sha256": sha(manifest_path),
        "implementation_sha256": sha(Path(__file__)),
        "input_sha256": sha(packet_path),
        "hkl_sha256": sha(hkl_path),
        "options": options,
        "maximum_atoms": a["input_only_resource_atom_cap"],
        "kernel_seconds": elapsed,
        "model": model,
        "effort": asdict(result.shared_search),
    }
    args.output.mkdir(parents=True)
    saved = args.output / "prediction-before-reference.json"
    saved.write_text(json.dumps(prediction, indent=2) + "\n")
    if sha(ROOT / case["files"]["reference"]["path"]) != case["files"]["reference"]["sha256"]:
        raise ValueError("reference checksum changed")
    audit = None if model is None else audit_case(case, model, ROOT)
    report = {
        **prediction,
        "prediction_before_reference_sha256": sha(saved),
        "reference_audit": audit,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "id": args.case,
                "grid_shape": options["grid_shape"],
                "correct": audit and audit["correct"],
                "kernel_seconds": elapsed,
            }
        )
    )


if __name__ == "__main__":
    main()
