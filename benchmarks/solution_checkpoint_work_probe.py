"""Development-only fixed-cycle probe; save all four predictions before reference audit."""

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
    parser.add_argument("--case", required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.cycles <= 800:
        parser.error("a new output directory and cycles in [1,800] are required")
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
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from phasesmith.io.hkl import read_hkl
    from phasesmith.io.space_groups import space_group_by_number
    from solution_corpus.audit import audit_case
    from solution_corpus.run import INPUT_KEYS

    manifest_path = ROOT / "validation/solution-corpus-v5.json"
    manifest = json.loads(manifest_path.read_text())
    case = next(c for c in manifest["cases"] if c["id"] == args.case)
    if case["split"] != "development":
        raise ValueError("fixed-work probe is restricted to development cases")
    report = json.loads((args.control / "comparison-verified.json").read_text())
    if report["holdout_opened"] or sha(Path(_core.__file__)) != report["native_sha256"]:
        raise ValueError("same-build development control required")
    row = next(r for r in report["runs"] if r["id"] == args.case and r["method"] == "phasesmith")
    directory = args.control / row["directory"]
    packet_path, hkl_path = directory / "inputs.json", directory / "sample.hkl"
    packet = json.loads(packet_path.read_text())
    if set(packet) != INPUT_KEYS:
        raise ValueError("only whitelisted count-free input fields are accepted")
    options = {**row["options"], "trials": 1, "cycles": args.cycles}
    data = read_hkl(hkl_path)
    results = []
    for mode, adaptive, sampling in [
        ("control", False, False),
        ("adaptive", True, False),
        ("sampling", False, True),
        ("combined", True, True),
    ]:
        start = time.perf_counter()
        result = solve_structure_elements(
            data,
            cell=UnitCell(*packet["cell"]),
            elements=packet["elements"],
            space_group=space_group_by_number(packet["space_group_number"]).space_group,
            options=SolutionOptions(**options),
            maximum_atoms=row["input_only_resource_atom_cap"],
            symmetry_recycling=True,
            tangent_refinement=True,
            wilson_normalization=True,
            phase_quality_selection=True,
            phase_origin_selection=True,
            chemical_assignment=True,
            iterative_refinement=True,
            adaptive_checkpoints=adaptive,
            independent_origin_sampling=sampling,
        )
        elapsed = time.perf_counter() - start
        completed = [len(h) for h in result.histories]
        if completed != [args.cycles]:
            raise ValueError("probe did not complete the declared identical Fourier work")
        model = (
            None
            if result.best is None
            else dict(
                species=list(result.best.species),
                fractional_xyz=result.best.fractional_xyz.tolist(),
                cell=list(result.cell.as_tuple()),
                space_group=packet["space_group"],
                space_group_number=packet["space_group_number"],
                reported_r1=result.best.r1,
            )
        )
        results.append(
            dict(
                mode=mode,
                model=model,
                kernel_seconds=elapsed,
                completed_cycles=completed,
                effort=asdict(result.shared_search),
            )
        )
    prediction = dict(
        schema="phasesmith.phase-checkpoint-work-probe.v1",
        id=args.case,
        qualification=(
            "One development case selected after a deadline-sensitive regression. "
            "One start and fixed Fourier cycles, original total native time/atom caps. "
            "A trajectory diagnostic, not a 35-case panel or a quiet-host speed estimate."
        ),
        holdout_outcomes_opened=False,
        native_sha256=report["native_sha256"],
        manifest_sha256=sha(manifest_path),
        implementation_sha256=sha(Path(__file__)),
        input_sha256=sha(packet_path),
        hkl_sha256=sha(hkl_path),
        options=options,
        maximum_atoms=row["input_only_resource_atom_cap"],
        results=results,
    )
    args.output.mkdir(parents=True)
    saved = args.output / "predictions-before-reference.json"
    saved.write_text(json.dumps(prediction, indent=2) + "\n")
    if sha(ROOT / case["files"]["reference"]["path"]) != case["files"]["reference"]["sha256"]:
        raise ValueError("reference checksum changed")
    audits = {r["mode"]: audit_case(case, r["model"], ROOT) for r in results if r["model"]}
    audit = {
        **prediction,
        "predictions_before_reference_sha256": sha(saved),
        "reference_audits": audits,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({mode: a["correct"] for mode, a in audits.items()}))


if __name__ == "__main__":
    main()
