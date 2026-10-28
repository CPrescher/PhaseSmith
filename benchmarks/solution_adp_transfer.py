"""Explicit checksum-pinned blind HKL transfer benchmark. Never regenerate silently.

CIF reflection loops are translated without modifying signed measured F² or sigma.
Reference coordinates are opened only after solving and saving the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from pathlib import Path

import gemmi
import numpy as np
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import SolutionOptions, solve_structure, write_solution
from phasesmith.io.hkl import HklData
from phasesmith.io.space_groups import space_group_by_symbol
from structure_solution import _reference_error, _verify

ROOT = Path(__file__).resolve().parents[1]


def reflection_data(path):
    block = gemmi.cif.read_file(str(path)).sole_block()
    tags = ["index_h", "index_k", "index_l", "F_squared_meas", "F_squared_sigma"]
    rows = np.array([[float(v) for v in row] for row in block.find("_refln_", tags)])
    if not len(rows) or np.any(rows[:, :3] != np.rint(rows[:, :3])):
        raise ValueError("missing or noninteger reflection indices")
    return HklData(
        rows[:, :3].astype(np.int64),
        rows[:, 3],
        rows[:, 4],
        source_name=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--phasing", choices=("charge_flip", "dual_space", "portfolio"), default="charge_flip"
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    source = json.loads((ROOT / "validation/structure-solution-adp-transfer.json").read_text())
    options = SolutionOptions(
        trials=8, cycles=1200, max_seconds=60, r1_target=0.06, phasing=args.phasing
    )
    from phasesmith import _core

    implementation = sorted((ROOT / "crates/phasesmith-solution/src").glob("*.rs")) + [
        ROOT / name
        for name in (
            "Cargo.lock",
            "crates/phasesmith-py/src/solution.rs",
            "crates/phasesmith-py/src/solution_symmetry.rs",
            "crates/phasesmith-py/src/solution_dual.rs",
            "python/phasesmith/experimental/solution_dual_reference.py",
            "python/phasesmith/experimental/solution.py",
            "python/phasesmith/experimental/solution_symmetry.py",
            "python/phasesmith/experimental/solution_reference.py",
            "benchmarks/solution_adp_transfer.py",
            "benchmarks/structure_solution.py",
        )
    ]
    provenance = {
        "implementation_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in implementation
        },
        "source_manifest_sha256": hashlib.sha256(
            (ROOT / "validation/structure-solution-adp-transfer.json").read_bytes()
        ).hexdigest(),
        "native_extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "qualification": "Fixed settings across two previously selected cases, "
        "reused as development "
        "evidence for alternative phasing methods; "
        "reference coordinates read only after saving blind results. "
        "R1 uses unweighted non-resonant amplitudes, unlike deposited refinement objectives.",
    }
    records = []
    for case in source["cases"]:
        code = case["query_metadata"]["file"]
        directory = ROOT / f"validation/data/structure-solution/cod-{code}"
        directory.mkdir(exist_ok=True, parents=True)
        for file in case["files"]:
            path = directory / file["file"]
            if args.download_data and not path.exists():
                with urllib.request.urlopen(file["url"], timeout=30) as response:
                    data = response.read(file["bytes"] + 1)
                if len(data) != file["bytes"] or hashlib.sha256(data).hexdigest() != file["sha256"]:
                    raise ValueError("download differs from pinned source")
                with path.open("xb") as stream:
                    stream.write(data)
            _verify(path, file)
        inputs = case["solver_inputs"]
        cell = UnitCell(*inputs["cell_angstrom_degrees"])
        data = reflection_data(directory / f"{code}.hkl")
        start = time.perf_counter()
        result = solve_structure(
            data,
            cell=cell,
            composition=inputs["composition_full_cell_non_hydrogen"],
            space_group=space_group_by_symbol(inputs["known_space_group"]).space_group,
            options=options,
        )
        elapsed = time.perf_counter() - start
        write_solution(result, args.output / code)
        record = {
            "cod": code,
            "input_rows": len(data.hkl),
            "termination": result.termination,
            "target_met": result.target_met,
            "elapsed_seconds": elapsed,
            "r1": result.best.r1 if result.best else None,
            "symmetry_refined": bool(result.best and result.best.refinement),
            "candidate_count": len(result.candidates),
            "phasing_trials": result.to_record()["phasing_trials"],
            "all_candidates": [
                {"trial": c.trial, "r1": c.r1, "symmetry_refined": c.refinement is not None}
                for c in result.candidates
            ],
            "diagnostic": result.best.refinement_diagnostic if result.best else None,
        }
        if result.best:
            record["post_solve_reference"] = _reference_error(
                result.best, directory / f"{code}.cif", cell, inputs["known_space_group"]
            )
            if result.best.refinement:
                record["p1_r1"] = result.best.refinement.p1_r1
                record["parameter_count"] = result.best.refinement.parameter_count
                record["asymmetric_sites"] = len(result.best.refinement.species)
        records.append(record)
        (args.output / "benchmark.json").write_text(
            json.dumps(
                {
                    **provenance,
                    "platform": platform.platform(),
                    "options": result.to_record()["options"],
                    "cases": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
