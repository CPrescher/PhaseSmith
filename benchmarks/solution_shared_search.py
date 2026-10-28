"""Explicit measured-data shared search; reference coordinates are read after all runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import gemmi
import numpy as np
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import (
    SolutionOptions,
    solve_structure_elements,
    write_solution,
)
from phasesmith.io.hkl import read_hkl
from phasesmith.io.space_groups import space_group_by_symbol
from solution_adp_transfer import reflection_data
from structure_solution import _reference_error

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hkl", type=Path, required=True)
    parser.add_argument("--hkl-format", choices=("cif", "hkl4"), default="cif")
    parser.add_argument("--cell", nargs=6, type=float, required=True)
    parser.add_argument("--space-group", required=True)
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-after-solving", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument(
        "--phasing", choices=("charge_flip", "dual_space", "portfolio"), default="charge_flip"
    )
    parser.add_argument("--no-staged-refinement", action="store_true")
    parser.add_argument("--no-integrated-peak-density", action="store_true")
    parser.add_argument("--screening-iterations", type=int, default=8)
    parser.add_argument("--refinement-shortlist", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be in [1,10]")
    cell = UnitCell(*args.cell)
    group = space_group_by_symbol(args.space_group).space_group
    data = reflection_data(args.hkl) if args.hkl_format == "cif" else read_hkl(args.hkl)
    options = SolutionOptions(
        trials=1,
        cycles=800,
        max_seconds=args.max_seconds,
        r1_target=0.06,
        completion=False,
        phasing=args.phasing,
    )
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "experiment-source.py").write_bytes(Path(__file__).read_bytes())
    sources = (
        sorted((ROOT / "crates/phasesmith-solution/src").glob("*.rs"))
        + sorted((ROOT / "crates/phasesmith-py/src").glob("solution*.rs"))
        + [
            ROOT / name
            for name in (
                "Cargo.lock",
                "python/phasesmith/experimental/solution.py",
                "python/phasesmith/experimental/solution_shared_reference.py",
                "python/phasesmith/experimental/solution_peak_density_reference.py",
                "benchmarks/solution_shared_search.py",
                "benchmarks/solution_adp_transfer.py",
                "benchmarks/structure_solution.py",
            )
        ]
    )
    header = {
        "schema": "phasesmith.shared-element-search-benchmark.v1",
        "elements_only": args.elements,
        "cell": args.cell,
        "space_group": args.space_group,
        "input_sha256": hashlib.sha256(args.hkl.read_bytes()).hexdigest(),
        "input_rows": len(data.hkl),
        "options": asdict(options),
        "repeats": args.repeats,
        "staged_refinement": not args.no_staged_refinement,
        "integrated_peak_density": not args.no_integrated_peak_density,
        "screening_iterations": args.screening_iterations,
        "refinement_shortlist": args.refinement_shortlist,
        "platform": platform.platform(),
        "native_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "implementation_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "qualification": (
            "Reused development data; exact cell and symmetry supplied. "
            "No atom counts or reference coordinates enter the search. "
            "References opened only after saving every run. "
            "Wall times are not a thread-matched SHELXT comparison."
        ),
    }
    (args.output / "inputs.json").write_text(json.dumps(header, indent=2) + "\n")
    records = []
    results = []
    for repeat in range(args.repeats):
        start = time.perf_counter()
        result = solve_structure_elements(
            data,
            cell=cell,
            elements=args.elements,
            space_group=group,
            options=options,
            staged_refinement=not args.no_staged_refinement,
            integrated_peak_density=not args.no_integrated_peak_density,
            screening_iterations=args.screening_iterations,
            refinement_shortlist=args.refinement_shortlist,
        )
        elapsed = time.perf_counter() - start
        write_solution(result, args.output / f"repeat-{repeat}")
        record = {
            "repeat": repeat,
            "solve_seconds": elapsed,
            "termination": result.termination,
            "r1": result.best.r1 if result.best else None,
            "inferred_counts": dict(result.composition),
            "shared_search": asdict(result.shared_search),
            "target_met": result.target_met,
        }
        records.append(record)
        results.append(result)
        (args.output / "report-before-reference.json").write_text(
            json.dumps({**header, "runs": records}, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
    if args.reference_after_solving:
        reference = gemmi.make_small_structure_from_block(
            gemmi.cif.read_file(str(args.reference_after_solving))[-1]
        )
        expected = Counter()
        for site in reference.sites:
            if site.element.name != "H":
                count = len(
                    group.expand_sites([[site.fract.x, site.fract.y, site.fract.z]]).fractional_xyz
                )
                expected[site.element.name] += count
        for result, record in zip(results, records, strict=True):
            record["counts_match_reference_after_solving"] = dict(result.composition) == dict(
                expected
            )
            if result.best and record["counts_match_reference_after_solving"]:
                record["post_solve_geometry"] = _reference_error(
                    result.best, args.reference_after_solving, cell, args.space_group
                )
        header["reference_sha256_after_solving"] = hashlib.sha256(
            args.reference_after_solving.read_bytes()
        ).hexdigest()
        header["report_before_reference_sha256"] = hashlib.sha256(
            (args.output / "report-before-reference.json").read_bytes()
        ).hexdigest()
    report = {
        **header,
        "runs": records,
        "median_solve_seconds": float(np.median([r["solve_seconds"] for r in records])),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
