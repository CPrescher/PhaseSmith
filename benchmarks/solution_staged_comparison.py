"""Paired full/staged comparison, with structure/reference audits deferred until all solves."""

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
import phasesmith as ps
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import (
    SolutionOptions,
    solve_structure_elements,
    write_solution,
)
from phasesmith.io.cif import read_cif
from phasesmith.io.hkl import read_hkl
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering import XrayNonResonant
from solution_adp_transfer import reflection_data
from structure_solution import _reference_error

ROOT = Path(__file__).resolve().parents[1]


def checked(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"input differs from pinned checksum: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must lie in [1,10]")
    manifest_path = ROOT / "validation/structure-solution-staged-comparison.json"
    cases = json.loads(manifest_path.read_text())["cases"]
    prepared = {}
    for case in cases:
        path = ROOT / case["hkl"]
        checked(path, case["hkl_sha256"])
        data = reflection_data(path) if case["hkl_format"] == "cif" else read_hkl(path)
        prepared[case["name"]] = (
            data,
            UnitCell(*case["cell"]),
            space_group_by_symbol(case["space_group"]).space_group,
        )
    options = SolutionOptions(
        trials=1, cycles=800, max_seconds=60, r1_target=0.06, completion=False
    )
    sources = (
        sorted((ROOT / "crates/phasesmith-solution/src").glob("*.rs"))
        + sorted((ROOT / "crates/phasesmith-py/src").glob("solution*.rs"))
        + [
            ROOT / p
            for p in (
                "Cargo.lock",
                "python/phasesmith/experimental/solution.py",
                "python/phasesmith/experimental/solution_shared_reference.py",
                "benchmarks/solution_staged_comparison.py",
                "benchmarks/solution_adp_transfer.py",
                "benchmarks/structure_solution.py",
            )
        ]
    )
    header = {
        "schema": "phasesmith.staged-refinement-comparison.v1",
        "platform": platform.platform(),
        "repeats": args.repeats,
        "options": asdict(options),
        "integrated_peak_density": False,
        "screening_iterations": 8,
        "refinement_shortlist": 3,
        "cases": cases,
        "native_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "implementation_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "qualification": (
            "Serial paired solves, alternating full/staged order by repeat. "
            "Shared inputs and settings; no counts/coordinates supplied. "
            "References read only after all paired solves and reports are saved. "
            "Reused development data; no general coverage guarantee."
        ),
    }
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "experiment-source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "inputs.json").write_text(json.dumps(header, indent=2) + "\n")
    records = []
    solutions = []
    for repeat in range(args.repeats):
        for case in cases:
            data, cell, group = prepared[case["name"]]
            for staged in (False, True) if repeat % 2 == 0 else (True, False):
                label = "staged" if staged else "full"
                destination = args.output / case["name"] / f"{label}-{repeat}"
                start = time.perf_counter()
                result = solve_structure_elements(
                    data,
                    cell=cell,
                    space_group=group,
                    elements=case["elements"],
                    options=options,
                    staged_refinement=staged,
                    integrated_peak_density=False,
                )
                elapsed = time.perf_counter() - start
                write_solution(result, destination)
                record = {
                    "case": case["name"],
                    "repeat": repeat,
                    "mode": label,
                    "seconds": elapsed,
                    "r1": result.best.r1 if result.best else None,
                    "counts": dict(result.composition),
                    "target_met": result.target_met,
                    "termination": result.termination,
                    "effort": asdict(result.shared_search),
                    "directory": str(destination.relative_to(args.output)),
                }
                records.append(record)
                solutions.append((case, result, record))
                (args.output / "report-before-reference.json").write_text(
                    json.dumps({**header, "runs": records}, indent=2) + "\n"
                )
                print(json.dumps(record), flush=True)
    for case in cases:
        path = ROOT / case["reference_after_solving"]
        checked(path, case["reference_sha256"])
        source = gemmi.make_small_structure_from_block(gemmi.cif.read_file(str(path))[-1])
        expected = Counter()
        ops = gemmi.find_spacegroup_by_name(case["space_group"]).operations()
        for site in source.sites:
            if site.element.name == "H":
                continue
            points = []
            for op in ops:
                point = np.mod(op.apply_to_xyz([site.fract.x, site.fract.y, site.fract.z]), 1)
                if not any(np.linalg.norm((point - p + 0.5) % 1 - 0.5) < 1e-8 for p in points):
                    points.append(point)
            expected[site.element.name] += len(points)
        for item, result, record in solutions:
            if item["name"] != case["name"]:
                continue
            record["counts_match_reference_after_solving"] = dict(expected) == dict(
                result.composition
            )
            if result.best and record["counts_match_reference_after_solving"]:
                record["post_solve_geometry"] = _reference_error(
                    result.best, path, result.cell, case["space_group"]
                )
            cif_checks = {}
            if result.best:
                for cif in sorted((args.output / record["directory"]).glob("*.cif")):
                    values = ps.calculate_structure_factor_values(
                        read_cif(cif.read_text()).structure,
                        result.expanded_hkl,
                        np.ones(len(result.expanded_hkl), dtype=np.int64),
                        XrayNonResonant(),
                    )
                    r1 = float(
                        np.abs(
                            result.best.amplitude_scale * np.sqrt(values.f_squared)
                            - result.amplitudes
                        ).sum()
                        / result.amplitudes.sum()
                    )
                    cif_checks[cif.name] = {
                        "r1": r1,
                        "absolute_r1_difference": abs(r1 - result.best.r1),
                        "sha256": hashlib.sha256(cif.read_bytes()).hexdigest(),
                    }
            record["cif_checks"] = cif_checks
    summaries = []
    passed = True
    for case in cases:
        name = case["name"]
        full = [r for r in records if r["case"] == name and r["mode"] == "full"]
        staged = [r for r in records if r["case"] == name and r["mode"] == "staged"]
        base = float(np.median([r["seconds"] for r in full]))
        new = float(np.median([r["seconds"] for r in staged]))
        valid = all(
            r["target_met"]
            and r["counts_match_reference_after_solving"]
            and r.get("post_solve_geometry", {}).get(
                "max_element_matched_error_angstrom", float("inf")
            )
            < 0.02
            and len(r["cif_checks"]) == 2
            and all(c["absolute_r1_difference"] < 2e-10 for c in r["cif_checks"].values())
            for r in full + staged
        )
        pairs = []
        for before, after in zip(full, staged, strict=True):
            pair_valid = (
                after["counts"] == before["counts"]
                and before["r1"] is not None
                and after["r1"] is not None
                and after["r1"] <= before["r1"] + 1e-7
                and after.get("post_solve_geometry", {}).get(
                    "max_element_matched_error_angstrom", float("inf")
                )
                <= before.get("post_solve_geometry", {}).get(
                    "max_element_matched_error_angstrom", float("inf")
                )
                + 1e-4
            )
            valid &= pair_valid
            pairs.append(
                {
                    "repeat": before["repeat"],
                    "speedup": before["seconds"] / after["seconds"],
                    "quality_preserved": pair_valid,
                }
            )
        summaries.append(
            {
                "case": name,
                "full_median_seconds": base,
                "staged_median_seconds": new,
                "speedup": base / new,
                "quality_preserved": bool(valid),
                "pairs": pairs,
            }
        )
        passed &= valid
    report = {
        **header,
        "runs": records,
        "summary": summaries,
        "all_quality_gates_passed": bool(passed),
        "report_before_reference_sha256": hashlib.sha256(
            (args.output / "report-before-reference.json").read_bytes()
        ).hexdigest(),
        "quality_gates": {
            "matching_full_cell_counts": True,
            "maximum_reference_error_angstrom": 0.02,
            "paired_maximum_error_increase_angstrom": 1e-4,
            "paired_r1_increase": 1e-7,
            "exported_cif_r1_difference": 2e-10,
        },
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summaries, "passed": bool(passed)}, indent=2), flush=True)
    if not passed:
        raise SystemExit("structure/quality comparison failed; see saved report")


if __name__ == "__main__":
    main()
