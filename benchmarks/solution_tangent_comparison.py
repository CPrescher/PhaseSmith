"""Paired tangent/annealing/quartet comparison; reference audits follow all saved solves."""

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
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--symmetry-recycling", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.trials <= 8:
        parser.error("trials must lie in [1,8]")
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
        trials=args.trials,
        cycles=800,
        max_seconds=60,
        r1_target=0.06,
        completion=False,
        phasing="dual_space",
    )
    sources = (
        sorted((ROOT / "crates/phasesmith-solution/src").glob("*.rs"))
        + sorted((ROOT / "crates/phasesmith-py/src").glob("solution*.rs"))
        + [
            ROOT / p
            for p in (
                "Cargo.lock",
                "crates/phasesmith-solution/benches/solution.rs",
                "python/phasesmith/experimental/solution.py",
                "python/phasesmith/experimental/solution_shared_reference.py",
                "python/phasesmith/experimental/solution_peak_density_reference.py",
                "benchmarks/solution_tangent_comparison.py",
                "python/phasesmith/experimental/solution_reference.py",
                "python/phasesmith/experimental/solution_recycling_reference.py",
                "python/phasesmith/experimental/solution_tangent_reference.py",
                "python/phasesmith/experimental/solution_count_free_reference.py",
                "benchmarks/solution_adp_transfer.py",
                "benchmarks/structure_solution.py",
            )
        ]
    )
    header = {
        "schema": "phasesmith.tangent-comparison.v1",
        "platform": platform.platform(),
        "repeats": args.repeats,
        "options": asdict(options),
        "options_by_method": {
            method: {
                **asdict(options),
                "tangent_refinement": method == "tangent",
                "symmetry_recycling": args.symmetry_recycling,
            }
            for method in ("baseline", "tangent")
        },
        "methods": ["baseline", "tangent"],
        "dual_checkpoints": True,
        "cached_refinement": True,
        "recycling_cycles": 10,
        "symmetry_recycling": args.symmetry_recycling,
        "dual_peak_sigma": 2.0,
        "dual_peak_cap": 128,
        "integrated_peak_density": True,
        "staged_refinement": True,
        "tangent_controls": {
            "maximum_strong_variables": 256,
            "minimum_strong_e": 1.2,
            "minimum_strong_i_over_sigma": 2.0,
            "maximum_weak_e_upper_2sigma": 0.9,
            "maximum_triplets_per_target": 64,
            "maximum_cross_pairs_per_group": 32,
            "maximum_quartet_combinations": 2_000_000,
            "maximum_quartet_candidates": 32_768,
            "maximum_quartets": 4096,
            "maximum_quartets_per_variable": 16,
            "frozen_fraction": 0.4,
            "annealed_cycles": 12,
            "initial_temperature_radians": 0.35,
            "temperature_decay": 0.85,
            "maximum_perturbation_radians": float(np.pi / 3),
            "initial_blend": 0.25,
            "maximum_backtracking_halvings": 6,
            "quartet_acceptance_tolerance": 1e-12,
        },
        "peak_density_radius_angstrom": 0.7,
        "screening_iterations": 8,
        "refinement_shortlist": 3,
        "cases": cases,
        "native_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "implementation_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "qualification": (
            "Serial paired solves, alternating baseline/tangent order by repeat. "
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
            for label in ("baseline", "tangent") if repeat % 2 == 0 else ("tangent", "baseline"):
                destination = args.output / case["name"] / f"{label}-{repeat}"
                start = time.perf_counter()
                result = solve_structure_elements(
                    data,
                    cell=cell,
                    space_group=group,
                    elements=case["elements"],
                    options=options,
                    staged_refinement=True,
                    integrated_peak_density=True,
                    dual_checkpoints=True,
                    cached_refinement=True,
                    symmetry_recycling=args.symmetry_recycling,
                    tangent_refinement=label == "tangent",
                    recycling_cycles=10,
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
                    "phasing_trials": [asdict(t) for t in result.phasing_trials],
                    "rejected_trials": result.rejected_trials,
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
        full = [r for r in records if r["case"] == name and r["mode"] == "baseline"]
        staged = [r for r in records if r["case"] == name and r["mode"] == "tangent"]
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
                    "wall_time_ratio_baseline_over_tangent": before["seconds"] / after["seconds"],
                    "solve_speedup": (before["seconds"] / after["seconds"]) if pair_valid else None,
                    "quality_preserved": pair_valid,
                }
            )
        summaries.append(
            {
                "case": name,
                "baseline_median_seconds": base,
                "tangent_median_seconds": new,
                "wall_time_ratio_baseline_over_tangent": base / new,
                "solve_speedup": base / new if valid else None,
                "baseline_correct": sum(
                    r["counts_match_reference_after_solving"]
                    and r.get("post_solve_geometry", {}).get(
                        "max_element_matched_error_angstrom", float("inf")
                    )
                    < 0.02
                    for r in full
                ),
                "tangent_correct": sum(
                    r["counts_match_reference_after_solving"]
                    and r.get("post_solve_geometry", {}).get(
                        "max_element_matched_error_angstrom", float("inf")
                    )
                    < 0.02
                    for r in staged
                ),
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
        print(
            "Tangent refinement does not preserve all baseline quality gates; see saved report.",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
