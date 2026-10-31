"""Bounded element-only count experiment; supplied cell and symmetry remain priors.

One sampled orbit multiplicity per required element is tried. This is experimental
benchmark orchestration around the count-constrained native solver, not a general
unknown-composition solver. No reference structure is read by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering import xray_species_metadata
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure, write_solution
from solution_adp_transfer import reflection_data


def count_hypotheses(group, elements, maximum_atoms):
    """Sample orbit sizes, then enumerate one size per distinct required element.

    Eighths, thirds and two generic fractions sample special/general positions.
    This is not an exhaustive Wyckoff table for arbitrary settings.
    """
    if (
        isinstance(maximum_atoms, bool)
        or not isinstance(maximum_atoms, int)
        or not 1 <= maximum_atoms <= 128
    ):
        raise ValueError("maximum_atoms must be an integer in [1,128]")
    if (
        not elements
        or len(elements) > 6
        or any(not isinstance(e, str) for e in elements)
        or len(set(elements)) != len(elements)
    ):
        raise ValueError("elements must contain 1..6 distinct symbols")
    records = [xray_species_metadata(e) for e in elements]
    if any(
        r is None or r.atomic_number <= 1 or re.fullmatch(r"[A-Z][a-z]?", r.key) is None
        for r in records
    ):
        raise ValueError("elements must be neutral non-hydrogen element symbols")
    ordered = tuple(r.key for r in sorted(records, key=lambda r: -r.atomic_number))
    grid = (*(i / 8 for i in range(8)), 1 / 3, 2 / 3, 0.137, 0.271)
    sizes = sorted(
        {
            len(group.expand_sites(np.array([p])).fractional_xyz)
            for p in itertools.product(grid, repeat=3)
        }
    )
    allowed = (
        counts
        for counts in itertools.product(sizes, repeat=len(ordered))
        if sum(counts) <= maximum_atoms
    )
    proposals = list(itertools.islice(allowed, 513))
    if len(proposals) > 512:
        raise ValueError("more than 512 count hypotheses; reduce elements/atom bound")
    proposals.sort(key=lambda counts: (sum(counts), counts))
    return sizes, [dict(zip(ordered, counts, strict=True)) for counts in proposals]


def has_overlap(xyz, cell, cutoff=0.5):
    """Exact periodic cutoff check with reciprocal-metric image bounds.

    The cutoff is strict: exactly cutoff is accepted. Bounds include every lattice
    image that could be within cutoff, including skew cells; >4096 images fails.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be a finite (N,3) array")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    geometry = cell.geometry()
    bounds = cutoff * np.sqrt(np.diag(geometry.reciprocal_metric))
    ranges = [range(math.ceil(-0.5 - b), math.floor(0.5 + b) + 1) for b in bounds]
    if math.prod(map(len, ranges)) > 4096:
        raise ValueError("periodic cutoff check exceeds 4096 images")
    pairs = np.triu_indices(len(xyz), 1)
    delta = (xyz[:, None, :] - xyz[None, :, :])[pairs]
    delta -= np.rint(delta)
    for shift in itertools.product(*ranges):
        cartesian = (delta + shift) @ geometry.direct_basis.T
        if np.any(np.einsum("ij,ij->i", cartesian, cartesian) < cutoff**2):
            return True
    return False


def penalized_score(objective, families, parameters):
    """Descriptive penalty, not a calibrated likelihood or probability."""
    return families * math.log(max(objective, 1e-300)) + parameters * math.log(families)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hkl", type=Path, required=True, help="measured CIF reflection loop")
    parser.add_argument("--cell", nargs=6, type=float, required=True)
    parser.add_argument("--space-group", required=True)
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-volume-per-atom", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds-per-hypothesis", type=float, default=12)
    args = parser.parse_args()
    if not np.isfinite(args.minimum_volume_per_atom) or args.minimum_volume_per_atom <= 0:
        parser.error("minimum volume per atom must be finite and positive")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in [1,8]")
    cell = UnitCell(*args.cell)
    group = space_group_by_symbol(args.space_group).space_group
    bound = min(128, math.ceil(cell.geometry().volume_angstrom3 / args.minimum_volume_per_atom))
    sizes, proposals = count_hypotheses(group, args.elements, bound)
    if not proposals:
        parser.error("no count hypotheses within bound")
    data = reflection_data(args.hkl)
    options = SolutionOptions(
        trials=1,
        cycles=1200,
        phasing="dual_space",
        completion=True,
        max_seconds=args.seconds_per_hypothesis,
        r1_target=0.06,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "experiment-source.py").write_bytes(Path(__file__).read_bytes())
    header = {
        "schema": "phasesmith.element-count-experiment.v1",
        "elements_only": args.elements,
        "cell": args.cell,
        "space_group": args.space_group,
        "source_sha256": data.source_sha256,
        "minimum_volume_per_atom_angstrom3": args.minimum_volume_per_atom,
        "maximum_atoms": bound,
        "sampled_orbit_sizes": sizes,
        "hypotheses": proposals,
        "workers": args.workers,
        "options": asdict(options),
        "ranking": (
            "symmetry refined, no pair <0.5 A; minimize N*log(amplitude objective)"
            "+(ADP/coordinate parameters+scale)*log(N), N=merged families; descriptive only"
        ),
        "qualification": (
            "Reused development data. One sampled orbit count per element; "
            "supplied cell and exact symmetry. No source composition or coordinates read. "
            "Each hypothesis uses the existing count-constrained solver. "
            "Time caps can affect determinism."
        ),
        "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "support_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in (
                "benchmarks/solution_adp_transfer.py",
                "benchmarks/structure_solution.py",
                "python/phasesmith/single_crystal/solution.py",
                "python/phasesmith/single_crystal/solution_symmetry.py",
                "Cargo.lock",
            )
        },
    }
    (args.output / "inputs.json").write_text(json.dumps(header, indent=2) + "\n")
    start = time.perf_counter()

    def run(index, counts):
        began = time.perf_counter()
        record = {"index": index, "counts": counts}
        try:
            result = solve_structure(
                data, cell=cell, composition=counts, space_group=group, options=options
            )
            write_solution(result, args.output / f"hypothesis-{index:03d}")
            candidate = result.best
            valid = candidate is not None and candidate.refinement is not None
            overlap = has_overlap(candidate.fractional_xyz, cell) if candidate else None
            parameters = candidate.refinement.parameter_count + 1 if valid else None
            score = (
                penalized_score(candidate.objective, result.family_count, parameters)
                if valid and not overlap
                else None
            )
            record.update(
                termination=result.termination,
                r1=candidate.r1 if candidate else None,
                objective=candidate.objective if candidate else None,
                symmetry_refined=valid,
                overlap=overlap,
                parameters=parameters,
                family_count=result.family_count,
                ranking_score=score,
            )
        except (ValueError, RuntimeError) as error:
            record["error"] = str(error)
        record["seconds"] = time.perf_counter() - began
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, index, counts) for index, counts in enumerate(proposals)]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
            ranking = sorted(
                (r for r in records if r.get("ranking_score") is not None),
                key=lambda r: (r["ranking_score"], r["index"]),
            )
            report = {
                **header,
                "elapsed_seconds": time.perf_counter() - start,
                "completed": len(records),
                "records": sorted(records, key=lambda r: r["index"]),
                "ranking": header["ranking"],
                "ranked_indices": [r["index"] for r in ranking],
            }
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
