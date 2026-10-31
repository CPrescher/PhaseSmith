"""Bounded PhaseSmith search on exposed enstatite intensities; no raw reduction.

The search subprocess receives only measured HKLs and explicit input priors.
Reference coordinates, source SHELX instructions and reference counts are excluded.
All inputs remain external. Writes a new output directory, never source files.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_single_crystal_data import DEFAULT_MANIFEST, verify_dataset, verify_file  # noqa: E402


def source_reflections(path):
    """Explicitly split the source HKL; never parse appended model instructions."""
    from phasesmith.io.hkl import parse_hkl

    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    for index, line in enumerate(lines):
        # The registered source uses the SHELX 3I4 fixed-width indices.
        if line.strip() and [int(line[start : start + 4]) for start in (0, 4, 8)] == [
            0,
            0,
            0,
        ]:
            if float(line[12:20]) != 0 or float(line[20:28]) != 0:
                raise ValueError("nonzero source HKL terminator values")
            reflection_text = "".join(lines[: index + 1])
            trailer = "".join(lines[index + 1 :])
            return parse_hkl(reflection_text), reflection_text, trailer
    raise ValueError("registered source HKL lacks the expected terminator")


def worker(packet_path):
    from phasesmith.crystallography import UnitCell
    from phasesmith.experimental.solution import (
        SolutionOptions,
        solve_structure_elements,
        write_solution,
    )
    from phasesmith.io.hkl import read_hkl
    from phasesmith.io.space_groups import space_group_by_number

    packet = json.loads(packet_path.read_text())
    if set(packet) != {
        "cell",
        "space_group_number",
        "elements",
        "maximum_atoms",
        "options",
        "controls",
    }:
        raise ValueError("unexpected search packet fields")
    result = solve_structure_elements(
        read_hkl(packet_path.parent / "search.hkl"),
        cell=UnitCell(*packet["cell"]),
        space_group=space_group_by_number(packet["space_group_number"]).space_group,
        elements=packet["elements"],
        maximum_atoms=packet["maximum_atoms"],
        options=SolutionOptions(**packet["options"]),
        **packet["controls"],
    )
    write_solution(result, packet_path.parent / "candidate")
    summary = {
        "target_met": result.target_met,
        "termination": result.termination,
        "returned_full_cell_counts": dict(Counter(result.best.species)) if result.best else None,
        "amplitude_r1": result.best.r1 if result.best else None,
        "options": asdict(result.options),
    }
    (packet_path.parent / "search-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run(args):
    import numpy as np
    from phasesmith import _core
    from phasesmith.experimental.solution import solve_structure_elements
    from phasesmith.io.hkl import HklData
    from solution_corpus.run import resource_atom_cap

    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    verification = verify_dataset(manifest, args.data_root)
    trailer = None
    if args.observations is None:
        observations, reflection_text, trailer = source_reflections(args.data_root / "dom1_v9.hkl")
        intensity_source = "external_source_integration"
        intensity_hash = next(r["sha256"] for r in manifest["files"] if r["path"] == "dom1_v9.hkl")
    else:
        record = dict(manifest["derived_observations"], path=args.observations.name)
        verified = verify_file(args.observations.parent, record)
        intensity_hash = verified["sha256"]
        with np.load(args.observations, allow_pickle=False) as arrays:
            observations = HklData(arrays["hkl"], arrays["intensity"], arrays["sigma"])
        # Full precision round-trip for PhaseSmith, preserving row order and signs.
        reflection_text = "".join(
            f"{hh} {kk} {ll} {intensity:.17g} {sigma:.17g} {batch}\n"
            for (hh, kk, ll), intensity, sigma, batch in zip(
                observations.hkl,
                observations.intensity,
                observations.sigma,
                observations.batch,
                strict=True,
            )
        )
        intensity_source = "archived_phasesmith_raw_count_reduction"
    inputs = manifest["search_inputs"]
    cap, measured_rows = resource_atom_cap(
        observations, inputs["cell_angstrom_degrees"], inputs["space_group_symbol"]
    )
    freeze = json.loads((ROOT / "validation/solution-holdout-freeze-v1.json").read_text())
    parameters = inspect.signature(solve_structure_elements).parameters
    controls = {k: v for k, v in freeze["controls"].items() if k in parameters}
    controls.update(symmetry_recycling=True, tangent_refinement=True)
    options = dict(freeze["solver_options"], phasing="dual_space", r1_target=0.06)
    packet = {
        "cell": inputs["cell_angstrom_degrees"],
        "space_group_number": inputs["space_group_number"],
        "elements": inputs["elements"],
        "maximum_atoms": cap,
        "options": options,
        "controls": controls,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "search.hkl").write_text(reflection_text, encoding="ascii")
    if trailer is not None:
        (args.output / "source-trailer.txt").write_text(trailer, encoding="ascii")
    packet_path = args.output / "search-input.json"
    packet_path.write_text(json.dumps(packet, indent=2) + "\n")
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    with (args.output / "search.log").open("w") as log:
        try:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(packet_path)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=18,
                check=False,
            )
            execution = {"returncode": process.returncode, "hard_timeout": False}
        except subprocess.TimeoutExpired:
            execution = {"returncode": None, "hard_timeout": True}
    summary_path = args.output / "search-summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    with Path(_core.__file__).open("rb") as stream:
        native_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "schema": "phasesmith.enstatite-search-probe.v1",
        "dataset_id": manifest["dataset_id"],
        "scope": "Exposed-data search from existing intensities; raw-frame workflow not executed",
        "input_verification": verification["input_verification"],
        "intensity_source": intensity_source,
        "intensity_sha256": intensity_hash,
        "observation_rows": len(observations.hkl),
        "symmetry_expanded_measured_rows": measured_rows,
        "input_only_atom_cap": cap,
        "native_sha256": native_sha,
        "solver_checkout_commit": subprocess.check_output(
            ["git", "-C", str(Path(_core.__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "search_controls": packet,
        "prepared_hkl_sha256": hashlib.sha256(reflection_text.encode("ascii")).hexdigest(),
        "manifest_sha256": hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest(),
        "execution": execution,
        "search": summary,
        # Only the parent reads the reference count, after the search has stopped.
        "post_search_assessment": {
            "reference_full_cell_atoms": manifest["assessment"]["reference_full_cell_non_h_atoms"],
            "capacity_sufficient_for_reference": cap
            >= manifest["assessment"]["reference_full_cell_non_h_atoms"],
            "coordinate_recovery": "not_assessed",
            "standalone_structure_refinement": "not_implemented",
            "complete_raw_to_structure_workflow": "not_executed",
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if summary is not None and execution["returncode"] == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--observations", type=Path, help="registered archived PhaseSmith NPZ")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if args.data_root is None or args.output is None:
        parser.error("--data-root and --output are required")
    args.output = args.output.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
