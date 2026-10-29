"""Measure phase coherence and search work from two saved, element-only solves.

Uses supplied symmetry and saved maps only; no reference formula or coordinates.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from phasesmith.io.space_groups import space_group_by_symbol


def phase_coherence(directory, group):
    metadata_path = directory / "result.json"
    arrays_path = directory / "density-and-observations.npz"
    metadata = json.loads(metadata_path.read_text())
    with np.load(arrays_path) as arrays:
        rho = arrays["density"]
        hkl = arrays["expanded_hkl"]
        coefficients = np.prod(rho.shape) * np.fft.ifftn(rho)
    shape = np.array(rho.shape)
    audit = metadata["shared_search"]
    setting = audit["recycling"][0]
    phasors = []
    for op in group.operations:
        mapped = hkl @ op.rotation
        for sign in (1, -1):
            h = sign * mapped
            values = coefficients[tuple((setting["hand"] * h % shape).T)] * np.exp(
                2j * np.pi * (h @ np.array(setting["origin_shift"]))
            )
            units = np.divide(
                values, np.abs(values), out=np.zeros_like(values), where=np.abs(values) > 1e-100
            )
            if sign == -1:
                units = units.conj()
            phasors.append(units * np.exp(2j * np.pi * (hkl @ np.array(op.translation, float))))
    coherence = np.abs(np.mean(phasors, axis=0))
    return {
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (metadata_path, arrays_path)
        },
        "expanded_observations": len(hkl),
        "source_to_symmetry_hand": setting["hand"],
        "source_to_symmetry_shift": setting["origin_shift"],
        "mean_symmetry_phase_coherence": float(np.mean(coherence)),
        "median_symmetry_phase_coherence": float(np.median(coherence)),
        "fraction_coherence_below_half": float(np.mean(coherence < 0.5)),
        "recycled_peak_counts": [a["peaks"] for a in audit["recycling"]],
        "joint_batch_orbits": [a["batch_orbits"] for a in audit["recycling"]],
        "rounds": audit["rounds"],
        "screening_refinements": audit["screening_refinements"],
        "final_refinements": audit["final_refinements"],
        "accepted_moves": audit["accepted_moves"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--tangent", required=True, type=Path)
    parser.add_argument("--space-group", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    group = space_group_by_symbol(args.space_group).space_group
    report = {
        "schema": "phasesmith.tangent-recycling-diagnosis.v1",
        "space_group": args.space_group,
        "qualification": "Saved development maps in each framework's first "
        "supplied-symmetry setting. "
        "Coherence is the length of the symmetry/Friedel-aligned mean unit phasor; "
        "it measures cancellation risk, not correctness. No reference structure is read.",
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline": phase_coherence(args.baseline, group),
        "unconstrained_tangent": phase_coherence(args.tangent, group),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
