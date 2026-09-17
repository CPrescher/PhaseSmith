#!/usr/bin/env python3
"""Diagnose initial QARR forward equivalence after powder Friedel averaging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import phasesmith as ps
import rietx as rx
from compare_rietx_qarr import FORWARD_RELATIVE_L2_LIMIT, ROOT, prepare
from investigate_rietx_workload import policy
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import verify_validation_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if rx.__version__ != "1.4.0":
        raise RuntimeError("requires audited rietx 1.4.0")
    root = ROOT / "validation/data/iucr-qarr-1g"
    files = verify_validation_dataset("iucr-qarr-1g", root)
    pattern, experiment, phases, structure, instrument = prepare(root)
    reference = rv.calculate(pattern, experiment, phases, support_fwhm=30).profile_y
    rows = {}
    for name, accuracy in [
        ("default", ps.ProfileAccuracy()),
        ("combined_1_percent", ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)),
    ]:
        py = rv.calculate(
            pattern, experiment, phases, support_fwhm=30, profile_accuracy=accuracy
        ).profile_y
        for external in ("default", "both"):
            with policy(external):
                ry = (
                    np.asarray(
                        rx.Refinement(structure, instrument, history=False).predict(pattern.x)
                    )
                    - pattern.background
                )
            delta = float(np.linalg.norm(py - ry) / np.linalg.norm(py))
            rows[f"{name}_vs_rietx_{external}"] = {
                "relative_l2": delta,
                "passes_forward_gate": delta <= FORWARD_RELATIVE_L2_LIMIT,
            }
    per_phase = {}
    for phase, rphase in zip(phases, structure.phases, strict=True):
        py = rv.calculate(pattern, experiment, (phase,), support_fwhm=30).profile_y
        with policy("both"):
            ry = (
                np.asarray(
                    rx.Refinement(rx.Structure(phases=[rphase]), instrument, history=False).predict(
                        pattern.x
                    )
                )
                - pattern.background
            )
        per_phase[phase.phase_id] = float(np.linalg.norm(py - ry) / np.linalg.norm(py))
    record = {
        "scope": (
            "Corrected production powder values, before refinement. rietx both retains axial "
            "terms and widens windows; support semantics still differ."
        ),
        "powder_intensity_convention": "friedel_pair_average",
        "rietx": rx.__version__,
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "forward_relative_l2_limit": FORWARD_RELATIVE_L2_LIMIT,
        "comparisons": rows,
        "default_per_phase_vs_rietx_both": per_phase,
        "samples": len(reference),
    }
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
