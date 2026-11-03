"""Approximate archived-geometry control over frozen, blind position observations.

The CrysAlis report supplies instrument values only. No UB, cell, structure or
reference HKLs enter indexing. Zero tilt is an explicit approximation pending
verification of the archived detector-rotation convention.
"""

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith.experimental.grains import GrainSearchOptions, find_grains
from phasesmith.experimental.reciprocal import PanelGeometry, correct_positions
from phasesmith.experimental.rotation import PositionTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text = args.summary.read_text(errors="replace")
    matches = list(
        re.finditer(r"DETECTOR DISTANCE \(MM\):\s*([\d.]+).*?X\s*([\d.]+)\s*Y\s*([\d.]+)", text)
    )
    match = matches[-1]
    distance, cx, cy = map(float, match.groups())
    with np.load(args.positions / "positions.npz") as f:
        arrays = dict(f)
    positions = PositionTable(**arrays, frame_hashes=())
    ids = np.load(args.positions / "selected-track-ids.npy")
    training = (positions.frame_ranges[ids, 0] // 5) % 5 != 0
    options = GrainSearchOptions(max_grains=2, workers=4)
    protocol = dict(
        source_summary=str(args.summary),
        source_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        source_line=text[: match.start()].count("\n") + 1,
        source_record=match.group(0),
        positions=str(args.positions),
        positions_sha256=hashlib.sha256(
            (args.positions / "positions.npz").read_bytes()
        ).hexdigest(),
        indexer_options=asdict(options),
        detector_tilt_assumption_degrees=0.0,
        wavelength_angstrom=0.414915,
        pixels_micrometres=139.0,
        limitations=(
            "Approximate instrument conversion, including unverified pixel origin; "
            "both row conventions and vertical-axis signs tested."
        ),
        reference_cell_or_hkl_supplied=False,
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for original in (False, True):
        for sign in (-1, 1):
            g = PanelGeometry.from_poni(
                distance=distance / 1000,
                poni1=cy * 139e-6,
                poni2=cx * 139e-6,
                pixel1=139e-6,
                pixel2=139e-6,
                wavelength=0.414915,
                axis=[sign, 0, 0],
            )
            if original:
                vectors = g.vectors.copy()
                vectors[0] += (3450 - 1) * vectors[2]
                vectors[2] *= -1
                g = PanelGeometry(vectors, g.wavelength)
            q = correct_positions(positions, g)
            started = time.perf_counter()
            result = find_grains(q.points[ids], training=training, options=options)
            name = f"rows-{int(original)}-sign-{sign:+d}"
            np.savez_compressed(
                args.output / (name + ".npz"),
                points=q.points[ids],
                training=result.training,
                labels=result.labels,
                bases=np.asarray([c.direct_basis for c in result.candidates]).reshape(-1, 3, 3),
                hkl=np.asarray([c.hkl for c in result.candidates]).reshape(-1, len(ids), 3),
            )
            rows.append(
                dict(
                    name=name,
                    original_rows=original,
                    axis_sign=sign,
                    seconds=time.perf_counter() - started,
                    candidates=[
                        dict(
                            cell=c.cell.tolist(),
                            direct_basis=c.direct_basis.tolist(),
                            accepted=bool(c.accepted),
                            training_count=int(np.count_nonzero(c.indexed & result.training)),
                            heldout_count=int(np.count_nonzero(c.indexed & ~result.training)),
                            validation_fraction=float(c.validation_fraction),
                            null_fraction=c.null_fraction,
                        )
                        for c in result.candidates
                    ],
                    message=result.message,
                    diagnostics=result.diagnostics,
                )
            )
    (args.output / "comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print([(r["name"], len(r["candidates"])) for r in rows])


if __name__ == "__main__":
    main()
