"""Bounded, blind vertical-orientation check for the older first Fe7C3 point.

No CrysAlis cell, UB, HKLs or structure enters detection/indexing. The supplied
Fit2D calibration, exact source frames and explicit phi-as-vertical-axis
hypothesis are frozen before comparing either row convention and both signs.
This is a positional diagnostic, not absolute geometry validation or integration.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import fabio
import numpy as np
import pyFAI
from phasesmith.experimental.grains import GrainSearchOptions, find_grains
from phasesmith.experimental.reciprocal import PanelGeometry, correct_positions
from phasesmith.experimental.rotation import locate_series
from phasesmith.experimental.spots import SpotOptions


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def geometry_from_poni(filename, *, axis_sign, shape, original_rows):
    ai = pyFAI.load(str(filename))
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = shape[0] - 1 - xy[:, 1] if original_rows else xy[:, 1]
    z, p1, p2 = ai.calc_pos_zyx(d1=y, d2=xy[:, 0], use_cython=False)
    rays = np.column_stack([p1, p2, z])
    vectors = np.array(
        [rays[0], rays[1] - rays[0], rays[2] - rays[0], [0, 0, 1], [axis_sign, 0, 0]]
    )
    return PanelGeometry(vectors, ai.wavelength * 1e10), dict(
        original_rows=original_rows,
        axis_sign=axis_sign,
        vectors=vectors.tolist(),
        wavelength_angstrom=ai.wavelength * 1e10,
        method="public pyFAI calc_pos_zyx; zero-based pixel centres; image rows explicitly mapped",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(args.source.glob(args.source.name + "_*.mar3450"))
    if len(files) < 16:
        raise ValueError("need at least 16 matching exposures")
    records = []
    bounds = []
    shape = None
    for path in files:
        with fabio.open(str(path)) as image:
            if shape is None:
                shape = image.data.shape
            if image.data.shape != shape:
                raise ValueError("inconsistent source frame shapes")
            # Use original uppercase textual headers, not misleading normalized StartPhi keys.
            bounds.append([float(image.header["PHI_START"]), float(image.header["PHI_END"])])
            records.append(
                dict(
                    file=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    phi_start=image.header["PHI_START"],
                    phi_end=image.header["PHI_END"],
                )
            )
    bounds = np.asarray(bounds)
    options = GrainSearchOptions(max_grains=2, workers=4)
    detection = SpotOptions()
    protocol = dict(
        schema="phasesmith.older-fe7c3-orientation.v1",
        source_records=records,
        calibration=str(args.calibration),
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        image_shape=shape,
        observed_angles=(
            "uppercase MAR PHI_START/PHI_END; mapped to the model rotation coordinate"
        ),
        explicit_axis_assumption=(
            "rotation around vertical sample axis; both signs tested; "
            "not a reconstructed arbitrary goniometer"
        ),
        image_loading=(
            "FabIO native pixels followed by exactly one vertical flip, matching Dioptas read_frame"
        ),
        tested_calibration_row_conventions=[False, True],
        detector_options=asdict(detection),
        indexer_options=asdict(options),
        workers="four frame detectors; four fitting workers; hypothesis jobs run sequentially",
        mask="zero padding and nonfinite pixels, explicitly excluded for position hunting",
        noise="nominal gain=1/read_noise=0; positional experiment, not calibrated intensity errors",
        selection=(
            "unflagged tracks; <=8 observations; whole five-frame blocks with "
            "every fifth block held out; reciprocal RMS<=0.03 in every hypothesis"
        ),
        ranking=(
            "training support only; report held-out/control support separately; "
            "no known-cell selection"
        ),
        limitations=(
            "either calibration-to-run association or assumed axis may be wrong; "
            "failure/ambiguity must remain explicit"
        ),
    )
    write(args.output / "protocol.json", protocol)
    masks = {}

    def load(i):
        with fabio.open(str(files[i])) as image:
            data = np.asarray(image.data)[::-1].copy()
        masks[i] = (data == 0) | ~np.isfinite(data)
        return data

    started = time.perf_counter()
    positions = locate_series(
        load,
        bounds,
        mask_loader=lambda i: masks.pop(i),
        spot_options=detection,
        workers=4,
        progress=lambda done, total: (
            print("positions", done, total, flush=True) if done % 8 == 0 else None
        ),
    )
    np.savez_compressed(
        args.output / "positions.npz",
        **{
            name: getattr(positions, name)
            for name in ("boxes", "frame_ranges", "values", "flags", "observations", "omega_bounds")
        },
    )
    write(
        args.output / "position-summary.json",
        dict(
            seconds=time.perf_counter() - started,
            positions=len(positions),
            frame_hashes=positions.frame_hashes,
        ),
    )
    start, end = positions.frame_ranges.T
    training = (start // 5) % 5 != 0
    keep = (
        (positions.flags == 0)
        & (positions.column("n_observations") <= 8)
        & (((end - 1) // 5) % 5 == (start // 5) % 5)
    )
    cases = []
    for original_rows in (False, True):
        for sign in (-1, 1):
            geometry, meta = geometry_from_poni(
                args.calibration, axis_sign=sign, shape=shape, original_rows=original_rows
            )
            corrected = correct_positions(positions, geometry)
            keep &= corrected.rms <= 0.03
            cases.append((original_rows, sign, geometry, meta, corrected))
    ids = np.flatnonzero(keep)
    np.save(args.output / "selected-track-ids.npy", ids)
    rows = []
    for original_rows, sign, _geometry, meta, corrected in cases:
        name = f"rows-{int(original_rows)}-sign-{sign:+d}"
        started = time.perf_counter()
        print("index", name, "positions", len(ids), flush=True)
        result = find_grains(
            corrected.points[ids],
            training=training[ids],
            options=options,
            progress=lambda done, total, name=name: print(name, done, total, flush=True),
        )
        np.savez_compressed(
            args.output / (name + ".npz"),
            points=corrected.points[ids],
            training=result.training,
            labels=result.labels,
            bases=np.asarray([c.direct_basis for c in result.candidates]).reshape(-1, 3, 3),
        )
        row = dict(
            name=name,
            original_rows=original_rows,
            axis_sign=sign,
            geometry=meta,
            seconds=time.perf_counter() - started,
            candidates=[
                dict(
                    cell=c.cell.tolist(),
                    direct_basis=c.direct_basis.tolist(),
                    accepted=bool(c.accepted),
                    training_count=int(np.count_nonzero(c.indexed & result.training)),
                    heldout_count=int(np.count_nonzero(c.indexed & ~result.training)),
                    training_fraction=c.training_fraction,
                    validation_fraction=c.validation_fraction,
                    null_fraction=c.null_fraction,
                    indexed_rms=float(np.sqrt(np.mean(c.residual[c.indexed] ** 2)))
                    if c.indexed.any()
                    else None,
                )
                for c in result.candidates
            ],
            message=result.message,
            diagnostics=result.diagnostics,
        )
        write(args.output / (name + ".json"), row)
        rows.append(row)
    write(
        args.output / "comparison.json",
        dict(
            protocol=protocol,
            selected_positions=len(ids),
            cases=rows,
            interpretation=(
                "positional hypothesis comparison only; do not force an orientation "
                "if neither passes or both remain plausible"
            ),
        ),
    )


if __name__ == "__main__":
    main()
