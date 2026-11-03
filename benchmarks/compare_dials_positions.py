"""Evaluate externally prepared peak lists with the same position-only indexer.

Optional research benchmark. DIALS runs in its own pinned environment through
dials_position_probe.py; no DIALS object or dependency enters PhaseSmith.
Input protocol and lossless image/geometry preparation are retained with the run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith.experimental.grains import GrainSearchOptions, find_grains
from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(case, method):
    with np.load(case / method / "positions.npz") as f:
        return {k: f[k] for k in f.files}


def selection(data, frames):
    first, last = data["ranges"].T
    block = (first // 5) % 5
    keep = (
        np.isfinite(data["xyz"]).all(axis=1)
        & (last - first <= 8)
        & (first > 0)
        & (last < frames)
        & (((last - 1) // 5) % 5 == block)
    )
    return keep, block != 0


def match_positions(a, b, pixel_radius=2.0, angle_radius=1.0):
    """Minimum-cost one-to-one matching, prioritizing maximum cardinality."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    from scipy.spatial import cKDTree

    if not len(a) or not len(b):
        return np.empty((0, 2), int), np.zeros(len(a), int), np.zeros(len(b), int)
    scale = np.array([pixel_radius, pixel_radius, angle_radius])
    neighbours = cKDTree(b / scale).query_ball_point(a / scale, np.sqrt(2))
    rows, columns, costs = [], [], []
    degrees_a, degrees_b = np.zeros(len(a), int), np.zeros(len(b), int)
    for i, indices in enumerate(neighbours):
        for j in indices:
            difference = a[i] - b[j]
            dxy = np.linalg.norm(difference[:2])
            if dxy <= pixel_radius and abs(difference[2]) <= angle_radius:
                rows.append(i)
                columns.append(j)
                costs.append(1 + (dxy / pixel_radius) ** 2 + (difference[2] / angle_radius) ** 2)
                degrees_a[i] += 1
                degrees_b[j] += 1
    rows.extend(range(len(a)))
    columns.extend(len(b) + np.arange(len(a)))
    costs.extend([4.0 * (len(a) + len(b))] * len(a))
    matrix = coo_matrix((costs, (rows, columns)), shape=(len(a), len(b) + len(a))).tocsr()
    x, y = min_weight_full_bipartite_matching(matrix)
    paired = y < len(b)
    return np.column_stack([x[paired], y[paired]]), degrees_a, degrees_b


def index(case, method):
    data = load(case, method)
    meta = json.loads((case / "images/geometry.json").read_text())
    keep, training = selection(data, len(meta["files"]))
    selected = np.flatnonzero(keep)
    options = GrainSearchOptions(grid_size=256)
    records = []
    for sign in (-1, 1):
        output = case / method / f"common-index-{sign:+d}"
        output.mkdir(exist_ok=False)
        vectors = np.array(meta["vectors"])
        vectors[-1] = [sign, 0, 0]
        geometry = PanelGeometry(vectors, meta["wavelength"])
        q = reciprocal_vectors(data["xyz"][keep], geometry)
        if method.startswith("dials") and sign == -1:
            error = float(np.max(np.abs(q - data["q"][keep])))
            assert error < 1e-12, error
        started = time.perf_counter()
        result = find_grains(
            q,
            training=training[keep],
            options=options,
            progress=lambda done, total, sign=sign: print(
                case.name, method, sign, done, total, flush=True
            ),
        )
        bases = np.array([c.direct_basis for c in result.candidates]).reshape(-1, 3, 3)
        np.savez(
            output / "arrays.npz",
            q=q,
            training=result.training,
            labels=result.labels,
            bases=bases,
            source_ids=selected,
        )
        candidates = []
        for c in result.candidates:
            candidates.append(
                dict(
                    cell=c.cell.tolist(),
                    volume=c.volume,
                    basis=c.direct_basis.tolist(),
                    accepted=bool(c.accepted),
                    training_count=int(np.count_nonzero(c.indexed & result.training)),
                    heldout_count=int(np.count_nonzero(c.indexed & ~result.training)),
                    training_fraction=c.training_fraction,
                    heldout_fraction=c.validation_fraction,
                    null_max_fraction=c.null_fraction,
                    indexed_rms=float(np.sqrt(np.mean(c.residual[c.indexed] ** 2)))
                    if c.indexed.any()
                    else None,
                )
            )
        record = dict(
            sign=sign,
            options=asdict(options),
            total_positions=len(data["xyz"]),
            common_selected=len(q),
            training=int(result.training.sum()),
            heldout=int((~result.training).sum()),
            seconds=time.perf_counter() - started,
            message=result.message,
            candidates=candidates,
            diagnostics=result.diagnostics,
        )
        write(output / "result.json", record)
        records.append({k: v for k, v in record.items() if k != "diagnostics"})
        print(json.dumps({k: v for k, v in records[-1].items() if k != "candidates"}), flush=True)
    write(case / method / "common-index.json", records)


def compare(case):
    native = load(case, "native")
    meta = json.loads((case / "images/geometry.json").read_text())
    result = []
    for method in ("dials-default", "dials-sigma5"):
        if not (case / method / "positions.npz").exists():
            continue
        other = load(case, method)
        for population in ("all", "common-quality"):
            na = np.flatnonzero(np.isfinite(native["xyz"]).all(axis=1))
            nb = np.flatnonzero(np.isfinite(other["xyz"]).all(axis=1))
            if population == "common-quality":
                na = np.flatnonzero(selection(native, len(meta["files"]))[0])
                nb = np.flatnonzero(selection(other, len(meta["files"]))[0])
            a, b = native["xyz"][na], other["xyz"][nb]
            pairs, da, db = match_positions(a, b)
            differences = a[pairs[:, 0]] - b[pairs[:, 1]]
            np.savez(
                case / f"matches-{method}-{population}.npz",
                pairs=np.column_stack([na[pairs[:, 0]], nb[pairs[:, 1]]]),
                differences=differences,
                native_degrees=da,
                dials_degrees=db,
            )
            dxy = np.linalg.norm(differences[:, :2], axis=1)
            domega = abs(differences[:, 2])
            row = dict(
                method=method,
                population=population,
                native_total=len(native["xyz"]),
                dials_total=len(other["xyz"]),
                native_nonfinite=int(np.count_nonzero(~np.isfinite(native["xyz"]).all(axis=1))),
                dials_nonfinite=int(np.count_nonzero(~np.isfinite(other["xyz"]).all(axis=1))),
                native=len(a),
                dials=len(b),
                matched=len(pairs),
                native_unmatched=len(a) - len(pairs),
                dials_unmatched=len(b) - len(pairs),
                native_multiple_neighbours=int(np.count_nonzero(da > 1)),
                dials_multiple_neighbours=int(np.count_nonzero(db > 1)),
                pixel_difference_median=float(np.median(dxy)) if len(dxy) else None,
                pixel_difference_p90=float(np.quantile(dxy, 0.9)) if len(dxy) else None,
                omega_difference_median=float(np.median(domega)) if len(domega) else None,
                omega_difference_p90=float(np.quantile(domega, 0.9)) if len(domega) else None,
            )
            result.append(row)
    write(case / "position-comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)


def paired_quality(case):
    """Paired reserved-position consistency; neither package is position truth."""
    native = load(case, "native")
    meta = json.loads((case / "images/geometry.json").read_text())

    def dominant(method):
        records = json.loads((case / method / "common-index.json").read_text())
        return max(
            ((r["sign"], c) for r in records for c in r["candidates"]),
            key=lambda item: item[1]["training_count"],
        )

    native_sign, nc = dominant("native")
    native_basis = np.array(nc["basis"])
    rows = []
    for method in ("dials-default", "dials-sigma5"):
        sign, dc = dominant(method)
        if sign != native_sign:
            rows.append(dict(method=method, comparison="dominant rotation signs differ"))
            continue
        other = load(case, method)
        with np.load(case / f"matches-{method}-common-quality.npz") as saved:
            pairs = saved["pairs"]
        _, ta = selection(native, len(meta["files"]))
        _, tb = selection(other, len(meta["files"]))
        pairs = pairs[~ta[pairs[:, 0]] & ~tb[pairs[:, 1]]]
        vectors = np.array(meta["vectors"])
        vectors[-1] = [sign, 0, 0]
        geometry = PanelGeometry(vectors, meta["wavelength"])
        qa = reciprocal_vectors(native["xyz"][pairs[:, 0]], geometry)
        qb = reciprocal_vectors(other["xyz"][pairs[:, 1]], geometry)
        dials_basis = np.array(dc["basis"])
        transform = np.rint(dials_basis @ np.linalg.inv(native_basis))
        relative_error = float(
            np.linalg.norm(dials_basis - transform @ native_basis) / np.linalg.norm(dials_basis)
        )
        relation = dict(
            integer_transform=transform.astype(int).tolist(),
            determinant=round(np.linalg.det(transform)),
            relative_basis_error=relative_error,
        )
        bases = [("native_training_basis", native_basis)]
        if abs(np.linalg.det(transform)) >= 1 and relative_error < 0.02:
            # A diagnostic basis conversion derived from training models only.
            # For det>1 this is a compact-cell hypothesis, not a DIALS solution.
            bases.append(
                ("dials_training_basis_in_native_cell", np.linalg.solve(transform, dials_basis))
            )
        for name, basis in bases:
            ha, hb = np.rint(qa @ basis.T), np.rint(qb @ basis.T)
            ea = np.linalg.norm(qa - ha @ np.linalg.inv(basis).T, axis=1)
            eb = np.linalg.norm(qb - hb @ np.linalg.inv(basis).T, axis=1)
            ia = (ea < 0.015) & np.any(ha != 0, axis=1)
            ib = (eb < 0.015) & np.any(hb != 0, axis=1)
            same = ia & ib & np.all(ha == hb, axis=1)
            rows.append(
                dict(
                    method=method,
                    reference=name,
                    paired_heldout=len(pairs),
                    both_indexed=int(np.count_nonzero(ia & ib)),
                    same_hkl=int(same.sum()),
                    native_only=int(np.count_nonzero(ia & ~ib)),
                    dials_only=int(np.count_nonzero(ib & ~ia)),
                    neither=int(np.count_nonzero(~ia & ~ib)),
                    native_rms_on_same_hkl=float(np.sqrt(np.mean(ea[same] ** 2)))
                    if same.any()
                    else None,
                    dials_rms_on_same_hkl=float(np.sqrt(np.mean(eb[same] ** 2)))
                    if same.any()
                    else None,
                    relation=relation,
                )
            )
    write(case / "paired-heldout-quality.json", rows)
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["index", "compare", "paired"])
    parser.add_argument("case", type=Path)
    parser.add_argument("--method", default="native")
    args = parser.parse_args()
    if args.action == "index":
        index(args.case, args.method)
    elif args.action == "compare":
        compare(args.case)
    else:
        paired_quality(args.case)


if __name__ == "__main__":
    main()
