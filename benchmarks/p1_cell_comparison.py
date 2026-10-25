"""Frozen position-only P1 primitive/Bravais comparison, no material/cell prior."""

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np
from phasesmith.experimental.cell_comparison import CellComparisonOptions, compare_cells
from phasesmith.experimental.cell_metrics import BRAVAIS
from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "runs/p1-bravais-comparison-20261025")
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
DIAG = ROOT / "runs/p1-blind-diagnosis-20261025"
access = []


def read(p):
    content = p.read_bytes()
    access.append(dict(path=str(p), sha256=hashlib.sha256(content).hexdigest()))
    return content


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


options = CellComparisonOptions(max_subcell_index=8)
protocol = dict(
    input_policy=(
        "Measured P1 positions, calibrated detector geometry, and the existing "
        "training-derived inferred basis only. No known cell, material, structure, "
        "saved UB, reference HKLs or other sample runs."
    ),
    primary=(
        "Consistent intensity-weighted x/y/omega centroids transformed to q; "
        "detector-space fixed-instrument fits."
    ),
    sensitivity="Prior equal-observation q centroids, reciprocal-space fits.",
    fitting=(
        "Training only; fixed inferred HKLs and fixed parent/hypothesis populations;"
        " all 14 Bravais types in bounded reduced axes."
    ),
    ranking=(
        "Smallest supported primitive hypothesis by exact HNF index; within it "
        "2*training radial Huber cost + p*log(3*N), fixed scales. Not a calibrated "
        "BIC or symmetry probability."
    ),
    validation=(
        "Original 1052/243 rotation-block split reused. Exploratory diagnostic, not "
        "a fresh independent benchmark. Held-out residuals do not select or refine "
        "models."
    ),
    options=asdict(options),
    sources=["diagnostic-inputs.npz", "P1_fft_grains.json geometry snapshot"],
)
if (OUT / "protocol.json").exists():
    assert json.loads((OUT / "protocol.json").read_text()) == protocol
else:
    write(OUT / "protocol.json", protocol)
z = np.load(BytesIO(read(DIAG / "diagnostic-inputs.npz")))
meta = json.loads(read(ROOT / "runs/fe7c3-fft-grains-20261025/P1_fft_grains.json"))
geo = next(g for g in meta["geometry"] if g["axis_sign"] == -1)
g = PanelGeometry(np.array(geo["vectors"]), geo["wavelength_angstrom"])
observed = z["values"][:, :3]
weighted_q = reciprocal_vectors(observed, g)
results = {}
for label, q, kwargs in [
    ("primary_detector", weighted_q, dict(observed=observed, geometry=g)),
    ("sensitivity_q_mean", z["q"], {}),
]:
    started = time.perf_counter()
    r = compare_cells(q, z["A"], training=z["train"], options=options, **kwargs)
    primitives = [
        dict(
            id=p.id,
            index=p.index,
            hnf=p.reciprocal_hnf.tolist(),
            cell=p.cell.tolist(),
            basis=p.direct_basis.tolist(),
            volume=p.volume,
            train=p.training_count,
            validation=p.validation_count,
            excluded_train=p.excluded_training,
            excluded_validation=p.excluded_validation,
            supported=p.supported,
            reason=p.reason,
        )
        for p in r.primitives
    ]
    settings = [
        dict(
            row=i,
            primitive_id=f.primitive_id,
            symbol=f.setting.symbol,
            index=f.setting.index,
            transform=f.setting.transform.tolist(),
            strain=f.setting.strain,
            cell=f.cell.tolist(),
            volume=float(abs(np.linalg.det(np.linalg.inv(f.reciprocal_basis).T))),
            basis=f.reciprocal_basis.tolist(),
            training=f.training_metrics,
            validation=f.validation_metrics,
            parameters=f.parameter_count,
            training_score=f.training_score,
            history=f.history.tolist(),
            converged=f.converged,
            status=f.status,
            recommended=i == r.recommended_setting_index,
        )
        for i, f in enumerate(r.settings)
    ]
    record = dict(
        seconds=time.perf_counter() - started,
        mode=r.mode,
        parent_train=int((r.parent_indexed & r.training).sum()),
        parent_validation=int((r.parent_indexed & ~r.training).sum()),
        recommended_primitive_id=r.recommended_primitive_id,
        recommended_setting_index=r.recommended_setting_index,
        primitives=primitives,
        settings=settings,
        type_status={
            symbol: sum(f.setting.symbol == symbol for f in r.settings) for symbol in BRAVAIS
        },
    )
    write(OUT / f"{label}.json", record)
    np.savez_compressed(
        OUT / f"{label}.npz",
        points=q,
        observed=observed,
        training=r.training,
        parent_hkl=r.parent_hkl,
        parent_indexed=r.parent_indexed,
        primitive_retained=np.array([p.retained for p in r.primitives]),
        setting_hkl=np.array([f.hkl for f in r.settings]),
        setting_eligible=np.array([f.eligible for f in r.settings]),
        setting_basis=np.array([f.reciprocal_basis for f in r.settings]),
    )
    with (OUT / f"{label}-primitives.csv").open("w", newline="") as stream:
        fields = [
            "id",
            "index",
            "volume",
            "train",
            "validation",
            "excluded_train",
            "excluded_validation",
            "supported",
            "reason",
            "cell",
            "hnf",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(primitives)
    with (OUT / f"{label}-settings.csv").open("w", newline="") as stream:
        fields = [
            "row",
            "primitive_id",
            "symbol",
            "index",
            "strain",
            "volume",
            "cell",
            "parameters",
            "training_score",
            "converged",
            "status",
            "training",
            "validation",
            "transform",
            "recommended",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(settings)
    results[label] = record
    print(
        label,
        json.dumps(
            {k: record[k] for k in ["seconds", "parent_train", "parent_validation", "type_status"]}
        ),
        flush=True,
    )
    print("recommended", settings[r.recommended_setting_index], flush=True)
write(OUT / "access-log.json", access)
