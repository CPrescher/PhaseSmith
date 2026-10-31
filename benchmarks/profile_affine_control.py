"""Frozen-pixel test of local scattering coordinates; original targets retained."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from benchmark_3d_profiles import load_case
from phasesmith.single_crystal.profiles import fit_profiles, learn_profiles, scattering_transforms
from phasesmith.single_crystal.reciprocal import PanelGeometry

root = Path("runs/learned-3d-profiles-20261025-final")
obs, model, a = load_case(root)
protocol = json.loads((root / "protocol.json").read_text())
meta = protocol["geometry"]
g = PanelGeometry(np.array(meta["vectors"]), meta["wavelength"])
A, zeta = scattering_transforms(obs.centres[:, :2], g)
obs = replace(obs, planar_transforms=A)
q = np.load(root / "matched.npz")
i, j = q["pairs"].T
b = np.load("runs/learned-3d-profiles-dials-20261025/arrays.npz")
sel = q["heldout"] & q["strong"] & q["input_quality"]
target = b["intensity.prf.value"][j]
for cutoff in [0, 0.005]:
    opts = replace(model.options, step=(0.025, 0.025, 0.25), cutoff=cutoff)
    learned = learn_profiles(obs, a["training"], detector_shape=model.detector_shape, options=opts)
    fitted = fit_profiles(obs, learned)
    r = fitted.values[i, 0] / np.maximum(abs(target), 1e-30) - 1
    record = dict(
        n=int(sel.sum()),
        median_relative=float(np.median(r[sel])),
        p90_absolute_relative=float(np.quantile(abs(r[sel]), 0.9)),
        counts=learned.counts.tolist(),
    )
    print("cutoff", cutoff, record, flush=True)
    out = Path(f"runs/profile-bias-investigation-20261025/affine-{cutoff}")
    out.mkdir(exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(
        out / "profiles.npz",
        profiles=learned.profiles,
        shifts=learned.shifts,
        counts=learned.counts,
        reference_ids=learned.reference_ids,
        values=fitted.values,
        flags=fitted.flags,
        model_ids=fitted.model_ids,
        planar=A,
    )
