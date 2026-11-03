"""Diagnostic interventions on frozen samples; oracle inputs never enter production."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from benchmark_3d_profiles import load_case
from phasesmith.single_crystal.profiles import fit_profiles


def main(root, output):
    output.mkdir(parents=True, exist_ok=True)
    obs, model, _native = load_case(root)
    oracle = Path("runs/learned-3d-profiles-dials-20261025")
    d = dict(np.load(oracle / "arrays.npz"))
    extra = dict(np.load(oracle / "measurement-diagnostics.npz"))
    match = dict(np.load(root / "matched.npz"))
    ni, di = match["pairs"].T
    selected = match["input_quality"] & match["strong"] & match["heldout"]
    target = d["intensity.prf.value"][di]
    records = {}
    arrays = {}

    def run(name, o=obs, m=model):
        fitted = fit_profiles(o, m)
        ratio = (fitted.values[ni, 0] - target) / np.maximum(abs(target), 1e-30)
        v = ratio[selected]
        records[name] = dict(
            n=len(v),
            median_relative=float(np.median(v)),
            p90_absolute_relative=float(np.quantile(abs(v), 0.9)),
        )
        arrays[name] = fitted.values
        print(name, records[name], flush=True)

    run("baseline")
    # External-centroid and external-background interventions diagnose, never train.
    centres = obs.centres.copy()
    centres[ni, :2] = extra["xyzobs.px.value"][di, :2] - 0.5
    zero_shifts = replace(model, shifts=np.zeros_like(model.shifts))
    run("oracle_spatial_centres", replace(obs, centres=centres), zero_shifts)
    bg = np.zeros(len(obs.centres))
    bg[ni] = extra["background.mean"][di]
    samples = obs.samples.copy()
    desired = bg[obs.summation.observation_ids]
    desired = np.repeat(desired, np.diff(obs.offsets))
    samples[:, 0] += samples[:, 2] - desired
    samples[:, 2] = desired
    fixed_bg = replace(obs, samples=samples, covariance=np.zeros_like(obs.covariance))
    run("oracle_constant_background", fixed_bg)
    run("oracle_centres_and_background", replace(fixed_bg, centres=centres), zero_shifts)
    for factor in (1.3, 1.6):
        opts = replace(model.options, step=(factor, factor, 0.25))
        run(f"spatial_width_{factor}", m=replace(model, options=opts))
    for factor in (1.3, 1.6):
        opts = replace(model.options, step=(1.0, 1.0, 0.25 * factor))
        run(f"rocking_width_{factor}", m=replace(model, options=opts))
    displacement = centres[ni, :2] - obs.centres[ni, :2]
    records["diagnostics"] = dict(
        median_centroid_displacement=np.median(displacement[selected], axis=0).tolist(),
        median_radial_centroid_displacement=float(
            np.median(np.linalg.norm(displacement[selected], axis=1))
        ),
    )
    (output / "interventions.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(output / "interventions.npz", **arrays)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.case, a.output)
