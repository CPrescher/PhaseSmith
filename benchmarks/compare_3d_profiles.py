"""Frozen-geometry learned-profile TIFF comparison with an isolated DIALS oracle.

prepare: source/cell protocol and native integration; compare: matched raw-count
summation and profile fields. No external fitted intensity enters training.
"""

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.integration import IntegrationOptions
from phasesmith.experimental.prediction import ScanPredictions, predict_scan
from phasesmith.experimental.profiles import (
    SCATTERING_VARIATIONS,
    ProfileOptions,
    fit_profiles,
    learn_profiles,
    prepare_profiles,
    rotation_factors,
    scattering_transforms,
)
from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors
from PIL import Image


def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(case, output, coordinates="detector", variations=False, pixel_radius=11):
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((case / "images/geometry.json").read_text())
    source = case / "dials-default/own-index--1/indexed.expt"
    crystal = json.loads(source.read_text())["crystal"][0]
    direct = np.array([crystal["real_space_" + s] for s in "abc"])
    basis = np.linalg.inv(direct).T
    geometry = PanelGeometry(np.array(meta["vectors"]), meta["wavelength"])
    height, width = meta["shape"]
    corners = np.array(
        [
            [-0.5, -0.5, 0],
            [width - 0.5, -0.5, 0],
            [-0.5, height - 0.5, 0],
            [width - 0.5, height - 0.5, 0],
        ]
    )
    qmax = float(np.linalg.norm(reciprocal_vectors(corners, geometry), axis=1).max()) * (1 + 1e-12)
    predicted = predict_scan(basis, geometry, meta["omega_bounds"], tuple(meta["shape"]), qmax=qmax)
    zeta = rotation_factors(predicted.values[:, :2], geometry)
    keep = abs(zeta) >= 0.05
    excluded = dict(
        ids=np.flatnonzero(~keep).tolist(),
        hkl=predicted.hkl[~keep].tolist(),
        zeta=zeta[~keep].tolist(),
    )
    predicted = ScanPredictions(
        predicted.hkl[keep], predicted.values[keep], predicted.frame_indices[keep], predicted.qmax
    )
    zeta = zeta[keep]
    scales = np.column_stack([np.ones((len(zeta), 2)), zeta])
    widths = np.minimum(10.0, (2.6 if variations else 1.625) / abs(zeta) + 0.5)
    training_frames = (np.arange(len(meta["files"])) // 20) % 4 != 0
    np.save(output / "training-frames.npy", training_frames)
    options = IntegrationOptions(
        pixel_radius=pixel_radius,
        omega_half_width=4.5,
        background_model="plane",
        background_gap=3,
        background_margin=6,
    )
    planar = None
    profile_options = ProfileOptions(shape=(19, 19, 13), step=(1.0, 1.0, 0.25))
    if coordinates == "scattering":
        planar, _ = scattering_transforms(predicted.values[:, :2], geometry)
        profile_options = ProfileOptions(
            shape=(19, 19, 13),
            step=(0.025, 0.025, 0.25),
            cutoff=0.0,
            reference_weighting="inverse_variance",
        )
    protocol = dict(
        schema="phasesmith.3d-profile-comparison.v1",
        case=str(case.resolve()),
        excluded_near_spindle=excluded,
        coordinates=coordinates,
        variation_factors=SCATTERING_VARIATIONS if variations else None,
        coordinate_model=(
            "first-order local scattering-plane Jacobian and signed zeta rocking coordinate"
            if coordinates == "scattering"
            else "detector x/y and signed zeta rocking coordinate"
        ),
        common_crystal=str(source.resolve()),
        common_crystal_sha256=sha(source),
        common_crystal_basis=basis.tolist(),
        calibration=meta["geometry_provenance"],
        geometry=meta,
        options=asdict(options),
        profile_options=asdict(profile_options),
        raw_tiff_files=[dict(file=f, sha256=sha(f)) for f in meta["files"]],
        heldout=(
            "every fourth 20-frame block reserved for profile validation; common geometry "
            "predates this intensity split"
        ),
        training=(
            "all exposures of a reference shoebox must be training frames; native "
            "strong/isolated/completeness gates apply"
        ),
        native_sha256=sha(_core.__file__),
        script_sha256=sha(__file__),
        conventions=(
            "raw detector counts; no LP/QE/scaling; compare DIALS intensity.prf.value and "
            "intensity.sum.value, not scaled intensity"
        ),
        primary_population=(
            "heldout, matched HKL/crossing, finite, learned-foreground input-quality flags "
            "clear, both observed masses>=0.99; no profile-residual exclusion; rectangular "
            "sum flags retained"
        ),
        diagnostic_gate=(
            ">=30 common strong heldout pairs: absolute median relative bias<=0.10, p90 "
            "absolute relative difference<=0.30, Pearson correlation>=0.95"
        ),
        validation_reused_for_development=True,
        no_intensity_rescaling=True,
        centres=(
            "frozen common crystal predictions plus training-local median offsets; no "
            "target-by-target intensity peak search"
        ),
    )
    write(output / "protocol.json", protocol)
    # Freeze the independent implementation inputs, not only a dirty Git revision.
    snapshot = output / "source-snapshot"
    for pattern in (
        "crates/phasesmith-spots/src/*.rs",
        "crates/phasesmith-py/src/*.rs",
        "python/phasesmith/experimental/*.py",
        "benchmarks/compare_3d_profiles.py",
        "benchmarks/benchmark_3d_profiles.py",
    ):
        for path in Path(".").glob(pattern):
            target = snapshot / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    protocol["source_sha256"] = {
        str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()
    }
    write(output / "protocol.json", protocol)
    masks = {}

    def load(i):
        with Image.open(meta["files"][i]) as image:
            a = np.asarray(image)[::-1].copy()
        cache = np.load(case / "images" / f"frame_{i + 1:04d}.npy")
        if not np.array_equal(a, cache):
            raise ValueError("TIFF and external pixel adapter differ")
        masks[i] = (a == 0) | ~np.isfinite(a)
        return a

    start = time.perf_counter()
    observations = prepare_profiles(
        load,
        meta["omega_bounds"],
        predicted,
        mask_loader=lambda i: masks.pop(i),
        options=options,
        saturation=65535,
        scales=scales,
        planar_transforms=planar,
        angular_half_widths=widths,
        progress=lambda d, t: print("prepare", d, t, flush=True) if d % 20 == 0 else None,
    )
    prepared_seconds = time.perf_counter() - start
    t = observations.summation
    ng = len(t.observation_ids)
    train_group = training_frames[t.frame_indices]
    group_count = np.bincount(t.observation_ids, minlength=len(t))
    count_training = np.bincount(t.observation_ids, weights=train_group, minlength=len(t))
    training = (count_training == group_count) & (group_count > 0)
    heldout = (count_training == 0) & (group_count > 0)
    start = time.perf_counter()
    model = learn_profiles(
        observations, training, detector_shape=tuple(meta["shape"]), options=profile_options
    )
    learn_seconds = time.perf_counter() - start
    print("references", model.counts.tolist(), flush=True)
    start = time.perf_counter()
    fit = fit_profiles(
        observations, model, variation_factors=SCATTERING_VARIATIONS if variations else None
    )
    fit_seconds = time.perf_counter() - start
    assert not np.any(heldout[model.reference_ids])
    np.savez_compressed(
        output / "model.npz",
        profiles=model.profiles,
        counts=model.counts,
        shifts=model.shifts,
        reference_ids=model.reference_ids,
    )
    extra = (
        {}
        if fit.coefficients is None
        else dict(coefficients=fit.coefficients, coefficient_covariance=fit.coefficient_covariance)
    )
    np.savez_compressed(
        output / "native.npz",
        **extra,
        hkl=t.hkl,
        predicted=t.predicted,
        summed=t.values,
        zeta=zeta,
        angular_half_widths=widths,
        sum_flags=t.flags,
        profile=fit.values,
        profile_flags=fit.flags,
        model_ids=fit.model_ids,
        reference_membership=fit.reference_membership,
        training=training,
        heldout=heldout,
        boxes=t.boxes,
        observation_ids=t.observation_ids,
        frame_indices=t.frame_indices,
        frame_values=t.frame_values,
        frame_flags=t.frame_flags,
        omega_bounds=t.omega_bounds,
    )
    # Keep reproducible prepared samples outside the package and standard test suite.
    np.savez_compressed(
        output / "observations.npz",
        samples=observations.samples,
        valid=observations.valid,
        offsets=observations.offsets,
        covariance=observations.covariance,
        centres=observations.centres,
        scales=observations.scales,
        planar_transforms=observations.planar_transforms,
        background_models=t.background_models,
    )
    summary = dict(
        predictions=len(t),
        groups=ng,
        samples=len(observations.samples),
        training=int(training.sum()),
        heldout=int(heldout.sum()),
        references=model.counts.tolist(),
        reference_ids=model.reference_ids.tolist(),
        preparation_seconds=prepared_seconds,
        learning_seconds=learn_seconds,
        fitting_seconds=fit_seconds,
        fingerprint=observations.fingerprint(),
        frame_hashes=t.frame_hashes,
        model_volume_sums=model.profiles.sum(axis=1).tolist(),
        profile_unflagged=int(np.count_nonzero(fit.flags == 0)),
        summation_unflagged=int(np.count_nonzero(t.flags == 0)),
    )
    write(output / "native-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def stats(a, b, mask):
    ids = np.flatnonzero(mask & np.isfinite(a) & np.isfinite(b) & (b != 0))
    if not len(ids):
        return dict(n=0)
    x, y = a[ids], b[ids]
    r = (x - y) / np.abs(y)
    return dict(
        n=len(ids),
        median_relative=float(np.median(r)),
        median_absolute_relative=float(np.median(abs(r))),
        p90_absolute_relative=float(np.quantile(abs(r), 0.9)),
        pearson=float(np.corrcoef(x, y)[0, 1])
        if len(ids) > 1 and np.std(x) > 0 and np.std(y) > 0
        else None,
    )


def compare(output, dials):
    with np.load(output / "native.npz") as f:
        a = dict(f)
    with np.load(dials / "arrays.npz") as f:
        b = dict(f)
    by_hkl = {}
    for i, h in enumerate(b["miller_index"]):
        by_hkl.setdefault(tuple(h), []).append(i)
    angles = np.rad2deg(b["xyzcal.mm"][:, 2])
    pairs = []
    unmatched = []
    for i, (h, xyz) in enumerate(zip(a["hkl"], a["predicted"], strict=True)):
        candidates = [j for j in by_hkl.get(tuple(h), []) if abs(angles[j] - xyz[2]) < 1e-6]
        if len(candidates) == 1:
            pairs.append((i, candidates[0]))
        else:
            unmatched.append(i)
    pairs = np.array(pairs, np.int64).reshape(-1, 2)
    i, j = pairs.T
    delta = np.column_stack(
        [a["predicted"][i, :2] - (b["xyzcal.px"][j, :2] - 0.5), a["predicted"][i, 2] - angles[j]]
    )
    if not np.all(abs(delta) < 1e-7):
        raise ValueError("geometry/half-pixel/rotation mismatch")
    hard = 1 | 2 | 4 | 8 | 16 | 256 | 1024 | 4096 | 524288
    base = ((a["profile_flags"][i] & hard) == 0) & ~b["quality.overloaded"][j]
    base &= (
        b["quality.integrated_prf"][j] & (b["partiality"][j] >= 0.99) & (a["profile"][i, 2] >= 0.99)
    )
    valid = base & np.isfinite(a["profile"][i, :2]).all(axis=1) & (a["profile"][i, 1] > 0)
    # Strength selection uses the pre-profile native summed measurement, not fitted agreement.
    strong = (a["summed"][i, 0] > 5 * a["summed"][i, 1]) & (b["intensity.prf.value"][j] > 0)
    heldout = a["heldout"][i]
    results = {}
    for name, population in [
        ("all_input_quality", valid),
        ("training_strong", valid & strong & a["training"][i]),
        ("heldout_strong", valid & strong & heldout),
        ("heldout_profile_quality", valid & strong & heldout & (a["profile_flags"][i] == 0)),
    ]:
        results[name] = stats(a["profile"][i, 0], b["intensity.prf.value"][j], population)
    sums = (
        base & b["quality.integrated_sum"][j] & strong & heldout & (b["intensity.sum.value"][j] > 0)
    )
    results["heldout_summation"] = stats(a["summed"][i, 0], b["intensity.sum.value"][j], sums)
    primary = results["heldout_strong"]
    passes = (
        primary["n"] >= 30
        and abs(primary["median_relative"]) <= 0.1
        and primary["p90_absolute_relative"] <= 0.3
        and primary["pearson"] is not None
        and primary["pearson"] >= 0.95
    )
    record = dict(
        matched=len(pairs),
        native_unmatched=len(unmatched),
        dials_predictions=len(b["miller_index"]),
        max_prediction_error=np.max(abs(delta), axis=0).tolist(),
        results=results,
        predeclared_diagnostic_gate=bool(passes),
        scale_fitted_or_applied=False,
        caveat=(
            "Different native/DIALS profile and background models; comparison is not "
            "absolute intensity truth."
        ),
    )
    write(output / "comparison.json", record)
    np.savez_compressed(
        output / "matched.npz",
        pairs=pairs,
        prediction_differences=delta,
        input_quality=valid,
        strong=strong,
        heldout=heldout,
    )
    with (output / "matched-intensities.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "native_id",
                "dials_id",
                "h",
                "k",
                "l",
                "heldout",
                "input_quality",
                "strong",
                "I_native_profile",
                "sigma_native_profile",
                "I_dials_profile",
                "sigma_dials_profile",
                "I_native_sum",
                "I_dials_sum",
                "native_flags",
                "dials_flags",
                "native_mass",
                "dials_partiality",
            ]
        )
        for k, (ni, dj) in enumerate(pairs):
            w.writerow(
                [
                    ni,
                    dj,
                    *a["hkl"][ni],
                    int(heldout[k]),
                    int(valid[k]),
                    int(strong[k]),
                    *a["profile"][ni, :2],
                    b["intensity.prf.value"][dj],
                    np.sqrt(b["intensity.prf.variance"][dj])
                    if b["intensity.prf.variance"][dj] >= 0
                    else np.nan,
                    a["summed"][ni, 0],
                    b["intensity.sum.value"][dj],
                    int(a["profile_flags"][ni]),
                    int(b["flags"][dj]),
                    a["profile"][ni, 2],
                    b["partiality"][dj],
                ]
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = valid & strong & heldout
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].scatter(b["intensity.prf.value"][j[sel]], a["profile"][i[sel], 0], s=10, alpha=0.6)
    axes[0].set(
        xscale="symlog",
        yscale="symlog",
        xlabel="DIALS profile intensity (raw counts)",
        ylabel="PhaseSmith profile intensity (raw counts)",
    )
    lim = axes[0].get_xlim()
    axes[0].plot(lim, lim, color="grey", linestyle="--")
    axes[1].scatter(
        b["intensity.prf.value"][j[sel]],
        (a["profile"][i[sel], 0] / b["intensity.prf.value"][j[sel]] - 1) * 100,
        s=10,
        alpha=0.6,
    )
    axes[1].set(xscale="log", xlabel="DIALS profile intensity", ylabel="Relative difference (%)")
    axes[1].axhline(0, color="grey", linestyle="--")
    fig.suptitle(
        f"Fe₇C₃ TIFF: {int(sel.sum())} reserved strong pairs; fixed common crystal; no fitted scale"
    )
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--case", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--coordinates", choices=("detector", "scattering"), default="detector")
    a.add_argument("--variations", action="store_true")
    a.add_argument("--pixel-radius", type=int, default=11)
    b = sub.add_parser("compare")
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--dials", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.case, args.output, args.coordinates, args.variations, args.pixel_radius)
    else:
        compare(args.output, args.dials)
