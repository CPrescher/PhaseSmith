"""Frozen TIFF support study and realistic multi-peak background benchmark.

Training-only aperture advice; reserved angular blocks are reported without
tuning. No source cell, reduced reference intensities or MAR images are used.
"""

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.integration import IntegrationOptions, integrate_series
from phasesmith.experimental.integration_quality import assess_integration, assess_support
from phasesmith.experimental.prediction import ScanPredictions
from PIL import Image


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(a, b, keep):
    good = (
        keep & np.isfinite(a.values[:, :2]).all(axis=1) & np.isfinite(b.values[:, :2]).all(axis=1)
    )
    good &= a.values[:, 0] > 0
    r = (b.values[good, 0] - a.values[good, 0]) / a.values[good, 0]
    return dict(
        n=int(good.sum()),
        median_absolute=float(np.median(abs(r))) if len(r) else None,
        p90_absolute=float(np.quantile(abs(r), 0.9)) if len(r) else None,
        median_signed=float(np.median(r)) if len(r) else None,
    )


def synthetic():
    y, x = np.indices((512, 512))
    xy = np.array([(a, b) for b in range(24, 490, 32) for a in range(24, 490, 32)])
    data = 60 + 0.01 * x + 0.02 * y
    peak = np.zeros_like(data)
    for a, b in xy:
        peak += 300 * np.exp(-0.5 * (((x - a) / 2.4) ** 2 + ((y - b) / 2.0) ** 2))
    data += peak
    hkl = np.column_stack([np.arange(1, len(xy) + 1), np.zeros((len(xy), 2), int)])
    predictions = ScanPredictions(
        hkl, np.column_stack([xy, np.full(len(xy), 3.5)]), np.full(len(xy), 3), 1.0
    )
    bounds = np.column_stack([np.arange(7), np.arange(1, 8)])
    timings = []
    for model, r, gap in (("mean", 5, 0), ("mean", 9, 0), ("plane", 9, 3)):
        options = IntegrationOptions(
            pixel_radius=r,
            omega_half_width=0.4,
            background_model=model,
            background_gap=gap,
            background_margin=6,
        )
        previous = None
        for repeat in range(4):
            start = time.perf_counter()
            t = integrate_series(lambda _: data, bounds, predictions, options=options)
            seconds = time.perf_counter() - start
            if previous is not None:
                assert t.values.tobytes() == previous.values.tobytes()
            previous = t
            timings.append(dict(model=model, radius=r, gap=gap, repeat=repeat, seconds=seconds))
        exact = 300 * 2 * np.pi * 2.4 * 2.0
        # Infinite Gaussian normalisation is effectively exact at this sampling density.
        error = t.values[:, 0] / exact - 1
        timings.append(
            dict(
                model=model,
                radius=r,
                gap=gap,
                reflections=len(t),
                measurements=len(t.frame_indices),
                median_absolute_error=float(np.median(abs(error))),
                byte_exact_repeats=True,
            )
        )
    return timings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta = json.loads(args.geometry.read_text())
    calibration = meta["geometry_provenance"]
    assert sha(calibration["calibration_file"]) == calibration["calibration_sha256"]
    files = [Path(f) for f in meta["files"]]
    assert all(f.suffix.lower() == ".tif" for f in files)
    with Image.open(files[0]) as image:
        dtype = np.asarray(image).dtype
    storage_ceiling = float(np.iinfo(dtype).max) if dtype.kind in "iu" else None
    prior = json.loads((args.previous / "report.json").read_text())["measured"]["rows"][0]
    with np.load(args.previous / (prior["name"] + ".npz")) as f:
        old = dict(f)
    bounds = np.array(meta["omega_bounds"])
    np.testing.assert_array_equal(bounds, old["omega_bounds"])
    nearest = np.argmin(abs(old["predicted"][:, 2, None] - bounds.mean(axis=1)), axis=1)
    pred = ScanPredictions(old["hkl"], old["predicted"], nearest, 1.0)
    radii = (5, 7, 9, 11, 13, 15, 17)
    windows = [
        IntegrationOptions(
            pixel_radius=r,
            omega_half_width=2.5,
            background_model="plane",
            background_gap=3,
            background_margin=6,
        )
        for r in radii
    ]
    windows += [
        IntegrationOptions(
            pixel_radius=11,
            omega_half_width=2.5,
            background_model="plane",
            background_gap=g,
            background_margin=6,
        )
        for g in (1, 5)
    ]
    windows += [IntegrationOptions(pixel_radius=11, omega_half_width=2.5), IntegrationOptions()]
    windows += [
        IntegrationOptions(
            pixel_radius=11,
            omega_half_width=4.5,
            background_model="plane",
            background_gap=3,
            background_margin=6,
        )
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "protocol.json",
        dict(
            source_geometry=str(args.geometry.resolve()),
            geometry_sha256=sha(args.geometry),
            calibration=calibration,
            source_files=[dict(path=str(f), sha256=sha(f)) for f in files],
            native_sha256=sha(_core.__file__),
            script_sha256=sha(__file__),
            windows=[asdict(o) for o in windows],
            selection=(
                "Fixed previous unflagged I>5sigma population; "
                "advisory boundary flags do not hide window changes."
            ),
            split=(
                "All contributing frames must be within training or reserved blocks; "
                "10-frame blocks, every fourth reserved."
            ),
            advice=(
                "First radius with two consecutive steps: training median absolute "
                "change<=2%, p90<=10%, n>=30. Otherwise no recommendation."
            ),
            interpretation=(
                "Conditional finite-window detector counts, "
                "not corrected crystallographic intensities."
            ),
            no_automatic_rejection_or_intensity_correction=True,
            storage_ceiling=storage_ceiling,
            saturation_policy=(
                "flag stored ceiling; legacy baseline only keeps its original unspecified limit"
            ),
            support_assessment=dict(
                nominal="plane radius 11, half-width 2.5, gap 3, ring 6",
                controls="radii 13/15, gaps 1/5, angular half-width 4.5",
                threshold="each difference <=5%, baseline >5sigma, all flags/endpoints clear",
                status="descriptive follow-up after the pilot; not a new blind holdout validation",
            ),
        ),
    )
    benchmark = synthetic()
    masks = {}
    storage_counts = {}

    def load(i):
        with Image.open(files[i]) as im:
            a = np.asarray(im)[::-1].copy()
        if a.dtype != dtype:
            raise ValueError("source storage types differ")
        storage_counts[i] = int(np.count_nonzero(a == storage_ceiling)) if storage_ceiling else 0
        masks[i] = (a == 0) | ~np.isfinite(a)
        return a

    rows, tables = [], []
    names = (
        "hkl",
        "predicted",
        "grain_ids",
        "values",
        "flags",
        "observation_ids",
        "frame_indices",
        "boxes",
        "frame_values",
        "frame_flags",
        "omega_bounds",
    )
    for index, o in enumerate(windows):
        label = f"{index:02d}-{o.background_model}-r{o.pixel_radius}-g{o.background_gap}"
        start = time.perf_counter()
        t = integrate_series(
            load,
            bounds,
            pred,
            mask_loader=lambda i: masks.pop(i),
            options=o,
            saturation=None if index == 10 else storage_ceiling,
        )
        assert t.frame_hashes == tuple(prior["frame_hashes"])
        record = {n: getattr(t, n) for n in names}
        if t.background_models is not None:
            record["background_models"] = t.background_models
        np.savez_compressed(args.output / (label + ".npz"), **record)
        if index == 10:
            for n in names:
                assert record[n].tobytes() == old[n].tobytes(), n
        quality = assess_integration(t)
        row = dict(
            name=label,
            options=asdict(o),
            seconds=time.perf_counter() - start,
            unflagged=int(np.count_nonzero(t.flags == 0)),
            spatial_boundary=int(np.count_nonzero(t.flags & 2048)),
            poor_background=int(np.count_nonzero(t.flags & 4096)),
            saturation=int(np.count_nonzero(t.flags & 4)),
            angular_endpoint=int(np.count_nonzero(quality.reflection_issues & 64)),
        )
        rows.append(row)
        tables.append(t)
        print(label, row, flush=True)
    baseline = tables[10]
    strong = (baseline.flags == 0) & (baseline.values[:, 0] > 5 * baseline.values[:, 1])
    # Freeze one common population across all radius hypotheses; do not hide changes by flags.
    valid = strong.copy()
    for t in tables[: len(radii)]:
        valid &= ((t.flags & np.uint32(~2048 & 0xFFFFFFFF)) == 0) & np.isfinite(t.values[:, 0])
        valid &= t.values[:, 0] > 0
    counts = np.bincount(tables[0].observation_ids, minlength=len(baseline))
    reserved_frames = (np.arange(len(files)) // 10) % 4 == 0
    held_counts = np.bincount(
        tables[0].observation_ids,
        weights=reserved_frames[tables[0].frame_indices],
        minlength=len(baseline),
    )
    train = valid & (held_counts == 0) & (counts > 0)
    held = valid & (held_counts == counts) & (counts > 0)
    changes = []
    for a, b in pairwise(tables[: len(radii)]):
        changes.append(dict(training=metrics(a, b, train), heldout=metrics(a, b, held)))
    passes = [
        v["training"]["n"] >= 30
        and v["training"]["median_absolute"] <= 0.02
        and v["training"]["p90_absolute"] <= 0.1
        for v in changes
    ]
    chosen = next((i for i in range(len(passes) - 1) if passes[i] and passes[i + 1]), None)
    report = dict(
        frames=len(files),
        reflections=len(baseline),
        baseline_reproduced_byte_exact=True,
        baseline_strong=int(strong.sum()),
        common_training=int(train.sum()),
        common_heldout=int(held.sum()),
        rows=rows,
        radial_steps=changes,
        training_recommended_radius=None if chosen is None else radii[chosen],
        synthetic=benchmark,
        corrections_applied=[],
        storage_ceiling=storage_ceiling,
        ceiling_pixels_by_frame=[storage_counts[i] for i in range(len(files))],
    )
    nominal = tables[3]
    control_ids = (4, 5, 7, 8, 11)
    controls = [tables[i] for i in control_ids]
    assessment = assess_support(nominal, controls)
    report["tested_support_checks"] = dict(
        nominal=rows[3]["name"],
        alternates=[rows[i]["name"] for i in control_ids],
        max_relative_change=0.05,
        strong_sigma=5.0,
        passes=int(assessment.passes.sum()),
        interpretation="support checks only; full measurements and failed checks retained",
    )
    np.savez_compressed(
        args.output / "support-assessment.npz",
        relative_changes=assessment.relative_changes,
        reasons=assessment.reasons,
        passes=assessment.passes,
    )
    with (args.output / "support-checks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "h",
                "k",
                "l",
                "passes_tested_checks",
                "I",
                "sigma",
                "flags",
                *[f"relative_change_{i}" for i in range(len(controls))],
                *[f"reasons_{i}" for i in range(len(controls))],
            ]
        )
        for i in range(len(nominal)):
            writer.writerow(
                [
                    i,
                    *nominal.hkl[i],
                    int(assessment.passes[i]),
                    *nominal.values[i, :2],
                    int(nominal.flags[i]),
                    *assessment.relative_changes[i],
                    *assessment.reasons[i],
                ]
            )
    write(args.output / "report.json", report)
    with (args.output / "reflections.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "h",
                "k",
                "l",
                "training",
                "heldout",
                *[f"{row['name']}_{c}" for row in rows for c in ("I", "sigma", "flags")],
            ]
        )
        for i in range(len(baseline)):
            writer.writerow(
                [
                    i,
                    *baseline.hkl[i],
                    int(train[i]),
                    int(held[i]),
                    *[v for t in tables for v in (*t.values[i, :2], int(t.flags[i]))],
                ]
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in ("training", "heldout"):
        axes[0].plot(
            radii[:-1],
            [100 * v[name]["median_absolute"] if v[name]["n"] else np.nan for v in changes],
            "o-",
            label=name,
        )
    axes[0].axhline(2, color="grey", linestyle="--")
    axes[0].set(
        xlabel="Smaller radius in each +2-pixel step", ylabel="Median absolute intensity change (%)"
    )
    axes[0].legend()
    axes[1].plot(
        radii,
        [r["spatial_boundary"] for r in rows[: len(radii)]],
        "o-",
        label="Signal at box boundary",
    )
    axes[1].plot(
        radii,
        [r["poor_background"] for r in rows[: len(radii)]],
        "o-",
        label="Background needs review",
    )
    axes[1].set(xlabel="Signal radius (pixels)", ylabel="Predicted crossings")
    axes[1].legend()
    fig.suptitle("Fe₇C₃ TIFF: guarded local planes; fixed cell, calibration and angular support")
    fig.savefig(args.output / "aperture-check.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "synthetic")}, indent=2))


if __name__ == "__main__":
    main()
