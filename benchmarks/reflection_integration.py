"""Full multi-reflection grid benchmark and opt-in measured Fe7C3 example.

Measured input uses the preserved position-only comparison's native candidate,
selected by training support before intensity access. No reference structure or
published cell enters the inverse. New output directories only.
"""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.single_crystal.integration import IntegrationOptions, integrate_series
from phasesmith.single_crystal.prediction import ScanPredictions, predict_scan, refine_crystal
from phasesmith.single_crystal.reciprocal import PanelGeometry, reciprocal_vectors


def save_table(path, t):
    np.savez_compressed(
        path,
        hkl=t.hkl,
        predicted=t.predicted,
        grain_ids=t.grain_ids,
        values=t.values,
        flags=t.flags,
        observation_ids=t.observation_ids,
        frame_indices=t.frame_indices,
        boxes=t.boxes,
        frame_values=t.frame_values,
        frame_flags=t.frame_flags,
        omega_bounds=t.omega_bounds,
    )


def synthetic(output):
    rng = np.random.default_rng(20261025)
    image = rng.poisson(10, (512, 512)).astype(float)
    xy = np.array([(x, y) for y in range(20, 500, 24) for x in range(20, 500, 24)])
    hkl = np.column_stack([np.arange(1, len(xy) + 1), np.zeros((len(xy), 2), int)])
    p = ScanPredictions(
        hkl, np.column_stack([xy, np.full(len(xy), 4.0)]), np.full(len(xy), 4, np.int64), 1.0
    )
    bounds = np.column_stack([np.arange(8), np.arange(1, 9)])
    rows = []
    baseline = None
    for _repeat in range(4):
        started = time.perf_counter()
        t = integrate_series(lambda _: image, bounds, p)
        seconds = time.perf_counter() - started
        if baseline is not None:
            assert t.values.tobytes() == baseline.values.tobytes()
            assert t.frame_values.tobytes() == baseline.frame_values.tobytes()
        baseline = t
        rows.append(seconds)
    return dict(
        reflections=len(t),
        observations=len(t.frame_values),
        shape=list(image.shape),
        frames=8,
        warmup_seconds=rows[0],
        seconds=rows[1:],
        median_seconds=float(np.median(rows[1:])),
        byte_exact_repeats=True,
        options=asdict(t.options),
    )


def measured(source, output):
    from PIL import Image

    meta_path = source / "images/geometry.json"
    meta = json.loads(meta_path.read_text())
    choices = []
    for sign in (-1, 1):
        folder = source / "native" / f"common-index-{sign:+d}"
        report = json.loads((folder / "result.json").read_text())
        for i, c in enumerate(report["candidates"]):
            if c["accepted"]:
                choices.append((c["training_count"], -c["volume"], sign, i, c, folder))
    _, _, sign, index, candidate, folder = max(choices, key=lambda x: x[:4])
    with np.load(folder / "arrays.npz") as f:
        data = dict(f)
    with np.load(source / "native/positions.npz") as f:
        observed = f["xyz"][data["source_ids"]]
    basis = np.linalg.inv(candidate["basis"]).T
    hkl = np.rint(data["q"] @ np.asarray(candidate["basis"]).T).astype(np.int64)
    eligible = data["labels"] == index + 1
    vectors = np.array(meta["vectors"])
    vectors[-1] = [sign, 0, 0]
    geometry = PanelGeometry(vectors, meta["wavelength"])
    fit = refine_crystal(
        observed, hkl, basis, geometry, training=data["training"], eligible=eligible
    )
    shape = tuple(map(int, meta["shape"]))
    height, width = shape
    corners = np.array(
        [
            [-0.5, -0.5, 0],
            [width - 0.5, -0.5, 0],
            [-0.5, height - 0.5, 0],
            [width - 0.5, height - 0.5, 0],
        ]
    )
    qmax = float(np.linalg.norm(reciprocal_vectors(corners, geometry), axis=1).max())
    predictions = predict_scan(
        fit.reciprocal_basis, geometry, meta["omega_bounds"], shape, qmax=qmax
    )
    protocol = dict(
        candidate_source=str(folder),
        candidate_index=index,
        axis_sign=sign,
        candidate_selection=(
            "accepted model with largest training count; smaller volume "
            "breaks ties; before intensity access"
        ),
        position_only_refinement=fit.metrics,
        source_geometry_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        inverse_priors="no supplied material, structure, published cell or reflection intensities",
        windows=[
            dict(pixel_radius=5, omega_half_width=1.0, background_margin=4),
            dict(pixel_radius=7, omega_half_width=1.5, background_margin=6),
        ],
        mask=(
            "explicit zero-padding and nonfinite exclusion; "
            "no calibrated seat mask available in this frozen example"
        ),
        noise="gain=1, read_noise=0 variance floor=1, no assumed detector saturation",
        interpretation="measured workflow and support sensitivity, not calibrated intensity truth",
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    masks = {}

    def load(i):
        with Image.open(meta["files"][i]) as f:
            image = np.asarray(f)[::-1].copy()
        masks[i] = (image == 0) | ~np.isfinite(image)
        return image

    tables = []
    rows = []
    for window in protocol["windows"]:
        options = IntegrationOptions(**window)
        started = time.perf_counter()
        t = integrate_series(
            load,
            meta["omega_bounds"],
            predictions,
            mask_loader=lambda i: masks.pop(i),
            options=options,
            progress=lambda done, total: (
                print(f"{done}/{total}", flush=True) if done % 20 == 0 else None
            ),
        )
        seconds = time.perf_counter() - started
        name = f"radius-{options.pixel_radius}-omega-{options.omega_half_width}"
        save_table(output / (name + ".npz"), t)
        rows.append(
            dict(
                name=name,
                seconds=seconds,
                reflections=len(t),
                observations=len(t.frame_values),
                unflagged=int(np.count_nonzero(t.flags == 0)),
                finite=int(np.isfinite(t.values[:, 0]).sum()),
                positive_3sigma_unflagged=int(
                    np.count_nonzero((t.flags == 0) & (t.values[:, 0] > 3 * t.values[:, 1]))
                ),
                frame_hashes=t.frame_hashes,
                options=asdict(options),
            )
        )
        tables.append(t)
    a, b = tables
    common = (
        (a.flags == 0)
        & (b.flags == 0)
        & np.isfinite(a.values[:, 0])
        & np.isfinite(b.values[:, 0])
        & (a.values[:, 0] > 5 * a.values[:, 1])
    )
    relative = (b.values[common, 0] - a.values[common, 0]) / a.values[common, 0]
    sensitivity = dict(
        common_strong_unflagged=int(common.sum()),
        median_signed_fraction=float(np.median(relative)) if len(relative) else None,
        median_absolute_fraction=float(np.median(abs(relative))) if len(relative) else None,
        p90_absolute_fraction=float(np.quantile(abs(relative), 0.9)) if len(relative) else None,
    )
    # Plot fixed examples chosen by baseline intensity only, with all contributions visible.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = np.flatnonzero((a.flags == 0) & np.isfinite(a.values[:, 0]))
    ids = ids[np.argsort(a.values[ids, 0])[-6:][::-1]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, i in zip(axes.flat, ids, strict=False):
        for t, label in [(a, "5 px / ±1°"), (b, "7 px / ±1.5°")]:
            curve = t.rocking_curve(int(i))
            ax.errorbar(
                curve["omega"],
                curve["intensity"],
                yerr=curve["sigma"],
                xerr=np.abs(np.diff(curve["omega_bounds"], axis=1)).ravel() / 2,
                marker="o",
                capsize=2,
                label=label,
            )
        ax.axvline(a.predicted[i, 2], color="grey", linestyle="--")
        ax.set_title(f"({' '.join(map(str, map(int, a.hkl[i])))}) · I={a.values[i, 0]:.0f}")
        ax.set_xlabel("Omega (degrees)")
        ax.set_ylabel("Net counts / exposure")
        ax.legend(fontsize=7)
    fig.suptitle("Fe7C3 P1: measured rocking curves and finite-window sensitivity")
    fig.savefig(output / "rocking-curves.png", dpi=160)
    plt.close(fig)
    return dict(protocol=protocol, rows=rows, sensitivity=sensitivity)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--measured", type=Path, help="Preserved Fe7C3 position-comparison case folder")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        platform=platform.platform(),
        native_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        synthetic=synthetic(args.output),
    )
    if args.measured:
        report["measured"] = measured(args.measured, args.output)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "measured"}, indent=2))


if __name__ == "__main__":
    main()
