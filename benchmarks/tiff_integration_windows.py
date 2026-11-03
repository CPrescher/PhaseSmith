"""Re-measure the preserved TIFF example with independently varied windows.

Run with the Dioptas image-reader environment. Frozen positional predictions,
calibration, exposure bounds and masks are reused; no cell or intensity fit is
performed. Results are support diagnostics, not corrected crystallographic I.
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
from phasesmith.experimental.integration import IntegrationOptions, integrate_series
from phasesmith.experimental.integration_quality import assess_integration
from phasesmith.experimental.prediction import ScanPredictions
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def comparison(a, b, population):
    keep = population & (a.flags == 0) & (b.flags == 0)
    keep &= np.isfinite(a.values[:, 0]) & np.isfinite(b.values[:, 0]) & (a.values[:, 0] != 0)
    relative = (b.values[keep, 0] - a.values[keep, 0]) / a.values[keep, 0]
    return dict(
        n=int(keep.sum()),
        median_signed_fraction=float(np.median(relative)) if len(relative) else None,
        median_absolute_fraction=float(np.median(abs(relative))) if len(relative) else None,
        p90_absolute_fraction=float(np.quantile(abs(relative), 0.9)) if len(relative) else None,
        changed_at_least_ten_percent=int(np.count_nonzero(abs(relative) >= 0.1)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    meta = json.loads(args.geometry.read_text())
    previous = json.loads((args.previous / "report.json").read_text())["measured"]["rows"]
    calibration = meta["geometry_provenance"]
    if sha(calibration["calibration_file"]) != calibration["calibration_sha256"]:
        raise ValueError("the preserved TIFF calibration changed")
    files = [Path(p) for p in meta["files"]]
    if any(p.suffix.lower() not in (".tif", ".tiff") for p in files):
        raise ValueError("this check accepts only the preserved TIFF series")
    with np.load(args.previous / (previous[0]["name"] + ".npz")) as z:
        frozen = dict(z)
    bounds = np.asarray(meta["omega_bounds"])
    np.testing.assert_array_equal(bounds, frozen["omega_bounds"])
    centres = bounds.mean(axis=1)
    nearest = np.argmin(abs(frozen["predicted"][:, 2, None] - centres), axis=1)
    predictions = ScanPredictions(frozen["hkl"], frozen["predicted"], nearest, 1.0)
    windows = [
        IntegrationOptions(pixel_radius=r, omega_half_width=w, background_margin=4)
        for r in (5, 7, 9)
        for w in (1.0, 1.5, 2.5)
    ]
    windows += [
        IntegrationOptions(pixel_radius=5, omega_half_width=1, background_margin=6),
        IntegrationOptions(pixel_radius=7, omega_half_width=1.5, background_margin=6),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "protocol.json",
        dict(
            schema="phasesmith.tiff-integration-window-check.v1",
            geometry_file=str(args.geometry.resolve()),
            geometry_sha256=sha(args.geometry),
            calibration=calibration,
            previous_measurements=str(args.previous.resolve()),
            previous_report_sha256=sha(args.previous / "report.json"),
            source_files=[dict(path=str(p), sha256=sha(p)) for p in files],
            native_sha256=sha(_core.__file__),
            predictions="fixed from previous position-only cell; no reindexing or fitting",
            loading="native TIFF pixels reversed vertically once, as in Dioptas",
            mask="zero padding and nonfinite pixels excluded; no calibrated seat/diamond mask",
            noise=dict(gain=1.0, read_noise=0.0, saturation=None),
            windows=[asdict(o) for o in windows],
            population=(
                "baseline unflagged I>5sigma; paired valid subsets and "
                "common-valid population both reported"
            ),
            comparison=(
                "signed finite-window counts; correlated differences are not significance tests"
            ),
            limitations=[
                "Changing the pixel radius also moves the local background annulus.",
                "Full intersecting exposures contribute; widths can select identical frames.",
                "No automatic window choice, rejection, extrapolation, or intensity corrections.",
                "Apparent stability does not establish completeness or absence of contamination.",
                "CrysAlis MAR data and the separate 0.414915 A calibration are outside this check.",
            ],
        ),
    )
    masks = {}

    def load(i):
        with Image.open(files[i]) as im:
            a = np.asarray(im)[::-1].copy()
        if a.shape != tuple(meta["shape"]):
            raise ValueError("TIFF detector shape changed")
        masks[i] = (a == 0) | ~np.isfinite(a)
        return a

    tables, rows = [], []
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
    for o in windows:
        label = f"r{o.pixel_radius}-w{o.omega_half_width:g}-b{o.background_margin}"
        started = time.perf_counter()
        t = integrate_series(
            load, bounds, predictions, mask_loader=lambda i: masks.pop(i), options=o
        )
        if t.frame_hashes != tuple(previous[0]["frame_hashes"]):
            raise ValueError("TIFF pixels or masks differ from the frozen example")
        if not tables:
            for name in names:
                if getattr(t, name).tobytes() != frozen[name].tobytes():
                    raise ValueError(f"baseline reproduction failed: {name}")
        if label == "r7-w1.5-b6":
            with np.load(args.previous / (previous[1]["name"] + ".npz")) as old:
                for name in names:
                    if getattr(t, name).tobytes() != old[name].tobytes():
                        raise ValueError(f"wider-window reproduction failed: {name}")
        np.savez_compressed(args.output / (label + ".npz"), **{n: getattr(t, n) for n in names})
        quality = assess_integration(t)
        rows.append(
            dict(
                name=label,
                options=asdict(o),
                seconds=time.perf_counter() - started,
                unflagged=int(np.count_nonzero(t.flags == 0)),
                endpoint_review=int(np.count_nonzero(quality.reflection_issues & 64)),
                frame_hashes=t.frame_hashes,
            )
        )
        tables.append(t)
        print(label, rows[-1]["seconds"], flush=True)
    base = tables[0]
    population = (base.flags == 0) & (base.values[:, 0] > 5 * base.values[:, 1])
    common = population.copy()
    for t in tables:
        common &= (t.flags == 0) & np.isfinite(t.values[:, 0])
    for row, t in zip(rows, tables, strict=True):
        row["paired_vs_baseline"] = comparison(base, t, population)
        row["common_vs_baseline"] = comparison(base, t, common)
    edges = []
    for ia, a in enumerate(windows[:9]):
        for ib, b in enumerate(windows[:9]):
            kind = None
            if b.pixel_radius == a.pixel_radius + 2 and b.omega_half_width == a.omega_half_width:
                kind = "pixel_radius"
            if b.pixel_radius == a.pixel_radius and (a.omega_half_width, b.omega_half_width) in (
                (1, 1.5),
                (1.5, 2.5),
            ):
                kind = "angular_half_width"
            if kind:
                edges.append(
                    dict(
                        change=kind,
                        source=rows[ia]["name"],
                        target=rows[ib]["name"],
                        common_population=comparison(tables[ia], tables[ib], common),
                    )
                )
    with (args.output / "reflections.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "reflection_id",
                "h",
                "k",
                "l",
                "baseline_strong",
                "all_windows_valid",
                *[f"{row['name']}_{c}" for row in rows for c in ("I", "sigma", "flags")],
            ]
        )
        for i in range(len(base)):
            writer.writerow(
                [
                    i,
                    *base.hkl[i],
                    int(population[i]),
                    int(common[i]),
                    *[v for t in tables for v in (*t.values[i, :2], int(t.flags[i]))],
                ]
            )
    report = dict(
        frames=len(files),
        reflections=len(base),
        baseline_strong=int(population.sum()),
        common_strong=int(common.sum()),
        baseline_byte_exact=True,
        previous_wider_byte_exact=True,
        rows=rows,
        independent_changes=edges,
        corrections_applied=[],
        active_example="Fe7C3_P1_C3_D1s TIFF / 0.3100 A",
    )
    ranked = np.flatnonzero(common)
    ranked = ranked[np.argsort(base.values[ranked, 0], kind="stable")]
    inspect_ids = ranked[[0, len(ranked) // 2, -1]].tolist() if len(ranked) else []
    report["inspection_selection"] = dict(
        rule="lowest, middle and highest baseline intensity among the fixed common population",
        reflection_ids=inspect_ids,
    )
    write(args.output / "report.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for j, r in enumerate((5, 7, 9)):
        axes[0].plot(
            (1, 1.5, 2.5),
            [
                100 * rows[3 * j + k]["common_vs_baseline"]["median_signed_fraction"]
                for k in range(3)
            ],
            "o-",
            label=f"radius {r} px",
        )
    axes[0].set(
        xlabel="Angular half-width (degrees)",
        ylabel="Median intensity change from baseline (%)",
        title=f"Same {common.sum()} strong, unflagged reflections",
    )
    axes[0].legend()
    for j, r in enumerate((5, 7, 9)):
        axes[1].plot(
            (1, 1.5, 2.5),
            [rows[3 * j + k]["endpoint_review"] for k in range(3)],
            "o-",
            label=f"radius {r} px",
        )
    axes[1].set(
        xlabel="Angular half-width (degrees)",
        ylabel="Crossings needing endpoint review",
        title=f"All {len(base)} predicted crossings",
    )
    axes[1].legend()
    fig.suptitle("Fe₇C₃ TIFF: fixed predictions and calibration; no corrections")
    fig.savefig(args.output / "window-check.png", dpi=160)
    plt.close(fig)
    from matplotlib.patches import Rectangle

    if inspect_ids:
        fig, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
        for (image_ax, curve_ax), i in zip(axes, inspect_ids, strict=True):
            frame = int(nearest[i])
            data = load(frame)
            masks.pop(frame)
            x, y, omega = base.predicted[i]
            x0, y0 = max(0, int(x) - 18), max(0, int(y) - 18)
            x1, y1 = min(data.shape[1], int(x) + 19), min(data.shape[0], int(y) + 19)
            image_ax.imshow(
                np.log1p(np.maximum(data[y0:y1, x0:x1], 0)),
                origin="lower",
                extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
                cmap="magma",
            )
            for ti, color in ((0, "cyan"), (6, "lime")):
                t = tables[ti]
                k = np.flatnonzero((t.observation_ids == i) & (t.frame_indices == frame))[0]
                bx0, by0, bx1, by1 = t.boxes[k]
                image_ax.add_patch(
                    Rectangle(
                        (bx0 - 0.5, by0 - 0.5),
                        bx1 - bx0,
                        by1 - by0,
                        fill=False,
                        edgecolor=color,
                        label=f"radius {t.options.pixel_radius} px",
                    )
                )
            image_ax.plot(x, y, "+", color="white")
            image_ax.set_title(f"Reflection {i}, exposure {frame + 1}; log(1 + raw counts)")
            image_ax.legend(fontsize=8)
            for ti in (2, 8):
                t = tables[ti]
                c = t.rocking_curve(i)
                curve_ax.errorbar(
                    c["omega"],
                    c["intensity"],
                    yerr=c["sigma"],
                    xerr=np.abs(np.diff(c["omega_bounds"], axis=1)).ravel() / 2,
                    fmt="o-",
                    label=f"radius {t.options.pixel_radius} px",
                )
            curve_ax.axvline(omega, color="grey", linestyle="--")
            curve_ax.set(
                xlabel="Omega (degrees)",
                ylabel="Net counts per exposure",
                title=f"HKL {tuple(map(int, base.hkl[i]))}; angular half-width 2.5°",
            )
            curve_ax.legend(fontsize=8)
        fig.suptitle(
            "TIFF inspection: baseline intensity ranks; boxes show signal support; no correction"
        )
        fig.savefig(args.output / "spot-and-rocking-inspection.png", dpi=160)
        plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "independent_changes")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
