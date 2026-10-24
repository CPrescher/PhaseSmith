"""Opt-in measured-frame pilot. Freeze selection/settings before reading selected pixels.

Requires optional Fabio/Matplotlib for this external-data audit only. Raw inputs
and numerical records stay under an ignored, newly created run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith.experimental import spots_reference
from phasesmith.experimental.spots import SpotOptions, detect_spots


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot(path, image, table, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(image, [5, 99.5])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    colors = np.where(table.flags == 0, "#3ddc97", "#ff6868")
    ax.scatter(
        table.column("x"),
        table.column("y"),
        s=25,
        facecolors="none",
        edgecolors=colors,
        linewidths=0.7,
    )
    ax.set(title=title, xlabel="Detector column (px)", ylabel="Detector row (px)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--exclude-zero-padding", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source / "Fe7C3_SC"
    selected = []
    for prefix in ["Fe7C3_P1_C3_D1s", "Fe7C3_P2_C2_D1s", "Fe7C3_P7_C2_D1s"]:
        series = sorted(source.glob(prefix + "_*.tif"))
        if not series:
            raise ValueError(f"missing prescribed series {prefix}")
        selected.extend(series[i] for i in [0, len(series) // 2, len(series) - 1])
    for prefix in ["Fe7C3_P1_still", "Fe7C3_P1_h1", "Fe7C3_P7_h1"]:
        series = sorted(source.glob(prefix + "_*.tif"))
        if not series:
            raise ValueError(f"missing prescribed series {prefix}")
        selected.append(series[len(series) // 2])
    selected.append(source / "Fe7C3_P1_C1_D1w_001.tif")
    if args.all_frames:
        selected = sorted(source.glob("*.tif"))
    options = SpotOptions()
    protocol = dict(
        selected=[str(p) for p in selected],
        selection="all TIFF frames in Fe7C3_SC, sorted by filename"
        if args.all_frames
        else "first/middle/last of three narrow series; "
        "middle still/P1_h1/P7_h1; fixed P1_C1 wide frame; no outcome-dependent replacements",
        options=asdict(options),
        gain=1.0,
        read_noise=0.0,
        saturation=65535,
        ceiling_convention="conservative uint16 storage-ceiling flag; "
        "not detector full-well calibration",
        noise="uncalibrated digital-count approximation; no calibrated significance/error claim",
        mask="nonfinite plus zero-valued pixels; explicit padding sensitivity"
        if args.exclude_zero_padding
        else "only nonfinite; no supplied detector/DAC mask",
        inverse_inputs="TIFF pixels only; no ub, indexed txt, material, cell, SG or structure",
        failure_policy="count and retain every selected frame; never replace failures",
        validation="repeat complete native records exactly; "
        "independent extraction of all saved boxes; "
        "independent detection on two fixed 96x96 quarter-position crops",
        tolerance=dict(rtol=1e-12, atol=1e-8),
        expected_frames=len(selected),
        copy_inputs=not args.all_frames,
        crops="two fixed crops per frame"
        if not args.all_frames
        else "two fixed crops on the first frame of every filename series",
        claim="measured-image engineering/inspection pilot; "
        "not labelled peak recall/precision or I(hkl) validation",
    )
    save(args.output / "protocol.json", protocol)
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / "crates/phasesmith-py/Cargo.toml",
        root / "crates/phasesmith-py/src/spots.rs",
        root / "crates/phasesmith-spots/src/lib.rs",
        Path(__file__),
        *sorted((root / "python/phasesmith/experimental").glob("*.py")),
    ]
    manifest = {}
    for p in files:
        out = args.output / "source" / p.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        manifest[str(p.relative_to(root))] = sha(out)
    from phasesmith import _core

    manifest["native_extension_sha256"] = sha(Path(_core.__file__))
    save(args.output / "source-manifest.json", manifest)
    import fabio

    rows, examined_series = [], set()
    for index, original in enumerate(selected):
        case = args.output / f"frame-{index:02d}"
        case.mkdir()
        series_key = re.sub(r"_\d+$", "", original.stem)
        first_in_series = series_key not in examined_series
        examined_series.add(series_key)
        row = dict(index=index, file=original.name, series=series_key, status="failed")
        try:
            copied = original if args.all_frames else case / original.name
            if not args.all_frames:
                shutil.copy2(original, copied)
            row["sha256"] = sha(copied)
            if row["sha256"] != sha(original):
                raise ValueError("input copy checksum differs")
            with fabio.open(str(copied)) as frame:
                image = np.array(frame.data, dtype=np.float64, order="C", copy=True)
            mask = ~np.isfinite(image)
            if args.exclude_zero_padding:
                mask |= image == 0
            started = time.perf_counter()
            result = detect_spots(image, mask=mask, options=options, saturation=65535)
            elapsed = time.perf_counter() - started
            np.savez_compressed(
                case / "selected-spots.npz",
                boxes=result.boxes,
                values=result.values,
                flags=result.flags,
            )
            row.update(
                status="complete",
                spots=len(result),
                flagged=int(np.count_nonzero(result.flags)),
                rejected_small=result.rejected_small_components,
                time_s=elapsed,
                storage_ceiling_pixels=int(np.count_nonzero(image >= 65535)),
                masked_pixels=int(np.count_nonzero(mask)),
            )
            # Save primary records before any reference/visual assessment.
            save(case / "primary.json", row)
            repeated = detect_spots(image, mask=mask, options=options, saturation=65535)
            for field in ["boxes", "values", "flags"]:
                np.testing.assert_array_equal(getattr(result, field), getattr(repeated, field))
            reference = spots_reference.extract_spots(
                image, result.boxes, mask=mask, options=options, saturation=65535
            )
            np.testing.assert_array_equal(
                result.flags & np.uint32(~32 & 0xFFFFFFFF), reference.flags
            )
            np.testing.assert_allclose(result.values, reference.values, rtol=1e-12, atol=1e-8)
            for fraction in [0.25, 0.75] if first_in_series or not args.all_frames else []:
                y, x = int(image.shape[0] * fraction), int(image.shape[1] * fraction)
                crop = image[y - 48 : y + 48, x - 48 : x + 48]
                crop_mask = mask[y - 48 : y + 48, x - 48 : x + 48]
                actual = detect_spots(crop, mask=crop_mask, options=options, saturation=65535)
                ref = spots_reference.detect_spots(
                    crop, mask=crop_mask, options=options, saturation=65535
                )
                np.testing.assert_array_equal(actual.boxes, ref.boxes)
                np.testing.assert_array_equal(actual.flags, ref.flags)
                np.testing.assert_allclose(actual.values, ref.values, rtol=1e-12, atol=1e-8)
            row["verification"] = "passed"
            if row["sha256"] != sha(original):
                raise ValueError("original input checksum changed during processing")
            if first_in_series or not args.all_frames:
                plot(
                    case / "overlay.png",
                    image,
                    result,
                    original.name
                    + f"\n{len(result)} candidates; {row['flagged']} flagged · gain uncalibrated",
                )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            row.setdefault("verification", "failed")
        rows.append(row)
        save(case / "outcome.json", row)
        save(args.output / "outcomes.json", rows)
        print(json.dumps(row), flush=True)
    html = [
        "<!doctype html><meta charset='utf-8'><title>Measured detector spot pilot</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}</style>",
        f"<h1>Measured detector spot pilot</h1><p>{len(selected)} prescribed frames; "
        "original TIFFs preserved. "
        "No cell or structure enters detection. Gain/read noise are uncalibrated; "
        "counts are not I(hkl). Green: unflagged; red: quality flags. "
        "These flags do not identify physical phases or all overlaps.</p>",
    ]
    for row in rows:
        html.append(f"<h2>{row['file']}</h2><pre>{json.dumps(row, indent=2)}</pre>")
        if (args.output / f"frame-{row['index']:02d}" / "overlay.png").exists():
            html.append(f"<img src='frame-{row['index']:02d}/overlay.png'>")
    (args.output / "index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
