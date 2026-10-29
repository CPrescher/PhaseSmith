"""Deterministic multi-reflection QC benchmark and preserved Fe7C3 example.

Reports over existing intensity records only; no source structure or calibration
is introduced into indexing or intensity integration by this command.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith.experimental.integration import IntegrationOptions, ReflectionTable
from phasesmith.experimental.integration_quality import assess_integration


def synthetic():
    nf, n = 200, 10_000
    centres = np.repeat(np.arange(nf), 50)
    pairs = [(f, np.flatnonzero(abs(centres - f) <= 2)) for f in range(nf)]
    frames = np.concatenate([np.full(len(ids), f, np.int64) for f, ids in pairs])
    ids = np.concatenate([ids for _, ids in pairs])
    m = len(ids)
    fv = np.zeros((m, 11))
    fv[:, :2] = 12
    fv[:, 2] = 200 * np.exp(-0.5 * ((frames - centres[ids]) / 0.8) ** 2)
    fv[:, 3] = 4
    fv[:, 4] = 10
    fv[:, 9] = 9
    fv[:, 10] = 20
    values = np.zeros((n, 6))
    values[:, 0] = np.bincount(ids, weights=fv[:, 2], minlength=n)
    flags = np.zeros(m, np.uint32)
    flags[::999] = 2
    table = ReflectionTable(
        np.column_stack([np.arange(1, n + 1), np.zeros((n, 2), int)]),
        np.column_stack([np.full((n, 2), 12), centres + 0.5]),
        np.zeros(n, np.int64),
        values,
        np.zeros(n, np.uint32),
        ids,
        frames,
        np.tile([11, 11, 14, 14], (m, 1)),
        fv,
        flags,
        np.column_stack([np.arange(nf), np.arange(1, nf + 1)]),
        (),
        IntegrationOptions(),
        "synthetic independent variance",
        (1.0, 0.0, None),
    )
    timings = []
    previous = None
    for _ in range(4):
        started = time.perf_counter()
        report = assess_integration(table)
        timings.append(time.perf_counter() - started)
        if previous is not None:
            assert report.frame_values.tobytes() == previous.frame_values.tobytes()
            assert report.reflection_values.tobytes() == previous.reflection_values.tobytes()
        previous = report
    return dict(
        frames=nf,
        reflections=n,
        measurements=m,
        warmup_seconds=timings[0],
        seconds=timings[1:],
        median_seconds=float(np.median(timings[1:])),
        byte_exact_repeats=True,
    )


def load_recorded(folder, row):
    with np.load(folder / (row["name"] + ".npz")) as z:
        arrays = dict(z)
    return ReflectionTable(
        **arrays,
        frame_hashes=tuple(row["frame_hashes"]),
        options=IntegrationOptions(**row["options"]),
        noise_model="poisson_plus_read_noise_with_variance_floor",
        noise_parameters=(1.0, 0.0, None),
    )


def measured(folder, output):
    original = json.loads((folder / "report.json").read_text())
    a, b = [load_recorded(folder, row) for row in original["measured"]["rows"]]
    before = a.values.tobytes()
    started = time.perf_counter()
    report = assess_integration(a, comparison=b)
    seconds = time.perf_counter() - started
    assert a.values.tobytes() == before
    np.savez_compressed(
        output / "quality.npz",
        frame_values=report.frame_values,
        frame_issues=report.frame_issues,
        reflection_values=report.reflection_values,
        reflection_issues=report.reflection_issues,
    )
    lines = [",".join(["frame_index", *report.columns, "issues"])]
    for i, (v, f) in enumerate(zip(report.frame_values, report.frame_issues, strict=True)):
        lines.append(",".join(map(str, [i, *v, int(f)])))
    (output / "quality-frames.csv").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(report.frame_values)) + 1
    for ax, name, label in zip(
        axes.flat,
        (
            "masked_pixel_fraction",
            "median_background",
            "median_position_offset",
            "n_window_changed",
        ),
        (
            "Excluded signal pixel fraction",
            "Median local background / pixel",
            "Strong spot offset (pixels)",
            "Reflections sensitive to wider windows",
        ),
        strict=True,
    ):
        ax.plot(x, report.column(name), "-o", markersize=3)
        ax.set_xlabel("Exposure index")
        ax.set_ylabel(label)
    fig.suptitle("Fe₇C₃ P1: image-by-image integration review — no intensity correction applied")
    fig.savefig(output / "image-quality.png", dpi=160)
    plt.close(fig)
    return dict(
        source=str(folder),
        source_report_sha256=hashlib.sha256((folder / "report.json").read_bytes()).hexdigest(),
        seconds=seconds,
        frames=len(report.frame_values),
        frames_needing_review=int(np.count_nonzero(report.frame_issues)),
        window_sensitive_reflections=int(np.count_nonzero(report.reflection_issues & 256)),
        endpoint_review_reflections=int(np.count_nonzero(report.reflection_issues & 64)),
        thresholds=asdict(report.options),
        intensities_unchanged=True,
        calibration=(
            "preserved 0.3100 A PVT-EOS experiment geometry; "
            "no older CrysAlis calibration substituted"
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recorded", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(synthetic=synthetic())
    if args.recorded:
        result["measured"] = measured(args.recorded, args.output)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
