"""Serial/parallel position hunt: complete byte-exact records and wall-clock timing.

Run the unchanged build first with --legacy-api --workers 1 --records DIR.
Run the candidate with --compare DIR and a separate --records directory. Optional
--measured-source is the Fe7C3_SC directory; no indexed/structure files are read.
Use separate processes/PYTHONPATH for baseline and candidate extension builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.rotation import locate_series

FIELDS = ("boxes", "frame_ranges", "values", "flags", "observations", "omega_bounds")


def digest_array(h, array):
    h.update(str((array.shape, array.dtype)).encode())
    h.update(memoryview(np.ascontiguousarray(array)).cast("B"))


def assert_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        a, b = actual[name], expected[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert a.tobytes() == b.tobytes(), f"byte mismatch: {name}"


def cases(source, selected):
    for size, frames in [(512, 32), (2048, 16)]:
        name = f"synthetic-{frames}x{size}"
        if selected and name not in selected:
            continue
        rng = np.random.default_rng(20261025)
        image = rng.poisson(5, (size, size)).astype(np.uint16)
        for y in range(24, size - 24, 48):
            for x in range(24, size - 24, 48):
                image[y : y + 3, x : x + 3] += 100
        bounds = np.column_stack([np.arange(frames) * 0.5, (np.arange(frames) + 1) * 0.5])
        yield (
            name,
            lambda i, image=image: np.roll(image, i % 3, axis=1),
            lambda _, image=image: image == 0,
            bounds,
            {},
        )
    if source is None:
        return
    from PIL import Image

    for prefix in ["Fe7C3_P1_C3_D1s", "Fe7C3_P2_C2_D1s", "Fe7C3_P7_C2_D1s"]:
        if selected and prefix not in selected:
            continue
        files = sorted(source.glob(prefix + "_*.tif"))
        if not files:
            raise ValueError(f"missing measured scan: {prefix}")
        manifest, bounds = [], []
        for path in files:
            sidecar = next(
                p
                for p in (
                    path.with_name(path.name + ".txt"),
                    path.with_name(path.parent.name + path.name + ".txt"),
                )
                if p.exists()
            )
            # The reviewed Dioptas exports freeze the explicit sidecar convention.
            text = sidecar.read_text()
            fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
            fields = {k.strip(): v.strip() for k, v in fields.items()}
            start, width = float(fields["omega0"]), float(fields["omegaR"])
            bounds.append([start, start + width])
            manifest.append(
                dict(
                    file=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                )
            )
        cached = {}

        def load(i, files=files, cached=cached):
            with Image.open(files[i]) as frame:
                data = np.asarray(frame)[::-1].copy()
            cached[i] = (data == 0) | ~np.isfinite(data)
            return data

        yield (
            prefix,
            load,
            lambda i, cached=cached: cached.pop(i),
            np.asarray(bounds),
            {"sources": manifest},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--legacy-api", action="store_true")
    parser.add_argument("--measured-source", type=Path)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.records.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        platform=platform.platform(),
        processor=platform.processor(),
        native_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        seed=20261025,
        repeats=args.repeats,
        warmup="one full excluded run per configuration",
        equality="dtype, shape and bytes, including NaNs; all final arrays, hashes and frame spots",
        measured_mask="zero padding and nonfinite",
        synthetic_mask="fixed zero-valued pixels of the base synthetic image",
        noise="gain=1, read_noise=0, saturation=65535",
        rows=[],
    )
    for name, loader, mask_loader, bounds, provenance in cases(args.measured_source, args.cases):
        expected = None
        if args.compare:
            with np.load(args.compare / f"{name}.npz") as saved:
                expected = dict(saved)
        for workers in args.workers:
            if args.legacy_api and workers != 1:
                raise ValueError("legacy API only supports one worker")
            timings = []
            for repeat in range(args.repeats + 1):
                frame_hash = hashlib.sha256()

                def finished(i, table, frame_hash=frame_hash):
                    frame_hash.update(
                        str((i, table.rejected_small_components, table.noise_model)).encode()
                    )
                    for field in ("boxes", "values", "flags"):
                        digest_array(frame_hash, getattr(table, field))

                started = time.perf_counter()
                table = locate_series(
                    loader,
                    bounds,
                    mask_loader=mask_loader,
                    saturation=65535,
                    frame_finished=finished,
                    **({} if args.legacy_api else {"workers": workers}),
                )
                elapsed = time.perf_counter() - started
                actual = {field: getattr(table, field) for field in FIELDS}
                actual["frame_hashes"] = np.array(table.frame_hashes)
                actual["spot_digest"] = np.array(frame_hash.hexdigest())
                if repeat == 0 and args.compare:
                    np.savez(args.records / f"{name}-w{workers}.npz", **actual)
                if expected is None:
                    expected = {k: v.copy() for k, v in actual.items()}
                    np.savez(args.records / f"{name}.npz", **actual)
                else:
                    assert_exact(actual, expected)
                if repeat:
                    timings.append(elapsed)
            row = dict(
                case=name,
                workers=workers,
                frames=len(bounds),
                tracks=len(table),
                observations=len(table.observations),
                seconds=timings,
                median_s=float(np.median(timings)),
                exact=True,
                **provenance,
            )
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
