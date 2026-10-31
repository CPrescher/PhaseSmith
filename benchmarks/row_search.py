"""Realistic multigrain row grouping and exact frozen-artifact replay.

No original DAFi executable/source is accessed. --artifact reads only our frozen
coordinate NPZs and independent Rust grouping records; it never regenerates them.
"""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
from phasesmith.experimental.rows import RowSearchOptions, group_rows


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def identity(result):
    return dict(
        labels=digest(result.labels),
        groups=[digest(g) for g in result.groups],
        mean=digest(result.mean),
        scale=digest(result.scale),
    )


def frozen_replays(artifact, output):
    records = []
    names = {f.name for f in fields(RowSearchOptions)}
    manifest = json.loads((artifact / "evidence/input-manifest.json").read_text())
    for info in manifest:
        tag = info["tag"]
        for scope in ["all", "training"] if tag.startswith("dataset") else ["all"]:
            suffix = "all-adaptive-final" if scope == "all" else "training-adaptive"
            job = artifact / "evidence/jobs" / f"{tag}-{suffix}"
            expected = json.loads((job / "result.json").read_text())
            with np.load(artifact / "evidence/inputs" / tag / "points.npz") as z:
                q = z["points"][np.load(job / "input-rows.npy")]
            options = RowSearchOptions(
                **{k: v for k, v in expected["options"].items() if k in names}
            )
            start = time.perf_counter()
            actual = group_rows(q, options=options)
            wall = time.perf_counter() - start
            np.testing.assert_array_equal(actual.labels, expected["labels"])
            for a, b in zip(actual.groups, expected["groups"], strict=True):
                np.testing.assert_array_equal(a, b)
            for name in ("mean", "scale"):
                np.testing.assert_array_equal(
                    getattr(actual, name).view(np.uint64),
                    np.asarray(expected[name]).view(np.uint64),
                )
            for name, value in actual.diagnostics.items():
                if not name.endswith("seconds"):
                    assert value == expected[name], (tag, scope, name)
            assert actual.complete
            record = dict(
                tag=tag,
                scope=scope,
                n=len(q),
                groups=len(actual.groups),
                exact=True,
                wall_seconds=wall,
                compute_seconds=actual.diagnostics["compute_seconds"],
                previous_compute_seconds=expected["compute_seconds"],
                identity=identity(actual),
            )
            records.append(record)
            output.write_text(json.dumps(records, indent=2) + "\n")
            print(tag, scope, "EXACT", f"{wall:.4f}s", flush=True)
    assert len(records) == 32
    return records


def benchmark(repeats):
    rng = np.random.default_rng(33)
    parts = []
    for edges in ([4.3, 5.1, 7.7], [5.2, 6.2, 8.3]):
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        basis = np.diag(edges) @ rotation
        h = np.unique(rng.integers(-6, 7, (1400, 3)), axis=0)
        h = h[np.any(h != 0, axis=1)]
        parts.append(h @ np.linalg.inv(basis).T + rng.normal(0, 0.0005, h.shape))
    q = np.concatenate([*parts, rng.uniform(-1, 1, (600, 3))])
    options = RowSearchOptions()
    records, expected = [], None
    for workers in (1, 8):
        for repeat in range(repeats):
            start = time.perf_counter()
            result = group_rows(q, options=replace(options, workers=workers))
            wall = time.perf_counter() - start
            signature = identity(result)
            assert result.complete
            if expected is not None:
                assert signature == expected
            expected = signature
            records.append(
                dict(
                    workers=workers,
                    repeat=repeat + 1,
                    wall_seconds=wall,
                    compute_seconds=result.diagnostics["compute_seconds"],
                    groups=len(result.groups),
                    identity=signature,
                )
            )
    return dict(
        platform=platform.platform(),
        positions=len(q),
        options=asdict(options),
        exact_worker_and_repeat_results=True,
        runs=records,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be in [1,10]")
    if args.artifact:
        frozen_replays(args.artifact, args.output)
    else:
        result = benchmark(args.repeats)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
