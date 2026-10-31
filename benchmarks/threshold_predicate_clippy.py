"""Compare exact spot records and realistic detector timings across native builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def worker(site, supplied):
    sys.path.insert(0, str(site))
    import numpy as np
    from phasesmith import _core
    from phasesmith.experimental.spots import SpotOptions, detect_spots

    assert Path(_core.__file__).resolve().is_relative_to(site.resolve())
    rng = np.random.default_rng(20261031)
    image = rng.poisson(5, (768, 1024)).astype(float)
    for y in range(32, 736, 48):
        for x in range(32, 992, 48):
            image[y - 1 : y + 2, x - 1 : x + 2] += 100
    mask = rng.random(image.shape) < 0.001
    image[mask] = np.nan
    variance = np.full(image.shape, 4.5) if supplied else None
    options = SpotOptions(threshold_sigma=4.5)
    durations = []
    digests = []
    for i in range(9):
        start = time.perf_counter()
        result = detect_spots(
            image, mask=mask, variance=variance, gain=1.7, read_noise=0.5, options=options
        )
        elapsed = time.perf_counter() - start
        h = hashlib.sha256()
        for a in [result.values, result.boxes, result.flags]:
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
        h.update(str(result.rejected_small_components).encode())
        digests.append(h.hexdigest())
        if i:
            durations.append(elapsed)
    assert len(result) > 100 and len(set(digests)) == 1
    return dict(
        native_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        supplied_variance=supplied,
        dimensions=list(image.shape),
        spots=len(result),
        complete_record_sha256=digests[0],
        samples_seconds=durations,
        median_seconds=float(np.median(durations)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-site", type=Path)
    p.add_argument("--candidate-site", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--supplied-variance", action="store_true")
    args = p.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.supplied_variance)))
        return
    if not args.baseline_site or not args.candidate_site or not args.output:
        p.error("both preserved package paths and a new output path are required")
    if args.output.exists():
        p.error("output must be new")
    rows = []
    for supplied in [False, True]:
        order = ["baseline", "candidate"] if not supplied else ["candidate", "baseline"]
        pair = {}
        for mode in order:
            site = getattr(args, mode + "_site").resolve()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(site)]
            if supplied:
                command.append("--supplied-variance")
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            pair[mode] = json.loads(subprocess.check_output(command, env=env, text=True))
        assert (
            pair["baseline"]["complete_record_sha256"]
            == pair["candidate"]["complete_record_sha256"]
        )
        pair["candidate_over_baseline"] = (
            pair["candidate"]["median_seconds"] / pair["baseline"]["median_seconds"]
        )
        rows.append(pair)
    args.output.write_text(
        json.dumps(
            dict(
                schema="phasesmith.threshold-predicate-comparison.v1",
                qualification=(
                    "Sequential isolated builds, one warmup and eight measured calls "
                    "per noise path. Counterbalanced path order, complete record "
                    "equality, multi-spot 768x1024 images. Local observations, not a "
                    "general speed guarantee."
                ),
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
