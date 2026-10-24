"""Reproducible multi-spot end-to-end measurement; no external images or structures."""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
from phasesmith.experimental.spots import SpotOptions, detect_spots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    seed = 2026102408
    rng = np.random.default_rng(seed)
    image = rng.poisson(5, (args.size, args.size)).astype(float)
    for y in range(24, args.size - 24, 64):
        for x in range(24, args.size - 24, 64):
            yy, xx = np.mgrid[-5:6, -5:6]
            image[y - 5 : y + 6, x - 5 : x + 6] += rng.poisson(
                1000 * np.exp(-(xx**2 + yy**2) / 2) / (2 * np.pi)
            )
    options = SpotOptions()
    detect_spots(image, options=options)
    timings, previous = [], None
    for _ in range(args.repetitions):
        start = time.perf_counter()
        result = detect_spots(image, options=options)
        timings.append(time.perf_counter() - start)
        if previous is not None:
            for name in ("boxes", "values", "flags"):
                np.testing.assert_array_equal(getattr(result, name), getattr(previous, name))
        previous = result
    from dataclasses import asdict

    report = dict(
        seed=seed,
        shape=image.shape,
        platform=platform.platform(),
        options=asdict(options),
        native_workers=1,
        spots=len(result),
        flagged=int(np.count_nonzero(result.flags)),
        exact_repeated_records=True,
        times_s=timings,
        median_s=float(np.median(timings)),
    )
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
