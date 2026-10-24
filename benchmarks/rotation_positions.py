"""Position-first realistic multi-frame/multi-spot benchmark, including detector work."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.rotation import locate_series

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--frames", type=int, default=32)
args = parser.parse_args()
seed = 20261024
rng = np.random.default_rng(seed)
image = rng.poisson(5, (args.size, args.size)).astype(float)
for y in range(24, args.size - 24, 32):
    for x in range(24, args.size - 24, 32):
        image[y : y + 3, x : x + 3] += 100
bounds = np.column_stack([np.arange(args.frames) * 0.5, (np.arange(args.frames) + 1) * 0.5])
timings = []
previous = None
for _ in range(3):
    started = time.perf_counter()
    result = locate_series(lambda i: np.roll(image, i % 3, axis=1), bounds)
    timings.append(time.perf_counter() - started)
    if previous is not None:
        for field in ["values", "observations", "flags", "boxes", "frame_ranges"]:
            np.testing.assert_array_equal(getattr(result, field), getattr(previous, field))
    previous = result
report = dict(
    seed=seed,
    platform=platform.platform(),
    shape=[args.frames, args.size, args.size],
    tracks=len(result),
    observations=len(result.observations),
    times_s=timings,
    median_s=float(np.median(timings)),
    exact_repeat=True,
    native_workers=1,
)
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
