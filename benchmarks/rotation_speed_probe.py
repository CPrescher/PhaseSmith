"""Time native position hunting on the comparison's lossless, common image arrays."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.experimental.rotation import locate_series

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument("--read-noise", type=float, default=0.0)
parser.add_argument("--frames", type=int)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
with np.load(args.case / "native/table.npz") as saved:
    expected = dict(saved)
bounds = expected["omega_bounds"]
if args.frames:
    bounds = bounds[: args.frames]
if args.reference:
    with np.load(args.reference) as saved:
        expected = dict(saved)
elif args.frames or args.gain != 1.0 or args.read_noise != 0.0:
    expected = None
baseline_compared = expected is not None
meta = json.loads((args.case / "native/metadata.json").read_text())
cache = {}


def load(i):
    data = np.load(args.case / f"images/frame_{i + 1:04d}.npy")
    cache[i] = data == 0
    return data


times = []
for repeat in range(args.repeats + 1):
    started = time.perf_counter()
    result = locate_series(
        load,
        bounds,
        mask_loader=lambda i: cache.pop(i),
        saturation=65535.0,
        workers=args.workers,
        gain=args.gain,
        read_noise=args.read_noise,
    )
    seconds = time.perf_counter() - started
    if expected is None:
        expected = {
            name: getattr(result, name).copy()
            for name in ("boxes", "frame_ranges", "values", "flags", "observations", "omega_bounds")
        }
    for field, array in expected.items():
        actual = getattr(result, field)
        assert actual.dtype == array.dtype and actual.shape == array.shape
        assert actual.tobytes() == array.tobytes(), field
    assert list(result.frame_hashes) == meta["frame_image_mask_hashes"][: len(bounds)]
    if repeat:
        times.append(seconds)
record = dict(
    workers=args.workers,
    seconds=float(np.median(times)),
    times_s=times,
    positions=len(result),
    exact_baseline=baseline_compared,
    exact_repeats=True,
    warmups=1,
    gain=args.gain,
    read_noise=args.read_noise,
    frames=len(bounds),
    native_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
)
np.savez(args.output / "table.npz", **expected)
(args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
