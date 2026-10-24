"""Position-only multigrain/FFT benchmark: realistic cloud, exact repeated output."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith.experimental.grains import GrainSearchOptions, find_grains

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
a = parser.parse_args()
rng = np.random.default_rng(33)
parts = []
for edges in ([4.3, 5.1, 7.7], [5.2, 6.2, 8.3]):
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    basis = np.diag(edges) @ rotation
    h = np.unique(rng.integers(-6, 7, (1400, 3)), axis=0)
    h = h[np.any(h != 0, axis=1)]
    parts.append(h @ np.linalg.inv(basis).T + rng.normal(0, 0.0005, h.shape))
q = np.concatenate([*parts, rng.uniform(-1, 1, (600, 3))])
options = GrainSearchOptions(max_length=15, grid_size=192, max_grains=4)
times = []
cpus = []
previous = None
for _ in range(3):
    start = time.perf_counter()
    cpu = time.process_time()
    result = find_grains(q, options=options)
    times.append(time.perf_counter() - start)
    cpus.append(time.process_time() - cpu)
    assert sum(c.accepted for c in result.candidates) == 2
    bases = np.array([c.direct_basis for c in result.candidates])
    if previous is not None:
        np.testing.assert_array_equal(bases, previous[0])
        np.testing.assert_array_equal(result.labels, previous[1])
    previous = bases, result.labels
record = dict(
    platform=platform.platform(),
    positions=len(q),
    options=asdict(options),
    wall_seconds=times,
    cpu_seconds=cpus,
    exact_repeat=True,
    accepted=2,
)
Path(a.output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
