"""Position-only primitive/Bravais benchmark over 5000 synthetic reflections."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith.experimental.cell_comparison import CellComparisonOptions, compare_cells

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
a = parser.parse_args()
rng = np.random.default_rng(615)
basis = np.diag([4.3, 6.1, 9.7])
h = rng.integers(-12, 13, (5000, 3))
h = h[np.any(h != 0, axis=1)]
q = h @ np.linalg.inv(basis).T + rng.normal(0, 0.0008, h.shape)
q = np.concatenate([q, rng.uniform(-1, 1, (400, 3))])
training = np.arange(len(q)) % 5 != 0
options = CellComparisonOptions(max_subcell_index=8)
times = []
previous = None
for _ in range(3):
    start = time.perf_counter()
    result = compare_cells(q, basis, training=training, options=options)
    times.append(time.perf_counter() - start)
    assert result.recommended.setting.symbol == "oP"
    bases = np.array([s.reciprocal_basis for s in result.settings])
    if previous is not None:
        np.testing.assert_array_equal(bases, previous)
    previous = bases
record = dict(
    platform=platform.platform(),
    positions=len(q),
    options=asdict(options),
    primitive_hypotheses=len(result.primitives),
    settings=len(result.settings),
    wall_seconds=times,
    exact_repeat=True,
)
Path(a.output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
