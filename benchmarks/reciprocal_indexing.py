"""Realistic multi-position geometry/Fourier/cell benchmark; deterministic repeats."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.indexing import IndexingOptions, find_lattices
from phasesmith.experimental.reciprocal import PanelGeometry, lattice_fourier, reciprocal_vectors

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
rng = np.random.default_rng(20261024)
basis = np.array([[4.7, 0, 0], [0.8, 6.3, 0], [-0.5, 0.2, 8.1]])
rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
basis = basis @ rotation
h = rng.integers(-8, 9, (2400, 3))
q = h @ np.linalg.inv(basis).T + rng.normal(0, 0.001, h.shape)
q = np.r_[q, rng.uniform(-1.5, 1.5, (600, 3))]
g = PanelGeometry.from_poni(
    distance=0.2, poni1=0.08, poni2=0.08, pixel1=8e-5, pixel2=8e-5, wavelength=0.4, axis=[-1, 0, 0]
)
samples = rng.uniform([0, 0, -100], [2048, 2048, -20], (10000, 3))
vectors = rng.uniform(-15, 15, (160, 3))
timings = []
cpu_timings = []
old = None
for _ in range(3):
    started = time.perf_counter()
    cpu_started = time.process_time()
    converted = reciprocal_vectors(samples, g)
    geometry_time = time.perf_counter() - started
    geometry_cpu = time.process_time() - cpu_started
    started = time.perf_counter()
    cpu_started = time.process_time()
    moments = lattice_fourier(q, vectors)
    fourier_time = time.perf_counter() - started
    fourier_cpu = time.process_time() - cpu_started
    started = time.perf_counter()
    cpu_started = time.process_time()
    result = find_lattices(q, options=IndexingOptions(max_length=25, grid_size=256))
    timings.append([geometry_time, fourier_time, time.perf_counter() - started])
    cpu_timings.append([geometry_cpu, fourier_cpu, time.process_time() - cpu_started])
    assert result.candidates[0].accepted
    if old is not None:
        for a, b in zip((converted, moments, result.candidates[0].direct_basis), old, strict=True):
            np.testing.assert_array_equal(a, b)
    old = (converted, moments, result.candidates[0].direct_basis)
record = dict(
    platform=platform.platform(),
    positions=len(q),
    geometry_observations=len(samples),
    vectors=len(vectors),
    times_s=timings,
    cpu_times_s=cpu_timings,
    median_cpu_s=np.median(cpu_timings, axis=0).tolist(),
    median_s=np.median(timings, axis=0).tolist(),
    exact_repeat=True,
    native_workers=1,
    seed=20261024,
)
Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
