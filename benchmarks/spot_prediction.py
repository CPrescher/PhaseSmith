"""Multi-reflection native value/Jacobian and full positional refinement benchmark."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.prediction import predict_spots, refine_crystal
from phasesmith.experimental.reciprocal import PanelGeometry

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
rng = np.random.default_rng(20261024)
g = PanelGeometry.from_poni(
    distance=0.2, poni1=0.08, poni2=0.08, pixel1=8e-5, pixel2=8e-5, wavelength=0.4, axis=[-1, 0, 0]
)
b = np.linalg.inv(np.array([[4.7, 0, 0], [0.8, 6.3, 0], [-0.5, 0.2, 8.1]])).T
h = rng.integers(-7, 8, (5000, 3), dtype=np.int64)
hint = np.full(len(h), -90.0)
p = predict_spots(h, b, g, hint)
valid = p.flags == 0
h, observed = h[valid], p.values[valid].copy()
observed += rng.normal(size=observed.shape) * [0.15, 0.15, 0.03]
training = np.arange(len(h)) % 5 != 0
start = b + rng.normal(0, 1e-4, b.shape)
times, cpu_times, old = [], [], None
for _ in range(3):
    t, cpu = time.perf_counter(), time.process_time()
    prediction = predict_spots(h, b, g, observed[:, 2])
    prediction_time, prediction_cpu = time.perf_counter() - t, time.process_time() - cpu
    t, cpu = time.perf_counter(), time.process_time()
    result = refine_crystal(observed, h, start, g, training=training)
    times.append([prediction_time, time.perf_counter() - t])
    cpu_times.append([prediction_cpu, time.process_time() - cpu])
    assert result.metrics["validation_improved"]
    if old is not None:
        np.testing.assert_array_equal(prediction.values, old[0])
        np.testing.assert_array_equal(prediction.jacobian, old[1])
        np.testing.assert_array_equal(result.reciprocal_basis, old[2])
    old = prediction.values, prediction.jacobian, result.reciprocal_basis
record = dict(
    platform=platform.platform(),
    reflections=len(h),
    times_s=times,
    cpu_times_s=cpu_times,
    median_s=np.median(times, axis=0).tolist(),
    median_cpu_s=np.median(cpu_times, axis=0).tolist(),
    exact_repeat=True,
    native_workers=1,
    seed=20261024,
    fields=["prediction_with_27_derivatives", "full_training_only_refinement"],
)
Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
