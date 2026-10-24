"""Multi-reflection native value/Jacobian and full positional refinement benchmark."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.orthorhombic import find_orthorhombic_settings, project_orthorhombic
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
orth_basis = project_orthorhombic(b)
exact = predict_spots(h, orth_basis, g, observed[:, 2])
good = exact.flags == 0
orth_h, orth_observed = h[good], exact.values[good].copy()
orth_observed += rng.normal(size=orth_observed.shape) * [0.15, 0.15, 0.03]
orth_start = orth_basis + rng.normal(0, 1e-4, b.shape)
orth_training = np.arange(len(orth_h)) % 5 != 0
orth_times, setting_times, previous = [], [], None
for _ in range(3):
    t = time.perf_counter()
    settings = find_orthorhombic_settings(np.linalg.inv(orth_start).T)
    setting_times.append(time.perf_counter() - t)
    assert settings
    t = time.perf_counter()
    result = refine_crystal(
        orth_observed, orth_h, orth_start, g, training=orth_training, symmetry="orthorhombic"
    )
    orth_times.append(time.perf_counter() - t)
    if previous is not None:
        np.testing.assert_array_equal(result.reciprocal_basis, previous)
    previous = result.reciprocal_basis
record["orthorhombic"] = dict(
    reflections=len(orth_h),
    median_refinement_s=float(np.median(orth_times)),
    median_setting_search_s=float(np.median(setting_times)),
    exact_repeat=True,
)
Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
