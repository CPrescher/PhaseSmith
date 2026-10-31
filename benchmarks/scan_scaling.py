"""Many-family scan-scale performance and equation parity; no raw data dependency."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental import scan_scaling_reference as reference
from phasesmith.experimental.scan_scaling import fit_scan_scale


def main(output):
    rng = np.random.default_rng(20261028)
    # 64x64 reflection-family grid, four different rotation observations each.
    groups = np.repeat(np.arange(64 * 64, dtype=np.int64), 4)
    omega = rng.uniform(-35, 36, len(groups))
    x = (omega - 0.5) / 35.5
    J = np.exp(rng.uniform(7, 11, 64 * 64))
    intensity = J[groups] * np.exp(-0.25 * x)
    sigma = np.sqrt(intensity)
    intensity += rng.normal(0, sigma)
    training = groups % 5 != 0
    times = []
    baseline = None
    for _ in range(6):
        start = time.perf_counter()
        model = fit_scan_scale(
            intensity, sigma, omega, groups, omega_range=(-35, 36), training=training
        )
        values = model.apply(intensity, sigma, omega)
        times.append(time.perf_counter() - start)
        if baseline is not None:
            assert model.coefficient == baseline[0] and values.tobytes() == baseline[1]
        baseline = (model.coefficient, values.tobytes())
    start = time.perf_counter()
    ref = reference.fit(intensity, sigma, x, groups, training)
    reference_seconds = time.perf_counter() - start
    np.testing.assert_allclose(model.coefficient, ref[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(model.variance, ref[1], rtol=1e-10, atol=1e-12)
    report = dict(
        observations=len(groups),
        families=64 * 64,
        references=len(model.reference_ids),
        warmup_seconds=times[0],
        native_seconds=times[1:],
        native_median_seconds=float(np.median(times[1:])),
        independent_reference_seconds=reference_seconds,
        coefficient=model.coefficient,
        known_coefficient=-0.25,
        variance=model.variance,
        converged=model.converged,
        byte_exact_repeats=True,
        reference_parity=True,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
