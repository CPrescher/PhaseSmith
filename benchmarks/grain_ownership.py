"""Bounded ownership/normal hot-loop timing on 20000 detector observations.

Distances differ by design; this measures computational overhead, not accuracy.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.single_crystal.grains import GrainSearchOptions
from phasesmith.single_crystal.ownership import (
    GrainOwnershipOptions,
    assign_observed,
    observation_charts,
)
from phasesmith.single_crystal.reciprocal import PanelGeometry, reciprocal_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(4521)
    g = PanelGeometry.from_poni(
        distance=0.2,
        poni1=0.08,
        poni2=0.08,
        pixel1=8e-5,
        pixel2=8e-5,
        wavelength=0.4,
        axis=[-1, 0, 0],
    )
    observed = rng.uniform([100, 100, -200], [2000, 2000, -30], (20000, 3))
    q = reciprocal_vectors(observed, g)
    models = np.array(
        [np.diag([4.1, 5.3, 7.7]) @ np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(8)]
    )
    reciprocal = np.linalg.inv(models).transpose(0, 2, 1)
    search = GrainSearchOptions(workers=2)
    options = GrainOwnershipOptions()
    charts, valid = observation_charts(observed, g, options)
    engine = _core.ExperimentalLatticeEngine(2)
    h = np.rint(q @ models[0].T)
    operations = dict(
        reciprocal_ownership=lambda: engine.assign(q, models, reciprocal, 0.015, 0.8),
        observation_ownership=lambda: assign_observed(
            engine, q, models, charts, valid, search, options
        ),
        fused_normal=lambda: engine.observation_normal(
            q, h, charts, valid, reciprocal[0], np.array([2.0, 2.0, 0.3]), 2.5
        ),
    )
    results = {}
    for name, fn in operations.items():
        previous = fn()
        wall = []
        cpu = []
        for _ in range(25):
            t, p = time.perf_counter(), time.process_time()
            current = fn()
            wall.append(time.perf_counter() - t)
            cpu.append(time.process_time() - p)
            for a, b in zip(previous, current, strict=True):
                np.testing.assert_array_equal(a, b)
        results[name] = dict(
            wall_seconds=wall,
            cpu_seconds=cpu,
            wall_median=float(np.median(wall)),
            cpu_median=float(np.median(cpu)),
            exact_repeat=True,
        )
    Path(args.output).write_text(
        json.dumps(
            dict(
                points=len(q),
                models=len(models),
                search=asdict(search),
                ownership=asdict(options),
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )
    print({k: round(v["wall_median"] * 1000, 3) for k, v in results.items()})


if __name__ == "__main__":
    main()
