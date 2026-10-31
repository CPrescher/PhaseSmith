"""Exact before/after Fourier kernels on full and conditioned track clouds.

Example: PYTHONPATH=python .venv/bin/python benchmarks/grain_fourier.py \
  --legacy-binary /absolute/task-snapshot/core-before-fourier.abi3.so \
  --points /absolute/frozen/points.npz --output /absolute/results.json
No candidate, chemistry, comparison group, or acceptance label is an input.
Imports and input loading are excluded. Complete inverses require separate timing.
"""

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from phasesmith import _core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-binary", type=Path, required=True)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    name = "_fourier_benchmark_before"
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + "._core", args.legacy_binary)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    engines = [old.ExperimentalLatticeEngine(4), _core.ExperimentalLatticeEngine(4)]
    if args.points:
        with np.load(args.points) as data:
            full = data["points"][data["training"]]
    else:
        full = np.random.default_rng(742971).uniform(-1.6, 1.6, (12000, 3))
    phase = full @ np.array([2.71, 3.18, 4.09])
    conditional = full[abs(phase - np.rint(phase)) < 0.08]
    records = []
    for kind, q in [("full", full), ("conditional", conditional)]:
        for n in (128, 256):
            step = min(2 * np.max(abs(q)) / (n - 4), 1 / (2.1 * 15))
            radius = 1.5 / (n * step)
            expected = None
            # Alternate order; each inverse/kernel owns its own worker budget.
            for repeat in range(args.repeats):
                for j in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    start = time.perf_counter()
                    result = engines[j].windowed_maxima(q, n, step, 1.5, 15, 400)
                    elapsed = time.perf_counter() - start
                    if expected is None:
                        expected = result
                    for x, y in zip(expected[:2], result[:2], strict=True):
                        np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
                    assert expected[2] == result[2]
                    records.append(
                        dict(
                            stage="fft",
                            cloud=kind,
                            points=len(q),
                            grid=n,
                            repeat=repeat,
                            engine=j,
                            seconds=elapsed,
                        )
                    )
            expected = None
            for repeat in range(args.repeats):
                for j in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    initial = result[0]
                    start = time.perf_counter()
                    refined = engines[j].refine_fourier(q, initial, radius)
                    elapsed = time.perf_counter() - start
                    if expected is None:
                        expected = refined
                    for x, y in zip(expected, refined, strict=True):
                        np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
                    records.append(
                        dict(
                            stage="ascent",
                            cloud=kind,
                            points=len(q),
                            grid=n,
                            repeat=repeat,
                            engine=j,
                            seconds=elapsed,
                        )
                    )
    args.output.write_text(
        json.dumps(
            dict(
                exact_all_outputs=True,
                legacy_binary=str(args.legacy_binary.resolve()),
                points=str(args.points.resolve()) if args.points else "deterministic synthetic",
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
