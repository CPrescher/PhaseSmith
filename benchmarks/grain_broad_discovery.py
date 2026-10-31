"""Equal-output bounded FFT and prepared fused-moment stage benchmarks.

Twelve thousand tracks include two minority lattices plus contamination.
Conditional subsets use the same strict integer-phase criterion as discovery.
These are kernel stages, never a one-second complete-inverse claim.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from phasesmith import _core
from phasesmith.experimental.grains import _fourier_evaluator


def main(args):
    rng = np.random.default_rng(842027)
    q = rng.uniform(-1.6, 1.6, (12000, 3))
    for start, edges in [(0, [4.7, 6.3, 8.1]), (300, [5.2, 7.1, 9.3])]:
        a = np.diag(edges) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = rng.integers(-5, 6, (300, 3))
        q[start : start + 300] = h @ np.linalg.inv(a).T + rng.normal(0, 0.0008, (300, 3))
    phase = q @ a[0]
    conditional = q[abs(phase - np.rint(phase)) < 0.08]
    engine = _core.ExperimentalLatticeEngine(4)
    rows = []
    for repeat in range(args.repeats):
        for label, points in [("full", q), ("conditional", conditional)]:
            for n in (128, 256):
                step = min(2 * np.max(abs(points)) / (n - 4), 1 / (2.1 * 15))
                start = perf_counter()
                old = engine.build_field(points, n, step).maxima(1.5, 15, 400)
                dense = perf_counter() - start
                start = perf_counter()
                new = engine.windowed_maxima(points, n, step, 1.5, 15, 400)
                windowed = perf_counter() - start
                np.testing.assert_array_equal(old[0], new[0])
                np.testing.assert_array_equal(old[1], new[1])
                assert old[2] == new[2]
                rows.append(dict(repeat=repeat, scope="field + maxima", cloud=label,
                                 points=len(points), grid=n, dense=dense, windowed=windowed,
                                 exact_values_vectors_total=True))
        start = perf_counter()
        prepared = engine.prepare_fourier(conditional)
        prepare_seconds = perf_counter() - start
        for count in (1, 4, 32, 400):
            vectors = rng.uniform(-8, 8, (count, 3))
            start = perf_counter()
            for _ in range(20):
                old = engine.fourier(conditional, vectors)
            regular = (perf_counter() - start) / 20
            start = perf_counter()
            for _ in range(20):
                new = prepared.evaluate(vectors)
            reuse = (perf_counter() - start) / 20
            np.testing.assert_array_equal(old, new)
            mixed = _fourier_evaluator(engine, conditional)
            start = perf_counter()
            for _ in range(20):
                hybrid = mixed(vectors)
            mixed_seconds = (perf_counter() - start) / 20
            np.testing.assert_array_equal(old, hybrid)
            rows.append(dict(repeat=repeat, scope="fused moments", points=len(conditional),
                             vectors=count, ordinary=regular, prepared=reuse,
                             hybrid=mixed_seconds, one_time_prepare=prepare_seconds,
                             exact_all_13_moments=True))
    result = dict(platform=platform.platform(), workers=4, seed=842027, runs=rows,
                  scope="Kernel stages only; excludes search, ownership, fits and validation.")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    main(parser.parse_args())
