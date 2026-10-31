"""Many-track fused ownership fields: equal equations and deterministic masks.

This is a stage benchmark, not a complete grain-discovery timing. Synthetic
oriented minority lattices are mixed with diffuse contamination; no experimental
reference cells or acceptance labels enter it.
"""

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from phasesmith import _core
from phasesmith.experimental.grains import GrainSearchOptions
from phasesmith.experimental.ownership import GrainOwnershipOptions, _physical_fields


def main(args):
    rng = np.random.default_rng(842076)
    q = rng.uniform(-1.6, 1.6, (12000, 3))
    bases = []
    for i in range(64):
        a = np.diag(rng.uniform(3.5, 12, 3)) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        bases.append(a)
        if i < 3:
            h = rng.integers(-5, 6, (100, 3))
            q[i * 100 : (i + 1) * 100] = h @ np.linalg.inv(a).T + rng.normal(0, 0.0004, (100, 3))
    bases = np.array(bases)
    charts = np.broadcast_to(np.diag([1000.0, 1000.0, 100.0]), (len(q), 3, 3)).copy()
    valid = rng.uniform(size=len(q)) > 0.05
    search = GrainSearchOptions(workers=4)
    own = GrainOwnershipOptions()
    engine = _core.ExperimentalLatticeEngine(search.workers)
    times = []
    for repeat in range(args.repeats):
        for mode in ("numpy-orchestration", "native-fields"):
            start = perf_counter()
            fields = _physical_fields(
                engine,
                q,
                bases,
                charts,
                valid,
                search,
                replace(own, native_fields=mode == "native-fields"),
            )
            elapsed = perf_counter() - start
            if mode == "numpy-orchestration":
                reference = fields
            else:
                np.testing.assert_allclose(fields[0], reference[0], rtol=3e-12, atol=3e-12)
                np.testing.assert_array_equal(fields[1], reference[1])
                np.testing.assert_array_equal(fields[2], reference[2])
            times.append(dict(repeat=repeat, mode=mode, seconds=elapsed))
    result = dict(
        seed=842076,
        points=len(q),
        models=len(bases),
        pairs=len(q) * len(bases),
        workers=search.workers,
        platform=platform.platform(),
        runs=times,
        equation_rtol=3e-12,
        equation_atol=3e-12,
        masks_exact=True,
        scope="Ownership fields only. Excludes discovery, fits, merge, validation and imports.",
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    main(parser.parse_args())
