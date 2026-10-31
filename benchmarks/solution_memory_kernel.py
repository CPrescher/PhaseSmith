"""Fixed many-reflection native values/Jacobian timing and numerical fingerprint.

Run sequentially in preserved and changed release environments. This uses the
same private native model boundary in both; it does not run a structure search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from phasesmith import _core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 3:
        parser.error("at least one warmup and three repetitions required")
    index = np.arange(6400, dtype=np.int64)
    hkl = np.column_stack((1 + index % 16, (index // 16) % 16, index // 256)).ravel()
    atoms = np.arange(24, dtype=np.float64)
    xyz = np.column_stack(((atoms * 0.137) % 1, (atoms * 0.241) % 1, (atoms * 0.319) % 1)).ravel()
    observed = np.ones(6400)

    def evaluate():
        return _core._solution_model(
            [20.0, 21.0, 22.0, 90.0, 90.0, 90.0], hkl, observed, ["O"] * 24, xyz, 0.01
        )

    def fingerprint(result):
        scalars = np.array([result[k] for k in ("objective", "r1", "amplitude_scale")])
        digest = hashlib.sha256(scalars.tobytes())
        for key in ("residual", "jacobian"):
            digest.update(result[key].tobytes())
        return digest.hexdigest()

    expected = fingerprint(evaluate())
    for _ in range(args.warmups):
        if fingerprint(evaluate()) != expected:
            raise ValueError("non-deterministic warmup")
    samples = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        result = evaluate()
        samples.append(time.perf_counter() - started)
        if fingerprint(result) != expected:
            raise ValueError("non-deterministic measured output")
    with Path(_core.__file__).open("rb") as stream:
        native_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "schema": "phasesmith.solution-memory-kernel.v1",
        "reflections": 6400,
        "atoms": 24,
        "scope": "native fused values/analytical Jacobian including Python array boundary",
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "native_sha256": native_sha,
        "scientific_sha256": expected,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "p95_seconds": float(np.quantile(samples, 0.95)),
    }
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: report[k] for k in ("scientific_sha256", "median_seconds", "p95_seconds")})
    )


if __name__ == "__main__":
    main()
