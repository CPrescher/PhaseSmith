"""Measure the complete fixed-amplitude plus strong-shape real MgSiO3 fit.

Uses the preserved observed shoebox case and observation-only strong selection.
No raw frames, original HKLs or atomic references enter this isolated probe.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from phasesmith.experimental.profile_refinement import ShapeFitOptions, refine_profiles
from phasesmith.experimental.profiles import ProfileExecution
from phasesmith.experimental.reduction_io import load_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[4, 6, 8, 10])
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    obs, bank = load_case(a.case)
    t = obs.summation
    choices = np.zeros(len(t), np.int64)
    choices[(t.flags == 0) & (t.values[:, 0] > 50 * t.values[:, 1]) & (t.values[:, 1] > 0)] = 1
    records = []
    hashes = []
    for workers in a.workers:
        seconds = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            result = refine_profiles(
                obs,
                bank,
                choices,
                options=ShapeFitOptions(workers=workers),
                execution=ProfileExecution(workers, 256 * 1024**2),
            )
            seconds.append(time.perf_counter() - start)
            digest = hashlib.sha256()
            for array in [
                result.integration.values,
                result.integration.flags,
                result.parameters,
                result.converged,
                result.baseline.values,
            ]:
                digest.update(array.tobytes())
            hashes.append(digest.hexdigest())
        records.append(
            dict(
                workers=workers,
                seconds=seconds,
                median_seconds=float(np.median(seconds)),
                sha256=hashes[-1],
            )
        )
        print(json.dumps(records[-1]), flush=True)
    assert len(set(hashes)) == 1
    record = dict(
        schema="phasesmith.profile-worker-scaling.v1",
        strong=int((choices != 0).sum()),
        reflections=len(t),
        samples=len(obs.samples),
        records=records,
        all_outputs_bitwise_identical=True,
        maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    with a.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
