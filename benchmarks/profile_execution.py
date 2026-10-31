"""Measure full real-bank learning and baseline integration in isolated processes.

Inputs are archived experimental shoeboxes and a frozen bank, never raw frames or
atomic references. Each output is exclusive. The full raw workflow is validated
separately after each optimization stage.
"""

import argparse
import hashlib
import inspect
import json
import resource
import time
from pathlib import Path

import numpy as np
from phasesmith.single_crystal.profiles import fit_profiles, learn_profiles
from phasesmith.single_crystal.reduction_io import load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--part", choices=("learn", "fit", "both"), default="both")
    args = parser.parse_args()
    obs, bank = load_case(args.case)
    references = np.zeros(len(obs.centres), bool)
    references[bank.reference_ids] = True
    controls = {}
    if "execution" in inspect.signature(learn_profiles).parameters:
        from phasesmith.single_crystal.profiles import ProfileExecution

        controls["execution"] = ProfileExecution(args.workers, args.cache_bytes)
    record = dict(
        schema="phasesmith.profile-execution-benchmark.v1",
        samples=len(obs.samples),
        reflections=len(obs.centres),
        teachers=len(bank.reference_ids),
        controls={"workers": args.workers, "cache_bytes": args.cache_bytes},
        execution_api=bool(controls),
        parts={},
    )
    for part in ("learn", "fit") if args.part == "both" else (args.part,):
        seconds = []
        hashes = []
        stats = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            if part == "learn":
                result = learn_profiles(
                    obs,
                    references,
                    detector_shape=bank.detector_shape,
                    options=bank.options,
                    **controls,
                )
                payload = (
                    result.profiles.tobytes()
                    + result.counts.tobytes()
                    + result.shifts.tobytes()
                    + result.reference_ids.tobytes()
                    + result.reference_centres.tobytes()
                )
                np.testing.assert_array_equal(result.profiles, bank.profiles)
            else:
                result = fit_profiles(obs, bank, **controls)
                payload = (
                    result.values.tobytes() + result.flags.tobytes() + result.model_ids.tobytes()
                )
            seconds.append(time.perf_counter() - start)
            hashes.append(hashlib.sha256(payload).hexdigest())
            stats.append(getattr(result, "execution", None))
        assert len(set(hashes)) == 1
        record["parts"][part] = dict(
            seconds=seconds,
            median_seconds=float(np.median(seconds)),
            sha256=hashes[0],
            deterministic=True,
            execution=stats[-1],
        )
    record["maximum_resident_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
