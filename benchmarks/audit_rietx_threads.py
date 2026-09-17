#!/usr/bin/env python3
"""Audit requested versus actual rietx concurrency on the real QARR workflow.

Run each worker setting in a fresh process, with BLAS/OpenMP limited to one.
Requires the optional rietx benchmark environment plus threadpoolctl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import compare_rietx_qarr as comparison
import phasesmith as ps
import rietx as rx
from rietx._about import COMPILED_THREADS_ENV
from rietx.model import compiled
from threadpoolctl import threadpool_info


def run(root: Path, threads: int, repetitions: int) -> dict:
    if rx.__version__ != "1.4.0":
        raise RuntimeError("audit uses rietx 1.4.0 internals")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"launch with {name}=1")
    os.environ[COMPILED_THREADS_ENV] = str(threads)
    compiled.warm(block=True)
    if not compiled.enabled() or compiled._POOL is not None:
        raise RuntimeError("requires compiled kernels and a fresh process")

    rows = Counter()
    pool_calls = 0
    spread = compiled._spread
    pool = compiled._pool

    def traced_spread(fn, n_rows):
        rows[n_rows] += 1
        return spread(fn, n_rows)

    def traced_pool():
        nonlocal pool_calls
        pool_calls += 1
        return pool()

    # Warm the full workflow; collect dispatch evidence outside measured runs.
    with (
        patch.object(compiled, "_spread", traced_spread),
        patch.object(compiled, "_pool", traced_pool),
    ):
        expected = comparison.run_rietx(root)

    preparation = []
    original_prepare = comparison.prepare

    def timed_prepare(path):
        wall, cpu = time.perf_counter(), time.process_time()
        result = original_prepare(path)
        preparation.append(
            {"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall}
        )
        return result

    measurements = []
    for _ in range(repetitions):
        with patch.object(comparison, "prepare", timed_prepare):
            wall, cpu = time.perf_counter(), time.process_time()
            result = comparison.run_rietx(root)
            cpu = time.process_time() - cpu
            wall = time.perf_counter() - wall
        if result != expected:
            raise RuntimeError("scientific results changed between repetitions")
        prep = preparation[-1]
        fit_wall = wall - prep["wall_seconds"]
        fit_cpu = cpu - prep["cpu_seconds"]
        measurements.append(
            {
                "workflow_wall_seconds": wall,
                "workflow_cpu_seconds": cpu,
                "preparation": prep,
                "after_preparation_wall_seconds": fit_wall,
                "after_preparation_cpu_seconds": fit_cpu,
                "after_preparation_cpu_to_wall": fit_cpu / fit_wall,
            }
        )
    source = Path(compiled.__file__)
    return {
        "schema": "phasesmith.rietx-thread-audit.v1",
        "rietx_version": rx.__version__,
        "requested_workers": threads,
        "compiled_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parallel_row_threshold": compiled._THREAD_MIN_ROWS,
        "warmup_spread_row_histogram": dict(sorted(rows.items())),
        "warmup_pool_requests": pool_calls,
        "pool_created_after_measurements": compiled._POOL is not None,
        "loaded_native_threadpools": threadpool_info(),
        "shared_phasesmith_preparation_default_workers": ps.ExecutionPolicy().threads,
        "repetitions": repetitions,
        "median_after_preparation_cpu_to_wall": statistics.median(
            row["after_preparation_cpu_to_wall"] for row in measurements
        ),
        "measurements": measurements,
        "scientific_results": expected,
        "notes": [
            "Dispatch instrumentation runs only during the warmup, outside measured runs.",
            "CPU time sums all process threads; CPU/wall estimates average busy cores, "
            "not affinity.",
            "After-preparation timing includes all three rietx fits and final result reporting.",
            "Shared preparation uses the existing PhaseSmith default of two workers; "
            "the original workflow timing is not a strict single-core measurement.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=comparison.ROOT / "validation/data/iucr-qarr-1g",
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 1:
        parser.error("threads and repetitions must be positive")
    record = run(args.data_directory, args.threads, args.repetitions)
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ("measurements", "scientific_results")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
