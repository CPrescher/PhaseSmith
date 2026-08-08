#!/usr/bin/env python3
"""Benchmark checksum-pinned native workflows on real powder patterns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import phasesmith
from phasesmith.radiation import RadiationProbe
from phasesmith.validation import (
    RealDataValidationReport,
    run_pbso4_cw_validation,
    run_qarr_1g_validation,
    run_sucrose_lebail_validation,
    verify_validation_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_IDS = ("aps-sucrose-11bmb", "iucr-qarr-1g", "gsasii-pbso4-cw")


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile used by the other benchmarks."""

    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("percentiles require values and a fraction in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[min(len(ordered) - 1, index)]


def timing_summary(values: list[float]) -> dict[str, Any]:
    """Summarize repeated wall-clock measurements without hiding samples."""

    if not values or any(value < 0.0 for value in values):
        raise ValueError("timings must be non-empty and nonnegative")
    return {
        "timings_ms": values,
        "minimum_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
    }


def scientific_record(report: RealDataValidationReport) -> dict[str, Any]:
    """Return the deterministic scientific result, excluding host timing."""

    record = report.to_record()
    record.pop("elapsed_seconds")
    return record


def scientific_fingerprint(record: dict[str, Any]) -> str:
    """Hash a canonical scientific record for compact regression comparisons."""

    encoded = json.dumps(
        record,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_once(
    operation: Callable[[], RealDataValidationReport],
) -> tuple[RealDataValidationReport, float]:
    started = time.perf_counter_ns()
    report = operation()
    wall_ms = (time.perf_counter_ns() - started) / 1.0e6
    if report.status != "passed":
        raise RuntimeError(f"real-data benchmark {report.dataset_id!r} returned {report.status!r}")
    return report, wall_ms


def benchmark_operation(
    dataset_id: str,
    operation: Callable[[], RealDataValidationReport],
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure one complete workflow and require exact scientific repeatability."""

    if warmups < 0 or repetitions <= 0:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    cold_report, cold_wall_ms = _run_once(operation)
    if cold_report.dataset_id != dataset_id:
        raise RuntimeError("real-data runner returned the wrong dataset identity")
    expected = scientific_record(cold_report)

    for _ in range(warmups):
        warmup_report, _ = _run_once(operation)
        if scientific_record(warmup_report) != expected:
            raise RuntimeError(f"{dataset_id} warmup changed the scientific result")

    reports: list[RealDataValidationReport] = []
    wall_ms: list[float] = []
    for _ in range(repetitions):
        report, elapsed = _run_once(operation)
        if scientific_record(report) != expected:
            raise RuntimeError(f"{dataset_id} repetitions are not deterministic")
        reports.append(report)
        wall_ms.append(elapsed)

    return {
        "dataset_id": dataset_id,
        "scientific_fingerprint_sha256": scientific_fingerprint(expected),
        "scientific_result": expected,
        "timing": {
            "cold": {
                "driver_wall_ms": cold_wall_ms,
                "reported_workflow_ms": 1_000.0 * cold_report.elapsed_seconds,
            },
            "warm": {
                "driver_wall": timing_summary(wall_ms),
                "reported_workflow": timing_summary(
                    [1_000.0 * report.elapsed_seconds for report in reports]
                ),
            },
        },
        "discarded_warmups": warmups,
        "measured_repetitions": repetitions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_IDS,
        help="dataset to benchmark; repeat the option to select several (default: all)",
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=REPOSITORY_ROOT / "validation" / "data",
        help="parent directory containing one checksum-pinned directory per dataset",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        action="append",
        help=(
            "QARR worker count; repeat to compare settings, zero selects automatic (default: 1, 2)"
        ),
    )
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")
    selected_datasets = DATASET_IDS if arguments.dataset is None else tuple(arguments.dataset)
    if len(set(selected_datasets)) != len(selected_datasets):
        raise ValueError("dataset selections must be unique")
    selected_threads = (1, 2) if arguments.threads is None else tuple(arguments.threads)
    if len(set(selected_threads)) != len(selected_threads) or any(
        threads < 0 for threads in selected_threads
    ):
        raise ValueError("thread selections must be unique and nonnegative")

    cases = []
    qarr_scientific_result: dict[str, Any] | None = None
    for dataset_id in selected_datasets:
        directory = arguments.data_directory / dataset_id
        verify_validation_dataset(dataset_id, directory)
        if dataset_id == "aps-sucrose-11bmb":
            result = benchmark_operation(
                dataset_id,
                lambda directory=directory: run_sucrose_lebail_validation(directory),
                warmups=arguments.warmups,
                repetitions=arguments.repetitions,
            )
            result["configuration"] = {"threads": None}
            cases.append(result)
            continue

        if dataset_id == "gsasii-pbso4-cw":
            for probe in (RadiationProbe.X_RAY, RadiationProbe.NEUTRON):
                execution = phasesmith.ExecutionPolicy(threads=1)
                result = benchmark_operation(
                    f"{dataset_id}-{probe.value}",
                    lambda directory=directory, probe=probe, execution=execution: (
                        run_pbso4_cw_validation(directory, probe, execution=execution)
                    ),
                    warmups=arguments.warmups,
                    repetitions=arguments.repetitions,
                )
                result["configuration"] = {
                    "threads": execution.threads,
                    "probe": probe.value,
                }
                cases.append(result)
            continue

        for threads in selected_threads:
            execution = phasesmith.ExecutionPolicy(threads=None if threads == 0 else threads)
            result = benchmark_operation(
                dataset_id,
                lambda directory=directory, execution=execution: run_qarr_1g_validation(
                    directory, execution=execution
                ),
                warmups=arguments.warmups,
                repetitions=arguments.repetitions,
            )
            result["configuration"] = {"threads": execution.threads}
            if qarr_scientific_result is None:
                qarr_scientific_result = result["scientific_result"]
            elif result["scientific_result"] != qarr_scientific_result:
                raise RuntimeError("QARR thread configurations changed the scientific result")
            cases.append(result)

    record = {
        "schema_version": 1,
        "scope": "checksum_pinned_real_data_native_workflows",
        "build_mode": phasesmith._core.BUILD_MODE,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "cases": cases,
    }
    output = json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
