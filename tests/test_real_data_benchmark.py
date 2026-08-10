from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from phasesmith.validation.real_data import RealDataValidationReport, ValidationCheck

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_benchmark() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "real_data_benchmark_test", REPOSITORY_ROOT / "benchmarks" / "real_data.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def report(*, elapsed: float, measured: float = 0.125) -> RealDataValidationReport:
    return RealDataValidationReport(
        dataset_id="test-data",
        status="passed",
        sample_count=101,
        reflection_count=7,
        elapsed_seconds=elapsed,
        checks=(
            ValidationCheck(
                "residual", "passed", "stable", measured=measured, criterion="value <= 1"
            ),
        ),
        notes=("deterministic scientific note",),
    )


def test_scientific_fingerprint_excludes_host_timing() -> None:
    benchmark = load_benchmark()
    first = benchmark.scientific_record(report(elapsed=0.1))
    second = benchmark.scientific_record(report(elapsed=9.9))

    assert first == second
    assert "elapsed_seconds" not in first
    assert benchmark.scientific_fingerprint(first) == benchmark.scientific_fingerprint(second)


def test_benchmark_reports_cold_warm_and_exact_repeatability() -> None:
    benchmark = load_benchmark()
    calls = 0

    def operation() -> RealDataValidationReport:
        nonlocal calls
        calls += 1
        return report(elapsed=0.001 * calls)

    result = benchmark.benchmark_operation("test-data", operation, warmups=2, repetitions=3)

    assert calls == 6
    assert result["measured_repetitions"] == 3
    assert result["discarded_warmups"] == 2
    assert len(result["timing"]["warm"]["driver_wall"]["timings_ms"]) == 3
    assert result["timing"]["warm"]["reported_workflow"]["timings_ms"] == [4.0, 5.0, 6.0]
    assert len(result["scientific_fingerprint_sha256"]) == 64


def test_benchmark_rejects_scientific_drift() -> None:
    benchmark = load_benchmark()
    calls = 0

    def operation() -> RealDataValidationReport:
        nonlocal calls
        calls += 1
        return report(elapsed=0.1, measured=0.125 if calls == 1 else 0.126)

    with pytest.raises(RuntimeError, match="repetitions are not deterministic"):
        benchmark.benchmark_operation("test-data", operation, warmups=0, repetitions=1)


def test_timing_controls_are_validated() -> None:
    benchmark = load_benchmark()

    with pytest.raises(ValueError, match="nonnegative"):
        benchmark.benchmark_operation(
            "test-data", lambda: report(elapsed=0.1), warmups=-1, repetitions=1
        )
    with pytest.raises(ValueError, match="non-empty"):
        benchmark.timing_summary([])
