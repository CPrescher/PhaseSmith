from __future__ import annotations

import pytest
from phasesmith import CalculationOptions, ExecutionPolicy
from phasesmith.execution import execution_pool
from phasesmith.refinement.lebail import LeBailOptions


@pytest.mark.parametrize("threads", (0, -1, True, 1.5, "2"))
def test_execution_policy_rejects_invalid_thread_counts(threads: object) -> None:
    with pytest.raises(ValueError, match="threads"):
        ExecutionPolicy(threads=threads)  # type: ignore[arg-type]


@pytest.mark.parametrize("minimum", (0, -1, True, 1.5, "2"))
def test_execution_policy_rejects_invalid_parallel_threshold(minimum: object) -> None:
    with pytest.raises(ValueError, match="minimum_parallel_tasks"):
        ExecutionPolicy(minimum_parallel_tasks=minimum)  # type: ignore[arg-type]


def test_execution_policy_bounds_workers_and_skips_small_task_sets() -> None:
    policy = ExecutionPolicy(threads=8, minimum_parallel_tasks=3)
    assert policy.resolved_budget() >= 1
    assert policy.python_worker_count(2) == 1
    assert policy.python_worker_count(3) == policy.resolved_threads(3)
    assert policy.resolved_threads(0) == 1
    assert policy.resolved_threads(2) == 1
    assert 1 <= policy.resolved_threads(3) <= 3
    with pytest.raises(ValueError, match="task_count"):
        policy.resolved_threads(-1)


def test_execution_pool_uses_serial_sentinel_below_threshold() -> None:
    with execution_pool(ExecutionPolicy(threads=None), 1) as executor:
        assert executor is None


def test_calculation_and_lebail_options_require_explicit_execution_policy() -> None:
    policy = ExecutionPolicy(threads=2)
    assert CalculationOptions(execution=policy).execution is policy
    assert LeBailOptions(execution=policy).execution is policy
    with pytest.raises(TypeError, match="execution"):
        CalculationOptions(execution=2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="execution"):
        LeBailOptions(execution=2)  # type: ignore[arg-type]
