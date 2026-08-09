from __future__ import annotations

from dataclasses import asdict
from pickle import dumps, loads

import pytest
from phasesmith import CalculationOptions, CancellationToken, ExecutionPolicy
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


def test_execution_policy_keeps_native_runtime_out_of_dataclass_records() -> None:
    policy = ExecutionPolicy(threads=1, minimum_parallel_tasks=3)
    assert asdict(policy) == {"threads": 1, "minimum_parallel_tasks": 3}
    assert policy == ExecutionPolicy(threads=1, minimum_parallel_tasks=3)
    assert loads(dumps(policy)) == policy


def test_execution_pool_uses_serial_sentinel_below_threshold() -> None:
    with execution_pool(ExecutionPolicy(threads=None), 1) as executor:
        assert executor is None


def test_cancellation_token_shares_first_reason_with_native_solver_handle() -> None:
    token = CancellationToken()

    assert token.request("native-stop") is True
    assert token.request("ignored") is False
    assert token.reason == "native-stop"
    assert token._native.reason == "native-stop"


def test_calculation_and_lebail_options_require_explicit_execution_policy() -> None:
    policy = ExecutionPolicy(threads=2)
    assert CalculationOptions(execution=policy).execution is policy
    assert LeBailOptions(execution=policy).execution is policy
    with pytest.raises(TypeError, match="execution"):
        CalculationOptions(execution=2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="execution"):
        LeBailOptions(execution=2)  # type: ignore[arg-type]
