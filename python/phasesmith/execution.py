"""Explicit, bounded execution controls for scriptable PhaseSmith workloads."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from os import cpu_count


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Control CPU concurrency without changing numerical ordering.

    ``threads=1`` is the conservative default for embedding applications. ``None``
    selects the available logical CPU count. Work is parallelized only when at
    least ``minimum_parallel_tasks`` independent tasks are available.
    """

    threads: int | None = 1
    minimum_parallel_tasks: int = 2

    def __post_init__(self) -> None:
        if self.threads is not None and (
            isinstance(self.threads, bool)
            or not isinstance(self.threads, int)
            or self.threads <= 0
        ):
            raise ValueError("threads must be None or a positive integer")
        if (
            isinstance(self.minimum_parallel_tasks, bool)
            or not isinstance(self.minimum_parallel_tasks, int)
            or self.minimum_parallel_tasks <= 0
        ):
            raise ValueError("minimum_parallel_tasks must be a positive integer")

    def resolved_threads(self, task_count: int) -> int:
        """Return the bounded worker count for a known number of tasks."""

        if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 0:
            raise ValueError("task_count must be a non-negative integer")
        if task_count < self.minimum_parallel_tasks:
            return 1
        available = max(1, cpu_count() or 1)
        requested = available if self.threads is None else self.threads
        return max(1, min(requested, available, task_count))


@contextmanager
def execution_pool(
    policy: ExecutionPolicy,
    task_count: int,
) -> Iterator[Executor | None]:
    """Yield one reusable pool, or ``None`` when serial execution is selected."""

    if not isinstance(policy, ExecutionPolicy):
        raise TypeError("policy must be ExecutionPolicy")
    workers = policy.resolved_threads(task_count)
    if workers == 1:
        yield None
        return
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="phasesmith",
    ) as executor:
        yield executor
