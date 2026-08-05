"""Small progress and cooperative-cancellation contracts for integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One operation-boundary progress notification with plain diagnostics."""

    stage: str
    completed: int
    total: int
    diagnostics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate monotonic counter bounds and stable stage labels."""

        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("progress stage must be a non-empty string")
        if self.total < 0 or self.completed < 0 or self.completed > self.total:
            raise ValueError("progress counters must satisfy 0 <= completed <= total")


@runtime_checkable
class ProgressCallback(Protocol):
    """Callable receiving progress only at calculation/iteration boundaries."""

    def __call__(self, event: ProgressEvent) -> None:
        """Consume one immutable progress event."""


@runtime_checkable
class CancellationCallback(Protocol):
    """Callable polled at safe Python orchestration boundaries."""

    def __call__(self) -> bool:
        """Return true when the current operation should stop cooperatively."""


class OperationCancelled(RuntimeError):
    """Raised when a calculation is cancelled before or after its native batch."""


def check_cancelled(cancellation: CancellationCallback | None) -> None:
    """Raise at a declared safe boundary when cancellation is requested."""

    if cancellation is not None and cancellation():
        raise OperationCancelled("operation cancelled")


def report_progress(progress: ProgressCallback | None, event: ProgressEvent) -> None:
    """Call an optional progress consumer synchronously."""

    if progress is not None:
        progress(event)
