"""Small progress and cooperative-cancellation contracts for integrations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock
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


class CancellationToken:
    """Thread-safe cooperative cancellation shared by scripts, GUIs, and CLIs."""

    __slots__ = ("_event", "_lock", "_reason")

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._reason: str | None = None

    def request(self, reason: str = "user_requested") -> bool:
        """Request cancellation and return true only for the first request."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must be a non-empty string")
        with self._lock:
            first = not self._event.is_set()
            if first:
                self._reason = reason
                self._event.set()
            return first

    @property
    def reason(self) -> str | None:
        """Return the first cancellation reason, if cancellation was requested."""

        with self._lock:
            return self._reason

    def __call__(self) -> bool:
        """Implement :class:`CancellationCallback`."""

        return self._event.is_set()


class OperationCancelled(RuntimeError):
    """Raised when a calculation is cancelled before or after its native batch."""

    def __init__(self, reason: str = "operation cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


def check_cancelled(cancellation: CancellationCallback | None) -> None:
    """Raise at a declared safe boundary when cancellation is requested."""

    if cancellation is not None and cancellation():
        reason = getattr(cancellation, "reason", None)
        raise OperationCancelled("operation cancelled" if reason is None else str(reason))


def report_progress(progress: ProgressCallback | None, event: ProgressEvent) -> None:
    """Call an optional progress consumer synchronously."""

    if progress is not None:
        progress(event)
