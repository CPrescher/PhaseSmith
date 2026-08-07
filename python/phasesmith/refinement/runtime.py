"""Method-independent bounded refinement runtime and structured event stream."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import IO, Protocol, TypeAlias, runtime_checkable

import numpy as np

from ..control import CancellationCallback
from .core import TerminationReason

DiagnosticValue: TypeAlias = str | bool | int | float | None
Clock: TypeAlias = Callable[[], float]


class RefinementEventKind(StrEnum):
    """Stable machine-readable event categories."""

    START = "start"
    TRIAL = "trial"
    STEP_ACCEPTED = "step_accepted"
    STEP_REJECTED = "step_rejected"
    ITERATION = "iteration"
    CHECKPOINT = "checkpoint"
    WARNING = "warning"
    TERMINATION = "termination"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class RefinementEvent:
    """One immutable boundary event suitable for console or JSON logging."""

    kind: RefinementEventKind
    stage: str
    attempted_iteration: int
    accepted_iterations: int
    evaluations: int
    elapsed_seconds: float
    message: str
    diagnostics: tuple[tuple[str, DiagnosticValue], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RefinementEventKind):
            raise TypeError("kind must be RefinementEventKind")
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        if min(self.attempted_iteration, self.accepted_iterations, self.evaluations) < 0:
            raise ValueError("event counters must be non-negative")
        if self.accepted_iterations > self.attempted_iteration:
            raise ValueError("accepted iterations cannot exceed attempted iterations")
        if not np.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be non-negative and finite")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")
        keys = []
        for key, value in self.diagnostics:
            if not isinstance(key, str) or not key:
                raise ValueError("diagnostic keys must be non-empty strings")
            if not isinstance(value, (str, bool, int, float, type(None))):
                raise TypeError("diagnostic values must be JSON scalar values")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("floating-point diagnostics must be finite")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError("diagnostic keys must be unique within an event")

    def as_record(self) -> dict[str, object]:
        """Return a finite JSON-compatible plain record."""

        return {
            "kind": self.kind.value,
            "stage": self.stage,
            "attempted_iteration": self.attempted_iteration,
            "accepted_iterations": self.accepted_iterations,
            "evaluations": self.evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "message": self.message,
            "diagnostics": dict(self.diagnostics),
        }


@runtime_checkable
class RefinementLogger(Protocol):
    """Synchronous consumer of immutable refinement events."""

    def __call__(self, event: RefinementEvent) -> None:
        """Consume one event at an orchestration boundary."""


@runtime_checkable
class CheckpointCallback(Protocol):
    """Application-owned durable checkpoint sink."""

    def __call__(self, checkpoint: object) -> None:
        """Persist one complete accepted-state checkpoint."""


class ConsoleRefinementLogger:
    """Compact human-readable logger writing only to a supplied stream."""

    def __init__(self, stream: IO[str] | None = None, *, include_trials: bool = False) -> None:
        self.stream = sys.stderr if stream is None else stream
        self.include_trials = include_trials

    def __call__(self, event: RefinementEvent) -> None:
        if not self.include_trials and event.kind in (
            RefinementEventKind.TRIAL,
            RefinementEventKind.STEP_REJECTED,
        ):
            return
        diagnostics = " ".join(f"{key}={value}" for key, value in event.diagnostics)
        suffix = "" if not diagnostics else f" {diagnostics}"
        self.stream.write(
            f"[{event.elapsed_seconds:9.3f}s] {event.kind.value} "
            f"iteration={event.attempted_iteration} accepted={event.accepted_iterations} "
            f"evaluations={event.evaluations} {event.message}{suffix}\n"
        )
        self.stream.flush()


class JsonLinesRefinementLogger:
    """Finite JSON-lines logger for batch systems and reproducible diagnostics."""

    def __init__(self, stream: IO[str]) -> None:
        self.stream = stream

    def __call__(self, event: RefinementEvent) -> None:
        self.stream.write(json.dumps(event.as_record(), sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()


@dataclass(frozen=True, slots=True)
class RefinementLimits:
    """Hard upper bounds checked independently of convergence criteria."""

    max_iterations: int = 100
    max_evaluations: int = 1_000
    max_runtime_seconds: float | None = None
    max_consecutive_rejections: int = 20

    def __post_init__(self) -> None:
        for name, value in (
            ("max_iterations", self.max_iterations),
            ("max_evaluations", self.max_evaluations),
            ("max_consecutive_rejections", self.max_consecutive_rejections),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_runtime_seconds is not None and (
            not np.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds <= 0.0
        ):
            raise ValueError("max_runtime_seconds must be positive and finite when present")


class RefinementStopped(RuntimeError):
    """Internal control-flow signal for a normal bounded termination."""

    def __init__(self, reason: TerminationReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class CheckpointCallbackError(RuntimeError):
    """Raised when an accepted state could not be sent to its checkpoint sink."""


class RefinementRuntime:
    """Stateful boundary guard shared by refinement method implementations."""

    def __init__(
        self,
        limits: RefinementLimits,
        *,
        cancellation: CancellationCallback | None = None,
        logger: RefinementLogger | None = None,
        checkpoint: CheckpointCallback | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        if not isinstance(limits, RefinementLimits):
            raise TypeError("limits must be RefinementLimits")
        if cancellation is not None and not isinstance(cancellation, CancellationCallback):
            raise TypeError("cancellation must implement CancellationCallback")
        if logger is not None and not isinstance(logger, RefinementLogger):
            raise TypeError("logger must implement RefinementLogger")
        if checkpoint is not None and not isinstance(checkpoint, CheckpointCallback):
            raise TypeError("checkpoint must implement CheckpointCallback")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.limits = limits
        self.cancellation = cancellation
        self.logger = logger
        self.checkpoint_callback = checkpoint
        self.clock = clock
        self.started_at = float(clock())
        if not np.isfinite(self.started_at):
            raise ValueError("clock must return finite values")
        self.attempted_iteration = 0
        self.accepted_iterations = 0
        self.evaluations = 0
        self.consecutive_rejections = 0
        self.logger_error: Exception | None = None

    @property
    def elapsed_seconds(self) -> float:
        elapsed = float(self.clock()) - self.started_at
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("refinement clock must be finite and monotonic")
        return elapsed

    def emit(
        self,
        kind: RefinementEventKind,
        stage: str,
        message: str,
        diagnostics: tuple[tuple[str, DiagnosticValue], ...] = (),
    ) -> RefinementEvent:
        event = RefinementEvent(
            kind,
            stage,
            self.attempted_iteration,
            self.accepted_iterations,
            self.evaluations,
            self.elapsed_seconds,
            message,
            diagnostics,
        )
        if self.logger is not None:
            try:
                self.logger(event)
            except Exception as error:  # logger output must not invalidate numerical state
                self.logger_error = error
                self.logger = None
        return event

    def check_boundary(self) -> None:
        """Stop at a safe boundary when cancellation or a hard budget is reached."""

        if self.cancellation is not None and self.cancellation():
            reason = getattr(self.cancellation, "reason", None)
            message = "user requested cancellation" if reason is None else str(reason)
            raise RefinementStopped(TerminationReason.CANCELLED, message)
        if (
            self.limits.max_runtime_seconds is not None
            and self.elapsed_seconds >= self.limits.max_runtime_seconds
        ):
            raise RefinementStopped(TerminationReason.MAX_RUNTIME, "runtime limit reached")
        if self.evaluations >= self.limits.max_evaluations:
            raise RefinementStopped(
                TerminationReason.MAX_EVALUATIONS, "model-evaluation limit reached"
            )

    def begin_iteration(self, attempted_iteration: int) -> None:
        if attempted_iteration <= self.attempted_iteration:
            raise ValueError("attempted iterations must increase strictly")
        if attempted_iteration > self.limits.max_iterations:
            raise RefinementStopped(TerminationReason.MAX_ITERATIONS, "iteration limit reached")
        self.attempted_iteration = attempted_iteration
        self.check_boundary()

    def begin_evaluation(self) -> None:
        self.check_boundary()
        self.evaluations += 1

    def accept_step(self, checkpoint: object | None = None) -> None:
        if self.accepted_iterations >= self.attempted_iteration:
            raise RuntimeError("at most one step may be accepted per attempted iteration")
        self.accepted_iterations += 1
        self.consecutive_rejections = 0
        if checkpoint is not None and self.checkpoint_callback is not None:
            try:
                self.checkpoint_callback(checkpoint)
            except Exception as error:
                raise CheckpointCallbackError(
                    "accepted state could not be written by the checkpoint callback"
                ) from error
            self.emit(
                RefinementEventKind.CHECKPOINT,
                "checkpoint",
                "accepted-state checkpoint completed",
            )

    def reject_step(self) -> None:
        self.consecutive_rejections += 1
        if self.consecutive_rejections >= self.limits.max_consecutive_rejections:
            raise RefinementStopped(
                TerminationReason.REPEATED_REJECTIONS,
                "consecutive rejected-step limit reached",
            )
