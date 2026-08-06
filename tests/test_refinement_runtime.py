from __future__ import annotations

import io
import json
import signal

import pytest
import rietveld
from rietveld.refinement import (
    CheckpointCallbackError,
    ConsoleRefinementLogger,
    JsonLinesRefinementLogger,
    RefinementEvent,
    RefinementEventKind,
    RefinementLimits,
    RefinementRuntime,
    RefinementStopped,
    TerminationReason,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def test_cancellation_token_is_thread_safe_callback_with_first_reason() -> None:
    token = rietveld.CancellationToken()
    assert not token()
    assert token.request("stop_button")
    assert not token.request("later_reason")
    assert token()
    assert token.reason == "stop_button"
    with pytest.raises(rietveld.OperationCancelled, match="stop_button") as raised:
        rietveld.control.check_cancelled(token)
    assert raised.value.reason == "stop_button"


def test_structured_events_feed_human_and_finite_json_logs() -> None:
    event = RefinementEvent(
        RefinementEventKind.ITERATION,
        "rietveld",
        3,
        2,
        12,
        1.25,
        "iteration completed",
        (("rwp", 0.052), ("accepted", True), ("warning", None)),
    )
    human = io.StringIO()
    machine = io.StringIO()
    ConsoleRefinementLogger(human)(event)
    JsonLinesRefinementLogger(machine)(event)
    assert "rwp=0.052" in human.getvalue()
    record = json.loads(machine.getvalue())
    assert record["kind"] == "iteration"
    assert record["diagnostics"]["accepted"] is True
    with pytest.raises(ValueError, match="finite"):
        RefinementEvent(
            RefinementEventKind.WARNING,
            "rietveld",
            0,
            0,
            0,
            0.0,
            "bad diagnostic",
            (("value", float("nan")),),
        )


def test_runtime_enforces_cancellation_runtime_evaluation_and_iteration_limits() -> None:
    clock = FakeClock()
    token = rietveld.CancellationToken()
    runtime = RefinementRuntime(
        RefinementLimits(max_iterations=2, max_evaluations=2, max_runtime_seconds=5.0),
        cancellation=token,
        clock=clock,
    )
    runtime.begin_iteration(1)
    runtime.begin_evaluation()
    runtime.begin_evaluation()
    with pytest.raises(RefinementStopped) as evaluations:
        runtime.begin_evaluation()
    assert evaluations.value.reason is TerminationReason.MAX_EVALUATIONS

    runtime = RefinementRuntime(
        RefinementLimits(max_iterations=2, max_evaluations=10, max_runtime_seconds=5.0),
        cancellation=token,
        clock=clock,
    )
    clock.value += 5.0
    with pytest.raises(RefinementStopped) as elapsed:
        runtime.begin_iteration(1)
    assert elapsed.value.reason is TerminationReason.MAX_RUNTIME

    clock.value += 1.0
    token = rietveld.CancellationToken()
    runtime = RefinementRuntime(RefinementLimits(max_iterations=1), cancellation=token, clock=clock)
    token.request("quit_key")
    with pytest.raises(RefinementStopped, match="quit_key") as cancelled:
        runtime.begin_iteration(1)
    assert cancelled.value.reason is TerminationReason.CANCELLED

    runtime = RefinementRuntime(RefinementLimits(max_iterations=1), clock=clock)
    runtime.begin_iteration(1)
    with pytest.raises(RefinementStopped) as iterations:
        runtime.begin_iteration(2)
    assert iterations.value.reason is TerminationReason.MAX_ITERATIONS


def test_runtime_bounds_rejections_and_isolates_logger_failure() -> None:
    clock = FakeClock()

    def broken_logger(event: RefinementEvent) -> None:
        del event
        raise OSError("closed log stream")

    runtime = RefinementRuntime(
        RefinementLimits(max_consecutive_rejections=2),
        logger=broken_logger,
        clock=clock,
    )
    runtime.emit(RefinementEventKind.START, "rietveld", "started")
    assert isinstance(runtime.logger_error, OSError)
    assert runtime.logger is None
    runtime.reject_step()
    with pytest.raises(RefinementStopped) as rejected:
        runtime.reject_step()
    assert rejected.value.reason is TerminationReason.REPEATED_REJECTIONS


def test_accepted_checkpoint_failure_is_typed_and_state_remains_accepted() -> None:
    def broken_checkpoint(checkpoint: object) -> None:
        assert checkpoint == {"iteration": 1}
        raise OSError("disk full")

    runtime = RefinementRuntime(
        RefinementLimits(),
        checkpoint=broken_checkpoint,
        clock=FakeClock(),
    )
    runtime.begin_iteration(1)
    with pytest.raises(CheckpointCallbackError, match="could not be written"):
        runtime.accept_step({"iteration": 1})
    assert runtime.accepted_iterations == 1


def test_terminal_controller_maps_q_and_two_interrupts_without_owning_numerics() -> None:
    output = io.StringIO()
    controller = rietveld.TerminalCancellationController(
        input_stream=io.StringIO(), output_stream=output, enable_quit_key=False
    )
    assert not controller.handle_key("x")
    assert controller.handle_key("Q")
    assert controller.token.reason == "quit_key"
    assert "Graceful" in output.getvalue()

    token = rietveld.CancellationToken()
    controller = rietveld.TerminalCancellationController(
        token, input_stream=io.StringIO(), output_stream=output, enable_quit_key=False
    )
    forced = []
    controller._previous_sigint = lambda signum, frame: forced.append((signum, frame))
    controller.handle_interrupt(signal.SIGINT, None)
    assert token.reason == "keyboard_interrupt"
    assert forced == []
    controller.handle_interrupt(signal.SIGINT, None)
    assert forced == [(signal.SIGINT, None)]


def test_terminal_controller_restores_previous_interrupt_handler() -> None:
    previous = signal.getsignal(signal.SIGINT)
    controller = rietveld.TerminalCancellationController(
        input_stream=io.StringIO(), output_stream=io.StringIO(), enable_quit_key=False
    )
    with controller:
        assert signal.getsignal(signal.SIGINT) == controller.handle_interrupt
    assert signal.getsignal(signal.SIGINT) == previous
