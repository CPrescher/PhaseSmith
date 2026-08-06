"""Optional terminal-to-cancellation adapter; numerical modules never import it."""

from __future__ import annotations

import os
import select
import signal
import sys
from threading import Event, Thread, current_thread, main_thread
from types import FrameType
from typing import IO

from .control import CancellationToken


class TerminalCancellationController:
    """Map `q` and interrupts to a shared cooperative cancellation token."""

    def __init__(
        self,
        token: CancellationToken | None = None,
        *,
        input_stream: IO[str] | None = None,
        output_stream: IO[str] | None = None,
        quit_key: str = "q",
        enable_quit_key: bool = True,
    ) -> None:
        if len(quit_key) != 1:
            raise ValueError("quit_key must be exactly one character")
        self.token = CancellationToken() if token is None else token
        if not isinstance(self.token, CancellationToken):
            raise TypeError("token must be CancellationToken")
        self.input_stream = sys.stdin if input_stream is None else input_stream
        self.output_stream = sys.stderr if output_stream is None else output_stream
        self.quit_key = quit_key
        self.enable_quit_key = enable_quit_key
        self._previous_sigint: object | None = None
        self._interrupt_count = 0
        self._stop = Event()
        self._thread: Thread | None = None
        self._terminal_attributes: object | None = None

    def _notify(self, message: str) -> None:
        self.output_stream.write(message + "\n")
        self.output_stream.flush()

    def handle_key(self, key: str) -> bool:
        """Handle one decoded key and return whether it requested cancellation."""

        if key.lower() != self.quit_key.lower():
            return False
        if self.token.request("quit_key"):
            self._notify("Graceful refinement stop requested; finishing the current batch.")
        return True

    def handle_interrupt(self, signum: int, frame: FrameType | None) -> None:
        """Request graceful cancellation once and delegate a second interrupt."""

        self._interrupt_count += 1
        if self._interrupt_count == 1:
            self.token.request("keyboard_interrupt")
            self._notify(
                "Graceful refinement stop requested; press Ctrl+C again to force interruption."
            )
            return
        previous = self._previous_sigint
        if callable(previous):
            previous(signum, frame)
        else:
            signal.default_int_handler(signum, frame)

    def _listen_posix(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self.input_stream], [], [], 0.1)
            if readable:
                key = self.input_stream.read(1)
                if key:
                    self.handle_key(key)

    def _listen_windows(self) -> None:  # pragma: no cover - exercised on Windows CI
        import msvcrt

        while not self._stop.wait(0.05):
            if msvcrt.kbhit():
                self.handle_key(msvcrt.getwch())

    def __enter__(self) -> TerminalCancellationController:
        if current_thread() is not main_thread():
            raise RuntimeError("terminal signal control must be entered on the main thread")
        self._previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handle_interrupt)
        try:
            if self.enable_quit_key and self.input_stream.isatty():
                if os.name == "nt":
                    target = self._listen_windows
                else:
                    import termios
                    import tty

                    descriptor = self.input_stream.fileno()
                    self._terminal_attributes = termios.tcgetattr(descriptor)
                    tty.setcbreak(descriptor)
                    target = self._listen_posix
                self._thread = Thread(target=target, name="rietveld-terminal-cancel", daemon=True)
                self._thread.start()
        except Exception:
            if self._terminal_attributes is not None:
                import termios

                termios.tcsetattr(
                    self.input_stream.fileno(), termios.TCSADRAIN, self._terminal_attributes
                )
            signal.signal(signal.SIGINT, self._previous_sigint)
            raise
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._terminal_attributes is not None:
            import termios

            termios.tcsetattr(
                self.input_stream.fileno(), termios.TCSADRAIN, self._terminal_attributes
            )
        if self._previous_sigint is not None:
            signal.signal(signal.SIGINT, self._previous_sigint)
