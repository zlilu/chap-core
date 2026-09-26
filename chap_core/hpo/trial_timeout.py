from __future__ import annotations

import os
import signal
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class HpoTrialTimeoutError(TimeoutError):
    """Raised when an HPO trial exceeds its configured wall-clock timeout."""


class HpoTrialCleanupError(RuntimeError):
    """
    Raised when a timed-out model command could not be safely cleaned up.

    This error must abort HPO rather than being treated as an ordinary failed
    trial, because the previous model process may still be running.
    """


# ContextVars representing dynamically scoped execution state
_trial_deadline: ContextVar[float | None] = ContextVar(
    "hpo_trial_deadline",
    default=None,
)

_trial_timeout_seconds: ContextVar[float | None] = ContextVar(
    "hpo_trial_timeout_seconds",
    default=None,
)


def current_trial_timeout_seconds() -> float | None:
    return _trial_timeout_seconds.get()


def _timeout_error() -> HpoTrialTimeoutError:
    timeout_seconds = current_trial_timeout_seconds()
    if timeout_seconds is None:
        return HpoTrialTimeoutError("HPO trial timed out")
    return HpoTrialTimeoutError(f"HPO trial exceeded timeout of {timeout_seconds:g} seconds")


def ensure_trial_timeout_runtime_supported() -> None:
    """Validate requirements of the CLI-local HPO timeout implementation."""
    if os.name != "posix":
        raise RuntimeError("HPO trial timeouts currently require a POSIX platform.")

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("HPO trial timeouts require SIGALRM and setitimer support.")

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("HPO trial timeouts currently require execution from the main thread.")


@contextmanager
def hpo_trial_timeout(timeout_seconds: float | None) -> Iterator[None]:
    """
    Apply one wall-clock deadline to an entire HPO objective evaluation.

    The SIGALRM timer covers Python-side work between model subprocesses.
    Model subprocesses temporarily suspend this timer and enforce the same
    absolute deadline through Popen.communicate(timeout=...).
    """
    if timeout_seconds is None:
        yield
        return

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    ensure_trial_timeout_runtime_supported()

    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("Cannot enable HPO trial timeout because ITIMER_REAL is already in use.")
    previous_handler = signal.getsignal(signal.SIGALRM)

    deadline = monotonic() + timeout_seconds
    deadline_token = _trial_deadline.set(deadline)
    timeout_token = _trial_timeout_seconds.set(timeout_seconds)

    def _handle_timeout(signum: int, frame: object) -> None:
        del signum, frame
        raise _timeout_error()

    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        _trial_timeout_seconds.reset(timeout_token)
        _trial_deadline.reset(deadline_token)


@contextmanager
def subprocess_trial_deadline() -> Iterator[float | None]:
    """
    Temporarily hand deadline enforcement from SIGALRM to a subprocess.
    Returns the absolute monotonic deadline. The caller should calculate
    deadline - monotonic() immediately before waiting on the subprocess.
    This prevents SIGALRM and Popen.communicate(timeout=...) from racing
    against each other.
    """
    deadline = _trial_deadline.get()
    if deadline is None:
        yield None
        return

    # communicate(timeout=...) owns timeout enforcement while the external model command is running.
    signal.setitimer(signal.ITIMER_REAL, 0.0)

    if deadline <= monotonic():
        raise _timeout_error()
    yield deadline
    # Code below runs only if the subprocess-owned section completes normally,
    # otherwise error and the outer hpo_trial_timeout() context will restore the signal state.
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise _timeout_error()

    # Continue enforcing the same original deadline for Python-side work.
    signal.setitimer(signal.ITIMER_REAL, remaining)
