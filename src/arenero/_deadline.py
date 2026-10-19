"""Cooperative deadlines for the limited virtual shell."""

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_end: ContextVar[float | None] = ContextVar("arenero_shell_deadline", default=None)


def remaining() -> float | None:
    """Check the active deadline.

    Returns:
        Remaining seconds, or None when no per-call deadline is active.

    Raises:
        TimeoutError: When the shell must stop before further work.
    """
    end = _end.get()
    if end is None:
        return None
    value = end - time.monotonic()
    if value <= 0:
        raise TimeoutError("virtual shell deadline exceeded; completed effects remain")
    return value


@contextmanager
def shell_deadline(timeout: int | None) -> Iterator[None]:
    """Set a call-local shell deadline without changing kernel policy.

    Args:
        timeout: Positive seconds; None or zero keeps the configured limits only.

    Yields:
        Control with a deadline inherited by nested calls and worker threads.
    """
    if timeout is not None and (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise ValueError("timeout must be a non-negative number or None")
    outer = _end.get()
    end = time.monotonic() + timeout if timeout else outer
    token = _end.set(min(outer, end) if outer is not None and end is not None else end)
    try:
        remaining()
        yield
    finally:
        _end.reset(token)
