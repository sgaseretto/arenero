"""Small, safe event records for Arenero integrations."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

EventCategory = Literal[
    "execution",
    "session",
    "filesystem",
    "network",
    "mcp",
    "hosted_tool",
    "rlm",
    "speculation",
    "runtime",
]


@dataclass(frozen=True)
class AreneroEvent:
    """Describe one safe observable Arenero event.

    Attributes:
        category: High-level subsystem that produced the event.
        action: Stable event action such as ``started`` or ``finished``.
        timestamp: Unix timestamp recorded by the host.
        data: Safe metadata that does not include code, file contents, or secrets.
    """

    category: EventCategory
    action: str
    timestamp: float
    data: Mapping[str, Any]


EventSink = Callable[[AreneroEvent], None]


class OpenTelemetryEventSink:
    """Write Arenero events to the current OpenTelemetry trace."""

    def __init__(self, tracer_name: str = "arenero") -> None:
        """Create an OpenTelemetry event sink.

        Args:
            tracer_name: Instrumentation scope used when no span is active.

        Raises:
            ImportError: If the optional OpenTelemetry package is unavailable.
        """

        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "OpenTelemetry support requires: uv add 'arenero[observability]'"
            ) from exc
        self._trace = trace
        self._tracer = trace.get_tracer(tracer_name)

    def __call__(self, event: AreneroEvent) -> None:
        """Add one event to an active span or create a short event span.

        Args:
            event: Safe Arenero event record.
        """

        name = f"arenero.{event.category}.{event.action}"
        attributes = _otel_attributes(event.data)
        current = self._trace.get_current_span()
        if current.is_recording():
            current.add_event(
                name, attributes=attributes, timestamp=int(event.timestamp * 1e9)
            )
            return
        with self._tracer.start_as_current_span(name) as span:
            span.set_attributes(attributes)


def emit_event(
    sink: EventSink | None,
    category: EventCategory,
    action: str,
    **data: Any,
) -> None:
    """Send one event without changing the operation result.

    Args:
        sink: Optional callback that accepts an ``AreneroEvent``.
        category: Subsystem that produced the event.
        action: Stable event action.
        **data: Safe metadata for the event.
    """
    if sink is None:
        return
    try:
        sink(AreneroEvent(category, action, time.time(), data))
    except Exception:  # noqa: BLE001 - observers must not break the sandbox
        return


def _otel_attributes(data: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only OpenTelemetry-compatible scalar event values.

    Args:
        data: Safe Arenero event metadata.

    Returns:
        Scalar attributes with an ``arenero.`` prefix.
    """

    supported = (str, bool, int, float)
    return {
        f"arenero.{key}": value
        for key, value in data.items()
        if value is not None and isinstance(value, supported)
    }
