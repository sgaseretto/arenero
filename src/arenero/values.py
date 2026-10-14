"""Convert public results to small, explicit JSON-compatible values."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from pathlib import PurePath
from typing import Any

from pydantic_monty import MontyClassProxy


def decode_value(value: Any) -> Any:
    """Explicitly decode Arenero bytes and complex records in JSON data.

    Args:
        value: Result value, not the complete execution envelope. Exact bytes
            and complex records are decoded recursively; other data stays data.

    Returns:
        Copied dictionaries/lists with native bytes and complex values.

    Raises:
        ValueError: If an exact recognized record has an invalid payload.
    """
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"encoding", "content"} and value["encoding"] == "base64":
        try:
            return base64.b64decode(value["content"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid Arenero Base64 record") from exc
    if set(value) == {"type", "real", "imag"} and value["type"] == "complex":
        if any(type(value[key]) not in {int, float} for key in ("real", "imag")):
            raise ValueError("complex record real and imag must be numbers")
        return complex(value["real"], value["imag"])
    return {key: decode_value(item) for key, item in value.items()}


def json_value(value: Any) -> Any:
    """Convert a result without changing values inside the interpreter.

    Args:
        value: Kernel, managed, or host result to represent.

    Returns:
        JSON data. Bytes use Base64, complex numbers use real/imaginary fields,
        dates use ISO text, and Monty instances expose copied attributes only.
        Other unsupported objects use their text representation.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "content": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, complex):
        return {"type": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, MontyClassProxy):
        return {
            "type": "monty_instance",
            "class": value.name,
            "attributes": json_value(value.attributes),
            "is_dataclass": value.is_dataclass,
        }
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    return repr(value)
