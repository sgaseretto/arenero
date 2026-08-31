"""Materialize MCP binary content into an Arenero virtual filesystem."""

from __future__ import annotations

import base64
import copy
import mimetypes
import posixpath
import uuid
from collections.abc import Callable
from typing import Any

ArtifactWriter = Callable[[str, bytes], int]


def materialize_mcp_artifacts(
    value: Any,
    writer: ArtifactWriter,
    directory: str = "/mcp-artifacts",
) -> tuple[Any, list[str]]:
    """Write MCP image, audio, and embedded-resource blobs to virtual files.

    Args:
        value: Normalized MCP result or a value that contains one.
        writer: Host callback that writes bytes to an absolute virtual path.
        directory: Absolute virtual directory for generated artifact files.

    Returns:
        A copied value with ``virtual_path`` fields and the written paths.

    Raises:
        ValueError: If ``directory`` is not an absolute virtual path.
    """
    if not directory.startswith("/"):
        raise ValueError("MCP artifact directory must be an absolute virtual path")
    copied = copy.deepcopy(value)
    paths: list[str] = []
    _materialize_value(copied, writer, directory.rstrip("/") or "/", paths)
    return copied, paths


def _materialize_value(
    value: Any,
    writer: ArtifactWriter,
    directory: str,
    paths: list[str],
) -> None:
    """Walk one copied result and replace recognized binary payloads.

    Args:
        value: Current copied value.
        writer: Virtual-file writer callback.
        directory: Destination virtual directory.
        paths: Mutable list that receives written paths.
    """
    if isinstance(value, list):
        for item in value:
            _materialize_value(item, writer, directory, paths)
        return
    if not isinstance(value, dict):
        return

    block_type = str(value.get("type", "")).lower()
    if block_type in {"image", "audio"} and isinstance(value.get("data"), str):
        _write_payload(value, "data", writer, directory, paths)
    elif block_type in {"resource", "embedded_resource"}:
        resource = value.get("resource")
        if isinstance(resource, dict) and isinstance(resource.get("blob"), str):
            _write_payload(resource, "blob", writer, directory, paths)
    for item in list(value.values()):
        _materialize_value(item, writer, directory, paths)


def _write_payload(
    block: dict[str, Any],
    field: str,
    writer: ArtifactWriter,
    directory: str,
    paths: list[str],
) -> None:
    """Decode and write one MCP base64 payload.

    Args:
        block: Content block that owns the payload.
        field: Payload field, either ``data`` or ``blob``.
        writer: Virtual-file writer callback.
        directory: Destination virtual directory.
        paths: Mutable list that receives written paths.
    """
    encoded = block[field]
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        block["artifact_error"] = f"invalid base64 payload: {exc}"
        return
    mime_type = block.get("mimeType") or block.get("mime_type")
    extension = _extension(str(mime_type) if mime_type else None)
    path = posixpath.join(directory, f"{uuid.uuid4().hex}{extension}")
    try:
        writer(path, payload)
    except Exception as exc:  # noqa: BLE001 - report materialization failure in data
        block["artifact_error"] = str(exc)
        return
    del block[field]
    block["virtual_path"] = path
    block["size_bytes"] = len(payload)
    paths.append(path)


def _extension(mime_type: str | None) -> str:
    """Return a short extension for one media type.

    Args:
        mime_type: Optional MIME type.

    Returns:
        Known extension or ``.bin``.
    """
    if mime_type == "image/jpeg":
        return ".jpg"
    return mimetypes.guess_extension(mime_type or "") or ".bin"
