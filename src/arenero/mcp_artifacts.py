"""Materialize MCP binary content into an Arenero virtual filesystem."""

from __future__ import annotations

import base64
import copy
import mimetypes
import posixpath
import uuid
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

ArtifactWriter = Callable[[str, bytes], int]
AsyncArtifactWriter = Callable[[str, bytes], Awaitable[int]]


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
    copied = _copy_artifacts(value, directory)
    paths: list[str] = []
    for block, field, path, payload in _artifact_payloads(copied, directory):
        try:
            writer(path, payload)
        except Exception as exc:  # noqa: BLE001 - preserve failed blocks as data
            block["artifact_error"] = str(exc)
        else:
            _finish_artifact(block, field, path, payload)
            paths.append(path)
    return copied, paths


async def async_materialize_mcp_artifacts(
    value: Any,
    writer: AsyncArtifactWriter,
    directory: str = "/mcp-artifacts",
) -> tuple[Any, list[str]]:
    """Save copied MCP binary blocks through an asynchronous virtual writer.

    Args:
        value: Normalized result or a value that contains one.
        writer: Async callback that writes bytes to an absolute virtual path.
        directory: Absolute destination directory.

    Returns:
        Copied result and successful paths. Failed blocks retain their payload.
    """
    copied = _copy_artifacts(value, directory)
    paths: list[str] = []
    for block, field, path, payload in _artifact_payloads(copied, directory):
        try:
            await writer(path, payload)
        except Exception as exc:  # noqa: BLE001 - cancellation is not swallowed
            block["artifact_error"] = str(exc)
        else:
            _finish_artifact(block, field, path, payload)
            paths.append(path)
    return copied, paths


def _copy_artifacts(value: Any, directory: str) -> Any:
    """Validate the destination and copy result data.

    Args:
        value: Original result data.
        directory: Absolute virtual directory.

    Returns:
        A deep copy so the caller's original result remains unchanged.
    """
    if not directory.startswith("/"):
        raise ValueError("MCP artifact directory must be an absolute virtual path")
    return copy.deepcopy(value)


def _artifact_payloads(
    value: Any, directory: str
) -> Iterator[tuple[dict[str, Any], str, str, bytes]]:
    """Find and decode recognized blocks for either writer implementation.

    Args:
        value: Copied result data.
        directory: Absolute virtual destination.

    Yields:
        Block, payload field, generated path, and decoded bytes.
    """
    if isinstance(value, list):
        for item in value:
            yield from _artifact_payloads(item, directory)
        return
    if not isinstance(value, dict):
        return

    block_type = str(value.get("type", "")).lower()
    block, field = value, "data"
    if block_type in {"image", "audio"} and isinstance(value.get("data"), str):
        pass
    elif block_type in {"resource", "embedded_resource"}:
        resource = value.get("resource")
        if isinstance(resource, dict) and isinstance(resource.get("blob"), str):
            block, field = resource, "blob"
    if isinstance(block.get(field), str) and (
        block_type in {"image", "audio"} or field == "blob"
    ):
        try:
            payload = base64.b64decode(block[field], validate=True)
        except (ValueError, TypeError) as exc:
            block["artifact_error"] = f"invalid base64 payload: {exc}"
        else:
            mime_type = block.get("mimeType") or block.get("mime_type")
            extension = _extension(str(mime_type) if mime_type else None)
            path = posixpath.join(directory, f"{uuid.uuid4().hex}{extension}")
            yield block, field, path, payload
    for item in list(value.values()):
        yield from _artifact_payloads(item, directory)


def _finish_artifact(
    block: dict[str, Any], field: str, path: str, payload: bytes
) -> None:
    """Replace one payload only after its writer succeeds.

    Args:
        block: Copied content block.
        field: Base64 payload field.
        path: Written virtual path.
        payload: Written bytes.
    """
    del block[field]
    block["virtual_path"] = path
    block["size_bytes"] = len(payload)


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
