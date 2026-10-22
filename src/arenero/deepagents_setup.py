"""Optional, root-aware settings and checks for Deep Agents adapters."""

from typing import Any

from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend

from .deepagents import CodeInterpreterMiddleware
from .deepagents_backend import (
    AreneroBackend,
    AreneroBackendMiddleware,
    AreneroFilesBackend,
)
from .doctor import _check


def deepagents_settings(
    backend: AreneroFilesBackend,
    *,
    interpreter: CodeInterpreterMiddleware | None = None,
    permissions: list[FilesystemPermission] | None = None,
    capture_output: bool = False,
) -> dict[str, Any]:
    """Build keyword settings for ``create_deep_agent`` without owning resources.

    Args:
        backend: File-only or executing adapter with its existing virtual root.
        interpreter: Optional caller-owned Python interpreter middleware. Use
            no direct file grants when graph permissions must control file calls.
        permissions: Native Deep Agents rules, passed through unchanged.
        capture_output: Enable native execute capture. Requires AreneroBackend.
            False keeps the backend's existing capture setting unchanged.

    Returns:
        Backend, middleware, and optional permissions keyword arguments. Artifact
        paths stay below the backend root. No routes or new mounts are added.

    Raises:
        ValueError: If capture is requested for a file-only backend.
    """
    if capture_output:
        if not isinstance(backend, AreneroBackend):
            raise ValueError("capture_output requires AreneroBackend")
        backend.enable_capture_offload = True
    middleware = []
    if interpreter is None or interpreter._backend is not backend:
        middleware.append(AreneroBackendMiddleware(backend))
    if interpreter is not None:
        middleware.append(interpreter)
    settings: dict[str, Any] = {
        "backend": CompositeBackend(
            default=backend, routes={}, artifacts_root=backend.root
        ),
        "middleware": middleware,
    }
    if permissions is not None:
        settings["permissions"] = permissions
    return settings


def check_deepagents_setup(
    backend: AreneroFilesBackend,
    *,
    artifacts_root: str = "/",
    interpreter: CodeInterpreterMiddleware | None = None,
    permissions: list[FilesystemPermission] | None = None,
) -> dict[str, Any]:
    """Check adapter wiring without executing code or changing permission rules.

    Args:
        backend: Underlying Arenero file-only or executing adapter.
        artifacts_root: Artifact root passed to CompositeBackend. Native setup
            defaults to '/'; ``deepagents_settings`` uses ``backend.root``.
        interpreter: Optional Python interpreter used by the same agent.
        permissions: Native filesystem rules, or None when none are supplied.

    Returns:
        Doctor-shaped checks and current kernel capabilities. Capabilities are
        deferred (None) if an MCP hub is configured, because the normal report
        can connect its servers. Required failures set ok to false. Optional
        failures are warnings, not access restrictions. Checks cover the default
        backend only, not custom routes or host tools.
    """
    try:
        backend._path(artifacts_root)
        aligned = True
    except ValueError:
        aligned = False
    executing = isinstance(backend, AreneroBackend)
    checks = [
        _check(
            "deepagents:artifacts_root",
            aligned,
            "Artifact root is inside the backend root."
            if aligned
            else "Use deepagents_settings or set artifacts_root=backend.root.",
        ),
        _check(
            "deepagents:backend_profile",
            not (executing and permissions is not None),
            "Use AreneroFilesBackend with filesystem permissions."
            if executing and permissions is not None
            else "Executing backend."
            if executing
            else "File-only backend.",
        ),
    ]
    if interpreter is not None:
        direct_files = interpreter._backend is not None or bool(
            interpreter.kernel.os_handler is not None or interpreter.kernel.mounts
        )
        checks.append(
            _check(
                "deepagents:direct_files",
                not (permissions is not None and direct_files),
                "Direct interpreter file grants bypass graph filesystem permissions. "
                "For permission-aware calls, omit backend/workspace/os_handler and "
                "select registered file tools with ptc."
                if permissions is not None and direct_files
                else "No conflicting direct file grants detected.",
                required=False,
            )
        )
        checks.append(
            _check(
                "deepagents:shared_backend",
                interpreter._backend is None or interpreter._backend is backend,
                "Interpreter and agent use the same backend."
                if interpreter._backend is backend
                else "Interpreter has no direct backend."
                if interpreter._backend is None
                else "Interpreter uses a different backend from the agent.",
                required=False,
            )
        )
    kernel = backend.sandbox.kernel
    defer_capabilities = kernel.mcp_hub is not None
    return {
        "ok": all(check["ok"] for check in checks if check["required"]),
        "checks": checks,
        "capabilities": None if defer_capabilities else kernel.capabilities(),
        "capabilities_deferred": defer_capabilities,
    }
