"""Configuration and runtime checks for Arenero."""

from __future__ import annotations

import importlib.util
import uuid
from typing import Any

from .kernel import PersistentMontyKernel


def run_doctor(kernel: PersistentMontyKernel) -> dict[str, Any]:
    """Check one configured kernel without keeping test Python state.

    Args:
        kernel: Open synchronous kernel to check.

    Returns:
        Overall status and individual check records.
    """
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "monty",
            True,
            f"pydantic-monty {kernel.monty_version}",
        )
    )
    execution = kernel.execute(
        f"doctor-{uuid.uuid4().hex}",
        "6 * 7",
        state_policy="never",
    )
    checks.append(
        _check(
            "worker",
            execution["ok"] and execution["result"] == 42,
            "Monty worker executed a test expression.",
            execution["error"],
        )
    )
    optional_packages = (
        ("deepagents", "deepagents"),
        ("langchain", "langchain"),
        ("mcp", "mcp"),
        ("pydantic_ai", "pydantic-ai"),
        ("opentelemetry", "observability"),
    )
    for package, extra in optional_packages:
        installed = importlib.util.find_spec(package) is not None
        checks.append(
            _check(
                f"optional-package:{package}",
                installed,
                "installed" if installed else f"not installed; use arenero[{extra}]",
                required=False,
            )
        )
    checks.append(_workspace_check(kernel))
    checks.append(_mcp_check(kernel))
    required_ok = all(check["ok"] for check in checks if check["required"])
    return {
        "ok": required_ok,
        "checks": checks,
        "capabilities": kernel.capabilities(),
    }


def _workspace_check(kernel: PersistentMontyKernel) -> dict[str, Any]:
    """Check virtual workspace read and write access when configured.

    Args:
        kernel: Open kernel to check.

    Returns:
        One doctor check record.
    """
    if kernel.workspace is None:
        return _check("workspace", True, "not configured", required=False)
    path = f"{kernel.workspace.virtual_path}/.arenero-doctor-{uuid.uuid4().hex}"
    try:
        kernel.upload_file(path, b"arenero")
        ok = kernel.download_file(path) == b"arenero"
        kernel.call_filesystem("doctor-cleanup", "delete", {"path": path})
        kernel.reset("doctor-cleanup")
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        return _check("workspace", False, "workspace test failed", str(exc))
    return _check("workspace", ok, "workspace read and write test completed")


def _mcp_check(kernel: PersistentMontyKernel) -> dict[str, Any]:
    """Check configured MCP server status without exposing credentials.

    Args:
        kernel: Open kernel to check.

    Returns:
        One doctor check record.
    """
    if kernel.mcp_hub is None:
        return _check("mcp", True, "not configured", required=False)
    try:
        status = kernel.mcp_hub.servers()
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        return _check("mcp", False, "MCP status check failed", str(exc))
    return _check("mcp", bool(status.get("ok")), "MCP server status loaded", status)


def _check(
    name: str,
    ok: bool,
    message: str,
    details: Any = None,
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Build one stable doctor check record.

    Args:
        name: Stable check name.
        ok: Whether the check passed.
        message: Short human-readable result.
        details: Optional safe diagnostic details.
        required: Whether failure changes the overall status.

    Returns:
        JSON-compatible check record.
    """
    return {
        "name": name,
        "ok": ok,
        "required": required,
        "message": message,
        "details": details,
    }
