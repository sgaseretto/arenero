"""Build machine-readable Arenero capability reports and model instructions."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from .modules import VirtualModuleConfig
from .store import InMemorySessionStore, SQLiteSessionStore, StatelessSessionStore

if TYPE_CHECKING:
    from .async_kernel import AsyncPersistentMontyKernel
    from .kernel import PersistentMontyKernel


def capability_report(
    kernel: PersistentMontyKernel | AsyncPersistentMontyKernel,
) -> dict[str, Any]:
    """Return the exact public capabilities of one configured kernel.

    Args:
        kernel: Configured synchronous or asynchronous kernel.

    Returns:
        JSON-compatible capability and limit data without credentials.
    """
    is_async = kernel.__class__.__name__ == "AsyncPersistentMontyKernel"
    store_mode, durable = _store_mode(kernel.store)
    filesystem_enabled = bool(kernel.os_handler is not None or kernel.mounts)
    shell_commands: list[str] = []
    network: dict[str, Any] = {"enabled": False, "reason": "async kernel has no shell"}
    if not is_async:
        from .kernel import PersistentMontyKernel
        from .shell import VirtualShell

        sync_kernel = cast(PersistentMontyKernel, kernel)
        shell_commands = VirtualShell(
            sync_kernel, "capability-report"
        ).available_commands()
        policy = sync_kernel.network_config
        network = {
            "enabled": policy.enabled,
            "allowed_methods": sorted(policy.allowed_methods),
            "allowed_url_prefixes": list(policy.allowed_url_prefixes),
            "all_public_urls_allowed": not policy.allowed_url_prefixes,
            "deny_non_global_addresses": policy.deny_non_global_addresses,
            "timeout_seconds": policy.timeout_seconds,
            "max_request_bytes": policy.max_request_bytes,
            "max_response_bytes": policy.max_response_bytes,
            "max_redirects": policy.max_redirects,
            "trusted_headers_configured": policy.trusted_header_provider is not None,
        }
    mcp_hub = kernel.mcp_hub
    mcp_limits: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] = []
    if mcp_hub is not None:
        mcp_limits = asdict(mcp_hub.config)
        allowed_tools = mcp_limits.get("allowed_tools")
        if allowed_tools is not None:
            mcp_limits["allowed_tools"] = sorted(allowed_tools)
        mcp_servers = mcp_hub.servers()["servers"]
    promoted_tools = mcp_hub.promoted_toolset() if mcp_hub is not None else None
    active_hosted_tools = kernel.hosted_tools.merged(promoted_tools)
    return {
        "runtime": {
            "name": "arenero",
            "monty_version": kernel.monty_version,
            "mode": "native-async" if is_async else "sync",
            "normal_cpython": False,
        },
        "persistence": {
            "mode": store_mode,
            "stores_python_state": kernel.store.stores_state,
            "durable_between_processes": durable,
            "session_ttl_seconds": kernel.config.session_ttl_seconds,
            "state_policies": ["always", "success", "never"],
            "default_state_policy": "always",
            "files_are_not_rolled_back": True,
        },
        "filesystem": {
            "enabled": filesystem_enabled,
            "mode": _filesystem_mode(kernel),
            "shared_between_sessions": filesystem_enabled,
            "workspace_virtual_path": (
                kernel.workspace.virtual_path if kernel.workspace is not None else None
            ),
            "helpers": [
                "read",
                "write",
                "edit",
                "mkdir",
                "delete",
                "ls",
                "glob",
                "grep",
            ],
            "limits": asdict(kernel.filesystem_config),
        },
        "virtual_modules": {
            "enabled": filesystem_enabled,
            "api": "execute_module",
            "static_only": True,
            "default_limits": asdict(VirtualModuleConfig()),
            "supports": [
                "top-level imports",
                ".py files",
                "__init__.py packages",
                "relative imports",
                "configured aliases",
                "same-call follow-up code",
            ],
            "does_not_support": [
                "dynamic imports",
                "wildcard imports",
                "nested imports",
                "circular imports",
                "site-packages",
                "native extensions",
                "type checking of generated bundle code",
            ],
        },
        "shell": {
            "enabled": not is_async,
            "commands": shell_commands,
            "supports_pipes": not is_async,
            "supports_redirection": not is_async,
            "is_real_os_shell": False,
        },
        "network": network,
        "mcp": {
            "enabled": mcp_hub is not None,
            "call_style": "await" if is_async else "sync",
            "functions": (
                [
                    "mcp_servers",
                    "mcp_search",
                    "mcp_describe",
                    "mcp_promote",
                    "mcp_add_server",
                    "mcp_remove_server",
                    "mcp_replace_server",
                    "mcp_refresh",
                    "mcp_call",
                ]
                if mcp_hub is not None
                else []
            ),
            "promoted_functions": (
                [tool.alias for tool in promoted_tools.tools]
                if promoted_tools is not None
                else []
            ),
            "servers": mcp_servers,
            "dynamic_server_configuration": mcp_hub is not None,
            "dynamic_server_scope": "shared MCP hub",
            "dynamic_server_transports": ["stdio", "streamable-http"],
            "dynamic_server_configuration_is_durable": False,
            "artifact_directory": getattr(kernel, "mcp_artifact_directory", None),
            "limits": mcp_limits,
        },
        "limits": _kernel_limits(kernel),
        "hosted_tools": {
            "count": len(active_hosted_tools.tools),
            "functions": [tool.alias for tool in active_hosted_tools.tools],
            "limits": asdict(active_hosted_tools.config),
            "catalog": active_hosted_tools.catalog(async_mode=is_async),
            "schema_warnings": active_hosted_tools.schema_warnings(),
            "operation_tools_may_add_functions": True,
            "dynamic_functions_are_type_checked": False,
        },
        "bootstrap": {
            "code_configured": bool(getattr(kernel, "bootstrap_code", None)),
            "files": list(getattr(kernel, "bootstrap_files", ())),
            "runs_for_new_sessions_only": True,
        },
        "repl_commands": (
            ["%run FILE", "%reset", "%limits", "%sessions", "%mcp"]
            if not is_async
            else []
        ),
        "important_limits": [
            "Monty is a Python subset, not CPython.",
            "Normal third-party imports and operating-system processes are unavailable.",
            "Use execute_file or %run for one flat virtual Python file.",
            "Use execute_module for a bounded static virtual module graph.",
            "Session snapshots do not contain virtual files or host functions.",
            "Dynamic MCP server configuration lasts only while its hub is alive.",
            "One execution at a time is allowed for each session.",
        ],
    }


def model_instructions(
    kernel: PersistentMontyKernel | AsyncPersistentMontyKernel,
) -> str:
    """Generate model instructions from one kernel capability report.

    Args:
        kernel: Configured synchronous or asynchronous kernel.

    Returns:
        Deterministic instructions with an embedded JSON capability report.
    """
    report = capability_report(kernel)
    return (
        "Use Arenero as a persistent Monty Python sandbox. Read the active "
        "capability report below before you act. Do not assume CPython, a real "
        "shell, packages, files, network access, or MCP access beyond this report. "
        "Check ok, error.code, error.hint, error.recovery, truncated, stats, and "
        "host_calls in each result.\n\n" + json.dumps(report, indent=2, sort_keys=True)
    )


def _store_mode(store: object) -> tuple[str, bool]:
    """Identify the built-in store mode without exposing its configuration.

    Args:
        store: Configured session store.

    Returns:
        Public mode name and durability flag.
    """
    if isinstance(store, SQLiteSessionStore):
        return "sqlite", True
    if isinstance(store, InMemorySessionStore):
        return "memory", False
    if isinstance(store, StatelessSessionStore):
        return "none", False
    return type(store).__name__, False


def _kernel_limits(
    kernel: PersistentMontyKernel | AsyncPersistentMontyKernel,
) -> dict[str, Any]:
    """Return safe kernel limits and omit callback or stub content.

    Args:
        kernel: Configured kernel.

    Returns:
        JSON-compatible limit data.
    """
    data = asdict(kernel.config)
    data.pop("type_check_stubs", None)
    allowed = data.get("allowed_shell_commands")
    if allowed is not None:
        data["allowed_shell_commands"] = sorted(allowed)
    return data


def _filesystem_mode(
    kernel: PersistentMontyKernel | AsyncPersistentMontyKernel,
) -> str:
    """Identify the active virtual filesystem configuration.

    Args:
        kernel: Configured kernel.

    Returns:
        ``workspace``, ``os-handler``, ``mounts``, or ``disabled``.
    """
    if kernel.workspace is not None:
        return "workspace"
    if kernel.os_handler is not None:
        return "os-handler"
    if kernel.mounts:
        return "mounts"
    return "disabled"
