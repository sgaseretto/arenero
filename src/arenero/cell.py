"""Shared Python and percent-cell utilities, with a sync-only virtual shell."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from pydantic_monty import MontyError

from .hosted_tools import HostedToolSet
from .kernel import PersistentMontyKernel, StatePolicy, _output_limit, _safe_result
from .shell import ShellResult, VirtualShell
from .values import trim_text as _trim

if TYPE_CHECKING:
    from .async_kernel import AsyncPersistentMontyKernel


def execute_repl_cell(
    kernel: PersistentMontyKernel,
    session_id: str,
    code: str,
    output_limit_chars: int | None = None,
    *,
    output_file: str | None = None,
    state_policy: StatePolicy = "always",
    restart: bool = False,
    hosted_tools: HostedToolSet | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one Python or notebook-style shell cell.

    Args:
        kernel: Kernel that owns the persistent session and virtual filesystem.
        session_id: Session used by both Python and shell execution.
        code: Monty Python, one ``!command``, or a ``%%shell`` cell.
        output_limit_chars: Optional result limit below the application hard cap.
        output_file: Optional absolute virtual path used when output is trimmed.
        state_policy: State save policy for Python and ``%run`` execution.
        restart: Reset Python state before an executable cell when true.
        hosted_tools: Optional tools for this operation only.
        inputs: Values bound once for Python or ``%run``; not shell/control inputs.

    Returns:
        Python execution fields with ``mode=python``, or shell execution fields
        with ``mode=shell``.
    """
    magic = _magic_command(
        kernel,
        session_id,
        code,
        output_limit_chars,
        output_file,
        state_policy,
        restart,
        hosted_tools,
        inputs,
    )
    if magic is not None:
        return magic

    command, error = _shell_command(code)
    if error is not None:
        return _shell_result(kernel, ShellResult(error, 2), output_limit_chars)
    if command is not None:
        if inputs:
            return _invalid_inputs()
        if restart:
            kernel.reset(session_id)
        result = VirtualShell(kernel, session_id).execute(
            command,
            output_limit_chars,
            output_file=output_file,
        )
        shell_result = _shell_result(kernel, result, output_limit_chars)
        shell_result["session_restarted"] = restart
        return shell_result

    execution: dict[str, Any] = dict(
        kernel.execute(
            session_id,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            inputs=inputs,
        )
    )
    return _execution_cell_result(execution, "python")


def _parse_magic(code: str) -> tuple[str, list[str]] | dict[str, Any] | None:
    """Parse and validate a percent command for both kernel modes.

    Args:
        code: Raw cell source, with one line required for a percent command.

    Returns:
        Command and arguments, a structured error, or None for ordinary source.
    """
    if not code.startswith("%") or code.startswith("%%"):
        return None
    hint = "Use %run FILE, %reset, %limits, %sessions, %mcp, %who, or %whos."
    if "\n" in code:
        return _command_result(
            False,
            None,
            "Percent commands must use one line.",
            "invalid_repl_command",
            hint,
        )
    try:
        arguments = shlex.split(code[1:])
    except ValueError as exc:
        return _command_result(
            False,
            None,
            str(exc),
            "invalid_repl_command",
            "Check the command quoting and try again.",
        )
    if not arguments:
        return _command_result(
            False, None, "Missing percent command.", "invalid_repl_command", hint
        )
    name, values = arguments[0], arguments[1:]
    counts = {
        "run": 1,
        "reset": 0,
        "limits": 0,
        "sessions": 0,
        "mcp": 0,
        "who": 0,
        "whos": 0,
    }
    if name not in counts or len(values) != counts[name]:
        return _command_result(
            False,
            None,
            f"Unsupported percent command or arguments: %{name}",
            "invalid_repl_command",
            hint,
        )
    return name, values


def _invalid_inputs() -> dict[str, Any]:
    """Return guidance when a non-Python cell receives host input values.

    Returns:
        Structured error with no execution, reset, or state save.
    """
    return _command_result(
        False,
        None,
        "inputs are supported only for Python and %run.",
        "invalid_cell_inputs",
        "Use a Python cell or %run for inputs.",
    )


async def execute_async_repl_cell(
    kernel: AsyncPersistentMontyKernel,
    session_id: str,
    code: str,
    output_limit_chars: int | None = None,
    *,
    output_file: str | None = None,
    state_policy: StatePolicy = "always",
    restart: bool = False,
    hosted_tools: HostedToolSet | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch native async Python and percent cells, without a second shell.

    Args:
        kernel: Native asynchronous kernel.
        session_id: Bound session key.
        code: Python source or one supported percent command.
        output_limit_chars: Optional public output limit.
        output_file: Optional virtual path for untrimmed execution output.
        state_policy: Python save policy: always, success, or never.
        restart: Reset Python state before executable source when true.
        hosted_tools: Optional tools for this operation only.
        inputs: Values bound once for Python or %run; other controls reject them.

    Returns:
        Common cell result. Shell cells are rejected without execution.
    """
    parsed = _parse_magic(code)
    if isinstance(parsed, dict):
        return parsed
    if parsed is None:
        if code.lstrip().startswith(("!", "%%")):
            return _command_result(
                False,
                None,
                "Native async shell cells are not supported.",
                "unsupported_shell_cell",
                "Use the sync or managed shell API.",
            )
        execution = await kernel.execute(
            session_id,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            inputs=inputs,
        )
        return _execution_cell_result(execution, "python")
    name, values = parsed
    if inputs and name != "run":
        return _invalid_inputs()
    if name == "run":
        execution = await kernel.execute_file(
            session_id,
            values[0],
            output_limit_chars=output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            inputs=inputs,
        )
        return _execution_cell_result(execution, "command")
    if name == "reset":
        return _command_result(True, {"reset": await kernel.reset(session_id)})
    if name == "limits":
        return _command_result(True, kernel.capabilities())
    if name == "sessions":
        info = await kernel.get_session_info(session_id)
        return _command_result(
            True, {"current_session": asdict(info) if info else None}
        )
    if name == "mcp":
        if kernel.mcp_hub is None:
            return _command_result(
                False,
                None,
                "No MCP hub is configured.",
                "mcp_unavailable",
                "Configure an McpHub before %mcp.",
            )
        return _command_result(True, await kernel.mcp_hub.async_servers())
    try:
        variables = await kernel.variables(session_id)
    except (MontyError, OSError, RuntimeError, ValueError) as exc:
        return _command_result(
            False,
            None,
            str(exc),
            "session_inspection_failed",
            "Check the session version and reset incompatible state.",
        )
    limit, capped = _output_limit(kernel.config, output_limit_chars)
    return _command_result(
        True,
        sorted(variables) if name == "who" else variables,
        output_limit_chars=limit,
        limit_was_capped=capped,
    )


def _execution_cell_result(execution: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Add shared cell fields without changing the execution result contract.

    Args:
        execution: Python or file execution fields from either kernel.
        mode: Python or command cell mode.

    Returns:
        Copied fields with cell output, exit status, and truncation flags.
    """
    result = dict(execution)
    result["mode"] = mode
    result["exit_code"] = 0 if result["ok"] else 1
    result["output"] = _python_output(result)
    result["truncation"] = {
        **result["truncated"],
        "output": any(result["truncated"].values()),
    }
    return result


def _magic_command(
    kernel: PersistentMontyKernel,
    session_id: str,
    code: str,
    output_limit_chars: int | None,
    output_file: str | None,
    state_policy: StatePolicy,
    restart: bool,
    hosted_tools: HostedToolSet | None,
    inputs: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Dispatch one supported single-line percent command.

    Args:
        kernel: Kernel that owns the session.
        session_id: Current bound session ID.
        code: Raw REPL cell source.
        output_limit_chars: Optional caller output limit.
        output_file: Optional virtual path for trimmed execution output.
        state_policy: State save policy for ``%run``.
        restart: Reset state before ``%run`` when true.
        hosted_tools: Optional tools for ``%run``.
        inputs: Values bound once for ``%run``; other controls reject nonempty inputs.

    Returns:
        Command result, or ``None`` when this is not a percent command.
    """
    parsed = _parse_magic(code)
    if parsed is None or isinstance(parsed, dict):
        return parsed
    name, values = parsed
    if inputs and name != "run":
        return _invalid_inputs()
    if name == "run" and len(values) == 1:
        execution: dict[str, Any] = dict(
            kernel.execute_file(
                session_id,
                values[0],
                output_limit_chars=output_limit_chars,
                output_file=output_file,
                state_policy=state_policy,
                restart=restart,
                hosted_tools=hosted_tools,
                inputs=inputs,
            )
        )
        return _execution_cell_result(execution, "command")
    if name == "reset" and not values:
        existed = kernel.reset(session_id)
        return _command_result(True, {"reset": existed})
    if name in {"who", "whos"} and not values:
        try:
            variables = kernel.variables(session_id)
        except (MontyError, OSError, RuntimeError, ValueError) as exc:
            return _command_result(
                False,
                None,
                str(exc),
                "session_inspection_failed",
                "Check the session version and reset incompatible state.",
            )
        limit, capped = _output_limit(kernel.config, output_limit_chars)
        return _command_result(
            True,
            sorted(variables) if name == "who" else variables,
            output_limit_chars=limit,
            limit_was_capped=capped,
        )
    if name == "limits" and not values:
        return _command_result(True, kernel.capabilities())
    if name == "sessions" and not values:
        info = kernel.get_session_info(session_id)
        return _command_result(
            True,
            {"current_session": asdict(info) if info is not None else None},
        )
    if name == "mcp" and not values:
        if kernel.mcp_hub is None:
            return _command_result(
                False,
                None,
                "No MCP hub is configured.",
                "mcp_unavailable",
                "Configure an McpHub on the kernel before you use %mcp.",
            )
        return _command_result(True, kernel.mcp_hub.servers())
    return _command_result(
        False,
        None,
        f"Unsupported percent command or arguments: %{name}",
        "invalid_repl_command",
        "Use %run FILE, %reset, %limits, %sessions, %mcp, %who, or %whos.",
    )


def _command_result(
    ok: bool,
    value: Any,
    message: str | None = None,
    code: str = "repl_command_error",
    hint: str = "Check the command and try again.",
    *,
    output_limit_chars: int | None = None,
    limit_was_capped: bool = False,
) -> dict[str, Any]:
    """Build one stable small REPL command result.

    Args:
        ok: Whether the command succeeded.
        value: Command result value.
        message: Optional error message.
        code: Stable error code.
        hint: Recovery guidance.
        output_limit_chars: Optional bound for inspection results; None is untrimmed.
        limit_was_capped: Whether the configured hard limit changed the request.

    Returns:
        Common REPL result fields.
    """
    output = "" if value is None else json.dumps(value, ensure_ascii=False)
    if message is not None:
        output = message
    trimmed = False
    if output_limit_chars is not None:
        value, trimmed = _safe_result(value, output_limit_chars)
        output, output_trimmed = _trim(output, output_limit_chars)
        trimmed = trimmed or output_trimmed
    error = None
    if not ok:
        error = {
            "type": "ReplCommandError",
            "code": code,
            "message": message or "REPL command failed",
            "hint": hint,
            "retryable": False,
            "recovery": "revise_code",
        }
    return {
        "mode": "command",
        "ok": ok,
        "result": value,
        "output": output,
        "stdout": output if ok else "",
        "stderr": "" if ok else output,
        "error": error,
        "exit_code": 0 if ok else 1,
        "truncated": {
            "result": trimmed,
            "stdout": trimmed,
            "stderr": False,
            "error": False,
        },
        "truncation": {
            "result": trimmed,
            "output": trimmed,
            "stdout": trimmed,
            "stderr": False,
            "error": False,
        },
        "output_limit_chars": output_limit_chars or 0,
        "limit_was_capped": limit_was_capped,
        "state_saved": False,
        "state_restored": False,
        "session_revision": None,
        "retryable": False,
        "output_file": None,
        "output_file_error": None,
        "state_policy": "never",
        "session_restarted": False,
        "stats": None,
        "mcp_artifacts": [],
        "host_calls": {"count": 0, "items": [], "truncated": False},
    }


def _shell_command(code: str) -> tuple[str | None, str | None]:
    """Parse notebook shell markers without parsing Python source.

    Args:
        code: Raw REPL cell text.

    Returns:
        Shell command and error text. Both are ``None`` for a Python cell.
    """
    lines = code.splitlines()
    if code.startswith("!"):
        if len(lines) > 1:
            return None, "Use %%shell for a shell cell with more than one line"
        return code[1:].strip(), None
    if lines and lines[0].strip() == "%%shell":
        _, separator, body = code.partition("\n")
        return body if separator else "", None
    return None, None


def _shell_result(
    kernel: PersistentMontyKernel,
    result: ShellResult,
    requested_limit: int | None,
) -> dict[str, Any]:
    """Convert an internal shell result to a REPL cell result.

    Args:
        kernel: Kernel that owns output limit configuration.
        result: Internal virtual shell result.
        requested_limit: Optional caller output limit.

    Returns:
        Stable shell-mode result fields for callers and agents.
    """
    limit, capped = _output_limit(kernel.config, requested_limit)
    ok = result.exit_code == 0
    error = None
    if not ok:
        error = {
            "type": "ShellError",
            "code": "shell_error",
            "message": result.output,
            "hint": "Use only commands and syntax listed by the virtual shell help command.",
            "retryable": False,
        }
    return {
        "mode": "shell",
        "ok": ok,
        "result": None,
        "output": result.output,
        "stdout": result.output if ok else "",
        "stderr": "" if ok else result.output,
        "error": error,
        "exit_code": result.exit_code,
        "truncated": result.truncated,
        "truncation": {
            "result": False,
            "output": result.truncated,
            "stdout": result.truncated if ok else False,
            "stderr": result.truncated if not ok else False,
            "error": result.truncated if not ok else False,
        },
        "output_limit_chars": limit,
        "limit_was_capped": capped,
        "state_saved": None,
        "state_restored": None,
        "session_revision": None,
        "retryable": False,
        "output_file": result.output_file,
        "output_file_error": result.output_file_error,
        "state_policy": "never",
        "stats": None,
        "mcp_artifacts": [],
    }


def _python_output(execution: dict[str, Any]) -> str:
    """Build one concise display field for a Python cell.

    Args:
        execution: Structured kernel execution result.

    Returns:
        Standard streams, error text, or serialized result text.
    """
    streams = execution["stdout"] + execution["stderr"]
    if streams:
        return streams
    if execution["error"] is not None:
        return execution["error"]["message"]
    value = execution["result"]
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)
