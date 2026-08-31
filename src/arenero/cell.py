import json
import shlex
from dataclasses import asdict
from typing import Any

from .hosted_tools import HostedToolSet
from .kernel import PersistentMontyKernel, StatePolicy
from .shell import ShellResult, VirtualShell


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
    )
    if magic is not None:
        return magic

    command, error = _shell_command(code)
    if error is not None:
        return _shell_result(kernel, ShellResult(error, 2), output_limit_chars)
    if command is not None:
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
        )
    )
    execution["mode"] = "python"
    execution["exit_code"] = 0 if execution["ok"] else 1
    execution["output"] = _python_output(execution)
    execution["truncation"] = {
        **execution["truncated"],
        "output": any(execution["truncated"].values()),
    }
    return execution


def _magic_command(
    kernel: PersistentMontyKernel,
    session_id: str,
    code: str,
    output_limit_chars: int | None,
    output_file: str | None,
    state_policy: StatePolicy,
    restart: bool,
    hosted_tools: HostedToolSet | None,
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

    Returns:
        Command result, or ``None`` when this is not a percent command.
    """
    if not code.startswith("%") or code.startswith("%%"):
        return None
    if "\n" in code:
        return _command_result(
            False,
            None,
            "Percent commands must use one line.",
            "invalid_repl_command",
            "Use one of %run, %reset, %limits, %sessions, or %mcp.",
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
            False,
            None,
            "Missing percent command.",
            "invalid_repl_command",
            "Use one of %run, %reset, %limits, %sessions, or %mcp.",
        )
    name, values = arguments[0], arguments[1:]
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
            )
        )
        execution["mode"] = "command"
        execution["exit_code"] = 0 if execution["ok"] else 1
        execution["output"] = _python_output(execution)
        execution["truncation"] = {
            **execution["truncated"],
            "output": any(execution["truncated"].values()),
        }
        return execution
    if name == "reset" and not values:
        existed = kernel.reset(session_id)
        return _command_result(True, {"reset": existed})
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
        "Use %run FILE, %reset, %limits, %sessions, or %mcp.",
    )


def _command_result(
    ok: bool,
    value: Any,
    message: str | None = None,
    code: str = "repl_command_error",
    hint: str = "Check the command and try again.",
) -> dict[str, Any]:
    """Build one stable small REPL command result.

    Args:
        ok: Whether the command succeeded.
        value: Command result value.
        message: Optional error message.
        code: Stable error code.
        hint: Recovery guidance.

    Returns:
        Common REPL result fields.
    """
    output = "" if value is None else json.dumps(value, ensure_ascii=False)
    if message is not None:
        output = message
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
            "result": False,
            "stdout": False,
            "stderr": False,
            "error": False,
        },
        "truncation": {
            "result": False,
            "output": False,
            "stdout": False,
            "stderr": False,
            "error": False,
        },
        "output_limit_chars": 0,
        "limit_was_capped": False,
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
        commands = [line.strip() for line in lines[1:] if line.strip()]
        return "; ".join(commands), None
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
    limit, capped = _output_limit(kernel, requested_limit)
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


def _output_limit(
    kernel: PersistentMontyKernel,
    requested: int | None,
) -> tuple[int, bool]:
    """Normalize a shell cell output limit.

    Args:
        kernel: Kernel that owns output limits.
        requested: Optional caller limit.

    Returns:
        Effective limit and whether the hard cap changed it.
    """
    if requested is None:
        return kernel.config.default_output_limit_chars, False
    normalized = max(0, requested)
    effective = min(normalized, kernel.config.max_output_limit_chars)
    return effective, effective != requested


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
