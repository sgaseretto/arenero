import json
from typing import Any

from .kernel import PersistentMontyKernel
from .shell import ShellResult, VirtualShell


def execute_repl_cell(
    kernel: PersistentMontyKernel,
    session_id: str,
    code: str,
    output_limit_chars: int | None = None,
    *,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Execute one Python or notebook-style shell cell.

    Args:
        kernel: Kernel that owns the persistent session and virtual filesystem.
        session_id: Session used by both Python and shell execution.
        code: Monty Python, one ``!command``, or a ``%%shell`` cell.
        output_limit_chars: Optional result limit below the application hard cap.
        output_file: Optional absolute virtual path used when output is trimmed.

    Returns:
        Python execution fields with ``mode=python``, or shell execution fields
        with ``mode=shell``.
    """
    command, error = _shell_command(code)
    if error is not None:
        return _shell_result(kernel, ShellResult(error, 2), output_limit_chars)
    if command is not None:
        result = VirtualShell(kernel, session_id).execute(
            command,
            output_limit_chars,
            output_file=output_file,
        )
        return _shell_result(kernel, result, output_limit_chars)

    execution: dict[str, Any] = dict(
        kernel.execute(
            session_id,
            code,
            output_limit_chars,
            output_file=output_file,
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
            "message": result.output,
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
