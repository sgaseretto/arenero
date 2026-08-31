import argparse
import ast
import codeop
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from .cell import execute_repl_cell
from .kernel import KernelConfig, PersistentMontyKernel
from .store import (
    InMemorySessionStore,
    SessionStore,
    SQLiteSessionStore,
    StatelessSessionStore,
)
from .workspace import WorkspaceConfig

EXIT_COMMANDS = {"exit", "exit()", "quit", "quit()"}


def read_cell(
    number: int,
    compiler: codeop.CommandCompiler,
    input_fn: Callable[[str], str],
) -> str | None:
    """Read one complete Python cell.

    Args:
        number: Input cell number shown in the prompt.
        compiler: Compiler used to detect complete input.
        input_fn: Function that reads one prompted line.

    Returns:
        Complete source, or ``None`` when the user requests exit.
    """
    lines: list[str] = []

    while True:
        prompt = f"In [{number}]: " if not lines else "   ...: "
        line = input_fn(prompt)

        if not lines and line.strip() in EXIT_COMMANDS:
            return None
        if not lines and not line.strip():
            continue

        lines.append(line)
        source = "\n".join(lines)

        if lines[0].strip() == "%%shell":
            if len(lines) > 1 and not line.strip():
                return "\n".join(lines[:-1])
            continue

        try:
            complete = compiler(source, "<arenero>", "single") is not None
        except (SyntaxError, OverflowError, ValueError):
            complete = True

        if complete:
            return source


def run_repl(
    kernel: PersistentMontyKernel,
    session_id: str = "cli",
    output_limit_chars: int | None = None,
    output_file: str | None = None,
    input_fn: Callable[[str], str] = input,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> None:
    """Run the interactive terminal REPL.

    Args:
        kernel: Kernel used to execute input cells.
        session_id: Persistent CLI session key.
        output_limit_chars: Optional output limit for each cell.
        output_file: Optional virtual path used when a cell output is trimmed.
        input_fn: Function that reads prompted input.
        stdout: Stream used for normal output.
        stderr: Stream used for errors.
    """
    compiler = codeop.CommandCompiler()
    compiler.compiler.flags |= ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

    print("Arenero -- a Python REPL powered by Monty", file=stdout)
    print(
        "Use !command or %%shell for the virtual shell. Use %limits for active capabilities. "
        "Type exit or quit, or press Ctrl-D to leave.",
        file=stdout,
    )

    number = 1
    while True:
        try:
            source = read_cell(number, compiler, input_fn)
        except EOFError:
            print(file=stdout)
            return
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt", file=stderr)
            continue

        if source is None:
            return

        execution = execute_repl_cell(
            kernel,
            session_id,
            source,
            output_limit_chars,
            output_file=output_file,
        )
        if execution["mode"] == "shell":
            output = execution["output"]
            if output:
                stream = stdout if execution["ok"] else stderr
                print(output, end="" if output.endswith("\n") else "\n", file=stream)
            number += 1
            continue
        if execution["stdout"]:
            print(execution["stdout"], end="", file=stdout)
        if execution["stderr"]:
            print(execution["stderr"], end="", file=stderr)
        if execution["error"] is not None:
            print(execution["error"]["message"], file=stderr)
        elif execution["result"] is not None:
            print(f"Out[{number}]: {execution['result']!r}", file=stdout)

        number += 1


def main() -> None:
    """Parse command-line options and run or administer REPL sessions."""
    parser = argparse.ArgumentParser(description="Run the Arenero REPL")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("doctor",),
        help="Optional administrative command",
    )
    parser.add_argument(
        "--store",
        choices=("memory", "sqlite", "none"),
        default="sqlite",
        help="Session storage mode (default: sqlite)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(".arenero/sessions.db"),
        help="SQLite session database",
    )
    parser.add_argument(
        "--session",
        default="cli",
        help="Persistent session name",
    )
    parser.add_argument(
        "--output-limit",
        type=int,
        help="Maximum characters returned for each output field",
    )
    parser.add_argument(
        "--output-file",
        help="Virtual path used when cell output is trimmed",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Persistent host directory mounted at /workspace",
    )
    parser.add_argument(
        "--type-check",
        action="store_true",
        help="Type-check each Monty cell before execution",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--reset",
        action="store_true",
        help="Reset the selected session and exit",
    )
    actions.add_argument(
        "--list-sessions",
        action="store_true",
        help="List saved sessions and exit",
    )
    actions.add_argument(
        "--prune-sessions",
        action="store_true",
        help="Delete expired sessions and exit",
    )
    actions.add_argument(
        "--doctor",
        action="store_true",
        help="Check the configured runtime and exit",
    )
    args = parser.parse_args()

    store = _create_store(args.store, args.db)
    workspace = WorkspaceConfig(args.workspace) if args.workspace else None
    config = KernelConfig(type_check=args.type_check)
    with PersistentMontyKernel(store, config, workspace=workspace) as kernel:
        if args.reset:
            kernel.reset(args.session)
            print(f"Reset session {args.session!r}.")
            return
        if args.list_sessions:
            for info in kernel.list_sessions():
                print(
                    f"{info.session_id}\trevision={info.revision}\t"
                    f"bytes={info.state_size_bytes}\tmonty={info.monty_version}\t"
                    f"updated={info.updated_at:.0f}"
                )
            return
        if args.prune_sessions:
            print(f"Deleted {kernel.delete_expired()} expired session(s).")
            return
        if args.doctor or args.command == "doctor":
            from .doctor import run_doctor

            report = run_doctor(kernel)
            print(json.dumps(report, indent=2, sort_keys=True))
            if not report["ok"]:
                raise SystemExit(1)
            return
        run_repl(
            kernel,
            session_id=args.session,
            output_limit_chars=args.output_limit,
            output_file=args.output_file,
        )


def _create_store(mode: str, database: Path) -> SessionStore:
    """Create the selected CLI session store.

    Args:
        mode: ``memory``, ``sqlite``, or ``none``.
        database: SQLite path used by ``sqlite`` mode.

    Returns:
        Selected session store.
    """
    if mode == "memory":
        return InMemorySessionStore()
    if mode == "none":
        return StatelessSessionStore()
    return SQLiteSessionStore(database)
