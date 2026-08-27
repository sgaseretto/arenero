import argparse
import ast
import codeop
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from .kernel import PersistentMontyKernel
from .store import (
    InMemorySessionStore,
    SessionStore,
    SQLiteSessionStore,
    StatelessSessionStore,
)

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

        try:
            complete = compiler(source, "<webmonty>", "single") is not None
        except (SyntaxError, OverflowError, ValueError):
            complete = True

        if complete:
            return source


def run_repl(
    kernel: PersistentMontyKernel,
    session_id: str = "cli",
    output_limit_chars: int | None = None,
    input_fn: Callable[[str], str] = input,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> None:
    """Run the interactive terminal REPL.

    Args:
        kernel: Kernel used to execute input cells.
        session_id: Persistent CLI session key.
        output_limit_chars: Optional output limit for each cell.
        input_fn: Function that reads prompted input.
        stdout: Stream used for normal output.
        stderr: Stream used for errors.
    """
    compiler = codeop.CommandCompiler()
    compiler.compiler.flags |= ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

    print("WebMonty -- a Python REPL powered by Monty", file=stdout)
    print("Type exit or quit, or press Ctrl-D to leave.", file=stdout)

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

        execution = kernel.execute(session_id, source, output_limit_chars)
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
    """Parse command-line options and run or reset a REPL session."""
    parser = argparse.ArgumentParser(description="Run the WebMonty REPL")
    parser.add_argument(
        "--store",
        choices=("memory", "sqlite", "none"),
        default="sqlite",
        help="Session storage mode (default: sqlite)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(".webmonty/sessions.db"),
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
        "--reset",
        action="store_true",
        help="Reset the selected session and exit",
    )
    args = parser.parse_args()

    store = _create_store(args.store, args.db)
    with PersistentMontyKernel(store) as kernel:
        if args.reset:
            kernel.reset(args.session)
            print(f"Reset session {args.session!r}.")
            return
        run_repl(
            kernel,
            session_id=args.session,
            output_limit_chars=args.output_limit,
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
