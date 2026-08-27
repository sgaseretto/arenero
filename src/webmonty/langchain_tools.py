from collections.abc import Callable
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import Field

from .kernel import PersistentMontyKernel
from .langchain_filesystem_tools import create_filesystem_tools

SessionIdResolver = Callable[[ToolRuntime], str | None]


def create_python_repl_tools(
    kernel: PersistentMontyKernel,
    session_id_resolver: SessionIdResolver | None = None,
) -> list[BaseTool]:
    """Create persistent Python tools for LangChain or LangGraph.

    Args:
        kernel: Kernel used by every generated tool.
        session_id_resolver: Optional function that maps tool runtime data to a
            kernel session ID.

    Returns:
        Execute-code, execute-file, filesystem, and reset tools.
    """
    resolve_session_id = session_id_resolver or _default_session_id

    @tool("python_repl")
    def python_repl(
        code: Annotated[
            str,
            Field(
                description=(
                    "Python code to run. Variables and functions remain available "
                    "in later calls in the same agent thread."
                )
            ),
        ],
        runtime: ToolRuntime,
        output_limit_chars: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional maximum characters for each result, stdout, stderr, "
                    "and error field. Use a smaller value when full output is not "
                    "needed. The application can enforce a lower hard maximum."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Run code in a persistent, isolated Python sandbox.

        Args:
            code: Monty-compatible Python code.
            runtime: LangChain runtime that identifies the current thread.
            output_limit_chars: Optional output limit requested by the model.

        Returns:
            A structured execution result.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return kernel.execute(session_id, code, output_limit_chars)

    @tool("python_repl_execute_file")
    def python_repl_execute_file(
        file_path: Annotated[
            str,
            Field(
                min_length=1,
                description=(
                    "Path to a Python file in the virtual filesystem. The file "
                    "runs in the persistent REPL session."
                ),
            ),
        ],
        runtime: ToolRuntime,
        code: Annotated[
            str | None,
            Field(
                description=(
                    "Optional follow-up code to run after the file succeeds. It "
                    "can use functions and variables defined by the file. This "
                    "tool returns the follow-up result when code is supplied."
                )
            ),
        ] = None,
        output_limit_chars: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional maximum characters for each result, stdout, stderr, "
                    "and error field. The application can enforce a lower hard "
                    "maximum."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Run a virtual Python file and optional follow-up code.

        Args:
            file_path: Path to a readable virtual Python file.
            runtime: LangChain runtime that identifies the current thread.
            code: Optional code that runs after the file succeeds.
            output_limit_chars: Optional output limit requested by the model.

        Returns:
            A structured result for the final code that ran.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return kernel.execute_file(
            session_id,
            file_path,
            code,
            output_limit_chars,
        )

    @tool("reset_python_repl")
    def reset_python_repl(runtime: ToolRuntime) -> dict[str, Any]:
        """Reset the persistent Python session for the current agent thread.

        Args:
            runtime: LangChain runtime that identifies the current thread.

        Returns:
            A result that reports whether a saved session existed.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        existed = kernel.reset(session_id)
        return {
            "ok": True,
            "reset": existed,
            "message": "The Python session was reset.",
        }

    filesystem_tools = create_filesystem_tools(kernel, resolve_session_id)
    return [
        python_repl,
        python_repl_execute_file,
        *filesystem_tools,
        reset_python_repl,
    ]


def _default_session_id(runtime: ToolRuntime) -> str | None:
    """Read the current thread ID from a LangChain tool runtime.

    Args:
        runtime: Runtime supplied by LangChain or LangGraph.

    Returns:
        The thread ID as text, or ``None`` when it is unavailable.
    """
    if runtime.execution_info is not None and runtime.execution_info.thread_id:
        return str(runtime.execution_info.thread_id)

    configurable = runtime.config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id is not None else None


def _missing_thread_id() -> dict[str, Any]:
    """Build the standard missing-thread error.

    Returns:
        A structured non-retryable error.
    """
    return {
        "ok": False,
        "error": {
            "type": "MissingThreadId",
            "message": "The graph call must include a thread_id.",
            "retryable": False,
        },
    }
