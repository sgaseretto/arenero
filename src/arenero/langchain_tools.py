from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Annotated, Any, Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import Field

from .async_kernel import AsyncPersistentMontyKernel
from .hosted_tools import HostedToolConfig
from .kernel import PersistentMontyKernel
from .langchain_filesystem_tools import create_filesystem_tools
from .langchain_hosted_tools import hosted_tools_from_langchain
from .sandbox import VIRTUAL_SHELL_TOOL_DESCRIPTION, AreneroSandbox

SessionIdResolver = Callable[[ToolRuntime], str | None]

PYTHON_REPL_TOOL_DESCRIPTION = (
    """Run one persistent Arenero REPL cell.
Normal input runs as Monty's limited Python subset. Prefix one shell command with
! or start a multiline shell cell with %%shell. Each nonempty line after %%shell
runs in sequence. Shell cells use the same session, working directory, Python
state, and virtual filesystem as Python cells. A shell cell can therefore use
python -c CODE to access existing REPL variables. Do not mix Python and shell in
one cell. A multiline ! cell is invalid; use %%shell. Python results contain
mode=python and the normal structured execution fields. Shell results contain
mode=shell, ok, output, exit_code, and truncated. Both modes also contain common
output, error, truncation, retryable, and output-file fields. Set output_file to
an absolute virtual path to save larger bounded output when the response is
trimmed.

"""
    + VIRTUAL_SHELL_TOOL_DESCRIPTION
)

ASYNC_PYTHON_REPL_TOOL_DESCRIPTION = """Run one persistent Monty Python cell on
Arenero's native asynchronous worker pool. State and virtual files remain
available in later calls with the same agent thread ID. This tool accepts Monty's
limited Python subset only; !COMMAND and %%shell cells are not supported. Use
fs_read, fs_write, fs_edit, fs_mkdir, fs_delete, fs_ls, fs_glob, and fs_grep for
virtual files. Await application async host functions. When MCP is configured,
also await mcp_servers(), mcp_search(), mcp_describe(), mcp_promote(),
mcp_refresh(), and mcp_call(). Check ok, error, error.recovery, truncated, and
host_calls before using a result. This small
native async adapter does not support output-file offloading.
"""

VIRTUAL_MODULE_TOOL_DESCRIPTION = """Resolve and run a limited static Python
module graph from the shared virtual filesystem. This is not CPython importlib.
It supports absolute or relative top-level imports of .py files, package entry
files named __init__.py, normal Monty standard-library imports, and optional
follow-up code in the same call. Public entry names remain in the persistent REPL.
It rejects wildcard imports, imports inside functions or control-flow blocks,
circular imports, dynamic imports, site-packages, native extensions, and paths
outside configured virtual roots. The default limits are 32 files, depth 8, and
1,000,000 combined UTF-8 source bytes. Generated module support code is not type
checked. File and hosted-tool side effects are not rolled back with Python state.
"""


def create_python_repl_tools(
    kernel: PersistentMontyKernel,
    session_id_resolver: SessionIdResolver | None = None,
    hosted_tools: Sequence[BaseTool] = (),
    hosted_tool_config: HostedToolConfig | None = None,
) -> list[BaseTool]:
    """Create persistent Python tools for LangChain or LangGraph.

    Args:
        kernel: Kernel used by every generated tool.
        session_id_resolver: Optional function that maps tool runtime data to a
            kernel session ID.
        hosted_tools: LangChain tools callable from inside the REPL.
        hosted_tool_config: Optional nested-call and record limits.

    Returns:
        Execute-code, execute-file, virtual-shell, filesystem, and reset tools.
    """
    resolve_session_id = session_id_resolver or _default_session_id
    repl_description = f"{PYTHON_REPL_TOOL_DESCRIPTION}\n\n{kernel.instructions()}"
    if kernel.mcp_hub is not None:
        repl_description = f"{repl_description}\n\n{kernel.mcp_hub.instructions}"
    hosted_catalog = hosted_tools_from_langchain(
        hosted_tools, limits=hosted_tool_config
    ).catalog(async_mode=False)
    if hosted_catalog:
        repl_description = f"{repl_description}\n\n{hosted_catalog}"

    @tool("python_repl", description=repl_description)
    def python_repl(
        code: Annotated[
            str,
            Field(
                description=(
                    "One Monty Python cell, one !COMMAND shell cell, or a %%shell "
                    "multiline shell cell. State and virtual files remain available "
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
                    "Optional maximum characters for Python result fields or the "
                    "final shell output. Use a smaller value when full output is "
                    "not needed. The application can enforce a lower hard maximum."
                ),
            ),
        ] = None,
        output_file: Annotated[
            str | None,
            Field(
                description=(
                    "Optional absolute virtual path. When output is trimmed, "
                    "Arenero saves the larger bounded output there."
                )
            ),
        ] = None,
        state_policy: Annotated[
            Literal["always", "success", "never"],
            Field(
                description=(
                    "State save policy. always keeps changes before errors; "
                    "success saves only successful cells; never does not save "
                    "Python state. File changes are never rolled back."
                )
            ),
        ] = "always",
        restart: Annotated[
            bool,
            Field(description="Reset Python state before this cell when true."),
        ] = False,
    ) -> dict[str, Any]:
        """Run one persistent Python or notebook-style shell cell.

        Args:
            code: Monty Python, one ``!command``, or a ``%%shell`` cell.
            runtime: LangChain runtime that identifies the current thread.
            output_limit_chars: Optional output limit requested by the model.
            output_file: Optional virtual path for larger bounded output.
            state_policy: State save policy for Python execution.
            restart: Reset Python state before this cell when true.

        Returns:
            A result with ``mode=python`` or ``mode=shell``.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return AreneroSandbox(kernel, session_id=session_id).execute_cell(
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools_from_langchain(
                hosted_tools,
                config=runtime.config,
                runtime=runtime,
                limits=hosted_tool_config,
            ),
        )

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
        output_file: Annotated[
            str | None,
            Field(
                description=(
                    "Optional absolute virtual path for larger bounded output "
                    "when returned fields are trimmed."
                )
            ),
        ] = None,
        state_policy: Annotated[
            Literal["always", "success", "never"],
            Field(description="State save policy for this file execution."),
        ] = "always",
        restart: Annotated[
            bool,
            Field(description="Reset Python state before this file when true."),
        ] = False,
    ) -> dict[str, Any]:
        """Run a virtual Python file and optional follow-up code.

        Args:
            file_path: Path to a readable virtual Python file.
            runtime: LangChain runtime that identifies the current thread.
            code: Optional code that runs after the file succeeds.
            output_limit_chars: Optional output limit requested by the model.
            output_file: Optional virtual path for larger bounded output.
            state_policy: State save policy for this operation.
            restart: Reset Python state before this file when true.

        Returns:
            A structured result for the final code that ran.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.execute_file(
                session_id,
                file_path,
                code,
                output_limit_chars,
                output_file=output_file,
                state_policy=state_policy,
                restart=restart,
                hosted_tools=hosted_tools_from_langchain(
                    hosted_tools,
                    config=runtime.config,
                    runtime=runtime,
                    limits=hosted_tool_config,
                ),
            )
        )

    @tool("python_repl_execute_module", description=VIRTUAL_MODULE_TOOL_DESCRIPTION)
    def python_repl_execute_module(
        entry_path: Annotated[
            str,
            Field(min_length=1, description="Absolute virtual entry .py file."),
        ],
        runtime: ToolRuntime,
        code: Annotated[
            str | None,
            Field(description="Optional follow-up code that can use public exports."),
        ] = None,
        output_limit_chars: Annotated[
            int | None,
            Field(ge=0, description="Optional maximum characters per output field."),
        ] = None,
        state_policy: Annotated[
            Literal["always", "success", "never"],
            Field(description="State save policy for this module execution."),
        ] = "always",
        restart: Annotated[
            bool,
            Field(description="Reset Python state before this module when true."),
        ] = False,
    ) -> dict[str, Any]:
        """Resolve and run a limited static virtual module graph.

        Args:
            entry_path: Absolute virtual entry ``.py`` file.
            runtime: LangChain runtime that identifies the current thread.
            code: Optional follow-up code.
            output_limit_chars: Optional caller output limit.
            state_policy: State save policy for this operation.
            restart: Reset Python state before the module when true.

        Returns:
            Structured module execution result.
        """

        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.execute_module(
                session_id,
                entry_path,
                code,
                output_limit_chars,
                state_policy=state_policy,
                restart=restart,
                hosted_tools=hosted_tools_from_langchain(
                    hosted_tools,
                    config=runtime.config,
                    runtime=runtime,
                    limits=hosted_tool_config,
                ),
            )
        )

    @tool("virtual_shell", description=VIRTUAL_SHELL_TOOL_DESCRIPTION)
    def virtual_shell(
        command: Annotated[
            str,
            Field(
                min_length=1,
                description=(
                    "One command list in the minimal Arenero shell grammar. "
                    "Use only the commands, flags, operators, and redirections "
                    "listed in the tool description. Use help to check the active "
                    "allowlist. This value is not an operating-system shell command."
                ),
            ),
        ],
        runtime: ToolRuntime,
        output_limit_chars: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional final output limit. The application hard maximum "
                    "still applies."
                ),
            ),
        ] = None,
        output_file: Annotated[
            str | None,
            Field(
                description=(
                    "Optional absolute virtual path for larger bounded shell "
                    "output when the response is trimmed."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Run one virtual shell command list for the current agent thread.

        Args:
            command: Text in the documented virtual shell grammar.
            runtime: LangChain runtime that identifies the current thread.
            output_limit_chars: Optional final output limit requested by the model.
            output_file: Optional virtual path for larger bounded output.

        Returns:
            Output, exit code, and truncation state.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        result = AreneroSandbox(kernel, session_id=session_id).execute(
            command,
            output_limit_chars=output_limit_chars,
            output_file=output_file,
        )
        return asdict(result)

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
        python_repl_execute_module,
        virtual_shell,
        *filesystem_tools,
        reset_python_repl,
    ]


def create_async_python_repl_tools(
    kernel: AsyncPersistentMontyKernel,
    session_id_resolver: SessionIdResolver | None = None,
    hosted_tools: Sequence[BaseTool] = (),
    hosted_tool_config: HostedToolConfig | None = None,
) -> list[BaseTool]:
    """Create native async Python tools for LangChain or LangGraph.

    Args:
        kernel: Native async kernel used by every generated tool.
        session_id_resolver: Optional runtime-to-session mapping function.
        hosted_tools: LangChain tools callable from inside the REPL.
        hosted_tool_config: Optional nested-call and record limits.

    Returns:
        Async execute-code, execute-file, execute-module, and reset tools.
    """
    resolve_session_id = session_id_resolver or _default_session_id
    repl_description = (
        f"{ASYNC_PYTHON_REPL_TOOL_DESCRIPTION}\n\n{kernel.instructions()}"
    )
    if kernel.mcp_hub is not None:
        repl_description = f"{repl_description}\n\n{kernel.mcp_hub.async_instructions}"
    hosted_catalog = hosted_tools_from_langchain(
        hosted_tools, limits=hosted_tool_config
    ).catalog(async_mode=True)
    if hosted_catalog:
        repl_description = f"{repl_description}\n\n{hosted_catalog}"

    @tool("python_repl", description=repl_description)
    async def python_repl(
        code: Annotated[
            str,
            Field(description="One Monty Python cell. Use await for async tools."),
        ],
        runtime: ToolRuntime,
        output_limit_chars: Annotated[
            int | None,
            Field(ge=0, description="Optional maximum characters per output field."),
        ] = None,
        state_policy: Annotated[
            Literal["always", "success", "never"],
            Field(description="State save policy for this execution."),
        ] = "always",
        restart: Annotated[
            bool,
            Field(description="Reset Python state before this cell when true."),
        ] = False,
    ) -> dict[str, Any]:
        """Run one persistent Monty cell.

        Args:
            code: Monty-compatible Python source.
            runtime: LangChain runtime that identifies the current thread.
            output_limit_chars: Optional caller output limit.
            state_policy: State save policy for this operation.
            restart: Reset Python state before this cell when true.

        Returns:
            Structured kernel execution data.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            await kernel.execute(
                session_id,
                code,
                output_limit_chars,
                state_policy=state_policy,
                restart=restart,
                hosted_tools=hosted_tools_from_langchain(
                    hosted_tools,
                    config=runtime.config,
                    runtime=runtime,
                    limits=hosted_tool_config,
                ),
            )
        )

    @tool("python_repl_execute_file")
    async def python_repl_execute_file(
        file_path: Annotated[
            str,
            Field(min_length=1, description="Virtual Python file to execute."),
        ],
        runtime: ToolRuntime,
        code: Annotated[
            str | None,
            Field(description="Optional follow-up code that can use file names."),
        ] = None,
        output_limit_chars: Annotated[
            int | None,
            Field(ge=0, description="Optional maximum characters per output field."),
        ] = None,
        state_policy: Annotated[
            Literal["always", "success", "never"],
            Field(description="State save policy for this file execution."),
        ] = "always",
        restart: Annotated[
            bool,
            Field(description="Reset Python state before this file when true."),
        ] = False,
    ) -> dict[str, Any]:
        """Run a virtual Python file and optional follow-up code.

        Args:
            file_path: Path to a readable virtual Python file.
            runtime: LangChain runtime that identifies the current thread.
            code: Optional follow-up code.
            output_limit_chars: Optional caller output limit.
            state_policy: State save policy for this operation.
            restart: Reset Python state before this file when true.

        Returns:
            Structured result for the final code that ran.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            await kernel.execute_file(
                session_id,
                file_path,
                code,
                output_limit_chars,
                state_policy=state_policy,
                restart=restart,
                hosted_tools=hosted_tools_from_langchain(
                    hosted_tools,
                    config=runtime.config,
                    runtime=runtime,
                    limits=hosted_tool_config,
                ),
            )
        )

    @tool("python_repl_execute_module", description=VIRTUAL_MODULE_TOOL_DESCRIPTION)
    async def python_repl_execute_module(
        entry_path: Annotated[
            str,
            Field(min_length=1, description="Absolute virtual entry .py file."),
        ],
        runtime: ToolRuntime,
        code: Annotated[
            str | None,
            Field(description="Optional follow-up code that can use public exports."),
        ] = None,
        output_limit_chars: Annotated[
            int | None,
            Field(ge=0, description="Optional maximum characters per output field."),
        ] = None,
        state_policy: Annotated[
            Literal["always", "success", "never"],
            Field(description="State save policy for this module execution."),
        ] = "always",
        restart: Annotated[
            bool,
            Field(description="Reset Python state before this module when true."),
        ] = False,
    ) -> dict[str, Any]:
        """Resolve and run a limited static virtual module graph.

        Args:
            entry_path: Absolute virtual entry ``.py`` file.
            runtime: LangChain runtime that identifies the current thread.
            code: Optional follow-up code.
            output_limit_chars: Optional caller output limit.
            state_policy: State save policy for this operation.
            restart: Reset Python state before the module when true.

        Returns:
            Structured module execution result.
        """

        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            await kernel.execute_module(
                session_id,
                entry_path,
                code,
                output_limit_chars,
                state_policy=state_policy,
                restart=restart,
                hosted_tools=hosted_tools_from_langchain(
                    hosted_tools,
                    config=runtime.config,
                    runtime=runtime,
                    limits=hosted_tool_config,
                ),
            )
        )

    @tool("reset_python_repl")
    async def reset_python_repl(runtime: ToolRuntime) -> dict[str, Any]:
        """Reset the current agent thread's Python state.

        Args:
            runtime: LangChain runtime that identifies the current thread.

        Returns:
            A result that reports whether saved state existed.
        """
        session_id = resolve_session_id(runtime)
        if session_id is None:
            return _missing_thread_id()
        existed = await kernel.reset(session_id)
        return {
            "ok": True,
            "reset": existed,
            "message": "The Python session was reset.",
        }

    return [
        python_repl,
        python_repl_execute_file,
        python_repl_execute_module,
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
            "code": "missing_thread_id",
            "message": "The graph call must include a thread_id.",
            "hint": "Set configurable.thread_id on the graph call.",
            "retryable": False,
            "recovery": "stop",
        },
    }
