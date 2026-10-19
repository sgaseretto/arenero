"""Optional Deep Agents middleware backed by Arenero and Monty."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, Self, cast
from uuid import uuid4

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import BaseModel, Field

from ._graph_calls import GraphCalls
from ._loop_thread import LoopThread
from ._subagents import task_bridge
from .async_kernel import AsyncPersistentMontyKernel
from .hosted_tools import HostedToolConfig, HostedToolSet
from .kernel import KernelConfig
from .langchain_hosted_tools import hosted_tools_from_langchain
from .rlm import RLM
from .workspace import WorkspaceConfig

logger = logging.getLogger(__name__)

PersistenceMode = Literal["thread", "turn", "call"]
PTCOption = Sequence[str | BaseTool]


class AreneroAgentState(AgentState):
    """Add private Monty snapshot fields to LangGraph agent state."""

    _arenero_snapshot: NotRequired[Annotated[bytes, PrivateStateAttr]]
    _arenero_snapshot_hmac: NotRequired[Annotated[bytes, PrivateStateAttr]]
    _arenero_slot_id: NotRequired[Annotated[str, PrivateStateAttr]]


class AreneroEvalInput(BaseModel):
    """Describe the model-visible input for the interpreter tool."""

    code: str = Field(
        description=(
            "Monty-compatible Python. Await hosted functions. Normal CPython "
            "packages, subprocesses, sockets, and dynamic imports are unavailable."
        )
    )
    output_limit_chars: int | None = Field(
        default=None,
        ge=0,
        description="Optional maximum characters for each returned output field.",
    )
    output_file: str | None = Field(
        default=None,
        description="Optional absolute virtual path for full bounded JSON when output is trimmed. Check output_file_error.",
    )


class AreneroModuleInput(BaseModel):
    """Describe one static virtual-module execution."""

    entry_path: str = Field(description="Absolute virtual path to the entry .py file.")
    code: str | None = Field(
        default=None,
        description=(
            "Optional follow-up code. It can use the entry module object and its "
            "public exports in the same call."
        ),
    )
    output_limit_chars: int | None = Field(
        default=None,
        ge=0,
        description="Optional maximum characters for each returned output field.",
    )
    output_file: str | None = Field(
        default=None,
        description="Optional absolute virtual path for full bounded JSON when output is trimmed. Check output_file_error.",
    )


class AreneroInterpreterMiddleware(
    AgentMiddleware[AreneroAgentState, ContextT, ResponseT]
):
    """Expose one persistent asynchronous Arenero interpreter to an agent.

    Args:
        kernel: Configured async kernel. A small in-memory kernel is made when
            omitted. The caller owns a supplied kernel.
        tool_name: Model-visible interpreter tool name.
        ptc: LangChain tool names or instances callable from Monty code.
        subagents: Expose a current Deep Agents ``task`` tool when true.
        rlm: Optional recursive model runner exposed inside Monty.
        mode: State lifetime: graph thread, one agent turn, or one tool call.
        hosted_tool_config: Programmatic call and concurrency limits.
        checkpoint_state: Put full Monty snapshots in private LangGraph state.
            This is valid only in thread mode.
        max_snapshot_bytes: Largest snapshot accepted in graph state.
        snapshot_signing_key: Optional HMAC key for checkpoint integrity.
        session_prefix: Stable prefix that separates graph session IDs.
    """

    state_schema = AreneroAgentState

    def __init__(
        self,
        *,
        kernel: AsyncPersistentMontyKernel | None = None,
        tool_name: str = "python_interpreter",
        ptc: PTCOption | None = None,
        subagents: bool = True,
        rlm: RLM | None = None,
        mode: PersistenceMode = "thread",
        hosted_tool_config: HostedToolConfig | None = None,
        checkpoint_state: bool = False,
        max_snapshot_bytes: int = 64 * 1024 * 1024,
        snapshot_signing_key: str | bytes | None = None,
        session_prefix: str = "arenero-agent",
    ) -> None:
        """Create middleware and its model-visible interpreter tool."""

        super().__init__()
        if mode not in {"thread", "turn", "call"}:
            raise ValueError("mode must be 'thread', 'turn', or 'call'")
        if checkpoint_state and mode != "thread":
            raise ValueError("checkpoint_state requires mode='thread'")
        if checkpoint_state and kernel is not None and not kernel.store.stores_state:
            raise ValueError("checkpoint_state requires a stateful kernel store")
        if max_snapshot_bytes < 1:
            raise ValueError("max_snapshot_bytes must be greater than 0")
        if not tool_name or not session_prefix:
            raise ValueError("tool_name and session_prefix must not be empty")
        self.kernel = kernel or AsyncPersistentMontyKernel()
        self._owns_kernel = kernel is None
        self._tool_name = tool_name
        self._ptc = tuple(ptc) if ptc is not None else None
        self._subagents = subagents
        self._rlm = rlm
        self._mode = mode
        self._hosted_tool_config = hosted_tool_config or HostedToolConfig()
        self._checkpoint_state = checkpoint_state
        self._max_snapshot_bytes = max_snapshot_bytes
        self._signing_key = (
            snapshot_signing_key.encode()
            if isinstance(snapshot_signing_key, str)
            else snapshot_signing_key
        )
        self._session_prefix = session_prefix
        self._fallback_thread_id = uuid4().hex
        self._turn_sessions: dict[str, str] = {}
        self.tools: list[BaseTool] = [self._build_tool(), self._build_module_tool()]

    async def __aenter__(self) -> Self:
        """Open the owned or supplied kernel and return this middleware.

        Returns:
            This open middleware.
        """

        await self.kernel.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close resources owned by this middleware.

        Args:
            *exc_info: Optional exception details from the context manager.
        """

        await self.aclose()

    async def aclose(self) -> None:
        """Close the kernel only when this middleware created it."""

        if self._owns_kernel:
            await self.kernel.close()

    def instructions(self, tools: Sequence[BaseTool] = ()) -> str:
        """Build current model instructions and hosted-function declarations.

        Args:
            tools: Current agent tools after other middleware filters them.

        Returns:
            Plain instructions for this interpreter configuration.
        """

        exposed = self._select_tools(tools)
        catalog = hosted_tools_from_langchain(
            exposed,
            limits=self._hosted_tool_config,
            include_code_execution=False,
        ).catalog(async_mode=True)
        persistence = {
            "thread": "Python state stays available across calls and agent turns.",
            "turn": "Python state stays available only during this agent turn.",
            "call": "Each interpreter call starts with empty Python state.",
        }[self._mode]
        sections = [
            (
                f"Use `{self._tool_name}` for sandboxed Python calculations. "
                "It uses Monty's Python subset, not CPython. "
                "Use agent.ainvoke() or agent.astream(); synchronous invocation "
                "cannot run this interpreter. "
                f"{persistence} Standard Monty modules work, but normal package "
                "imports, subprocesses, sockets, native extensions, and dynamic "
                "imports do not. Use the module tool for small static virtual-file "
                "imports. It supports top-level import statements, .py files, "
                "packages through __init__.py, and relative imports. It rejects "
                "wildcards, nested virtual-file imports, cycles, and third-party packages. "
                "Nested bundled-standard-module imports are allowed. "
                "Virtual files are separate from Python snapshots."
            ),
            self.kernel.instructions(),
        ]
        if catalog:
            sections.append(catalog)
            sections.append(
                "Programmatic calls bypass the parent ToolNode. They do not run "
                "parent per-tool approval middleware. Their effects are not rolled "
                "back when Python state is reset or a graph checkpoint is restored."
            )
        if self._rlm is not None:
            sections.append(self._rlm.catalog(self.kernel))
        return "\n\n".join(sections)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Add current interpreter instructions to a synchronous model call.

        Args:
            request: Current model request.
            handler: Next synchronous middleware or model function.

        Returns:
            Model response from the next handler.
        """

        prompt = self.instructions(
            [tool for tool in request.tools or () if isinstance(tool, BaseTool)]
        )
        return handler(
            request.override(
                system_message=append_to_system_message(request.system_message, prompt)
            )
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Add current interpreter instructions to an asynchronous model call.

        Args:
            request: Current model request.
            handler: Next asynchronous middleware or model function.

        Returns:
            Awaited model response from the next handler.
        """

        prompt = self.instructions(
            [tool for tool in request.tools or () if isinstance(tool, BaseTool)]
        )
        return await handler(
            request.override(
                system_message=append_to_system_message(request.system_message, prompt)
            )
        )

    async def abefore_agent(
        self,
        state: AreneroAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Start a turn and restore an optional checkpoint snapshot.

        Args:
            state: Current private and public graph state.
            runtime: Current LangGraph runtime.

        Returns:
            Private state fields to clear after a rejected snapshot, or ``None``.
        """

        thread_id = _resolve_thread_id(self._fallback_thread_id)
        if self._mode == "turn":
            self._turn_sessions[thread_id] = self._new_session_id(thread_id, "turn")
            return None
        if not self._checkpoint_state:
            return None
        session_id = self._thread_session_id(thread_id)
        payload = state.get("_arenero_snapshot")
        if payload is None:
            await self.kernel.reset(session_id)
            return None
        if len(payload) > self._max_snapshot_bytes or not self._snapshot_valid(
            payload, state, session_id
        ):
            await self.kernel.reset(session_id)
            return {"_arenero_snapshot": None, "_arenero_snapshot_hmac": None}
        async with self.kernel._lock_for(session_id):
            await asyncio.to_thread(self.kernel.store.delete, session_id)
            await asyncio.to_thread(
                self.kernel.store.save,
                session_id,
                payload,
                self.kernel.monty_version,
                None,
            )
        return None

    async def aafter_agent(
        self,
        state: AreneroAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """End a turn and save or remove its Monty state.

        Args:
            state: Current private and public graph state.
            runtime: Current LangGraph runtime.

        Returns:
            New private snapshot state, or ``None`` when not checkpointed.
        """

        thread_id = _resolve_thread_id(self._fallback_thread_id)
        if self._mode == "turn":
            session_id = self._turn_sessions.pop(thread_id, None)
            if session_id is not None:
                await self.kernel.reset(session_id)
            return None
        if not self._checkpoint_state:
            return None
        session_id = self._thread_session_id(thread_id)
        async with self.kernel._lock_for(session_id):
            stored = await asyncio.to_thread(self.kernel.store.load, session_id)
            await asyncio.to_thread(self.kernel.store.delete, session_id)
        if stored is None:
            return None
        if len(stored.state) > self._max_snapshot_bytes:
            logger.warning(
                "Dropping Arenero snapshot for %s because it is too large",
                session_id,
            )
            return {"_arenero_snapshot": None, "_arenero_snapshot_hmac": None}
        update: dict[str, Any] = {"_arenero_snapshot": stored.state}
        if self._signing_key is not None:
            update["_arenero_snapshot_hmac"] = self._sign(stored.state, session_id)
        return update

    def _build_tool(self) -> BaseTool:
        """Build the interpreter tool registered by this middleware.

        Returns:
            Structured LangChain tool with hidden runtime injection.
        """

        middleware = self

        def sync_eval(
            runtime: ToolRuntime,
            code: str,
            output_limit_chars: int | None = None,
            output_file: str | None = None,
        ) -> ToolMessage:
            """Return guidance when an agent uses synchronous invocation.

            Args:
                runtime: Active outer tool runtime.
                code: Unused Monty source.
                output_limit_chars: Unused requested output limit.
                output_file: Unused overflow destination.

            Returns:
                Error ToolMessage that requests asynchronous invocation.
            """

            content = json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "AsyncInvocationRequired",
                        "message": "This interpreter uses Async Monty.",
                        "hint": "Call agent.ainvoke() or agent.astream().",
                    },
                }
            )
            return ToolMessage(
                content=content,
                tool_call_id=runtime.tool_call_id,
                name=middleware._tool_name,
                status="error",
            )

        async def async_eval(
            runtime: ToolRuntime,
            code: str,
            output_limit_chars: int | None = None,
            output_file: str | None = None,
        ) -> ToolMessage:
            """Run one asynchronous persistent Monty cell.

            Args:
                runtime: Active outer tool runtime.
                code: Monty-compatible Python source.
                output_limit_chars: Optional caller output limit.
                output_file: Optional absolute virtual overflow destination.

            Returns:
                ToolMessage with the JSON execution result.
            """

            session_id = middleware._session_for_runtime(runtime)
            hosted = middleware._hosted_for_runtime(runtime, session_id)
            control = GraphCalls()
            try:
                result = await middleware.kernel.execute(
                    session_id,
                    code,
                    output_limit_chars,
                    hosted_tools=control.wrap(hosted),
                    output_file=output_file,
                )
            except asyncio.CancelledError:
                control.reraise()
                raise
            finally:
                if middleware._mode == "call":
                    await middleware.kernel.reset(session_id)
            return ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=runtime.tool_call_id,
                name=middleware._tool_name,
                status="success" if result["ok"] else "error",
            )

        return StructuredTool.from_function(
            name=self._tool_name,
            description=(
                "Run Monty's limited Python asynchronously. State lifetime follows "
                f"mode={self._mode}. Await hosted functions listed in the system "
                "instructions. Check ok, error, truncated, stats, and host_calls."
            ),
            func=sync_eval,
            coroutine=async_eval,
            infer_schema=False,
            args_schema=AreneroEvalInput,
            metadata={
                "arenero_kind": "code-execution",
                "code_arg_name": "code",
                "ls_code_input_language": "python",
            },
        )

    def _build_module_tool(self) -> BaseTool:
        """Build the static virtual-module interpreter tool.

        Returns:
            Structured asynchronous module execution tool.
        """

        middleware = self
        name = f"{self._tool_name}_module"

        def sync_module(
            runtime: ToolRuntime,
            entry_path: str,
            code: str | None = None,
            output_limit_chars: int | None = None,
            output_file: str | None = None,
        ) -> ToolMessage:
            """Return guidance for unsupported synchronous module execution.

            Args:
                runtime: Active outer tool runtime.
                entry_path: Unused virtual entry path.
                code: Unused follow-up code.
                output_limit_chars: Unused requested output limit.
                output_file: Unused overflow destination.

            Returns:
                Error ToolMessage that requests asynchronous invocation.
            """

            content = json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "AsyncInvocationRequired",
                        "message": "Virtual module execution uses Async Monty.",
                        "hint": "Call agent.ainvoke() or agent.astream().",
                    },
                }
            )
            return ToolMessage(
                content=content,
                tool_call_id=runtime.tool_call_id,
                name=name,
                status="error",
            )

        async def async_module(
            runtime: ToolRuntime,
            entry_path: str,
            code: str | None = None,
            output_limit_chars: int | None = None,
            output_file: str | None = None,
        ) -> ToolMessage:
            """Resolve and run one static virtual module graph.

            Args:
                runtime: Active outer tool runtime.
                entry_path: Absolute virtual entry ``.py`` file.
                code: Optional follow-up code.
                output_limit_chars: Optional caller output limit.
                output_file: Optional absolute virtual overflow destination.

            Returns:
                ToolMessage with the JSON execution result.
            """

            session_id = middleware._session_for_runtime(runtime)
            hosted = middleware._hosted_for_runtime(runtime, session_id)
            control = GraphCalls()
            try:
                result = await middleware.kernel.execute_module(
                    session_id,
                    entry_path,
                    code,
                    output_limit_chars,
                    hosted_tools=control.wrap(hosted),
                    output_file=output_file,
                )
            except asyncio.CancelledError:
                control.reraise()
                raise
            finally:
                if middleware._mode == "call":
                    await middleware.kernel.reset(session_id)
            return ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=runtime.tool_call_id,
                name=name,
                status="success" if result["ok"] else "error",
            )

        return StructuredTool.from_function(
            name=name,
            description=(
                "Resolve and run a limited static Python module graph from the "
                "virtual filesystem. This is not CPython importlib. Only top-level "
                "imports, .py files, __init__.py packages, configured roots, and "
                "bounded acyclic graphs work. Optional follow-up code runs in the "
                "same call and can use public entry exports."
            ),
            func=sync_module,
            coroutine=async_module,
            infer_schema=False,
            args_schema=AreneroModuleInput,
            metadata={
                "arenero_kind": "code-execution",
                "code_arg_name": "code",
                "ls_code_input_language": "python",
            },
        )

    def _hosted_for_runtime(
        self,
        runtime: ToolRuntime,
        session_id: str,
    ) -> HostedToolSet:
        """Adapt selected live tools for one interpreter call.

        Args:
            runtime: Current outer tool runtime.
            session_id: Active parent Monty session.

        Returns:
            Operation-scoped hosted tool definitions.
        """

        application_tools = hosted_tools_from_langchain(
            self._select_tools(runtime.tools),
            config=runtime.config,
            runtime=runtime,
            limits=self._hosted_tool_config,
            include_code_execution=False,
        )
        if self._rlm is None:
            return application_tools
        return self._rlm.hosted_tools(
            self.kernel,
            session_id,
            application_tools=application_tools,
        )

    def _select_tools(self, available: Sequence[BaseTool]) -> tuple[BaseTool, ...]:
        """Select live tools that code may call.

        Args:
            available: Current tools after upstream middleware changes.

        Returns:
            Deduplicated safe tools in stable order.
        """

        by_name = {tool.name: tool for tool in available}
        candidates: list[BaseTool] = []
        for entry in self._ptc or ():
            tool = by_name.get(entry) if isinstance(entry, str) else entry
            if tool is not None:
                candidates.append(tool)
        if self._subagents:
            task = by_name.get("task")
            if task is not None and {"description", "subagent_type"} <= set(task.args):
                candidates.append(task)
        selected: list[BaseTool] = []
        names: set[str] = set()
        for tool in candidates:
            metadata = dict(tool.metadata or {})
            is_code = (
                "code_arg_name" in metadata
                or metadata.get("arenero_kind") == "code-execution"
            )
            is_control = metadata.get("arenero_kind") == "control"
            if (
                tool.name == self._tool_name
                or is_code
                or is_control
                or tool.name in names
            ):
                continue
            selected.append(tool)
            names.add(tool.name)
        return tuple(selected)

    def _session_for_runtime(self, runtime: ToolRuntime) -> str:
        """Resolve the Monty session for one interpreter call.

        Args:
            runtime: Active outer tool runtime.

        Returns:
            Session ID that follows the configured persistence mode.
        """

        thread_id = _resolve_thread_id(self._fallback_thread_id, runtime)
        if self._mode == "thread":
            return self._thread_session_id(thread_id)
        if self._mode == "turn":
            return self._turn_sessions.setdefault(
                thread_id, self._new_session_id(thread_id, "turn")
            )
        return self._new_session_id(thread_id, "call")

    def _thread_session_id(self, thread_id: str) -> str:
        """Build a stable session ID for one graph thread.

        Args:
            thread_id: LangGraph thread ID.

        Returns:
            Stable kernel session ID.
        """

        digest = hashlib.sha256(thread_id.encode()).hexdigest()[:24]
        return f"{self._session_prefix}:thread:{digest}"[:255]

    def _new_session_id(self, thread_id: str, scope: str) -> str:
        """Build a unique session ID for a turn or call.

        Args:
            thread_id: LangGraph thread ID.
            scope: ``turn`` or ``call``.

        Returns:
            Unique bounded kernel session ID.
        """

        digest = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
        return f"{self._session_prefix}:{scope}:{digest}:{uuid4().hex[:12]}"[:255]

    def _sign(self, payload: bytes, session_id: str) -> bytes:
        """Sign one snapshot for one stable session.

        Args:
            payload: Monty snapshot bytes.
            session_id: Bound session identity.

        Returns:
            SHA-256 HMAC bytes.
        """

        assert self._signing_key is not None
        return hmac.new(
            self._signing_key,
            session_id.encode() + b"\0" + payload,
            hashlib.sha256,
        ).digest()

    def _snapshot_valid(
        self,
        payload: bytes,
        state: AreneroAgentState,
        session_id: str,
    ) -> bool:
        """Check snapshot integrity under the configured policy.

        Args:
            payload: Snapshot from graph state.
            state: State that can contain its HMAC.
            session_id: Session identity bound into the HMAC.

        Returns:
            ``True`` when unsigned snapshots are allowed or the HMAC matches.
        """

        if self._signing_key is None:
            return True
        tag = state.get("_arenero_snapshot_hmac")
        return isinstance(tag, bytes) and hmac.compare_digest(
            tag, self._sign(payload, session_id)
        )


def _resolve_thread_id(fallback: str, runtime: ToolRuntime | None = None) -> str:
    """Read a LangGraph thread ID or use a stable fallback.

    Args:
        fallback: Instance-scoped value used outside a configured graph.
        runtime: Optional active tool runtime.

    Returns:
        String thread identity.
    """

    config = runtime.config if runtime is not None else None
    if config is None:
        try:
            config = get_config()
        except RuntimeError:
            config = None
    value = (config or {}).get("configurable", {}).get("thread_id")
    return str(value) if value is not None else fallback


class CodeInterpreterMiddleware(AreneroInterpreterMiddleware):
    """Provide the QuickJS-style agent interface with Monty Python execution.

    Args:
        memory_limit: Monty heap cap in bytes, not host RSS.
        timeout: Interpreter time cap for each Python feed, excluding host waits.
        max_ptc_calls: Regular hosted calls per eval; None removes this adapter cap.
        tool_name: Interpreter tool name; defaults to eval.
        max_result_chars: Character cap for each returned field.
        capture_console: Collect stdout and stderr when true; otherwise discard.
        subagents: Expose the current Deep Agents task tool when true.
        max_subagent_concurrency: Active subagents per eval, separate from PTC.
        max_subagent_calls: Task dispatch cap per eval.
        ptc: Live tool names or explicit tool objects to expose as Python functions.
        mode: Python lifetime: thread, turn, or call. None means thread.
        max_snapshot_bytes: Graph snapshot cap; defaults to memory_limit.
        snapshot_signing_key: Optional HMAC key bound to the private graph slot.
        module_tool: Include the static virtual-module tool when true.
        tools_namespace: Also expose await tools.NAME(**kwargs) when true.
        backend: Optional shared AreneroBackend. Guest files then use its hosted
            fs_* functions, not direct mounts. The caller owns the backend.
        workspace: Optional direct mounted workspace, instead of a shared backend.
    """

    trace_policy = TracePolicy(
        process_inputs=omit_payload, process_outputs=omit_payload
    )

    def __init__(
        self,
        *,
        memory_limit: int = 64 * 1024 * 1024,
        timeout: float = 5.0,
        max_ptc_calls: int | None = 256,
        tool_name: str = "eval",
        max_result_chars: int = 4000,
        capture_console: bool = True,
        subagents: bool = True,
        max_subagent_concurrency: int = 32,
        max_subagent_calls: int = 100,
        ptc: PTCOption | None = None,
        mode: PersistenceMode | None = None,
        max_snapshot_bytes: int | None = None,
        snapshot_signing_key: str | bytes | None = None,
        module_tool: bool = False,
        tools_namespace: bool = True,
        backend: Any = None,
        workspace: WorkspaceConfig | None = None,
    ) -> None:
        """Create an optional adapter without changing the original middleware."""
        if max_ptc_calls is not None and max_ptc_calls < 1:
            raise ValueError("max_ptc_calls must be positive or None")
        if min(max_subagent_concurrency, max_subagent_calls) < 1:
            raise ValueError("subagent limits must be positive")
        if max_result_chars < 0:
            raise ValueError("max_result_chars must not be negative")
        if mode not in (None, "thread", "turn", "call") or not tool_name:
            raise ValueError("use a nonempty tool_name and thread, turn, or call mode")
        self._capture_console = capture_console
        self._namespace = tools_namespace
        self._max_ptc_calls = max_ptc_calls
        self._subagent_concurrency = max_subagent_concurrency
        self._subagent_calls = max_subagent_calls
        self._backend = backend
        if backend is not None and (module_tool or workspace is not None):
            raise ValueError(
                "a shared backend cannot use module_tool or direct workspace"
            )
        config = KernelConfig(
            max_memory_bytes=memory_limit,
            max_feed_duration_seconds=timeout,
            default_output_limit_chars=max_result_chars,
            max_output_limit_chars=max(max_result_chars, 1),
            max_state_bytes=memory_limit
            if max_snapshot_bytes is None
            else max_snapshot_bytes,
        )
        super().__init__(
            kernel=AsyncPersistentMontyKernel(config=config, workspace=workspace),
            tool_name=tool_name,
            ptc=ptc,
            subagents=subagents,
            mode=mode or "thread",
            checkpoint_state=(mode or "thread") == "thread",
            max_snapshot_bytes=memory_limit
            if max_snapshot_bytes is None
            else max_snapshot_bytes,
            snapshot_signing_key=snapshot_signing_key,
            hosted_tool_config=HostedToolConfig(
                max_calls_per_operation=2**31 - 1,
                max_parallel_calls=10 + max_subagent_concurrency,
            ),
        )
        self._bridge = LoopThread()
        self.kernel._collect_streams = capture_console
        self._owns_kernel = True
        self.tools = [self._compat_tool()]
        if module_tool:
            self.tools.append(self._compat_tool(module=True))

    def __enter__(self) -> Self:
        """Open the interpreter for synchronous graph calls.

        Returns:
            This middleware.
        """
        self._bridge.submit(self.kernel.open()).result()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the owned interpreter and loop.

        Args:
            *exc_info: Optional context exception details.
        """
        self.close()

    async def __aenter__(self) -> Self:
        """Open the same interpreter for asynchronous graph calls.

        Returns:
            This middleware.
        """
        await asyncio.wrap_future(self._bridge.submit(self.kernel.open()))
        return self

    def close(self) -> None:
        """Close the kernel before its loop; the caller owns any backend."""
        if self._bridge.thread.is_alive():
            self._bridge.submit(self.kernel.close()).result()
            self._bridge.stop()

    async def aclose(self) -> None:
        """Close resources without blocking the caller's event loop."""
        await asyncio.to_thread(self.close)

    def instructions(self, tools: Sequence[BaseTool] = ()) -> str:
        """Describe the Python interface and its active adapter limits.

        Args:
            tools: Current agent tools available for progressive composition.

        Returns:
            Model instructions with exact active function schemas.
        """
        text = (
            super()
            .instructions(tools)
            .replace(
                "Use agent.ainvoke() or agent.astream(); synchronous invocation "
                "cannot run this interpreter. ",
                "Use invoke/stream or ainvoke/astream. ",
            )
        )
        text = text.replace(
            "Use the module tool for small static virtual-file imports.",
            "The optional module tool can load small static virtual-file imports when enabled.",
        )
        text += (
            f"\nUse {self._tool_name}(code=...) with Python, not JavaScript. "
            "Results use <stdout>, <stderr>, <result>, and <error> text blocks. "
            "Send one eval at a time per graph thread. Python state is private "
            "graph checkpoint data; files and tool effects are separate. "
            f"Regular PTC cap: {self._max_ptc_calls}; task cap: {self._subagent_calls}; "
            f"task concurrency: {self._subagent_concurrency}. "
            "When a live task tool is present, "
            "task(description=..., subagent_type=..., response_schema=None, label=None) "
            "accepts a JSON schema of at most 4096 bytes, depth 5, 32 properties. "
            "Subagent events use replay-stable IDs; they do not guarantee exactly-once effects."
        )
        if self._namespace:
            text += "\nBare functions and await tools.NAME(**kwargs) both work."
        if self._backend is not None:
            text += "\nDirect pathlib/open files are disabled. Use the shared hosted fs_* functions.\n"
            text += self._backend.instructions() + "\n"
            text += (
                self._backend._hosted_tools()
                .catalog(async_mode=True)
                .replace("_arenero_backend_fs_", "fs_")
            )
        if len(self.tools) == 1:
            text += (
                "\nNo module tool is enabled. Use inline dependency-free Monty Python."
            )
        return text

    def before_agent(self, state: AreneroAgentState, runtime: Any) -> dict[str, Any]:
        """Assign a private slot before synchronous graph execution.

        Args:
            state: Current graph state.
            runtime: Graph runtime.

        Returns:
            Private slot update.
        """
        return self._bridge.submit(self._before(state)).result()

    async def abefore_agent(
        self, state: AreneroAgentState, runtime: Any
    ) -> dict[str, Any]:
        """Assign a private slot before asynchronous graph execution.

        Args:
            state: Current graph state.
            runtime: Graph runtime.

        Returns:
            Private slot update.
        """
        return await asyncio.wrap_future(self._bridge.submit(self._before(state)))

    async def _before(self, state: AreneroAgentState) -> dict[str, Any]:
        """Create an invocation slot without an instance-wide fallback.

        Args:
            state: Private graph state from the previous checkpoint.

        Returns:
            New or retained slot identity.
        """
        slot = state.get("_arenero_slot_id") if self._mode == "thread" else None
        return {"_arenero_slot_id": slot or uuid4().hex}

    def after_agent(self, state: AreneroAgentState, runtime: Any) -> None:
        """Evict working state after synchronous graph completion.

        Args:
            state: Completed graph state.
            runtime: Graph runtime.
        """
        self._bridge.submit(self._after(state)).result()

    async def aafter_agent(self, state: AreneroAgentState, runtime: Any) -> None:
        """Evict working state after asynchronous graph completion.

        Args:
            state: Completed graph state.
            runtime: Graph runtime.
        """
        await asyncio.wrap_future(self._bridge.submit(self._after(state)))

    async def _after(self, state: AreneroAgentState) -> None:
        """Remove the working copy; graph checkpoints own thread persistence.

        Args:
            state: Graph state with its private slot.
        """
        slot = state.get("_arenero_slot_id")
        if slot:
            await self.kernel.reset(self._thread_session_id(slot))

    def _compat_tool(self, module: bool = False) -> BaseTool:
        """Build one bridged tool with stable graph state updates.

        Args:
            module: Use static virtual-module input instead of a normal cell.

        Returns:
            Tool with both sync and async entry points.
        """
        name = self._tool_name + ("_module" if module else "")

        def invoke(runtime: ToolRuntime, **kwargs: Any) -> Any:
            """Run a cell from a synchronous ToolNode.

            Args:
                runtime: Injected parent runtime.
                **kwargs: Validated tool inputs.

            Returns:
                Message and private checkpoint update.
            """
            return self._bridge.submit(
                self._compat_eval(runtime, kwargs, module)
            ).result()

        async def ainvoke(runtime: ToolRuntime, **kwargs: Any) -> Any:
            """Run the same cell from an asynchronous ToolNode.

            Args:
                runtime: Injected parent runtime.
                **kwargs: Validated tool inputs.

            Returns:
                Message and private checkpoint update.
            """
            return await asyncio.wrap_future(
                self._bridge.submit(self._compat_eval(runtime, kwargs, module))
            )

        return StructuredTool.from_function(
            name=name,
            description="Run bounded Monty Python, not JavaScript or CPython. "
            "Await the hosted functions in the instructions. One eval per graph step. "
            "Read <stdout>, <stderr>, <result>, and <error> blocks.",
            func=invoke,
            coroutine=ainvoke,
            infer_schema=False,
            args_schema=AreneroModuleInput if module else AreneroEvalInput,
            metadata={
                "arenero_kind": "code-execution",
                "code_arg_name": "code",
                "ls_code_input_language": "python",
            },
        )

    async def _compat_eval(
        self, runtime: ToolRuntime, arguments: dict[str, Any], module: bool
    ) -> Any:
        """Restore graph state, run a feed, and checkpoint its completed state.

        Args:
            runtime: Injected graph state and current tool inventory.
            arguments: Validated tool arguments.
            module: Run the optional virtual module entry point when true.

        Returns:
            LangGraph command containing a ToolMessage and private state fields.
        """
        from langgraph.types import Command

        messages = runtime.state.get("messages", [])
        active_calls = getattr(messages[-1], "tool_calls", []) if messages else []
        if (
            sum(
                call.get("name") in {self._tool_name, self._tool_name + "_module"}
                for call in active_calls
            )
            > 1
        ):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content='<error type="ConcurrentEval">Send one eval per graph step.</error>',
                            status="error",
                            name=self._tool_name,
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        slot = runtime.state.get("_arenero_slot_id")
        if not slot:
            raise RuntimeError("before_agent must assign the private interpreter slot")
        session_id = self._thread_session_id(slot)
        if self._mode == "call":
            await self.kernel.reset(session_id)
        elif self._mode == "thread":
            await self.kernel.reset(session_id)
            payload = runtime.state.get("_arenero_snapshot")
            if (
                payload is not None
                and len(payload) <= self._max_snapshot_bytes
                and self._snapshot_valid(
                    payload, cast(AreneroAgentState, runtime.state), session_id
                )
            ):
                await asyncio.to_thread(
                    self.kernel.store.save,
                    session_id,
                    payload,
                    self.kernel.monty_version,
                    None,
                )
        selected = self._select_tools(runtime.tools)
        regular = [tool for tool in selected if tool.name != "task"]
        hosted = hosted_tools_from_langchain(
            regular,
            config=runtime.config,
            runtime=runtime,
            limits=self._hosted_tool_config,
        )
        if self._backend is not None:
            hosted = hosted.merged(
                self._backend._hosted_tools(config=self._hosted_tool_config)
            )
        gate = asyncio.Semaphore(10)
        calls = 0
        from dataclasses import replace

        from .hosted_tools import _invoke_tool

        wrapped = []
        for tool in hosted.tools:

            async def call(args, active=tool):
                """Apply the regular-call budget independently of tasks.

                Args:
                    args: Validated tool arguments.
                    active: Captured hosted function.

                Returns:
                    Hosted function result.
                """
                nonlocal calls
                calls += 1
                if self._max_ptc_calls is not None and calls > self._max_ptc_calls:
                    raise ValueError("regular PTC call limit exceeded")
                async with gate:
                    return await _invoke_tool(active, args)

            wrapped.append(replace(tool, ainvoke=call))
        task = next((tool for tool in selected if tool.name == "task"), None)
        if task is not None:
            wrapped.append(
                task_bridge(
                    task,
                    runtime,
                    concurrency=self._subagent_concurrency,
                    max_calls=self._subagent_calls,
                )
            )
        hosted = HostedToolSet(tuple(wrapped), self._hosted_tool_config)
        control = GraphCalls()
        code = arguments.get("code") if module else arguments["code"]
        prefix = ""
        if self._backend is not None:
            prefix = (
                "\n".join(
                    f"async def {tool.name}(**kwargs):\n    return await {tool.alias}(**kwargs)"
                    for tool in hosted.tools
                    if tool.alias.startswith("_arenero_backend_fs_")
                )
                + "\n"
            )
        if self._namespace and hosted.tools and code is not None:
            methods = "\n".join(
                f"    async def {tool.name if tool.alias.startswith('_arenero_backend_fs_') else tool.alias}(self, **kwargs):\n        return await {tool.alias}(**kwargs)"
                for tool in hosted.tools
            )
            prefix += f"class _AreneroTools:\n{methods}\ntools = _AreneroTools()\n"
        if code is not None:
            code = prefix + code
        kwargs: dict[str, Any] = {
            **arguments,
            "code": code,
            "hosted_tools": control.wrap(hosted),
        }
        if not module:
            kwargs["filesystem"] = self._backend is None
        try:
            operation = self.kernel.execute_module if module else self.kernel.execute
            result = await operation(session_id, **kwargs)
        except asyncio.CancelledError:
            control.reraise()
            raise
        message = ToolMessage(
            content=_compat_text(result, self._capture_console),
            tool_call_id=runtime.tool_call_id,
            name=self._tool_name + ("_module" if module else ""),
            status="success" if result["ok"] else "error",
            artifact=result,
        )
        update: dict[str, Any] = {"messages": [message]}
        if self._mode == "thread":
            stored = await asyncio.to_thread(self.kernel.store.load, session_id)
            payload = (
                stored.state
                if stored and len(stored.state) <= self._max_snapshot_bytes
                else None
            )
            update["_arenero_snapshot"] = payload
            update["_arenero_snapshot_hmac"] = (
                self._sign(payload, session_id)
                if payload and self._signing_key
                else None
            )
        elif self._mode == "call":
            await self.kernel.reset(session_id)
        return Command(update=update)


def _compat_text(result: Mapping[str, Any], capture_console: bool) -> str:
    """Format bounded execution fields as independent escaped text blocks.

    Args:
        result: Existing rich Arenero execution result.
        capture_console: Include stdout and stderr when true.

    Returns:
        Model-facing blocks; the unmodified rich result is in the artifact.
    """
    blocks = []
    if capture_console:
        for name in ("stdout", "stderr"):
            if result[name]:
                blocks.append(f"<{name}>\n{html.escape(result[name])}\n</{name}>")
    if result["error"]:
        error = result["error"]
        blocks.append(
            f'<error type="{html.escape(error["type"], quote=True)}">{html.escape(error["message"])}</error>'
        )
    else:
        blocks.append(
            "<result>"
            + html.escape(json.dumps(result["result"], ensure_ascii=False))
            + "</result>"
        )
    if any(result["truncated"].values()):
        blocks.append("<truncated>Output reached the configured limit.</truncated>")
    return "\n".join(blocks)
