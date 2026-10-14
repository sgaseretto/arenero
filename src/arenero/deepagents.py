"""Optional Deep Agents middleware backed by Arenero and Monty."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal, NotRequired, Self
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
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import BaseModel, Field

from .async_kernel import AsyncPersistentMontyKernel
from .hosted_tools import HostedToolConfig, HostedToolSet
from .langchain_hosted_tools import hosted_tools_from_langchain
from .rlm import RLM

logger = logging.getLogger(__name__)

PersistenceMode = Literal["thread", "turn", "call"]
PTCOption = Sequence[str | BaseTool]


class AreneroAgentState(AgentState):
    """Add private Monty snapshot fields to LangGraph agent state."""

    _arenero_snapshot: NotRequired[Annotated[bytes, PrivateStateAttr]]
    _arenero_snapshot_hmac: NotRequired[Annotated[bytes, PrivateStateAttr]]


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
        ) -> ToolMessage:
            """Return guidance when an agent uses synchronous invocation.

            Args:
                runtime: Active outer tool runtime.
                code: Unused Monty source.
                output_limit_chars: Unused requested output limit.

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
        ) -> ToolMessage:
            """Run one asynchronous persistent Monty cell.

            Args:
                runtime: Active outer tool runtime.
                code: Monty-compatible Python source.
                output_limit_chars: Optional caller output limit.

            Returns:
                ToolMessage with the JSON execution result.
            """

            session_id = middleware._session_for_runtime(runtime)
            hosted = middleware._hosted_for_runtime(runtime, session_id)
            try:
                result = await middleware.kernel.execute(
                    session_id,
                    code,
                    output_limit_chars,
                    hosted_tools=hosted,
                )
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
        ) -> ToolMessage:
            """Return guidance for unsupported synchronous module execution.

            Args:
                runtime: Active outer tool runtime.
                entry_path: Unused virtual entry path.
                code: Unused follow-up code.
                output_limit_chars: Unused requested output limit.

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
        ) -> ToolMessage:
            """Resolve and run one static virtual module graph.

            Args:
                runtime: Active outer tool runtime.
                entry_path: Absolute virtual entry ``.py`` file.
                code: Optional follow-up code.
                output_limit_chars: Optional caller output limit.

            Returns:
                ToolMessage with the JSON execution result.
            """

            session_id = middleware._session_for_runtime(runtime)
            hosted = middleware._hosted_for_runtime(runtime, session_id)
            try:
                result = await middleware.kernel.execute_module(
                    session_id,
                    entry_path,
                    code,
                    output_limit_chars,
                    hosted_tools=hosted,
                )
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
