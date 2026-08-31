"""Recursive Language Model support built on the asynchronous Monty kernel."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .async_kernel import AsyncPersistentMontyKernel
from .events import emit_event
from .hosted_tools import HostedToolConfig, HostedToolSet

_CODE_BLOCK = re.compile(r"```(?:python|repl)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BUDGET_PREFIXES = (
    "RLMCallLimitExceeded:",
    "RLMCostLimitExceeded:",
    "RLMPromptTooLarge:",
    "RLMResultTooLarge:",
    "RLMResponseTooLarge:",
    "RLMTimeout:",
    "RLMTokenLimitExceeded:",
)


@dataclass(frozen=True)
class RLMModelResponse:
    """Hold normalized text and optional usage from one model call.

    Attributes:
        content: Model response text.
        input_tokens: Reported input token count, or zero when unknown.
        output_tokens: Reported output token count, or zero when unknown.
        cost: Reported call cost, or zero when unknown.
    """

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def __post_init__(self) -> None:
        """Validate model response text and usage values."""

        if not isinstance(self.content, str):
            raise TypeError("RLM model response content must be a string")
        if self.input_tokens < 0 or self.output_tokens < 0 or self.cost < 0:
            raise ValueError("RLM model usage values must not be negative")


RLMModelCall = Callable[[str, str | None], Awaitable[str | RLMModelResponse]]


@dataclass(frozen=True)
class RLMConfig:
    """Set recursion, model-call, context, and result limits.

    Attributes:
        max_depth: Maximum child RLM depth. Zero allows only plain model calls.
        max_iterations: Maximum root-model and REPL steps for each RLM node.
        max_errors: Maximum consecutive REPL errors for each RLM node.
        max_model_calls: Maximum model calls across the complete recursion tree.
        max_parallel_calls: Maximum model calls that can run at the same time.
        max_context_chars: Maximum characters stored in one RLM context.
        max_prompt_chars: Maximum characters sent in one model call.
        max_response_chars: Maximum characters accepted from one model call.
        max_transcript_chars: Maximum recent transcript characters sent to a root.
        max_repl_output_chars: Maximum characters returned per Monty output field.
        max_total_tokens: Optional reported-token limit for the recursion tree.
        max_cost: Optional reported-cost limit for the recursion tree.
        timeout_seconds: Wall-time limit for the complete recursion tree.
    """

    max_depth: int = 1
    max_iterations: int = 20
    max_errors: int = 3
    max_model_calls: int = 64
    max_parallel_calls: int = 8
    max_context_chars: int = 10_000_000
    max_prompt_chars: int = 200_000
    max_response_chars: int = 100_000
    max_transcript_chars: int = 40_000
    max_repl_output_chars: int = 8_000
    max_total_tokens: int | None = None
    max_cost: float | None = None
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        """Validate all RLM limits."""

        if self.max_depth < 0:
            raise ValueError("max_depth must be at least 0")
        positive = {
            "max_iterations": self.max_iterations,
            "max_errors": self.max_errors,
            "max_model_calls": self.max_model_calls,
            "max_parallel_calls": self.max_parallel_calls,
            "max_context_chars": self.max_context_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "max_response_chars": self.max_response_chars,
            "max_transcript_chars": self.max_transcript_chars,
            "max_repl_output_chars": self.max_repl_output_chars,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"RLM limits must be greater than 0: {', '.join(invalid)}")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be greater than 0")
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValueError("max_cost must be greater than 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")


@dataclass(frozen=True)
class RLMUsage:
    """Report model and recursive work for one RLM completion.

    Attributes:
        model_calls: Total model calls across the recursion tree.
        recursive_calls: Child RLM calls that received their own REPL.
        input_tokens: Total reported input tokens.
        output_tokens: Total reported output tokens.
        total_tokens: Total reported input and output tokens.
        cost: Total reported cost.
        deepest_level: Deepest RLM level started.
        wall_time_seconds: Total elapsed wall time.
    """

    model_calls: int
    recursive_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    deepest_level: int
    wall_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible usage data.

        Returns:
            Usage fields as a dictionary.
        """

        return asdict(self)


@dataclass(frozen=True)
class RLMResult:
    """Return one final RLM response and its bounded execution metadata.

    Attributes:
        response: Final answer text.
        session_id: Ephemeral root Monty session identifier.
        iterations: Root RLM iteration count.
        usage: Usage across all recursive branches.
    """

    response: str
    session_id: str
    iterations: int
    usage: RLMUsage

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible result data.

        Returns:
            Response, session, iteration, and usage fields.
        """

        return {
            "response": self.response,
            "session_id": self.session_id,
            "iterations": self.iterations,
            "usage": self.usage.to_dict(),
        }


@dataclass
class _RunState:
    """Share budgets and usage across one recursive completion tree."""

    model_call: RLMModelCall
    config: RLMConfig
    event_sink: Any = None
    started_at: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    recursive_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    deepest_level: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _slots: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        """Create the configured model-call semaphore."""

        self._slots = asyncio.Semaphore(self.config.max_parallel_calls)

    def remaining_seconds(self) -> float:
        """Return remaining wall time.

        Returns:
            Positive seconds before the RLM deadline.
        """

        remaining = self.config.timeout_seconds - (time.monotonic() - self.started_at)
        if remaining <= 0:
            raise RuntimeError("RLMTimeout: recursive completion time limit exceeded")
        return remaining

    async def call(self, prompt: str, model: str | None, depth: int) -> str:
        """Call the configured model under shared limits.

        Args:
            prompt: Complete model input.
            model: Optional adapter-defined model name.
            depth: Current RLM depth.

        Returns:
            Model response text.
        """

        if len(prompt) > self.config.max_prompt_chars:
            raise RuntimeError(
                "RLMPromptTooLarge: model prompt exceeds max_prompt_chars"
            )
        async with self._lock:
            if self.model_calls >= self.config.max_model_calls:
                raise RuntimeError("RLMCallLimitExceeded: model call limit exceeded")
            self.model_calls += 1
            call_number = self.model_calls
            self.deepest_level = max(self.deepest_level, depth)
        emit_event(
            self.event_sink,
            "rlm",
            "model_call_started",
            call_number=call_number,
            depth=depth,
            model=model,
        )
        async with self._slots:
            try:
                raw = await asyncio.wait_for(
                    self.model_call(prompt, model),
                    timeout=self.remaining_seconds(),
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "RLMTimeout: recursive completion time limit exceeded"
                ) from exc
        response = _normalize_response(raw)
        if len(response.content) > self.config.max_response_chars:
            raise RuntimeError(
                "RLMResponseTooLarge: model response exceeds max_response_chars"
            )
        async with self._lock:
            self.input_tokens += response.input_tokens
            self.output_tokens += response.output_tokens
            self.cost += response.cost
            total_tokens = self.input_tokens + self.output_tokens
            if (
                self.config.max_total_tokens is not None
                and total_tokens > self.config.max_total_tokens
            ):
                raise RuntimeError(
                    "RLMTokenLimitExceeded: reported token limit exceeded"
                )
            if self.config.max_cost is not None and self.cost > self.config.max_cost:
                raise RuntimeError("RLMCostLimitExceeded: reported cost limit exceeded")
        emit_event(
            self.event_sink,
            "rlm",
            "model_call_completed",
            call_number=call_number,
            depth=depth,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
        )
        return response.content

    async def add_recursive_call(self, depth: int) -> None:
        """Record one child RLM call.

        Args:
            depth: Child RLM depth.
        """

        async with self._lock:
            self.recursive_calls += 1
            self.deepest_level = max(self.deepest_level, depth)

    def usage(self) -> RLMUsage:
        """Build current usage data.

        Returns:
            Immutable usage snapshot.
        """

        return RLMUsage(
            model_calls=self.model_calls,
            recursive_calls=self.recursive_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            cost=self.cost,
            deepest_level=self.deepest_level,
            wall_time_seconds=time.monotonic() - self.started_at,
        )


class RLM:
    """Run recursive model calls through isolated child Monty sessions.

    Args:
        model_call: Async adapter called as ``model_call(prompt, model_name)``.
        config: Optional recursive and model-call limits.
        hosted_tool_config: Optional limits for calls made from one Monty feed.
    """

    def __init__(
        self,
        model_call: RLMModelCall,
        *,
        config: RLMConfig | None = None,
        hosted_tool_config: HostedToolConfig | None = None,
    ) -> None:
        """Create an RLM runner around one async model adapter."""

        self.model_call = model_call
        self.config = config or RLMConfig()
        self.hosted_tool_config = hosted_tool_config or HostedToolConfig(
            max_total_call_seconds=self.config.timeout_seconds,
            max_parallel_calls=self.config.max_parallel_calls,
        )

    async def complete(
        self,
        kernel: AsyncPersistentMontyKernel,
        context: str,
        question: str = "",
        *,
        model: str | None = None,
        hosted_tools: HostedToolSet | None = None,
    ) -> RLMResult:
        """Run one complete root RLM operation.

        Args:
            kernel: Async kernel used for root and child REPL sessions.
            context: Full external text stored in the root REPL.
            question: Optional task shown to the root model.
            model: Optional adapter-defined model name.
            hosted_tools: Optional application tools available in every RLM node.

        Returns:
            Final response and usage for the complete recursion tree.
        """

        self._validate_context(context)
        state = _RunState(self.model_call, self.config, kernel.event_sink)
        session_id = _session_id(0)
        emit_event(
            kernel.event_sink,
            "rlm",
            "started",
            session_id=session_id,
            context_chars=len(context),
        )
        try:
            response, iterations = await self._complete_node(
                kernel,
                context,
                question,
                model,
                hosted_tools,
                state,
                depth=0,
                session_id=session_id,
            )
        except BaseException as exc:
            emit_event(
                kernel.event_sink,
                "rlm",
                "failed",
                session_id=session_id,
                error_type=type(exc).__name__,
            )
            raise
        usage = state.usage()
        emit_event(
            kernel.event_sink,
            "rlm",
            "completed",
            session_id=session_id,
            model_calls=usage.model_calls,
            recursive_calls=usage.recursive_calls,
            wall_time_seconds=usage.wall_time_seconds,
        )
        return RLMResult(response, session_id, iterations, usage)

    def hosted_tools(
        self,
        kernel: AsyncPersistentMontyKernel,
        parent_session_id: str,
        *,
        application_tools: HostedToolSet | None = None,
    ) -> HostedToolSet:
        """Create RLM functions for one existing Monty operation.

        Args:
            kernel: Async kernel used for child RLM sessions.
            parent_session_id: Active parent session used only for event lineage.
            application_tools: Optional tools inherited by child RLM nodes.

        Returns:
            Operation-scoped RLM and application hosted tools.
        """

        state = _RunState(self.model_call, self.config, kernel.event_sink)
        return self._toolset(
            kernel,
            parent_session_id,
            application_tools,
            state,
            depth=0,
        )

    def catalog(
        self,
        kernel: AsyncPersistentMontyKernel,
        *,
        async_mode: bool = True,
    ) -> str:
        """Return model instructions for the RLM hosted functions.

        Args:
            kernel: Kernel used only to build callable closures and schemas.
            async_mode: Render async declarations when true.

        Returns:
            Python declarations and concise RLM behavior notes.
        """

        state = _RunState(self.model_call, self.config, kernel.event_sink)
        toolset = self._toolset(
            kernel,
            "arenero-rlm-catalog",
            None,
            state,
            depth=0,
        )
        return (
            toolset.catalog(async_mode=async_mode)
            + "\nRLM notes: `llm_query` makes one plain model call. "
            "`rlm_query` gives the child its own temporary REPL until max_depth; "
            "at max_depth it becomes a plain model call. Batch forms can run in "
            "parallel. Use `rlm_usage` to inspect the current shared budget."
        )

    def capabilities(self) -> dict[str, Any]:
        """Return the active recursive model-call capability report.

        Returns:
            JSON-compatible function names, limits, and important boundaries.
        """

        return {
            "enabled": True,
            "functions": [
                "llm_query",
                "llm_query_batched",
                "rlm_query",
                "rlm_query_batched",
                "rlm_usage",
            ],
            "limits": asdict(self.config),
            "boundaries": [
                "recursive children use separate temporary Monty sessions",
                "recursive child sessions do not mount the kernel filesystem",
                "inherited hosted file tools can share files with children",
                "the active parent interpreter never calls itself",
                "token and cost limits require adapter-reported usage",
            ],
        }

    async def _complete_node(
        self,
        kernel: AsyncPersistentMontyKernel,
        context: str,
        question: str,
        model: str | None,
        application_tools: HostedToolSet | None,
        state: _RunState,
        *,
        depth: int,
        session_id: str,
    ) -> tuple[str, int]:
        """Run one RLM node in its own temporary session.

        Args:
            kernel: Shared async kernel.
            context: Text stored in this node's REPL.
            question: Task shown to this node's root model.
            model: Optional model name.
            application_tools: Optional tools inherited by this node.
            state: Shared recursion budget and usage state.
            depth: Current RLM level.
            session_id: Unique temporary Monty session.

        Returns:
            Final response and iteration count for this node.
        """

        self._validate_context(context)
        toolset = self._toolset(
            kernel,
            session_id,
            application_tools,
            state,
            depth=depth,
        )
        transcript: list[str] = []
        errors = 0
        try:
            initialized = await kernel.execute(
                session_id,
                "answer = {'content': '', 'ready': False}\nlen(context)",
                self.config.max_repl_output_chars,
                restart=True,
                inputs={"context": context, "question": question},
                filesystem=depth == 0,
            )
            if not initialized["ok"]:
                raise RuntimeError(
                    "RLMInitializationError: " + _execution_error_message(initialized)
                )
            for iteration in range(1, self.config.max_iterations + 1):
                state.remaining_seconds()
                prompt = self._root_prompt(
                    kernel,
                    toolset,
                    context_chars=len(context),
                    question=question,
                    transcript=transcript,
                    depth=depth,
                )
                response = await state.call(prompt, model, depth)
                code_blocks = _code_blocks(response)
                if not code_blocks:
                    return _final_text(response), iteration
                transcript.append(f"MODEL:\n{response}")
                for code in code_blocks:
                    execution = await kernel.execute(
                        session_id,
                        code.rstrip() + "\nanswer",
                        self.config.max_repl_output_chars,
                        hosted_tools=toolset,
                        filesystem=depth == 0,
                    )
                    transcript.append(
                        "REPL:\n"
                        + json.dumps(execution, ensure_ascii=False, default=str)
                    )
                    if not execution["ok"]:
                        message = _execution_error_message(execution)
                        if any(prefix in message for prefix in _BUDGET_PREFIXES):
                            raise RuntimeError(message)
                        errors += 1
                        if errors >= self.config.max_errors:
                            raise RuntimeError(
                                "RLMErrorLimitExceeded: consecutive REPL error "
                                "limit exceeded"
                            )
                        break
                    errors = 0
                    if execution["truncated"]["result"]:
                        raise RuntimeError(
                            "RLMResultTooLarge: increase max_repl_output_chars "
                            "or produce a smaller final answer"
                        )
                    answer = execution["result"]
                    if isinstance(answer, dict) and answer.get("ready") is True:
                        return str(answer.get("content", "")), iteration
                transcript = _trim_transcript(
                    transcript, self.config.max_transcript_chars
                )
            raise RuntimeError(
                "RLMIterationLimitExceeded: root iteration limit exceeded"
            )
        finally:
            await kernel.reset(session_id, filesystem=depth == 0)

    def _toolset(
        self,
        kernel: AsyncPersistentMontyKernel,
        parent_session_id: str,
        application_tools: HostedToolSet | None,
        state: _RunState,
        *,
        depth: int,
    ) -> HostedToolSet:
        """Build operation functions that close over one shared run state.

        Args:
            kernel: Shared async kernel.
            parent_session_id: Parent session used for lineage events.
            application_tools: Optional inherited tools.
            state: Shared recursion state.
            depth: Current RLM level.

        Returns:
            RLM functions merged with inherited application tools.
        """

        async def llm_query(prompt: str, model: str | None = None) -> str:
            """Make one plain model call without a child REPL.

            Args:
                prompt: Complete subquery sent to the model.
                model: Optional adapter-defined model name.

            Returns:
                Model response text.
            """

            return await state.call(prompt, model, depth)

        async def llm_query_batched(
            prompts: list[str], model: str | None = None
        ) -> list[str]:
            """Make plain model calls concurrently under the shared limit.

            Args:
                prompts: Complete subqueries sent to the model.
                model: Optional adapter-defined model name.

            Returns:
                Response text in input order.
            """

            return await asyncio.gather(
                *(state.call(prompt, model, depth) for prompt in prompts)
            )

        async def rlm_query(prompt: str, model: str | None = None) -> str:
            """Run a child RLM or a plain call at the depth limit.

            Args:
                prompt: Child task and context text.
                model: Optional adapter-defined model name.

            Returns:
                Child RLM or plain model response text.
            """

            if depth >= self.config.max_depth:
                return await state.call(prompt, model, depth)
            child_depth = depth + 1
            await state.add_recursive_call(child_depth)
            child_session_id = _session_id(child_depth)
            emit_event(
                kernel.event_sink,
                "rlm",
                "child_started",
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
                depth=child_depth,
            )
            response, _ = await self._complete_node(
                kernel,
                prompt,
                "Answer the request stored in context.",
                model,
                application_tools,
                state,
                depth=child_depth,
                session_id=child_session_id,
            )
            emit_event(
                kernel.event_sink,
                "rlm",
                "child_completed",
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
                depth=child_depth,
            )
            return response

        async def rlm_query_batched(
            prompts: list[str], model: str | None = None
        ) -> list[str]:
            """Run child RLM calls concurrently under shared limits.

            Args:
                prompts: Child task and context texts.
                model: Optional adapter-defined model name.

            Returns:
                Child responses in input order.
            """

            return await asyncio.gather(
                *(rlm_query(prompt, model) for prompt in prompts)
            )

        async def rlm_usage() -> dict[str, Any]:
            """Return current shared recursion usage.

            Returns:
                Model calls, recursion, tokens, cost, depth, and wall time.
            """

            return state.usage().to_dict()

        rlm_tools = HostedToolSet.from_callables(
            [
                llm_query,
                llm_query_batched,
                rlm_query,
                rlm_query_batched,
                rlm_usage,
            ],
            config=self.hosted_tool_config,
        )
        return rlm_tools.merged(application_tools)

    def _root_prompt(
        self,
        kernel: AsyncPersistentMontyKernel,
        toolset: HostedToolSet,
        *,
        context_chars: int,
        question: str,
        transcript: list[str],
        depth: int,
    ) -> str:
        """Build one root-model prompt without copying the external context.

        Args:
            kernel: Kernel that supplies current Monty instructions.
            toolset: Exact hosted tools for this RLM node.
            context_chars: Size of the context stored in Monty.
            question: Current task text.
            transcript: Bounded prior model and REPL observations.
            depth: Current RLM level.

        Returns:
            Complete root-model prompt.
        """

        history = "\n\n".join(transcript) if transcript else "(none)"
        return f"""You are an RLM controller. Use code to inspect external context.

RLM depth: {depth}
The complete input is stored in the Monty variable `context`.
Context size: {context_chars} characters.
The task is stored in `question` and is also shown here:
{question or "Answer the request contained in context."}

Write one or more fenced `python` or `repl` code blocks. Keep large context
parts inside Monty. Use slicing, search, loops, `llm_query`, or `rlm_query` to
analyze selected parts. Hosted functions accept keyword arguments only and must
be awaited. When the answer is ready, set:

answer["content"] = "final answer"
answer["ready"] = True

You can also return direct text without a code block when no REPL work is needed.

Direct kernel filesystem access is available only at depth 0. Recursive child
REPLs can still use explicitly inherited hosted file tools.

{kernel.instructions()}

{toolset.catalog(async_mode=True)}

Recent transcript:
{history}
"""

    def _validate_context(self, context: str) -> None:
        """Validate one external context value.

        Args:
            context: Text that will be stored in Monty.
        """

        if not isinstance(context, str):
            raise TypeError("RLM context must be a string")
        if len(context) > self.config.max_context_chars:
            raise ValueError("RLM context exceeds max_context_chars")


def _normalize_response(value: str | RLMModelResponse) -> RLMModelResponse:
    """Normalize one adapter response.

    Args:
        value: Plain text or a response with usage.

    Returns:
        Validated response object.
    """

    if isinstance(value, RLMModelResponse):
        return value
    if isinstance(value, str):
        return RLMModelResponse(value)
    raise TypeError("RLM model adapter must return str or RLMModelResponse")


def _code_blocks(response: str) -> list[str]:
    """Extract executable Python and REPL fences.

    Args:
        response: Root-model response text.

    Returns:
        Code block bodies in response order.
    """

    return [block.strip() for block in _CODE_BLOCK.findall(response) if block.strip()]


def _final_text(response: str) -> str:
    """Remove an optional final-answer marker.

    Args:
        response: Direct root-model response.

    Returns:
        Final answer text.
    """

    stripped = response.strip()
    if stripped.upper().startswith("FINAL:"):
        return stripped[6:].strip()
    return stripped


def _trim_transcript(items: list[str], limit: int) -> list[str]:
    """Keep the newest complete transcript items under a character limit.

    Args:
        items: Transcript entries in time order.
        limit: Maximum combined characters.

    Returns:
        Newest entries that fit, with a marker when older entries were removed.
    """

    kept: list[str] = []
    size = 0
    for item in reversed(items):
        if kept and size + len(item) > limit:
            break
        kept.append(item[-limit:] if not kept and len(item) > limit else item)
        size += len(kept[-1])
    kept.reverse()
    if len(kept) < len(items):
        kept.insert(0, "(older transcript entries were trimmed)")
    return kept


def _execution_error_message(result: Mapping[str, Any]) -> str:
    """Read one kernel error message.

    Args:
        result: Structured kernel result.

    Returns:
        Error message or a fallback description.
    """

    error = result.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "REPL error")
    return "REPL error"


def _session_id(depth: int) -> str:
    """Create a short unique RLM session identifier.

    Args:
        depth: RLM depth included for diagnostics.

    Returns:
        Session ID below Arenero's 255-character limit.
    """

    return f"arenero-rlm-{depth}-{uuid.uuid4().hex}"
