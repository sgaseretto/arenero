"""Speculative programmatic tool calls with a Monty-only shadow session."""

from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import json
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from itertools import islice
from typing import Any, Literal

from .async_kernel import AsyncPersistentMontyKernel
from .events import EventSink, emit_event
from .hosted_tools import HostedTool, HostedToolSet

SpeculationMode = Literal["off", "lazy", "stream"]
SpeculationKey = tuple[str, str]
_MAX_STATIC_SEQUENCE_SIZE = 1_000_000


@dataclass(frozen=True)
class SpeculativeToolPolicy:
    """Mark one hosted tool as safe for speculative execution.

    Args:
        name: Hosted-tool name or Python alias.
        deterministic: Reuse one result for identical calls when true.
        key: Optional function that returns stable identity material.
        gate: Optional function that approves each concrete call.
    """

    name: str
    deterministic: bool = False
    key: Callable[[Mapping[str, Any]], Any] | None = field(
        default=None, repr=False, compare=False
    )
    gate: Callable[[Mapping[str, Any]], bool] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the tool policy name."""

        if not self.name:
            raise ValueError("speculative tool name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return safe capability data for this policy.

        Returns:
            Public policy fields without callable objects.
        """

        return {
            "name": self.name,
            "deterministic": self.deterministic,
            "custom_key": self.key is not None,
            "gated": self.gate is not None,
        }


def _default_policies() -> tuple[SpeculativeToolPolicy, ...]:
    """Return safe default RLM tool policies.

    Returns:
        Policies for plain single and batched model calls.
    """

    return (
        SpeculativeToolPolicy("llm_query"),
        SpeculativeToolPolicy("llm_query_batched"),
    )


@dataclass(frozen=True)
class SpeculationConfig:
    """Configure speculative programmatic tool calling.

    Attributes:
        mode: ``off``, full-response ``lazy``, or token ``stream`` planning.
        tools: Tools the host asserts are pure enough to run early.
        max_inflight: Maximum speculative tool calls running at once.
        max_calls_per_turn: Maximum speculative dispatches in one model turn.
        max_argument_chars: Maximum canonical argument size for one dispatch.
        max_code_chars: Maximum generated code retained by the planner.
        max_loop_unroll: Maximum statically planned loop iterations.
        shadow: Use an isolated Monty snapshot to resolve state-dependent calls.
        shadow_output_chars: Maximum output kept from one shadow statement.
        shadow_drain_seconds: Maximum wait for the shadow after generation.
        cancel_unclaimed: Cancel unfinished unused calls at turn end.
    """

    mode: SpeculationMode = "off"
    tools: tuple[SpeculativeToolPolicy, ...] = field(default_factory=_default_policies)
    max_inflight: int = 8
    max_calls_per_turn: int = 64
    max_argument_chars: int = 100_000
    max_code_chars: int = 200_000
    max_loop_unroll: int = 64
    shadow: bool = True
    shadow_output_chars: int = 1_000
    shadow_drain_seconds: float = 30.0
    cancel_unclaimed: bool = True

    def __post_init__(self) -> None:
        """Validate speculation limits and unique policies."""

        if self.mode not in {"off", "lazy", "stream"}:
            raise ValueError("speculation mode must be 'off', 'lazy', or 'stream'")
        positive = {
            "max_inflight": self.max_inflight,
            "max_calls_per_turn": self.max_calls_per_turn,
            "max_argument_chars": self.max_argument_chars,
            "max_code_chars": self.max_code_chars,
            "max_loop_unroll": self.max_loop_unroll,
            "shadow_output_chars": self.shadow_output_chars,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                "speculation limits must be greater than 0: " + ", ".join(invalid)
            )
        if self.shadow_drain_seconds <= 0:
            raise ValueError("shadow_drain_seconds must be greater than 0")
        names = [policy.name for policy in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("speculative tool policy names must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible active settings.

        Returns:
            Public speculation settings and tool policies.
        """

        return {
            "mode": self.mode,
            "tools": [policy.to_dict() for policy in self.tools],
            "max_inflight": self.max_inflight,
            "max_calls_per_turn": self.max_calls_per_turn,
            "max_argument_chars": self.max_argument_chars,
            "max_code_chars": self.max_code_chars,
            "max_loop_unroll": self.max_loop_unroll,
            "shadow": self.shadow,
            "shadow_output_chars": self.shadow_output_chars,
            "shadow_drain_seconds": self.shadow_drain_seconds,
            "cancel_unclaimed": self.cancel_unclaimed,
        }


@dataclass(frozen=True)
class SpeculationUsage:
    """Report speculative work across one RLM completion.

    Attributes:
        turns: Model turns that enabled speculation.
        dispatched: Calls started before real execution requested them.
        hits: Real calls served from speculative work.
        misses: Real calls that ran through the normal path.
        already_ready: Hits complete before the real call arrived.
        failed: Speculative calls that raised an error.
        cancelled: Unused pending calls cancelled during cleanup.
        wasted: Unused calls that completed before cleanup.
        shadow_executions: Statements executed in Monty shadow sessions.
        shadow_errors: Shadow statements that failed without affecting real state.
        waited_seconds: Time real calls spent waiting for speculative results.
    """

    turns: int = 0
    dispatched: int = 0
    hits: int = 0
    misses: int = 0
    already_ready: int = 0
    failed: int = 0
    cancelled: int = 0
    wasted: int = 0
    shadow_executions: int = 0
    shadow_errors: int = 0
    waited_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible usage fields.

        Returns:
            Speculation counters and timing.
        """

        return asdict(self)


@dataclass
class _MutableUsage:
    """Collect mutable counters for one speculative turn."""

    dispatched: int = 0
    hits: int = 0
    misses: int = 0
    already_ready: int = 0
    failed: int = 0
    cancelled: int = 0
    wasted: int = 0
    shadow_executions: int = 0
    shadow_errors: int = 0
    waited_seconds: float = 0.0

    def freeze(self, turns: int = 1) -> SpeculationUsage:
        """Build an immutable turn report.

        Args:
            turns: One when speculation was active, otherwise zero.

        Returns:
            Usage with the supplied completed-turn count.
        """

        return SpeculationUsage(turns=turns, **asdict(self))


@dataclass
class _Speculation:
    """Hold one dispatched call and its claim state."""

    key: SpeculationKey
    sequence: int
    tool: HostedTool
    policy: SpeculativeToolPolicy
    arguments: dict[str, Any]
    source: Literal["planner", "shadow"]
    task: asyncio.Task[Any]
    adopted: bool = False
    claims: int = 0


@dataclass(frozen=True)
class _Plan:
    """Describe one statically resolved tool call."""

    tool: HostedTool
    policy: SpeculativeToolPolicy
    arguments: dict[str, Any]


@dataclass(frozen=True)
class _CodeBlock:
    """Hold one generated Python fence and its closure state."""

    source: str
    closed: bool


class SpeculativeTurn:
    """Coordinate one streamed model response and one real Monty execution.

    Args:
        kernel: Kernel that owns the real and shadow sessions.
        session_id: Saved real session to fork for shadow execution.
        tools: Complete real hosted-tool set.
        config: Speculation modes, policies, and limits.
        known_values: Plain values available to the safe static planner.
        event_sink: Optional safe event observer.
    """

    def __init__(
        self,
        kernel: AsyncPersistentMontyKernel,
        session_id: str,
        tools: HostedToolSet,
        config: SpeculationConfig,
        *,
        known_values: Mapping[str, Any] | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        """Create an idle speculative turn."""

        self.kernel = kernel
        self.session_id = session_id
        self.tools = tools
        self.config = config
        self.event_sink = event_sink
        self.known_values = dict(known_values or {})
        self.turn_id = uuid.uuid4().hex
        self.shadow_session_id = f"arenero-spec-{self.turn_id}"
        self._tool_by_alias = {tool.alias: tool for tool in tools.tools}
        self._policies = self._active_policies()
        self._calls: dict[SpeculationKey, deque[_Speculation]] = defaultdict(deque)
        self._all: list[_Speculation] = []
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(config.max_inflight)
        self._usage = _MutableUsage()
        self._sequence = 0
        self._buffer = ""
        self._planned_counts: Counter[SpeculationKey] = Counter()
        self._shadow_counts: dict[int, int] = defaultdict(int)
        self._shadow_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._shadow_task: asyncio.Task[None] | None = None
        self._shadow_started = False
        self._generation_finished = False
        self._closed = False
        self._real_tools = self._replace_tools(shadow=False)
        self._shadow_tools = self._replace_tools(shadow=True)

    @property
    def active(self) -> bool:
        """Return whether this turn has configured tools and an active mode."""

        return self.config.mode != "off" and bool(self._policies)

    @property
    def real_tools(self) -> HostedToolSet:
        """Return real tools whose eligible calls claim speculative results."""

        return self._real_tools

    async def start(self) -> None:
        """Create the optional isolated Monty shadow snapshot."""

        if not self.active:
            return
        emit_event(
            self.event_sink,
            "speculation",
            "turn_started",
            turn_id=self.turn_id,
            session_id=self.session_id,
            mode=self.config.mode,
        )
        if not self.config.shadow or not self.kernel.store.stores_state:
            return
        try:
            await self.kernel.fork_session(self.session_id, self.shadow_session_id)
        except (RuntimeError, ValueError):
            emit_event(
                self.event_sink,
                "speculation",
                "shadow_unavailable",
                turn_id=self.turn_id,
                session_id=self.session_id,
            )
            return
        self._shadow_started = True
        self._shadow_task = asyncio.create_task(
            self._run_shadow(), name=f"arenero-shadow-{self.turn_id[:8]}"
        )
        emit_event(
            self.event_sink,
            "speculation",
            "shadow_started",
            turn_id=self.turn_id,
            shadow_session_id=self.shadow_session_id,
        )

    async def feed(self, delta: str) -> None:
        """Feed one generated text delta to planning and shadow queues.

        Args:
            delta: New model response text.
        """

        if not self.active or not delta:
            return
        remaining = self.config.max_code_chars - len(self._buffer)
        if remaining <= 0:
            return
        self._buffer += delta[:remaining]
        if not any(token in delta for token in ("\n", ")", "]", "}", "```")):
            return
        await self._plan_current_code()
        if self._shadow_started:
            self._queue_shadow_statements()

    async def finish_generation(self) -> None:
        """Plan final text and drain the shadow before real execution."""

        if not self.active or self._generation_finished:
            return
        self._generation_finished = True
        await self._plan_current_code()
        if not self._shadow_started:
            return
        self._queue_shadow_statements(final=True)
        await self._shadow_queue.put(None)
        assert self._shadow_task is not None
        try:
            await asyncio.wait_for(
                asyncio.shield(self._shadow_task),
                timeout=self.config.shadow_drain_seconds,
            )
        except TimeoutError:
            self._shadow_task.cancel()
            await asyncio.gather(self._shadow_task, return_exceptions=True)
            self._usage.shadow_errors += 1
            emit_event(
                self.event_sink,
                "speculation",
                "shadow_timeout",
                turn_id=self.turn_id,
            )

    async def close(self) -> SpeculationUsage:
        """Stop the shadow, cancel unused work, and remove temporary state.

        Returns:
            Final speculative usage for this turn.
        """

        if self._closed:
            return self._usage.freeze(turns=int(self.active))
        self._closed = True
        if self._shadow_task is not None and not self._shadow_task.done():
            self._shadow_task.cancel()
            await asyncio.gather(self._shadow_task, return_exceptions=True)
        if self._shadow_started:
            await self.kernel.reset(self.shadow_session_id, filesystem=False)
        await self._discard_unclaimed()
        usage = self._usage.freeze(turns=int(self.active))
        emit_event(
            self.event_sink,
            "speculation",
            "turn_finished",
            turn_id=self.turn_id,
            dispatched=usage.dispatched,
            hits=usage.hits,
            misses=usage.misses,
            cancelled=usage.cancelled,
            wasted=usage.wasted,
        )
        return usage

    def _active_policies(self) -> dict[str, SpeculativeToolPolicy]:
        """Resolve configured policy names to active Python aliases.

        Returns:
            Alias-to-policy mapping for tools present in this operation.
        """

        result: dict[str, SpeculativeToolPolicy] = {}
        for policy in self.config.tools:
            for tool in self.tools.tools:
                if policy.name in {tool.name, tool.alias} and not tool.sequential:
                    result[tool.alias] = policy
                    break
        return result

    def _replace_tools(self, *, shadow: bool) -> HostedToolSet:
        """Build shadow-only or real claim hosted tools.

        Args:
            shadow: Return only speculative shadow wrappers when true.

        Returns:
            Hosted tools that dispatch, adopt, or claim through this turn.
        """

        wrapped: list[HostedTool] = []
        for tool in self.tools.tools:
            policy = self._policies.get(tool.alias)
            if policy is None:
                if not shadow:
                    wrapped.append(tool)
                continue

            async def invoke(
                arguments: dict[str, Any],
                _tool: HostedTool = tool,
                _policy: SpeculativeToolPolicy = policy,
            ) -> Any:
                """Dispatch a shadow call or claim a real call.

                Args:
                    arguments: Concrete keyword arguments from Monty.
                    _tool: Captured original hosted tool.
                    _policy: Captured speculative policy.

                Returns:
                    Original, speculative, or claimed tool result.
                """

                if shadow:
                    return await self._shadow_call(_tool, _policy, arguments)
                return await self._real_call(_tool, _policy, arguments)

            wrapped.append(
                HostedTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                    ainvoke=invoke,
                    python_name=tool.python_name,
                    sequential=tool.sequential,
                    metadata=tool.metadata,
                    kind=tool.kind,
                )
            )
        return HostedToolSet(tuple(wrapped), self.tools.config)

    async def _plan_current_code(self) -> None:
        """Resolve safe calls from all current Python fences."""

        plans: list[_Plan] = []
        for block in _code_blocks(self._buffer):
            source = _parseable_prefix(block.source)
            if source:
                plans.extend(
                    _Planner(
                        self._tool_by_alias,
                        self._policies,
                        self.known_values,
                        self.config.max_loop_unroll,
                    ).plan(source)
                )
        counts: Counter[SpeculationKey] = Counter()
        first_by_key: dict[SpeculationKey, _Plan] = {}
        for plan in plans:
            key = _speculation_key(plan.tool, plan.policy, plan.arguments)
            counts[key] += 1
            first_by_key.setdefault(key, plan)
        for key, needed in counts.items():
            current = self._planned_counts[key]
            target = 1 if first_by_key[key].policy.deterministic else needed
            while current < target:
                call = await self._dispatch(first_by_key[key], source="planner")
                if call is None:
                    break
                current += 1
            self._planned_counts[key] = current

    def _queue_shadow_statements(self, *, final: bool = False) -> None:
        """Queue newly closed top-level statements for Monty shadow execution.

        Args:
            final: Treat all current code fences as closed.
        """

        for block_index, block in enumerate(_code_blocks(self._buffer)):
            source = _parseable_prefix(block.source)
            if not source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            statements = list(tree.body)
            closed = block.closed or final
            if statements and not closed and _is_compound(statements[-1]):
                statements.pop()
            start = self._shadow_counts[block_index]
            for statement in statements[start:]:
                statement_source = ast.get_source_segment(source, statement)
                if statement_source:
                    self._shadow_queue.put_nowait(statement_source)
            self._shadow_counts[block_index] = len(statements)

    async def _run_shadow(self) -> None:
        """Execute queued statements only inside the forked Monty session."""

        while True:
            source = await self._shadow_queue.get()
            if source is None:
                return
            result = await self.kernel.execute(
                self.shadow_session_id,
                source,
                self.config.shadow_output_chars,
                state_policy="always",
                hosted_tools=self._shadow_tools,
                filesystem=False,
            )
            self._usage.shadow_executions += 1
            if not result["ok"]:
                self._usage.shadow_errors += 1
                emit_event(
                    self.event_sink,
                    "speculation",
                    "shadow_statement_failed",
                    turn_id=self.turn_id,
                    error_code=(result["error"] or {}).get("code"),
                )

    async def _shadow_call(
        self,
        tool: HostedTool,
        policy: SpeculativeToolPolicy,
        arguments: Mapping[str, Any],
    ) -> Any:
        """Adopt an early call or dispatch one for the Monty shadow.

        Args:
            tool: Original hosted tool.
            policy: Active speculation policy.
            arguments: Concrete Monty call arguments.

        Returns:
            Concrete tool result for the shadow state.
        """

        args = dict(arguments)
        key = _speculation_key(tool, policy, args)
        async with self._lock:
            call = self._adopt_locked(key, policy)
        if call is None:
            call = await self._dispatch(_Plan(tool, policy, args), source="shadow")
            if call is None:
                raise RuntimeError("speculative call budget exceeded in shadow")
            call.adopted = True
        return await call.task

    async def _real_call(
        self,
        tool: HostedTool,
        policy: SpeculativeToolPolicy,
        arguments: Mapping[str, Any],
    ) -> Any:
        """Claim matching speculative work or invoke the normal tool path.

        Args:
            tool: Original hosted tool.
            policy: Active speculation policy.
            arguments: Concrete real-call arguments.

        Returns:
            Claimed or normally invoked tool result.
        """

        args = dict(arguments)
        if policy.gate is not None and not policy.gate(args):
            self._usage.misses += 1
            return await _invoke_tool(tool, args)
        key = _speculation_key(tool, policy, args)
        async with self._lock:
            call = self._claim_locked(key, policy)
            if call is None:
                self._usage.misses += 1
        if call is None:
            emit_event(
                self.event_sink,
                "speculation",
                "claim_miss",
                turn_id=self.turn_id,
                tool=tool.alias,
            )
            return await _invoke_tool(tool, args)
        ready = call.task.done()
        self._usage.hits += 1
        if ready:
            self._usage.already_ready += 1
        started = time.monotonic()
        emit_event(
            self.event_sink,
            "speculation",
            "claim_hit",
            turn_id=self.turn_id,
            tool=tool.alias,
            already_ready=ready,
        )
        try:
            return await call.task
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a failed bet must not change real execution.
            self._usage.misses += 1
            return await _invoke_tool(tool, args)
        finally:
            self._usage.waited_seconds += time.monotonic() - started

    async def _dispatch(
        self,
        plan: _Plan,
        *,
        source: Literal["planner", "shadow"],
    ) -> _Speculation | None:
        """Start one eligible tool call under turn limits.

        Args:
            plan: Resolved tool and arguments.
            source: Planner or shadow dispatch source.

        Returns:
            New or reused speculation, or ``None`` when denied.
        """

        if plan.policy.gate is not None and not plan.policy.gate(plan.arguments):
            return None
        key = _speculation_key(plan.tool, plan.policy, plan.arguments)
        argument_chars = len(_canonical_json(plan.arguments))
        if argument_chars > self.config.max_argument_chars:
            return None
        async with self._lock:
            if plan.policy.deterministic:
                existing = self._existing_locked(key)
                if existing is not None:
                    return existing
            if len(self._all) >= self.config.max_calls_per_turn:
                return None
            self._sequence += 1
            task = asyncio.create_task(
                self._run_tool(plan.tool, plan.arguments),
                name=f"arenero-spec-{self.turn_id[:6]}-{self._sequence}",
            )
            call = _Speculation(
                key,
                self._sequence,
                plan.tool,
                plan.policy,
                dict(plan.arguments),
                source,
                task,
            )
            self._calls[key].append(call)
            self._all.append(call)
            self._usage.dispatched += 1
        emit_event(
            self.event_sink,
            "speculation",
            "dispatched",
            turn_id=self.turn_id,
            sequence=call.sequence,
            tool=plan.tool.alias,
            source=source,
        )
        return call

    async def _run_tool(self, tool: HostedTool, arguments: Mapping[str, Any]) -> Any:
        """Run one original tool under the inflight semaphore.

        Args:
            tool: Original hosted tool.
            arguments: Resolved keyword arguments.

        Returns:
            Original tool result.
        """

        try:
            async with self._slots:
                return await _invoke_tool(tool, dict(arguments))
        except asyncio.CancelledError:
            emit_event(
                self.event_sink,
                "speculation",
                "call_cancelled",
                turn_id=self.turn_id,
                tool=tool.alias,
            )
            raise
        except BaseException:
            self._usage.failed += 1
            emit_event(
                self.event_sink,
                "speculation",
                "call_failed",
                turn_id=self.turn_id,
                tool=tool.alias,
            )
            raise

    def _existing_locked(self, key: SpeculationKey) -> _Speculation | None:
        """Find reusable deterministic work while holding the store lock.

        Args:
            key: Canonical tool-call identity.

        Returns:
            Reusable call or ``None``.
        """

        for call in self._calls.get(key, ()):
            if not call.task.cancelled() and not (
                call.task.done() and call.task.exception() is not None
            ):
                return call
        return None

    def _adopt_locked(
        self, key: SpeculationKey, policy: SpeculativeToolPolicy
    ) -> _Speculation | None:
        """Adopt planner work for shadow use without consuming the real claim.

        Args:
            key: Canonical call identity.
            policy: Active policy.

        Returns:
            Matching call or ``None``.
        """

        if policy.deterministic:
            call = self._existing_locked(key)
            if call is not None:
                call.adopted = True
            return call
        for call in self._calls.get(key, ()):
            if not call.adopted and call.claims == 0:
                call.adopted = True
                return call
        return None

    def _claim_locked(
        self, key: SpeculationKey, policy: SpeculativeToolPolicy
    ) -> _Speculation | None:
        """Claim one matching call while holding the store lock.

        Args:
            key: Canonical call identity.
            policy: Active policy.

        Returns:
            Matching healthy call or ``None``.
        """

        if policy.deterministic:
            call = self._existing_locked(key)
            if call is not None:
                call.claims += 1
            return call
        for call in self._calls.get(key, ()):
            if call.claims == 0 and not call.task.cancelled():
                if call.task.done() and call.task.exception() is not None:
                    continue
                call.claims = 1
                return call
        return None

    async def _discard_unclaimed(self) -> None:
        """Cancel pending unused work and count completed waste."""

        pending: list[asyncio.Task[Any]] = []
        for call in self._all:
            if call.claims:
                continue
            if call.task.cancelled():
                self._usage.cancelled += 1
            elif call.task.done():
                self._usage.wasted += 1
                call.task.exception()
            elif self.config.cancel_unclaimed:
                call.task.cancel()
                pending.append(call.task)
                self._usage.cancelled += 1
            else:
                pending.append(call.task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


class _Planner:
    """Resolve pure argument expressions without executing generated code."""

    def __init__(
        self,
        tools: Mapping[str, HostedTool],
        policies: Mapping[str, SpeculativeToolPolicy],
        values: Mapping[str, Any],
        max_loop_unroll: int,
    ) -> None:
        """Create a bounded planner over known plain values."""

        self.tools = tools
        self.policies = policies
        self.values = dict(values)
        self.max_loop_unroll = max_loop_unroll
        self.plans: list[_Plan] = []

    def plan(self, source: str) -> list[_Plan]:
        """Plan calls from one parseable Python source prefix.

        Args:
            source: Generated Python source that parses without repair.

        Returns:
            Resolved calls in program order.
        """

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        for statement in tree.body:
            self._statement(statement, self.values)
        return self.plans

    def _statement(self, statement: ast.stmt, values: dict[str, Any]) -> None:
        """Plan one statement and update safe local values.

        Args:
            statement: Parsed statement.
            values: Current safe value environment.
        """

        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is None:
                return
            self._expression(value, values)
            if not _contains_speculative_call(value, self.policies):
                try:
                    resolved = _safe_eval(value, values)
                except _Unresolved:
                    return
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    try:
                        _bind_value(target, resolved, values)
                    except _Unresolved:
                        continue
            return
        if isinstance(statement, ast.Expr):
            self._expression(statement.value, values)
            return
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            try:
                items = islice(_safe_eval(statement.iter, values), self.max_loop_unroll)
            except (_Unresolved, TypeError):
                return
            for item in items:
                nested = dict(values)
                try:
                    _bind_value(statement.target, item, nested)
                except _Unresolved:
                    return
                for child in statement.body:
                    self._statement(child, nested)
            return
        if isinstance(statement, ast.If):
            try:
                selected = (
                    statement.body
                    if _safe_eval(statement.test, values)
                    else statement.orelse
                )
            except _Unresolved:
                return
            nested = dict(values)
            for child in selected:
                self._statement(child, nested)

    def _expression(self, expression: ast.expr, values: dict[str, Any]) -> None:
        """Find eligible calls inside one expression.

        Args:
            expression: Parsed expression.
            values: Current safe value environment.
        """

        if isinstance(expression, ast.Await):
            self._expression(expression.value, values)
            return
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            tool = self.tools.get(expression.func.id)
            policy = self.policies.get(expression.func.id)
            if tool is not None and policy is not None:
                arguments = _resolve_arguments(expression, tool, values)
                if arguments is not None:
                    self.plans.append(_Plan(tool, policy, arguments))
                return
        if isinstance(expression, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            self._comprehension(expression.elt, expression.generators, values)
            return
        if isinstance(expression, ast.DictComp):
            self._comprehension(expression.key, expression.generators, values)
            self._comprehension(expression.value, expression.generators, values)
            return
        for child in ast.iter_child_nodes(expression):
            if isinstance(child, ast.expr):
                self._expression(child, values)

    def _comprehension(
        self,
        element: ast.expr,
        generators: list[ast.comprehension],
        values: dict[str, Any],
    ) -> None:
        """Unroll one simple bounded comprehension.

        Args:
            element: Result expression to inspect.
            generators: Comprehension generators.
            values: Current safe value environment.
        """

        if len(generators) != 1 or generators[0].is_async:
            return
        generator = generators[0]
        try:
            items = islice(_safe_eval(generator.iter, values), self.max_loop_unroll)
        except (_Unresolved, TypeError):
            return
        for item in items:
            nested = dict(values)
            try:
                _bind_value(generator.target, item, nested)
                if not all(
                    _safe_eval(condition, nested) for condition in generator.ifs
                ):
                    continue
            except _Unresolved:
                continue
            self._expression(element, nested)


class _Unresolved(Exception):
    """Signal that the safe planner cannot resolve an expression."""


_PURE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "range": range,
    "repr": repr,
    "round": round,
    "str": str,
}
_PURE_STRING_METHODS = {
    "capitalize",
    "endswith",
    "lower",
    "lstrip",
    "rstrip",
    "startswith",
    "strip",
    "title",
    "upper",
}


def _safe_eval(node: ast.expr, values: Mapping[str, Any], depth: int = 0) -> Any:
    """Evaluate a small side-effect-free expression subset.

    Args:
        node: Expression to evaluate.
        values: Known plain values.
        depth: Current recursion depth.

    Returns:
        Resolved plain value.
    """

    if depth > 40:
        raise _Unresolved("expression depth")
    resolve = lambda child: _safe_eval(child, values, depth + 1)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise _Unresolved(node.id)
        return values[node.id]
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                parts.append(str(resolve(value.value)))
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                raise _Unresolved("formatted string")
        return "".join(parts)
    if isinstance(node, ast.BinOp):
        left, right = resolve(node.left), resolve(node.right)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                sequence, count = (
                    (left, right) if isinstance(right, int) else (right, left)
                )
                if (
                    isinstance(sequence, (str, bytes, list, tuple))
                    and isinstance(count, int)
                    and len(sequence) * max(count, 0) > _MAX_STATIC_SEQUENCE_SIZE
                ):
                    raise _Unresolved("large sequence multiplication")
                return left * right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise _Unresolved(type(node.op).__name__) from exc
        raise _Unresolved(type(node.op).__name__)
    if isinstance(node, ast.UnaryOp):
        value = resolve(node.operand)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise _Unresolved(type(node.op).__name__)
    if isinstance(node, ast.Subscript):
        return resolve(node.value)[resolve(node.slice)]
    if isinstance(node, ast.Slice):
        return slice(
            resolve(node.lower) if node.lower else None,
            resolve(node.upper) if node.upper else None,
            resolve(node.step) if node.step else None,
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [resolve(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(items)
        if isinstance(node, ast.Set):
            return set(items)
        return items
    if isinstance(node, ast.Dict):
        return {
            resolve(key): resolve(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.BoolOp):
        items = [resolve(value) for value in node.values]
        return all(items) if isinstance(node.op, ast.And) else any(items)
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left, right = resolve(node.left), resolve(node.comparators[0])
        operator = node.ops[0]
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.Gt):
            return left > right
        if isinstance(operator, ast.GtE):
            return left >= right
        if isinstance(operator, ast.In):
            return left in right
        if isinstance(operator, ast.NotIn):
            return left not in right
        raise _Unresolved(type(operator).__name__)
    if isinstance(node, ast.IfExp):
        return resolve(node.body) if resolve(node.test) else resolve(node.orelse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _PURE_FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise _Unresolved(node.func.id)
        try:
            return function(*(resolve(argument) for argument in node.args))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _Unresolved(node.func.id) from exc
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        value = resolve(node.func.value)
        if not isinstance(value, str) or node.func.attr not in _PURE_STRING_METHODS:
            raise _Unresolved(node.func.attr)
        if any(keyword.arg is None for keyword in node.keywords):
            raise _Unresolved("expanded keyword")
        try:
            return getattr(value, node.func.attr)(
                *(resolve(argument) for argument in node.args),
                **{
                    keyword.arg: resolve(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg
                },
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise _Unresolved(node.func.attr) from exc
    raise _Unresolved(type(node).__name__)


def _resolve_arguments(
    call: ast.Call,
    tool: HostedTool,
    values: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve positional and keyword arguments into hosted-tool keywords.

    Args:
        call: Parsed call expression.
        tool: Hosted tool that defines property order.
        values: Safe planner values.

    Returns:
        Resolved keyword arguments, or ``None`` when unsafe or invalid.
    """

    properties = list(tool.input_schema.get("properties", {}))
    if len(call.args) > len(properties):
        return None
    arguments: dict[str, Any] = {}
    try:
        for name, expression in zip(properties, call.args, strict=False):
            arguments[name] = _safe_eval(expression, values)
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in arguments:
                return None
            arguments[keyword.arg] = _safe_eval(keyword.value, values)
    except _Unresolved:
        return None
    if any(name not in properties for name in arguments):
        return None
    required = set(tool.input_schema.get("required", ()))
    if not required <= arguments.keys():
        return None
    return arguments


def _bind_value(target: ast.expr, value: Any, values: dict[str, Any]) -> None:
    """Bind a safe assignment or loop target.

    Args:
        target: Name or destructuring target.
        value: Plain value to bind.
        values: Mutable planner environment.
    """

    if isinstance(target, ast.Name):
        values[target.id] = value
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        try:
            items = list(value)
        except TypeError as exc:
            raise _Unresolved("destructure") from exc
        for child, item in zip(target.elts, items, strict=False):
            _bind_value(child, item, values)
        return
    raise _Unresolved("assignment target")


def _contains_speculative_call(
    node: ast.AST, policies: Mapping[str, SpeculativeToolPolicy]
) -> bool:
    """Return whether an AST subtree calls an eligible tool.

    Args:
        node: AST subtree.
        policies: Active alias policies.

    Returns:
        ``True`` when the subtree has an eligible direct call.
    """

    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in policies
        for child in ast.walk(node)
    )


def _code_blocks(text: str) -> list[_CodeBlock]:
    """Extract closed and open Python or REPL fences.

    Args:
        text: Current generated response text.

    Returns:
        Code blocks in response order.
    """

    blocks: list[_CodeBlock] = []
    current: list[str] | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.strip().lower()
        if current is None:
            if stripped.startswith(("```python", "```repl")):
                current = []
            continue
        if stripped == "```":
            blocks.append(_CodeBlock("".join(current), True))
            current = None
        else:
            current.append(line)
    if current is not None:
        blocks.append(_CodeBlock("".join(current), False))
    return blocks


def _parseable_prefix(source: str) -> str | None:
    """Return the longest line prefix accepted by Python's parser.

    Args:
        source: Incomplete generated source.

    Returns:
        Parseable prefix used only for AST inspection, or ``None``.
    """

    lines = source.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return None


def _is_compound(statement: ast.stmt) -> bool:
    """Return whether a final open-fence statement can still gain a body.

    Args:
        statement: Top-level parsed statement.

    Returns:
        ``True`` for compound statements.
    """

    return isinstance(
        statement,
        (
            ast.AsyncFunctionDef,
            ast.AsyncFor,
            ast.AsyncWith,
            ast.ClassDef,
            ast.For,
            ast.FunctionDef,
            ast.If,
            ast.Match,
            ast.Try,
            ast.While,
            ast.With,
        ),
    )


async def _invoke_tool(tool: HostedTool, arguments: dict[str, Any]) -> Any:
    """Invoke one original hosted tool on its supported path.

    Args:
        tool: Original hosted tool.
        arguments: Concrete keyword arguments.

    Returns:
        Raw tool result.
    """

    if tool.ainvoke is not None:
        return await tool.ainvoke(arguments)
    assert tool.invoke is not None
    return await asyncio.to_thread(tool.invoke, arguments)


def _speculation_key(
    tool: HostedTool,
    policy: SpeculativeToolPolicy,
    arguments: Mapping[str, Any],
) -> SpeculationKey:
    """Build stable identity for one concrete call.

    Args:
        tool: Hosted tool alias namespace.
        policy: Policy with an optional custom key.
        arguments: Concrete keyword arguments.

    Returns:
        Tool alias and short canonical hash.
    """

    material = policy.key(arguments) if policy.key is not None else arguments
    digest = hashlib.sha256(_canonical_json(material).encode()).hexdigest()
    return tool.alias, digest


def _canonical_json(value: Any) -> str:
    """Serialize common boundary values for stable call identity.

    Args:
        value: Tool arguments or custom key material.

    Returns:
        Deterministic JSON text.
    """

    def fallback(item: Any) -> Any:
        """Convert bytes and uncommon values to deterministic JSON forms.

        Args:
            item: Value that standard JSON cannot serialize.

        Returns:
            Stable JSON-compatible identity data.
        """

        if isinstance(item, bytes):
            return {"__bytes__": base64.b64encode(item).decode()}
        return {"__type__": type(item).__name__, "repr": repr(item)}

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=fallback,
    )
