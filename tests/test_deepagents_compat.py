"""Mocked graph and backend contract tests; no model or network provider."""

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from deepagents import create_deep_agent
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt
from pydantic_monty import OSAccess

from arenero import (
    AreneroSandbox,
    FilesystemConfig,
    KernelConfig,
    PersistentMontyKernel,
    WorkspaceConfig,
)
from arenero.deepagents import AreneroInterpreterMiddleware, CodeInterpreterMiddleware
from arenero.deepagents_backend import AreneroBackend
from tests.test_langchain_agent import ToolCallingFakeChatModel, named_tool_call


def _model(*messages: AIMessage) -> ToolCallingFakeChatModel:
    """Make a scripted tool-calling model.

    Args:
        *messages: Model responses in invocation order.

    Returns:
        Fake model with no provider credentials or requests.
    """
    return ToolCallingFakeChatModel(messages=iter(messages))


def _runtime(
    state: dict[str, Any], tools: list[Any], events: list | None = None
) -> ToolRuntime:
    """Make a runtime for direct adapter calls.

    Args:
        state: Private graph state.
        tools: Current tool inventory.
        events: Optional custom event collector.

    Returns:
        Stable parent runtime suitable for replay tests.
    """
    return ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=(events.append if events is not None else lambda value: None),
        tool_call_id="parent-eval",
        store=None,
        tools=tools,
    )


async def _eval(
    middleware: CodeInterpreterMiddleware,
    state: dict,
    code: str,
    tools: list | None = None,
) -> ToolMessage:
    """Run a tool and apply its private graph update.

    Args:
        middleware: Compatibility adapter.
        state: Mutable fake graph checkpoint.
        code: Monty Python source.
        tools: Optional live hosted tools.

    Returns:
        Model-facing ToolMessage with the original rich artifact.
    """
    operation = middleware.tools[0].coroutine
    assert operation is not None
    result = await operation(runtime=_runtime(state, tools or []), code=code)
    assert isinstance(result, Command)
    state.update(result.update)
    return result.update["messages"][0]


class CompatibilityGraphTests(unittest.IsolatedAsyncioTestCase):
    """Test complete model-driven graphs and the Python compatibility contract."""

    async def test_two_human_turns_restore_in_a_new_adapter(self) -> None:
        """Persist functions and data in the graph, not a middleware fallback."""
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "two-turns"}}
        async with CodeInterpreterMiddleware(snapshot_signing_key="test") as first:
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "one",
                        "eval",
                        {
                            "code": "value = 41\ndef add(x):\n    return value + x\nvalue"
                        },
                    ),
                    AIMessage(content="41"),
                ),
                middleware=[first],
                checkpointer=saver,
            )
            await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Store 41"}]}, config
            )
            checkpoint = agent.get_state(config).values
            self.assertIsInstance(checkpoint["_arenero_snapshot"], bytes)
            self.assertEqual(first.kernel.store.list_sessions(), [])
        async with CodeInterpreterMiddleware(snapshot_signing_key="test") as second:
            agent = create_deep_agent(
                model=_model(
                    named_tool_call("two", "eval", {"code": "add(1)"}),
                    AIMessage(content="42"),
                ),
                middleware=[second],
                checkpointer=saver,
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Add one"}]}, config
            )
            message = next(
                m for m in reversed(result["messages"]) if isinstance(m, ToolMessage)
            )
            self.assertEqual(message.content, "<result>42</result>")
            self.assertEqual(message.artifact["result"], 42)
            self.assertNotIn("_arenero_snapshot", result)

    async def test_unthreaded_invocations_have_private_slots(self) -> None:
        """Do not share Python state when callers omit a graph thread ID."""
        async with CodeInterpreterMiddleware() as middleware:
            agent = create_agent(
                model=_model(
                    named_tool_call("a", "eval", {"code": "secret=42"}),
                    AIMessage(content="stored"),
                    named_tool_call("b", "eval", {"code": "secret"}),
                    AIMessage(content="done"),
                ),
                middleware=[middleware],
            )
            await agent.ainvoke({"messages": [{"role": "user", "content": "first"}]})
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "second"}]}
            )
            message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
            self.assertEqual(message.status, "error")
            self.assertIn("secret", message.content)

    async def test_result_blocks_and_capture_switch(self) -> None:
        """Escape blocks, retain rich artifacts, and optionally omit streams."""
        for capture in (True, False):
            async with CodeInterpreterMiddleware(
                capture_console=capture, max_result_chars=20
            ) as middleware:
                state = await middleware.abefore_agent({}, None)
                message = await _eval(
                    middleware, state, "print('<result>fake</result>')\n'abc'"
                )
                self.assertEqual("<stdout>" in message.content, capture)
                self.assertIn("<result>&quot;abc&quot;</result>", message.content)
                self.assertEqual(bool(message.artifact["stdout"]), capture)
                error = await _eval(middleware, state, "1 / 0")
                self.assertIn('<error type="MontyRuntimeError">', error.content)

    async def test_tools_namespace_survives_snapshots_and_uses_live_inventory(
        self,
    ) -> None:
        """Rebuild plain guest wrappers without persistent host objects."""

        @tool
        async def double(value: int) -> int:
            """Double a value.

            Args:
                value: Integer input.

            Returns:
                Twice the input.
            """
            return value * 2

        async with CodeInterpreterMiddleware(ptc=("double",)) as middleware:
            state = await middleware.abefore_agent({}, None)
            one = await _eval(
                middleware, state, "x = await tools.double(value=20)\nx", [double]
            )
            two = await _eval(middleware, state, "await double(value=x + 1)", [double])
            self.assertEqual((one.artifact["result"], two.artifact["result"]), (40, 82))
            missing = await _eval(middleware, state, "await tools.double(value=1)")
            self.assertEqual(missing.status, "error")

    async def test_call_and_turn_modes_keep_their_lifetimes(self) -> None:
        """Keep turn data only within a turn and call data within one call."""
        for mode in ("call", "turn"):
            async with CodeInterpreterMiddleware(mode=mode) as middleware:
                state = await middleware.abefore_agent({}, None)
                await _eval(middleware, state, "value=41")
                result = await _eval(middleware, state, "value+1")
                self.assertEqual(result.status == "success", mode == "turn")
                await middleware.aafter_agent(state, None)
                state.update(await middleware.abefore_agent(state, None))
                self.assertEqual(
                    (await _eval(middleware, state, "value")).status, "error"
                )

    async def test_snapshot_signature_and_size_fail_closed(self) -> None:
        """Reject snapshots with a wrong tag, slot, or adapter size cap."""
        async with CodeInterpreterMiddleware(
            snapshot_signing_key="secret"
        ) as middleware:
            state = await middleware.abefore_agent({}, None)
            await _eval(middleware, state, "secret=42")
            original = dict(state)
            for change in (
                {"_arenero_snapshot_hmac": b"wrong"},
                {"_arenero_slot_id": "different"},
            ):
                changed = {**original, **change}
                self.assertEqual(
                    (await _eval(middleware, changed, "secret")).status, "error"
                )
            middleware._max_snapshot_bytes = 1
            self.assertEqual(
                (await _eval(middleware, original, "secret")).status, "error"
            )

    async def test_graph_interrupt_resume_preserves_dispatch_and_last_checkpoint(
        self,
    ) -> None:
        """Stop the guest on interrupt and replay from the last completed eval."""
        ids, events, completed = [], [], []

        @tool
        async def task(
            description: str, subagent_type: str, runtime: ToolRuntime
        ) -> Command:
            """Ask for approval and return a fake child response.

            Args:
                description: Task request.
                subagent_type: Dummy agent name.
                runtime: Injected child runtime.

            Returns:
                A fake Deep Agents task command after graph resume.
            """
            ids.append(runtime.tool_call_id)
            approved = interrupt("approve task")
            completed.append(approved)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="approved", tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        async with CodeInterpreterMiddleware() as middleware:
            model = _model(
                named_tool_call("seed", "eval", {"code": "value=40"}),
                named_tool_call(
                    "pause",
                    "eval",
                    {
                        "code": "value += 1\ntry:\n    await task(description='check', subagent_type='dummy', label='Check')\nexcept BaseException:\n    value=999\nvalue"
                    },
                ),
                AIMessage(content="done"),
            )
            agent = create_agent(
                model=model,
                tools=[task],
                middleware=[middleware],
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "interrupt"}}
            async for event in agent.astream(
                {"messages": [{"role": "user", "content": "start"}]},
                config,
                stream_mode="custom",
            ):
                events.append(event)
            self.assertTrue(agent.get_state(config).interrupts)
            self.assertEqual(completed, [])
            result = await agent.ainvoke(Command(resume="yes"), config)
            message = next(
                m for m in reversed(result["messages"]) if isinstance(m, ToolMessage)
            )
            self.assertEqual(message.artifact["result"], 41)
            self.assertEqual(ids, [ids[0], ids[0]])
            self.assertEqual(completed, ["yes"])
            self.assertEqual(events[0]["phase"], "start")
            self.assertNotIn("error", [e["phase"] for e in events])

    async def test_subagent_schema_events_and_separate_budgets(self) -> None:
        """Use task schemas and events without consuming the regular PTC cap."""
        seen, events = [], []

        @tool
        async def task(
            description: str, subagent_type: str, runtime: ToolRuntime
        ) -> Command:
            """Return structured fake subagent data.

            Args:
                description: Task text.
                subagent_type: Child kind.
                runtime: Injected response strategy and child ID.

            Returns:
                Command with final JSON text.
            """
            seen.append(runtime)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content='{"answer":42}', tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        @tool
        async def ping() -> int:
            """Return a fixed value.

            Returns:
                Integer one.
            """
            return 1

        async with CodeInterpreterMiddleware(
            ptc=("ping",), max_ptc_calls=1
        ) as middleware:
            state = await middleware.abefore_agent({}, None)
            runtime = _runtime(state, [task, ping], events)
            operation = middleware.tools[0].coroutine
            result = await operation(
                runtime=runtime,
                code="await ping()\na = await task(description='count', subagent_type='fake', response_schema={'type':'object','properties':{'answer':{'type':'integer'}}}, label='Count')\na['answer']",
            )
            message = result.update["messages"][0]
            self.assertEqual(message.artifact["result"], 42)
            self.assertIn(
                "__deepagents_subagent_response_format", seen[0].config["configurable"]
            )
            self.assertEqual([e["phase"] for e in events], ["start", "complete"])
            self.assertEqual(events[0]["label"], "Count")
            self.assertEqual(events[0]["id"], events[1]["id"])
            result = await _eval(
                middleware, state, "await ping()\nawait ping()", [ping]
            )
            self.assertEqual(result.status, "error")
            self.assertIn("regular PTC call limit", result.content)

    async def test_shared_backend_works_from_a_complete_agent_graph(self) -> None:
        """Use one workspace through outer file tools and nested Python calls."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            config=KernelConfig(min_worker_processes=1, max_worker_processes=1),
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            async with CodeInterpreterMiddleware(
                backend=backend, ptc=("read_file",)
            ) as middleware:
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            "write",
                            "write_file",
                            {"file_path": "/shared.txt", "content": "41"},
                        ),
                        named_tool_call(
                            "read",
                            "eval",
                            {
                                "code": "data = await fs_read(file_path='/shared.txt')\nvalue = int(data['file_data']['content']) + 1\nawait fs_write(file_path='/answer.txt', content=str(value))\nawait tools.read_file(file_path='/answer.txt')"
                            },
                        ),
                        AIMessage(content="42"),
                    ),
                    backend=backend,
                    middleware=[middleware],
                )
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": [{"role": "user", "content": "work"}]}),
                    10,
                )
                message = next(
                    m
                    for m in reversed(result["messages"])
                    if isinstance(m, ToolMessage)
                )
                self.assertEqual(message.status, "success", message.content)
                self.assertIn("42", message.content)
                self.assertEqual(
                    backend.download_files(["/answer.txt"])[0].content, b"42"
                )
                self.assertFalse(middleware.kernel._filesystem_enabled)

    async def test_optional_module_tool_runs_with_direct_workspace(self) -> None:
        """Keep the static module workaround optional and separate from backend files."""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "helper.py").write_text("def answer():\n    return 42\n")
            async with CodeInterpreterMiddleware(
                module_tool=True, workspace=WorkspaceConfig(directory)
            ) as middleware:
                self.assertEqual(
                    [t.name for t in middleware.tools], ["eval", "eval_module"]
                )
                state = await middleware.abefore_agent({}, None)
                result = await middleware.tools[1].coroutine(
                    runtime=_runtime(state, []),
                    entry_path="/workspace/helper.py",
                    code="answer()",
                )
                message = result.update["messages"][0]
                self.assertEqual(message.artifact["result"], 42, message.content)

    async def test_subagent_concurrency_call_cap_and_error_events(self) -> None:
        """Bound tasks separately and report child failures on the custom stream."""
        active, peak = 0, 0
        events = []

        @tool
        async def task(description: str, subagent_type: str) -> str:
            """Return a delayed fake task result.

            Args:
                description: Use fail to request a child error.
                subagent_type: Unused child kind.

            Returns:
                Task text on success.
            """
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.005)
                if description == "fail":
                    raise ValueError("child failed")
                return description
            finally:
                active -= 1

        async with CodeInterpreterMiddleware(
            max_subagent_concurrency=1, max_subagent_calls=3
        ) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(
                middleware,
                state,
                "import asyncio\nawait asyncio.gather(task(description='one', subagent_type='fake'), task(description='two', subagent_type='fake'), task(description='three', subagent_type='fake'))",
                [task],
            )
            self.assertEqual(result.artifact["result"], ["one", "two", "three"])
            self.assertEqual(peak, 1)
            operation = middleware.tools[0].coroutine
            failed = await operation(
                runtime=_runtime(state, [task], events),
                code="await task(description='fail', subagent_type='fake')",
            )
            self.assertEqual(failed.update["messages"][0].status, "error")
            self.assertEqual([event["phase"] for event in events], ["start", "error"])
            middleware._subagent_calls = 1
            capped = await _eval(
                middleware,
                state,
                "await task(description='one', subagent_type='fake')\nawait task(description='two', subagent_type='fake')",
                [task],
            )
            self.assertIn("subagent call limit", capped.content)
            self.assertEqual(capped.artifact["error"]["code"], "subagent_call_limit")

    async def test_schema_bounds_reject_before_dispatch(self) -> None:
        """Do not send oversized, deep, or wide schemas to a child model."""
        calls = []

        @tool
        async def task(description: str, subagent_type: str) -> str:
            """Record a child dispatch.

            Args:
                description: Task request.
                subagent_type: Child kind.

            Returns:
                Empty JSON object text.
            """
            calls.append(description)
            return "{}"

        async with CodeInterpreterMiddleware() as middleware:
            state = await middleware.abefore_agent({}, None)
            for schema in (
                "{'description': 'x' * 5000}",
                "{'properties': {str(i): {'type': 'string'} for i in range(33)}}",
                "{'items': {'items': {'items': {'items': {'items': {'items': {}}}}}}}",
            ):
                result = await _eval(
                    middleware,
                    state,
                    f"await task(description='no', subagent_type='fake', response_schema={schema})",
                    [task],
                )
                self.assertEqual(result.status, "error")
                self.assertIn("response_schema", result.content)
            self.assertEqual(calls, [])

    async def test_changed_task_payload_has_a_new_dispatch_identity(self) -> None:
        """Do not reuse an ID when replay code changes a task request."""
        ids = []

        @tool
        async def task(
            description: str, subagent_type: str, runtime: ToolRuntime
        ) -> str:
            """Return one fake response while recording the child ID.

            Args:
                description: Task request.
                subagent_type: Child kind.
                runtime: Injected dispatch runtime.

            Returns:
                Fixed text.
            """
            ids.append(runtime.tool_call_id)
            return "done"

        async with CodeInterpreterMiddleware() as middleware:
            state = await middleware.abefore_agent({}, None)
            for text in ("one", "one", "two"):
                await _eval(
                    middleware,
                    state,
                    f"await task(description='{text}', subagent_type='fake')",
                    [task],
                )
            self.assertEqual(ids[0], ids[1])
            self.assertNotEqual(ids[0], ids[2])

    async def test_external_cancellation_keeps_the_previous_snapshot(self) -> None:
        """Cancel a waiting tool and preserve the last completed graph state."""
        started = threading.Event()

        @tool
        async def wait_forever() -> str:
            """Wait for cancellation without a real provider.

            Returns:
                No normal result.
            """
            started.set()
            await asyncio.Future()
            return "unreachable"

        async with CodeInterpreterMiddleware(ptc=("wait_forever",)) as middleware:
            state = await middleware.abefore_agent({}, None)
            await _eval(middleware, state, "value=40")
            work = asyncio.create_task(
                _eval(
                    middleware, state, "value=999\nawait wait_forever()", [wait_forever]
                )
            )
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            work.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await work
            result = await _eval(middleware, state, "value")
            self.assertEqual(result.artifact["result"], 40)

    async def test_legacy_middleware_propagates_graph_interrupt(self) -> None:
        """Fix graph control flow without changing the existing tool interface."""

        @tool
        async def pause() -> None:
            """Raise a graph signal.

            Returns:
                No normal result.
            """
            raise GraphInterrupt(())

        async with AreneroInterpreterMiddleware(ptc=("pause",)) as middleware:
            operation = middleware.tools[0].coroutine
            with self.assertRaises(GraphInterrupt):
                await operation(
                    runtime=_runtime({}, [pause]),
                    code="try:\n    await pause()\nexcept BaseException:\n    123",
                )

    async def test_parallel_eval_calls_are_rejected_before_execution(self) -> None:
        """Reject a conflicting graph step without changing its Python state."""
        async with CodeInterpreterMiddleware() as middleware:
            state = await middleware.abefore_agent({}, None)
            state["messages"] = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "eval", "args": {"code": "value=1"}, "id": "a"},
                        {"name": "eval", "args": {"code": "value=2"}, "id": "b"},
                    ],
                )
            ]
            result = await _eval(middleware, state, "value=1")
            self.assertIn("ConcurrentEval", result.content)
            self.assertEqual(middleware.kernel.store.list_sessions(), [])

    async def test_namespace_can_be_disabled_and_ptc_cap_removed(self) -> None:
        """Keep bare Python calls with no adapter-level regular call cap."""

        @tool
        async def ping() -> int:
            """Return one.

            Returns:
                Integer one.
            """
            return 1

        async with CodeInterpreterMiddleware(
            ptc=("ping",), tools_namespace=False, max_ptc_calls=None
        ) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(
                middleware, state, "await ping() + await ping()", [ping]
            )
            self.assertEqual(result.artifact["result"], 2)
            self.assertEqual(
                (await _eval(middleware, state, "tools", [ping])).status, "error"
            )


class CompatibilitySyncTests(unittest.TestCase):
    """Exercise both synchronous graph invocation and streaming."""

    def test_sync_invoke_and_stream_share_graph_state(self) -> None:
        """Use one stable runtime loop across separate synchronous human turns."""
        with CodeInterpreterMiddleware() as middleware:
            agent = create_agent(
                model=_model(
                    named_tool_call("a", "eval", {"code": "value=41"}),
                    AIMessage(content="stored"),
                    named_tool_call("b", "eval", {"code": "value+1"}),
                    AIMessage(content="42"),
                ),
                middleware=[middleware],
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "sync"}}
            agent.invoke({"messages": [{"role": "user", "content": "store"}]}, config)
            chunks = list(
                agent.stream(
                    {"messages": [{"role": "user", "content": "add"}]},
                    config,
                    stream_mode="values",
                )
            )
            messages = chunks[-1]["messages"]
            result = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
            self.assertEqual(result.artifact["result"], 42)
        self.assertFalse(middleware._bridge.thread.is_alive())

    def test_existing_middleware_still_has_two_json_tools(self) -> None:
        """Do not change the old tool list or persistence default."""
        middleware = AreneroInterpreterMiddleware()
        self.assertEqual(
            [t.name for t in middleware.tools],
            ["python_interpreter", "python_interpreter_module"],
        )
        self.assertFalse(middleware._checkpoint_state)
        asyncio.run(middleware.aclose())


class BackendContractTests(unittest.IsolatedAsyncioTestCase):
    """Exercise native records, file matching, transfers, and cooperative timeouts."""

    def setUp(self) -> None:
        """Create one shared in-memory file authority."""
        self.kernel = PersistentMontyKernel(os_handler=OSAccess())
        self.addCleanup(self.kernel.close)
        self.backend = AreneroBackend(AreneroSandbox(self.kernel))

    async def test_native_protocol_and_all_async_file_operations(self) -> None:
        """Use inherited a* entry points with native result dataclasses."""
        self.assertIsInstance(self.backend, SandboxBackendProtocol)
        self.assertIsNone(
            (await self.backend.awrite("/a.txt", "one\ntwo\nthree\n")).error
        )
        page = await self.backend.aread("/a.txt", 1, 1)
        self.assertEqual(
            (page.start_line, page.end_line, page.next_offset, page.total_lines),
            (2, 2, 2, 3),
        )
        self.assertEqual(page.file_data["content"], "two\n")
        self.assertIn("created_at", page.file_data)
        self.assertIsNone((await self.backend.aedit("/a.txt", "two", "TWO")).error)
        self.assertEqual(len((await self.backend.als("/")).entries), 1)
        self.assertEqual(len((await self.backend.aglob("*.txt")).matches), 1)
        self.assertEqual((await self.backend.agrep("TWO")).matches[0]["line"], 2)
        self.assertIsNone((await self.backend.adelete("/a.txt")).error)
        self.assertIsNotNone((await self.backend.aread("/a.txt")).error)

    async def test_paging_empty_missing_and_binary_cases(self) -> None:
        """Do not inspect zero-line windows or emit invalid empty paging fields."""
        zero = await self.backend.aread("/missing", -5, 0)
        self.assertTrue(zero.no_lines_requested)
        self.backend.write("/empty", "")
        empty = self.backend.read("/empty")
        self.assertIsNone(empty.total_lines)
        self.backend.upload_files([("/binary", b"\xff\x00")])
        self.assertEqual(self.backend.read("/binary").file_data["encoding"], "base64")
        self.backend.write("/text", "a\nb")
        self.assertEqual(self.backend.read("/text", -5, 1).start_line, 1)
        self.assertIsNone(self.backend.read("/text", 9).next_offset)

    async def test_glob_uses_upstream_classes_braces_and_dot_rules(self) -> None:
        """Reuse the Deep Agents matcher instead of adding a second grammar."""
        for path in ("/a.py", "/b.txt", "/nested/c.py", "/.hidden/d.py", "/.env"):
            self.assertIsNone(self.backend.write(path, "ok").error)
        paths = lambda result: {e["path"] for e in result.matches}
        self.assertEqual(
            paths(self.backend.glob("{a,b}.{py,txt}")), {"/a.py", "/b.txt"}
        )
        self.assertEqual(paths(self.backend.glob("[ab].*")), {"/a.py", "/b.txt"})
        self.assertIn("/.hidden/d.py", paths(self.backend.glob("*.py")))
        self.assertNotIn("/.hidden/d.py", paths(self.backend.glob("**/*.py")))
        self.assertEqual(paths(self.backend.glob("/*.py")), {"/a.py"})
        self.assertEqual(paths(self.backend.glob(".env")), {"/.env"})
        self.assertIsNotNone(self.backend.glob("../*.py").error)

    async def test_grep_caps_are_total_and_exact(self) -> None:
        """Report truncation only when matches or scan data were omitted."""
        self.backend.write("/a.txt", "a.b\na.b\n")
        exact = self.backend.grep("a.b", max_count=2)
        self.assertFalse(exact.truncated)
        self.assertEqual(len(exact.matches), 2)
        self.assertTrue(self.backend.grep("a.b", max_count=1).truncated)
        self.assertEqual(self.backend.grep("a.*", path="/a.txt").matches, [])
        self.assertTrue(self.backend.grep("a.b", max_count=0).truncated)
        self.assertEqual(
            len((await self.backend.agrep("a.b", glob="*.txt")).matches), 2
        )

    async def test_transfers_preserve_order_binary_and_partial_success(self) -> None:
        """Keep transfer bytes raw and normalize known error codes."""
        uploaded = await self.backend.aupload_files(
            [("/image.bin", b"\x00\xff"), ("relative", b"no")]
        )
        self.assertIsNone(uploaded[0].error)
        self.assertEqual(uploaded[1].error, "invalid_path")
        downloaded = await self.backend.adownload_files(["/missing", "/image.bin"])
        self.assertEqual(downloaded[0].error, "file_not_found")
        self.assertEqual(downloaded[1].content, b"\x00\xff")

    async def test_timeout_stops_later_commands_without_rollback(self) -> None:
        """Stop at a command boundary and retain completed file effects."""
        from arenero.shell import VirtualShell

        original = VirtualShell._run

        def slow(shell, arguments, stdin, cwd):
            """Delay one command to reach the cooperative deadline.

            Args:
                shell: Active shell.
                arguments: Command words.
                stdin: Pipeline input.
                cwd: Current virtual directory.

            Returns:
                Original command result after a short delay.
            """
            result = original(shell, arguments, stdin, cwd)
            if arguments[0] == "cat":
                time.sleep(0.03)
            return result

        self.backend.write("/input", "ok")
        with patch.object(VirtualShell, "_run", slow):
            result = await self.backend.aexecute(
                "echo first > /first; cat /input; echo last > /last", timeout=0.02
            )
        self.assertEqual(result.exit_code, 124)
        self.assertIsNone(self.backend.read("/first").error)
        self.assertIsNotNone(self.backend.read("/last").error)
        self.assertEqual(self.backend.execute("echo ok", timeout=0).exit_code, 0)
        self.assertEqual(self.backend.execute("echo ok", timeout=-1).exit_code, 2)

    async def test_backend_root_and_transfer_directory_errors(self) -> None:
        """Keep absolute file paths within the selected backend root."""
        self.backend.write("/work/a", "ok")
        bound = AreneroBackend(self.backend.sandbox, root="/work")
        self.assertIsNotNone(bound.read("/outside").error)
        self.assertEqual(
            bound.download_files(["/outside"])[0].error, "permission_denied"
        )
        self.assertEqual(bound.download_files(["/work"])[0].error, "is_directory")
        self.assertEqual(bound.upload_files([("", b"bad")])[0].error, "invalid_path")
        self.assertEqual(bound.glob("*").matches[0]["path"], "/work/a")

    async def test_backend_reports_scan_and_character_caps(self) -> None:
        """Keep bounded searches honest instead of returning silent partial data."""
        config = FilesystemConfig(
            default_result_limit=2, max_result_limit=2, max_grep_chars=4
        )
        with PersistentMontyKernel(
            os_handler=OSAccess(), filesystem_config=config
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            for path in ("/a", "/b", "/c"):
                backend.write(path, "matching text")
            self.assertTrue(backend.glob("*").truncated)
            self.assertIsNotNone(backend.ls("/").error)
            self.assertTrue(backend.grep("matching").truncated)
            self.assertTrue(backend.grep("matching", path="/a").truncated)
