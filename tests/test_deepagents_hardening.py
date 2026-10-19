"""Adapter regression tests with fake models, real graphs, and local files."""

import asyncio
import json
import unittest
from typing import Annotated
from unittest.mock import patch

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic_monty import OSAccess

from arenero import (
    RLM,
    AreneroSandbox,
    AsyncPersistentMontyKernel,
    HostedToolConfig,
    HostedToolSet,
    KernelConfig,
    PersistentMontyKernel,
)
from arenero._streams import BoundedStreams
from arenero.deepagents import CodeInterpreterMiddleware, _tool_value
from arenero.deepagents_backend import AreneroBackend
from arenero.mcp import McpHub, McpServerConfig
from tests import test_mcp as mcp_tests
from tests.test_deepagents_compat import _eval, _model, _runtime
from tests.test_langchain_agent import named_tool_call


class AdapterHardeningTests(unittest.IsolatedAsyncioTestCase):
    """Test async affinity, configuration, persistence, and shared authority."""

    async def test_loop_bound_tool_in_a_model_driven_graph(self) -> None:
        """Run a loop-bound tool twice across human turns, without a provider."""
        loop = asyncio.get_running_loop()
        calls = []

        @tool
        async def loop_value() -> int:
            """Await a value on the application loop.

            Returns:
                Fixed value 21.
            """
            self.assertIs(asyncio.get_running_loop(), loop)
            pending = loop.create_future()
            loop.call_soon(pending.set_result, 21)
            calls.append(await pending)
            return calls[-1]

        async with CodeInterpreterMiddleware(ptc=("loop_value",)) as middleware:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "one",
                        "eval",
                        {"code": "value = await tools.loop_value()\nvalue"},
                    ),
                    AIMessage(content="21"),
                    named_tool_call(
                        "two", "eval", {"code": "value + await loop_value()"}
                    ),
                    AIMessage(content="42"),
                ),
                tools=[loop_value],
                middleware=[middleware],
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "loop-bound"}}
            for prompt in ("Remember it", "Add it again"):
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]}, config
                )
            messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
            self.assertEqual([m.artifact["result"] for m in messages], [21, 42])
            self.assertEqual(calls, [21, 21])

    async def test_cancellation_reaches_the_application_loop(self) -> None:
        """Cancel a graph tool without saving partial REPL state."""
        started, cancelled = asyncio.Event(), asyncio.Event()

        @tool
        async def pause() -> str:
            """Wait until cancelled.

            Returns:
                No normal result.
            """
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
            return "unreachable"

        async with CodeInterpreterMiddleware(ptc=("pause",)) as middleware:
            state = await middleware.abefore_agent({}, None)
            await _eval(middleware, state, "value = 42")
            work = asyncio.create_task(
                _eval(middleware, state, "value=999\nawait pause()", [pause])
            )
            await asyncio.wait_for(started.wait(), 2)
            work.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await work
            await asyncio.wait_for(cancelled.wait(), 2)
            result = await _eval(middleware, state, "value")
            self.assertEqual(result.artifact["result"], 42)

    async def test_message_result_does_not_leak_tool_envelopes(self) -> None:
        """Return only message content from an InjectedToolCallId tool."""

        @tool
        async def wrapped(
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> ToolMessage:
            """Return a framework message.

            Args:
                tool_call_id: Injected identity.

            Returns:
                Message with value-only text.
            """
            return ToolMessage(content="value-only", tool_call_id=tool_call_id)

        async with CodeInterpreterMiddleware(ptc=("wrapped",)) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(middleware, state, "await tools.wrapped()", [wrapped])
            self.assertEqual(result.artifact["result"], "value-only")
        message = ToolMessage(content="last", tool_call_id="id")
        command = Command(update={"messages": [message], "unrelated": "ignored"})
        for value in (message, command, [message, command]):
            self.assertEqual(_tool_value(value), "last")
        for value in ([1, 2], {"content": "plain"}, True, 42):
            self.assertIs(_tool_value(value), value)

    async def test_large_console_preserves_state_and_counts(self) -> None:
        """Discard excess printed output without discarding saved variables."""
        async with CodeInterpreterMiddleware(max_result_chars=40) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(
                middleware, state, "value=42\nprint('x' * 1100000)\nvalue"
            )
            self.assertTrue(result.artifact["ok"], result.content)
            self.assertTrue(result.artifact["truncated"]["stdout"])
            self.assertEqual(
                result.artifact["stats"]["original_output_chars"]["stdout"], 1100001
            )
            self.assertEqual(
                (await _eval(middleware, state, "value")).artifact["result"], 42
            )
        streams = BoundedStreams(5)
        streams("stdout", "ééé")
        streams("stdout", "end")
        streams("stderr", "abcdef")
        self.assertEqual(streams.output, [("stdout", "éé"), ("stderr", "abcde")])
        self.assertEqual(streams.original_chars, {"stdout": 6, "stderr": 6})

    async def test_version_mismatch_is_rejected_before_load(self) -> None:
        """Keep known version drift outside the native dump loader."""
        async with CodeInterpreterMiddleware(snapshot_signing_key="key") as middleware:
            state = await middleware.abefore_agent({}, None)
            await _eval(middleware, state, "value=42")
            self.assertEqual(
                state["_arenero_monty_version"], middleware.kernel.monty_version
            )
            state["_arenero_monty_version"] = "0.0.0"
            with patch.object(
                middleware.kernel.store,
                "save",
                side_effect=AssertionError("must not load"),
            ):
                result = await _eval(middleware, state, "value=999")
            self.assertEqual(result.artifact["error"]["code"], "session_version_error")
            self.assertIn("Recovery: restart_session", result.content)
            state["_arenero_monty_version"] = middleware.kernel.monty_version
            self.assertEqual(
                (await _eval(middleware, state, "value")).artifact["result"], 42
            )

    async def test_legacy_signature_and_new_metadata_tamper(self) -> None:
        """Accept old tags but do not accept removed metadata from a new tag."""
        async with CodeInterpreterMiddleware(snapshot_signing_key="key") as middleware:
            state = await middleware.abefore_agent({}, None)
            await _eval(middleware, state, "value=42")
            state.pop("_arenero_monty_version")
            result = await _eval(middleware, state, "value")
            self.assertEqual(result.status, "error")
            await _eval(middleware, state, "value=43")
            state.pop("_arenero_monty_version")
            state["_arenero_snapshot_hmac"] = middleware._sign(
                state["_arenero_snapshot"],
                middleware._thread_session_id(state["_arenero_slot_id"]),
            )
            self.assertEqual(
                (await _eval(middleware, state, "value")).artifact["result"], 43
            )

    async def test_configuration_bootstrap_and_configured_hosted_tool(self) -> None:
        """Reuse existing configuration with documented scalar precedence."""
        loop = asyncio.get_running_loop()
        events = []

        async def increment(value: int) -> int:
            """Add one on the application loop.

            Args:
                value: Original integer.

            Returns:
                Next integer.
            """
            self.assertIs(asyncio.get_running_loop(), loop)
            return value + 1

        async with CodeInterpreterMiddleware(
            kernel_config=KernelConfig(
                max_memory_bytes=16 * 1024 * 1024, max_feed_duration_seconds=2
            ),
            hosted_tools=HostedToolSet.from_callables([increment]),
            hosted_tool_config=HostedToolConfig(max_total_call_seconds=1),
            bootstrap_code="initial=40",
            event_sink=events.append,
        ) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(
                middleware, state, "await tools.increment(value=initial+1)"
            )
            self.assertEqual(result.artifact["result"], 42, result.content)
            self.assertEqual(
                middleware.kernel.config.max_memory_bytes, 16 * 1024 * 1024
            )
            self.assertEqual(middleware.kernel.config.max_feed_duration_seconds, 2)
            self.assertIn("increment", middleware.instructions())
            self.assertTrue(events)

    async def test_mcp_and_rlm_wiring(self) -> None:
        """Use existing local MCP and deterministic RLM functions in eval."""

        async def model(prompt: str, name: str | None) -> str:
            """Return a fake model response.

            Args:
                prompt: Plain query.
                name: Unused model selector.

            Returns:
                Uppercase query text.
            """
            return prompt.upper()

        with McpHub(
            [McpServerConfig.in_process("demo", mcp_tests.McpHubTests().make_server())]
        ) as hub:
            async with CodeInterpreterMiddleware(
                mcp_hub=hub, rlm=RLM(model)
            ) as middleware:
                state = await middleware.abefore_agent({}, None)
                result = await _eval(
                    middleware,
                    state,
                    "response=await mcp_call(name='demo__add', arguments={'a':20,'b':22})\n[response['ok'], await llm_query(prompt='done')]",
                )
                self.assertEqual(
                    result.artifact["result"], [True, "DONE"], result.content
                )
            self.assertTrue(hub.servers()["servers"][0]["connected"])

    async def test_shared_module_bootstrap_and_output_use_backend(self) -> None:
        """Load helper imports and write offloaded JSON with one file worker."""
        with PersistentMontyKernel(
            os_handler=OSAccess(), config=KernelConfig(max_worker_processes=1)
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            kernel.upload_file("/helper.py", b"def add(a, b):\n    return a+b\n")
            kernel.upload_file(
                "/main.py", b"from helper import add\nvalue=add(initial,2)\n"
            )
            kernel.upload_file(
                "/startup.py",
                b"initial=40\nawait fs_write(file_path='/boot.txt', content='ready')\n",
            )
            async with CodeInterpreterMiddleware(
                backend=backend,
                module_tool=True,
                bootstrap_files=("/startup.py",),
                max_result_chars=40,
            ) as middleware:
                state = await middleware.abefore_agent({}, None)
                operation = middleware.tools[1].coroutine
                command = await asyncio.wait_for(
                    operation(
                        runtime=_runtime(state, []),
                        entry_path="/main.py",
                        code="[value, 'x'*100]",
                        output_file="/output.json",
                    ),
                    4,
                )
                state.update(command.update)
                message = command.update["messages"][0]
                self.assertTrue(message.artifact["ok"], message.content)
                self.assertEqual(message.artifact["output_file"], "/output.json")
                saved = json.loads(kernel.download_file("/output.json"))
                self.assertEqual(saved["result"], [42, "x" * 100])
                self.assertEqual(kernel.download_file("/boot.txt"), b"ready")
                self.assertEqual(
                    (await _eval(middleware, state, "value")).artifact["result"], 42
                )
                denied = await _eval(middleware, state, "open('/helper.py').read()")
                self.assertFalse(denied.artifact["ok"])
                kernel.upload_file(
                    "/namespace.py", b"has_namespace = tools is not None\n"
                )
                command = await operation(
                    runtime=_runtime(state, []), entry_path="/namespace.py"
                )
                self.assertTrue(command.update["messages"][0].artifact["ok"])
                state.update(command.update)
                self.assertTrue(
                    (await _eval(middleware, state, "has_namespace")).artifact["result"]
                )

    async def test_error_guidance_and_aggregate_host_deadline(self) -> None:
        """Make PTC count and wall budget failures clear to the model."""

        @tool
        async def slow() -> int:
            """Wait longer than the test host budget.

            Returns:
                Fixed one on normal completion.
            """
            await asyncio.sleep(1)
            return 1

        async with CodeInterpreterMiddleware(
            ptc=("slow",),
            hosted_tool_config=HostedToolConfig(max_total_call_seconds=0.05),
        ) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(middleware, state, "await slow()", [slow])
            self.assertEqual(result.artifact["error"]["code"], "hosted_tool_time_limit")
            self.assertIn("Hint:", result.content)
            self.assertIn("Retryable:", result.content)
        async with CodeInterpreterMiddleware(
            ptc=("slow",), max_ptc_calls=1
        ) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(
                middleware, state, "await slow()\nawait slow()", [slow]
            )
            self.assertEqual(result.artifact["error"]["code"], "ptc_call_limit")
        async with CodeInterpreterMiddleware(timeout=0.01) as middleware:
            state = await middleware.abefore_agent({}, None)
            result = await _eval(middleware, state, "while True:\n    pass")
            self.assertEqual(result.artifact["error"]["code"], "feed_time_limit")
            self.assertIn("host wait budgets are separate", result.content)

    async def test_idle_locks_are_removed_without_breaking_waiters(self) -> None:
        """Keep contending calls serialized, then remove idle bookkeeping."""
        async with AsyncPersistentMontyKernel() as kernel:
            lock = kernel._lock_for("one")
            order = []
            await lock.acquire()

            async def wait() -> None:
                """Wait on the same lock and record entry order."""
                async with kernel._lock_for("one"):
                    order.append(len(order))
                    await asyncio.sleep(0)

            tasks = [asyncio.create_task(wait()) for _ in range(3)]
            await asyncio.sleep(0)
            self.assertIs(kernel._lock_for("one"), lock)
            lock.release()
            del lock
            await asyncio.gather(*tasks)
            self.assertEqual(order, [0, 1, 2])
            self.assertEqual(len(kernel._locks), 0)
        async with CodeInterpreterMiddleware(mode="call") as middleware:
            for _ in range(12):
                state = await middleware.abefore_agent({}, None)
                await _eval(middleware, state, "42")
                await middleware.aafter_agent(state, None)
            self.assertEqual(len(middleware.kernel._locks), 0)
            self.assertEqual(middleware.kernel.store.list_sessions(), [])
        with PersistentMontyKernel() as kernel:
            kernel.execute("temporary", "42")
            kernel.reset("temporary")
            self.assertEqual(len(kernel._locks), 0)

    async def test_real_subagent_graph_with_structured_fake_response(self) -> None:
        """Send response_schema through the real task and AutoStrategy path."""
        child = _model(
            named_tool_call("structured", "subagent_response", {"answer": 42})
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        }
        async with CodeInterpreterMiddleware() as middleware:
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "parent",
                        "eval",
                        {
                            "code": f"reply=await task(description='answer', subagent_type='analyst', response_schema={schema!r})\nreply['answer']"
                        },
                    ),
                    AIMessage(content="42"),
                ),
                subagents=[
                    {
                        "name": "analyst",
                        "description": "Answer numbers",
                        "system_prompt": "Return the answer",
                        "model": child,
                    }
                ],
                middleware=[middleware],
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Ask the analyst"}]}
            )
            message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
            self.assertEqual(message.artifact["result"], 42, message.content)

    async def test_real_child_tool_uses_the_application_loop(self) -> None:
        """Run a loop-bound provider stand-in inside the real subagent graph."""
        loop = asyncio.get_running_loop()

        @tool
        async def child_value() -> int:
            """Return a local value on the application loop.

            Returns:
                Integer 42.
            """
            self.assertIs(asyncio.get_running_loop(), loop)
            future = loop.create_future()
            loop.call_soon(future.set_result, 42)
            return await future

        async with CodeInterpreterMiddleware() as middleware:
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "parent",
                        "eval",
                        {
                            "code": "await tools.task(description='answer', subagent_type='analyst')"
                        },
                    ),
                    AIMessage(content="42"),
                ),
                subagents=[
                    {
                        "name": "analyst",
                        "description": "Answer numbers",
                        "system_prompt": "Use the tool",
                        "model": _model(
                            named_tool_call("child", "child_value", {}),
                            AIMessage(content="42"),
                        ),
                        "tools": [child_value],
                    }
                ],
                middleware=[middleware],
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Ask the analyst"}]}
            )
            message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
            self.assertEqual(message.artifact["result"], "42", message.content)

    async def test_real_child_graph_interrupt_resumes(self) -> None:
        """Pass approval through a real child graph without committing the feed."""

        @tool
        async def approve() -> str:
            """Ask the graph host for approval.

            Returns:
                Approval response.
            """
            return interrupt("approve child")

        async with CodeInterpreterMiddleware() as middleware:
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "parent",
                        "eval",
                        {
                            "code": "await task(description='answer', subagent_type='analyst')"
                        },
                    ),
                    AIMessage(content="approved"),
                ),
                subagents=[
                    {
                        "name": "analyst",
                        "description": "Ask approval",
                        "system_prompt": "Use approval",
                        "model": _model(
                            named_tool_call("child", "approve", {}),
                            AIMessage(content="approved"),
                        ),
                        "tools": [approve],
                    }
                ],
                middleware=[middleware],
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "child-approval"}}
            paused = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Ask the analyst"}]}, config
            )
            self.assertTrue(paused.get("__interrupt__"))
            resumed = await agent.ainvoke(Command(resume="yes"), config)
            message = next(m for m in resumed["messages"] if isinstance(m, ToolMessage))
            self.assertEqual(message.artifact["result"], "approved", message.content)


class AdapterConfigurationTests(unittest.TestCase):
    """Reject invalid configuration before a worker or loop starts."""

    def test_empty_key_reserved_names_and_task_selection(self) -> None:
        """Reject aliases and switches that would replace internal functions."""
        cases = [
            {"snapshot_signing_key": ""},
            {"snapshot_signing_key": b""},
            {"ptc": ("tools",)},
            {"ptc": ("_AreneroTools",)},
            {"ptc": ("__init__",)},
            {"ptc": ("_arenero_helper",)},
            {"ptc": ("fs_read",)},
            {"ptc": ("task",)},
            {"ptc": ("task",), "subagents": False},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                CodeInterpreterMiddleware(**options)
