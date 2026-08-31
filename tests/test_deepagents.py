"""Tests for optional Deep Agents interpreter middleware."""

import json
import unittest
from typing import Any

from deepagents import create_deep_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from pydantic_monty import MemoryFile, OSAccess

from arenero import RLM, AsyncPersistentMontyKernel, RLMConfig
from arenero.deepagents import AreneroInterpreterMiddleware


def _runtime(tools: list[Any]) -> ToolRuntime:
    """Build a small ToolRuntime for direct middleware tool tests.

    Args:
        tools: Live tools visible during the interpreter call.

    Returns:
        Runtime with a stable fallback session and tool-call ID.
    """

    return ToolRuntime(
        state={"messages": []},
        context=None,
        config={},
        stream_writer=lambda value: None,
        tool_call_id="outer-call",
        store=None,
        tools=tools,
    )


async def _call(
    middleware: AreneroInterpreterMiddleware,
    code: str,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Call the middleware coroutine and decode its ToolMessage.

    Args:
        middleware: Middleware under test.
        code: Monty source.
        tools: Optional live agent tools.

    Returns:
        Decoded execution result.
    """

    active = [*middleware.tools, *(tools or [])]
    coroutine = middleware.tools[0].coroutine
    assert coroutine is not None
    message = await coroutine(runtime=_runtime(active), code=code)
    assert isinstance(message, ToolMessage)
    return json.loads(message.content)


class DeepAgentsMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    """Check persistence, PTC, checkpoint, and module behavior."""

    async def test_thread_and_call_modes_have_distinct_lifetimes(self) -> None:
        """Keep thread state but remove call-scoped state after each call."""

        thread = AreneroInterpreterMiddleware(mode="thread")
        call = AreneroInterpreterMiddleware(mode="call")
        self.addAsyncCleanup(thread.aclose)
        self.addAsyncCleanup(call.aclose)

        await _call(thread, "value = 41")
        kept = await _call(thread, "value + 1")
        await _call(call, "value = 41")
        removed = await _call(call, "value + 1")

        self.assertEqual(kept["result"], 42)
        self.assertFalse(removed["ok"])

    async def test_middleware_compiles_in_a_deep_agent(self) -> None:
        """Use the installed Deep Agents middleware API to build a graph."""

        middleware = AreneroInterpreterMiddleware()
        self.addAsyncCleanup(middleware.aclose)
        agent = create_deep_agent(
            model=FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
            middleware=[middleware],
        )

        self.assertIn("model", agent.get_graph().nodes)
        self.assertEqual(
            [tool.name for tool in middleware.tools],
            ["python_interpreter", "python_interpreter_module"],
        )

    async def test_turn_mode_resets_after_the_agent_hook(self) -> None:
        """Keep state within a turn and remove it at the after-agent hook."""

        middleware = AreneroInterpreterMiddleware(mode="turn")
        self.addAsyncCleanup(middleware.aclose)

        await middleware.abefore_agent({}, None)
        await _call(middleware, "value = 41")
        kept = await _call(middleware, "value + 1")
        await middleware.aafter_agent({}, None)
        await middleware.abefore_agent({}, None)
        removed = await _call(middleware, "value + 1")

        self.assertEqual(kept["result"], 42)
        self.assertFalse(removed["ok"])

    async def test_programmatic_tool_gets_full_runtime(self) -> None:
        """Inject ToolRuntime when a hosted call bypasses ToolNode."""

        @tool
        async def runtime_probe(runtime: ToolRuntime) -> str:
            """Return the current outer tool-call ID.

            Args:
                runtime: Injected child tool runtime.

            Returns:
                Synthetic child tool-call ID.
            """

            return str(runtime.tool_call_id)

        middleware = AreneroInterpreterMiddleware(ptc=("runtime_probe",))
        self.addAsyncCleanup(middleware.aclose)

        result = await _call(
            middleware,
            "await runtime_probe()",
            [runtime_probe],
        )

        self.assertTrue(result["ok"])
        self.assertTrue(str(result["result"]).startswith("arenero_runtime_probe_"))
        self.assertEqual(result["host_calls"]["count"], 1)
        instructions = middleware.instructions([runtime_probe])
        self.assertIn("async def runtime_probe", instructions)
        self.assertIn("bypass the parent ToolNode", instructions)

    async def test_rlm_functions_are_available_inside_the_interpreter(self) -> None:
        """Expose a bounded plain model call through Deep Agents middleware."""

        async def model(prompt: str, model_name: str | None) -> str:
            """Return one deterministic nested response.

            Args:
                prompt: Nested model prompt.
                model_name: Unused optional model selector.

            Returns:
                Echoed nested response.
            """

            return f"nested:{prompt}"

        middleware = AreneroInterpreterMiddleware(
            rlm=RLM(model, config=RLMConfig(timeout_seconds=5))
        )
        self.addAsyncCleanup(middleware.aclose)

        result = await _call(
            middleware,
            "value = await llm_query(prompt='hello')\n"
            "usage = await rlm_usage()\n"
            "[value, usage['model_calls']]",
        )

        self.assertEqual(result["result"], ["nested:hello", 1])
        self.assertIn("async def rlm_query", middleware.instructions())

    async def test_task_bridge_returns_subagent_text(self) -> None:
        """Expose a Deep Agents-shaped task tool without leaking Command data."""

        @tool("task")
        async def task(
            description: str,
            subagent_type: str,
            runtime: ToolRuntime,
        ) -> Command:
            """Return a synthetic subagent response.

            Args:
                description: Requested subagent work.
                subagent_type: Selected subagent name.
                runtime: Injected child tool runtime.

            Returns:
                Command with one final task ToolMessage.
            """

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"{subagent_type}: {description}",
                            tool_call_id=str(runtime.tool_call_id),
                        )
                    ]
                }
            )

        middleware = AreneroInterpreterMiddleware(subagents=True)
        self.addAsyncCleanup(middleware.aclose)

        result = await _call(
            middleware,
            "await task(description='check it', subagent_type='worker')",
            [task],
        )

        self.assertEqual(result["result"], "worker: check it")

    async def test_checkpoint_round_trip_and_hmac_rejection(self) -> None:
        """Move a signed snapshot through private graph-state fields."""

        middleware = AreneroInterpreterMiddleware(
            checkpoint_state=True,
            snapshot_signing_key="secret",
        )
        self.addAsyncCleanup(middleware.aclose)
        await _call(middleware, "value = 41")

        update = await middleware.aafter_agent({}, None)
        assert update is not None
        await middleware.abefore_agent(update, None)
        restored = await _call(middleware, "value + 1")

        self.assertEqual(restored["result"], 42)
        tampered = dict(update)
        tampered["_arenero_snapshot"] = update["_arenero_snapshot"] + b"x"
        rejected = await middleware.abefore_agent(tampered, None)
        self.assertEqual(rejected["_arenero_snapshot"], None)

    async def test_module_tool_loads_virtual_imports(self) -> None:
        """Let an agent load a module graph and follow-up code in one call."""

        filesystem = OSAccess(
            files=[
                MemoryFile("/helper.py", "def double(value):\n    return value * 2\n"),
                MemoryFile("/main.py", "from helper import double\n"),
            ]
        )
        kernel = AsyncPersistentMontyKernel(os_handler=filesystem)
        middleware = AreneroInterpreterMiddleware(kernel=kernel)
        self.addAsyncCleanup(kernel.close)
        coroutine = middleware.tools[1].coroutine
        assert coroutine is not None

        message = await coroutine(
            runtime=_runtime(middleware.tools),
            entry_path="/main.py",
            code="double(21)",
        )
        result = json.loads(message.content)

        self.assertEqual(result["result"], 42)
        self.assertIn("top-level imports", middleware.tools[1].description)
