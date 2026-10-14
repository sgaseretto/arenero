"""Check the shared cleanup helpers through unchanged public entry points."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import unittest

from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic_monty import MontyError, OSAccess

from arenero import (
    AsyncPersistentMontyKernel,
    HostedToolConfig,
    HostedToolSet,
    InMemorySessionStore,
    KernelConfig,
    PersistentMontyKernel,
    create_async_python_repl_tools,
    create_python_repl_tools,
    execute_repl_cell,
)
from arenero.filesystem import FILESYSTEM_TYPE_CHECK_STUBS
from arenero.hosted_tools import HostedTool, _invoke_tool
from arenero.shell import VirtualShell
from arenero.values import error_text, trim_text


def _double(value: int) -> int:
    """Double a tool argument.

    Args:
        value: Integer to double.

    Returns:
        Twice the input value.
    """
    return value * 2


def _old_stubs_hash(
    kernel: PersistentMontyKernel | AsyncPersistentMontyKernel,
    *,
    async_mode: bool = False,
) -> str:
    """Reproduce the previous stable-stub cache hash for restore checks.

    Args:
        kernel: Test kernel with application declarations and configured tools.
        async_mode: Use the previous native async declarations when true.

    Returns:
        The SHA-256 digest produced before the cleanup.
    """
    host_stubs = "\n".join(
        value
        for value in (
            kernel.config.type_check_stubs,
            kernel.hosted_tools.type_check_stubs(async_mode=async_mode),
        )
        if value
    )
    stubs = FILESYSTEM_TYPE_CHECK_STUBS
    if host_stubs:
        stubs = f"{stubs}\n{host_stubs}"
    return hashlib.sha256(stubs.encode()).hexdigest()


def _tool_graph(tools: list) -> object:
    """Build a real tool node without a model or a configured thread ID.

    Args:
        tools: Existing Arenero LangChain tools.

    Returns:
        Compiled graph that injects ToolRuntime and validates tool arguments.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode(tools))
    builder.set_entry_point("tools")
    builder.add_edge("tools", END)
    return builder.compile()


class CleanupTests(unittest.TestCase):
    """Check common results, tool errors, limits, and stable sync policy."""

    def test_missing_thread_errors_match_for_repl_and_file_tools(self) -> None:
        """Return the same guidance without running code or touching files."""
        calls = [
            ("python_repl", {"code": "42"}),
            ("read_file", {"path": "/missing.txt"}),
            ("write_file", {"path": "/new.txt", "content": "text"}),
            ("grep_files", {"query": "text"}),
        ]
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            graph = _tool_graph(create_python_repl_tools(kernel))
            output = graph.invoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": name, "args": args, "id": name}
                                for name, args in calls
                            ],
                        )
                    ]
                }
            )
            results = [json.loads(item.content) for item in output["messages"][1:]]
            self.assertEqual(len(results), len(calls))
            self.assertTrue(all(result == results[0] for result in results))
            self.assertEqual(results[0]["error"]["code"], "missing_thread_id")
            self.assertEqual(results[0]["error"]["recovery"], "stop")
            self.assertIn("thread_id", results[0]["error"]["hint"])
            self.assertEqual(kernel.list_sessions(), [])
            with self.assertRaises(MontyError):
                kernel.download_file("/new.txt")

    def test_shared_trim_marker_and_error_fallback(self) -> None:
        """Keep zero, short, exact, and marker-sized text outputs unchanged."""
        text = "x" * 40
        for limit, expected, shortened in (
            (0, "", True),
            (3, "xxx", True),
            (16, "x" * 16, True),
            (20, "x" * 7 + "... <trimmed>", True),
            (40, text, False),
            (50, text, False),
        ):
            with self.subTest(limit=limit):
                self.assertEqual(trim_text(text, limit), (expected, shortened))
        self.assertEqual(error_text({"error": {"message": "failed"}}), "failed")
        self.assertEqual(error_text({"error": None}), "unknown Arenero error")

    def test_limits_match_across_python_cells_shell_and_inspection(self) -> None:
        """Keep default/capped limits and the shell's negative-limit rejection."""
        config = KernelConfig(default_output_limit_chars=20, max_output_limit_chars=30)
        with PersistentMontyKernel(config=config) as kernel:
            for request, limit, capped in (
                (None, 20, False),
                (0, 0, False),
                (3, 3, False),
                (20, 20, False),
                (100, 30, True),
            ):
                with self.subTest(request=request):
                    python = kernel.execute("agent", "print('x' * 80)", request)
                    cell = execute_repl_cell(
                        kernel, "agent", "!echo " + "x" * 80, request
                    )
                    inspected = kernel.inspect_values("agent", [], request)
                    invalid = kernel.execute("", "42", request)
                    for result in (python, cell, inspected, invalid):
                        self.assertEqual(result["output_limit_chars"], limit)
                        self.assertEqual(result["limit_was_capped"], capped)
                    self.assertEqual(python["stdout"], cell["output"])
            negative = kernel.execute("agent", "42", -1)
            self.assertEqual(negative["output_limit_chars"], 0)
            self.assertTrue(negative["limit_was_capped"])
            self.assertEqual(
                VirtualShell(kernel, "agent").execute("echo x", -1).exit_code, 2
            )

    def test_stable_stub_hash_and_saved_state_survive_reopen(self) -> None:
        """Keep the old policy digest without retaining a second stub string."""
        store = InMemorySessionStore()
        config = KernelConfig(type_check=True, type_check_stubs="seed: int")
        tools = HostedToolSet((HostedTool.from_callable(_double, name="double"),))
        with PersistentMontyKernel(store, config=config, hosted_tools=tools) as kernel:
            self.assertFalse(hasattr(kernel, "_type_check_stubs"))
            result = kernel.execute("agent", "answer = double(value=21)\nanswer")
            self.assertTrue(result["ok"], result["error"])
            policy = kernel.session_policy("agent")
            self.assertEqual(policy["saved"]["stubs_hash"], _old_stubs_hash(kernel))
        with PersistentMontyKernel(store, config=config, hosted_tools=tools) as kernel:
            self.assertEqual(kernel.session_policy("agent")["status"], "match")
            self.assertEqual(kernel.execute("agent", "answer")["result"], 42)


class AsyncCleanupTests(unittest.IsolatedAsyncioTestCase):
    """Check invocation, shared tool setup, policy, and the async no-lock path."""

    async def test_async_invoker_has_priority_and_sync_work_uses_a_thread(self) -> None:
        """Use async invocation when available and offload sync-only callbacks."""
        main_thread = threading.get_ident()

        async def async_invoke(arguments: dict) -> int:
            """Return a value through the preferred async path.

            Args:
                arguments: Tool input mapping.

            Returns:
                The supplied value plus one.
            """
            return arguments["value"] + 1

        def sync_invoke(arguments: dict) -> int:
            """Return the host thread identity.

            Args:
                arguments: Unused tool input mapping.

            Returns:
                Current host thread identifier.
            """
            return threading.get_ident()

        both = HostedTool("both", "Test", {}, invoke=sync_invoke, ainvoke=async_invoke)
        sync = HostedTool("sync", "Test", {}, invoke=sync_invoke)
        self.assertEqual(await _invoke_tool(both, {"value": 41}), 42)
        self.assertNotEqual(await _invoke_tool(sync, {}), main_thread)

    async def test_async_invocation_preserves_errors_and_cancellation(self) -> None:
        """Propagate errors and cancel an active async callback with its cleanup."""
        started = asyncio.Event()
        finished = asyncio.Event()

        async def wait(arguments: dict) -> None:
            """Wait until cancellation, then record callback cleanup.

            Args:
                arguments: Unused tool input mapping.
            """
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        def fail(arguments: dict) -> None:
            """Raise an error through the sync-only invocation path.

            Args:
                arguments: Unused tool input mapping.

            Raises:
                ValueError: Always, for the error propagation check.
            """
            raise ValueError("tool failed")

        task = asyncio.create_task(
            _invoke_tool(HostedTool("wait", "Wait", {}, ainvoke=wait), {})
        )
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())
        with self.assertRaisesRegex(ValueError, "tool failed"):
            await _invoke_tool(HostedTool("fail", "Fail", {}, invoke=fail), {})

    async def test_python_only_execution_and_reset_skip_the_file_lock(self) -> None:
        """Keep native async execution and reset usable while the file lock is held."""
        async with AsyncPersistentMontyKernel(os_handler=OSAccess()) as kernel:
            async with kernel.filesystem_guard():
                result = await asyncio.wait_for(
                    kernel.execute("agent", "answer = 42\nanswer", filesystem=False), 2
                )
                self.assertEqual(result["result"], 42)
                self.assertTrue(
                    await asyncio.wait_for(kernel.reset("agent", filesystem=False), 2)
                )
            self.assertIsNone(await kernel.get_session_info("agent"))
            self.assertEqual((await kernel.execute("agent", "6 * 7"))["result"], 42)

    async def test_shared_tool_setup_preserves_budgets_collisions_and_policy(
        self,
    ) -> None:
        """Keep operation tools temporary and preserve sync/async policy digests."""
        tools = HostedToolSet((HostedTool.from_callable(_double, name="double"),))
        extra = HostedTool.from_callable(_double, name="extra")
        operation = HostedToolSet((extra,), HostedToolConfig(max_calls_per_operation=1))
        config = KernelConfig(type_check=True, type_check_stubs="seed: int")
        for async_mode in (False, True):
            store = InMemorySessionStore()
            kernel_type = (
                AsyncPersistentMontyKernel if async_mode else PersistentMontyKernel
            )
            kernel = kernel_type(store, config=config, hosted_tools=tools)
            if async_mode:
                await kernel.open()
            try:
                if async_mode:
                    result = await kernel.execute(
                        "agent",
                        "await extra(value=21)\nawait extra(value=1)",
                        hosted_tools=operation,
                    )
                    policy = await kernel.session_policy("agent")
                    stable = await kernel.execute("agent", "await double(value=21)")
                    missing = await kernel.execute("agent", "await extra(value=1)")
                else:
                    result = kernel.execute(
                        "agent",
                        "extra(value=21)\nextra(value=1)",
                        hosted_tools=operation,
                    )
                    policy = kernel.session_policy("agent")
                    stable = kernel.execute("agent", "double(value=21)")
                    missing = kernel.execute("agent", "extra(value=1)")
                self.assertEqual(result["error"]["code"], "hosted_tool_call_limit")
                self.assertEqual(result["host_calls"]["count"], 1)
                self.assertEqual(policy["status"], "match")
                self.assertEqual(
                    policy["saved"]["stubs_hash"],
                    _old_stubs_hash(kernel, async_mode=async_mode),
                )
                self.assertEqual(stable["result"], 42)
                self.assertFalse(missing["ok"])
                kernel.external_lookup["extra"] = _double
                if async_mode:
                    collision = await kernel.execute(
                        "agent", "42", hosted_tools=operation
                    )
                else:
                    collision = kernel.execute("agent", "42", hosted_tools=operation)
                self.assertEqual(
                    collision["error"]["type"], "HostedToolConfigurationError"
                )
                self.assertIn("conflict", collision["error"]["message"])
            finally:
                if async_mode:
                    await kernel.close()
                else:
                    kernel.close()

    async def test_native_async_tools_use_the_same_missing_thread_error(self) -> None:
        """Keep the native async agent error equal to the sync adapter error."""
        async with AsyncPersistentMontyKernel() as kernel:
            graph = _tool_graph(create_async_python_repl_tools(kernel))
            output = await graph.ainvoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "python_repl",
                                    "args": {"code": "42"},
                                    "id": "call",
                                }
                            ],
                        )
                    ]
                }
            )
            result = json.loads(output["messages"][-1].content)
            self.assertEqual(result["error"]["code"], "missing_thread_id")
            self.assertEqual(result["error"]["recovery"], "stop")
            self.assertEqual(await kernel.list_sessions(), [])
