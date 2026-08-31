import asyncio
import tempfile
import unittest
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from arenero import (
    AsyncPersistentMontyKernel,
    InMemorySessionStore,
    KernelConfig,
    SQLiteSessionStore,
    WorkspaceConfig,
)
from arenero.mcp import McpHub, McpServerConfig


class AsyncKernelTests(unittest.IsolatedAsyncioTestCase):
    """Test native asynchronous sessions and batch execution."""

    async def test_state_persists_and_sessions_are_isolated(self) -> None:
        """Restore one session without exposing its state to another session."""
        async with AsyncPersistentMontyKernel() as kernel:
            await kernel.execute("one", "x = 21")
            restored = await kernel.execute("one", "x * 2")
            isolated = await kernel.execute("two", "x")

        self.assertEqual(restored["result"], 42)
        self.assertTrue(restored["state_restored"])
        self.assertFalse(isolated["ok"])
        self.assertFalse(isolated["state_restored"])

    async def test_shared_store_survives_async_kernel_restart(self) -> None:
        """Restore state after replacing the native async kernel."""
        store = InMemorySessionStore()
        async with AsyncPersistentMontyKernel(store) as kernel:
            await kernel.execute("thread", "value = 41")

        async with AsyncPersistentMontyKernel(store) as kernel:
            result = await kernel.execute("thread", "value + 1")

        self.assertEqual(result["result"], 42)
        self.assertTrue(result["state_restored"])

    async def test_different_sessions_run_concurrently(self) -> None:
        """Use two workers at the same time for two independent sessions."""
        active = 0
        maximum_active = 0
        both_started = asyncio.Event()

        async def hold(value: int) -> int:
            """Wait until two host calls are active and return the input."""
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                both_started.set()
            try:
                await asyncio.wait_for(both_started.wait(), timeout=1)
                return value
            finally:
                active -= 1

        config = KernelConfig(min_worker_processes=2, max_worker_processes=2)
        async with AsyncPersistentMontyKernel(
            config=config,
            external_lookup={"hold": hold},
        ) as kernel:
            results = await asyncio.gather(
                kernel.execute("one", "await hold(1)"),
                kernel.execute("two", "await hold(2)"),
            )

        self.assertEqual([result["result"] for result in results], [1, 2])
        self.assertEqual(maximum_active, 2)

    async def test_same_session_runs_in_order(self) -> None:
        """Keep at most one active operation for one session ID."""
        active = 0
        maximum_active = 0

        async def observe(value: int) -> int:
            """Record active calls and return after a short yield."""
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                await asyncio.sleep(0.02)
                return value
            finally:
                active -= 1

        config = KernelConfig(min_worker_processes=2, max_worker_processes=2)
        async with AsyncPersistentMontyKernel(
            config=config,
            external_lookup={"observe": observe},
        ) as kernel:
            results = await asyncio.gather(
                kernel.execute("same", "await observe(1)"),
                kernel.execute("same", "await observe(2)"),
            )

        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual(maximum_active, 1)
        self.assertEqual([result["session_revision"] for result in results], [1, 2])

    async def test_worker_checkout_timeout_returns_retryable_result(self) -> None:
        """Return bounded backpressure instead of raising to the batch caller."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> int:
            """Own the only worker until the test releases it."""
            started.set()
            await release.wait()
            return 1

        config = KernelConfig(
            max_worker_processes=1,
            worker_checkout_timeout_seconds=0.02,
        )
        async with AsyncPersistentMontyKernel(
            config=config,
            external_lookup={"hold": hold},
        ) as kernel:
            first = asyncio.create_task(kernel.execute("one", "await hold()"))
            await started.wait()
            busy = await kernel.execute("two", "2")
            release.set()
            await first

        self.assertEqual(busy["error"]["type"], "WorkerCheckoutTimeout")
        self.assertTrue(busy["retryable"])

    async def test_async_host_function_and_type_check_stub(self) -> None:
        """Await an application host function from type-checked Monty code."""

        async def double(value: int) -> int:
            """Return twice the supplied value."""
            await asyncio.sleep(0)
            return value * 2

        config = KernelConfig(
            type_check=True,
            type_check_stubs="async def double(value: int) -> int: ...",
        )
        async with AsyncPersistentMontyKernel(
            config=config,
            external_lookup={"double": double},
        ) as kernel:
            result = await kernel.execute("typed", "await double(21)")

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], 42)

    async def test_terminal_timeout_deletes_async_session_state(self) -> None:
        """Start fresh after a terminal interpreter timeout."""
        config = KernelConfig(
            max_duration_seconds=0.01,
            request_timeout_seconds=1,
        )
        async with AsyncPersistentMontyKernel(config=config) as kernel:
            await kernel.execute("thread", "value = 42")
            timed_out = await kernel.execute("thread", "while True:\n    pass")
            missing = await kernel.execute("thread", "value")

        self.assertFalse(timed_out["ok"])
        self.assertFalse(timed_out["state_saved"])
        self.assertFalse(missing["ok"])
        self.assertFalse(missing["state_restored"])

    async def test_sqlite_state_survives_async_kernel_restart(self) -> None:
        """Persist native async state in the optional SQLite store."""
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            async with AsyncPersistentMontyKernel(
                SQLiteSessionStore(database)
            ) as kernel:
                await kernel.execute("thread", "value = 40")
            async with AsyncPersistentMontyKernel(
                SQLiteSessionStore(database)
            ) as kernel:
                result = await kernel.execute("thread", "value + 2")

        self.assertEqual(result["result"], 42)

    async def test_virtual_helpers_and_execute_file(self) -> None:
        """Create a file with ``fs_write`` and execute it with follow-up code."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceConfig(Path(directory))
            async with AsyncPersistentMontyKernel(workspace=workspace) as kernel:
                written = await kernel.execute(
                    "thread",
                    "fs_write('/workspace/helper.py', "
                    "'def triple(value):\\n    return value * 3\\n')",
                )
                result = await kernel.execute_file(
                    "thread",
                    "/workspace/helper.py",
                    "triple(14)",
                )
                content = await kernel.execute(
                    "thread", "fs_read('/workspace/helper.py')"
                )

        self.assertEqual(written["result"], len(content["result"]))
        self.assertEqual(result["result"], 42)

    async def test_async_mcp_call_runs_inside_monty(self) -> None:
        """Await an MCP tool from the native asynchronous kernel."""
        server = MCPServer("async-demo")

        @server.tool(description="Add two integers")
        async def add(a: int, b: int) -> dict[str, int]:
            """Add two integers after yielding to the event loop."""
            await asyncio.sleep(0)
            return {"sum": a + b}

        hub = McpHub([McpServerConfig.in_process("demo", server)])
        try:
            async with AsyncPersistentMontyKernel(mcp_hub=hub) as kernel:
                result = await kernel.execute(
                    "mcp",
                    'response = await mcp_call("demo__add", {"a": 20, "b": 22})\n'
                    'response["content"][0]["text"]',
                )
        finally:
            await asyncio.to_thread(hub.close)

        self.assertTrue(result["ok"])
        self.assertIn('"sum": 42', result["result"])

    def test_invalid_worker_pool_configuration_is_rejected(self) -> None:
        """Reject inconsistent worker pool sizes before startup."""
        with self.assertRaises(ValueError):
            KernelConfig(min_worker_processes=2, max_worker_processes=1)


if __name__ == "__main__":
    unittest.main()
