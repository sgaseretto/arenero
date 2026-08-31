import asyncio
import json
import sys
import unittest
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from arenero import KernelConfig, PersistentMontyKernel
from arenero.integrations import create_python_repl_tools
from arenero.mcp import McpHub, McpHubConfig, McpServerConfig


class McpHubTests(unittest.TestCase):
    """Test MCP discovery and Monty host calls."""

    def make_server(self, name: str = "demo") -> MCPServer:
        """Create an in-process MCP server with deterministic tools.

        Args:
            name: Server implementation name.

        Returns:
            A configured in-process server.
        """
        server = MCPServer(name)

        @server.tool(description="Add two integers")
        def add(a: int, b: int) -> dict[str, int]:
            """Add two integers.

            Args:
                a: First integer.
                b: Second integer.

            Returns:
                A dictionary with the sum.
            """
            return {"sum": a + b}

        @server.tool(description="Return text of a requested size")
        def large_text(size: int) -> str:
            """Return repeated text.

            Args:
                size: Number of returned characters.

            Returns:
                Repeated ``x`` text.
            """
            return "x" * size

        @server.tool(description="Wait for a short test duration")
        async def slow(seconds: float) -> str:
            """Wait before returning.

            Args:
                seconds: Test wait duration.

            Returns:
                A completion marker.
            """
            await asyncio.sleep(seconds)
            return "done"

        return server

    def test_discovery_description_and_call(self) -> None:
        """Discover a tool, inspect its schema, and call it."""
        hub = McpHub([McpServerConfig.in_process("demo", self.make_server())])
        with hub:
            servers = hub.servers()
            search = hub.search("add integers")
            description = hub.describe("demo__add")
            result = hub.call("demo__add", {"a": 2, "b": 5})

        self.assertTrue(servers["servers"][0]["connected"])
        self.assertEqual(search["tools"][0]["name"], "demo__add")
        self.assertEqual(description["tool"]["input_schema"]["required"], ["a", "b"])
        self.assertTrue(result["ok"])
        self.assertIn('"sum": 7', result["content"][0]["text"])

    def test_mcp_event_sink_gets_safe_call_metadata(self) -> None:
        """Observe MCP call identity and status without arguments or results."""
        events = []
        with McpHub(
            [McpServerConfig.in_process("demo", self.make_server())],
            event_sink=events.append,
        ) as hub:
            result = hub.call("demo__add", {"a": 2, "b": 3})

        self.assertTrue(result["ok"])
        self.assertEqual(
            [event.action for event in events],
            ["call_started", "call_finished"],
        )
        self.assertNotIn("arguments", events[0].data)
        self.assertNotIn("result", events[1].data)

    def test_monty_can_compose_calls_and_persist_a_helper(self) -> None:
        """Keep a function that calls MCP across persistent Monty feeds."""
        with (
            McpHub([McpServerConfig.in_process("demo", self.make_server())]) as hub,
            PersistentMontyKernel(mcp_hub=hub) as kernel,
        ):
            first = kernel.execute(
                "thread",
                """def add_with_mcp(a, b):
    response = mcp_call("demo__add", {"a": a, "b": b})
    return response["content"][0]["text"]
add_with_mcp(4, 6)
""",
            )
            second = kernel.execute("thread", "add_with_mcp(10, 20)")

        self.assertTrue(first["ok"])
        self.assertIn('"sum": 10', first["result"])
        self.assertTrue(second["state_restored"])
        self.assertIn('"sum": 30', second["result"])

    def test_call_limit_is_per_kernel_operation(self) -> None:
        """Reset the MCP call counter for each outer kernel operation."""
        config = McpHubConfig(max_calls_per_operation=2)
        with (
            McpHub(
                [McpServerConfig.in_process("demo", self.make_server())],
                config,
            ) as hub,
            PersistentMontyKernel(mcp_hub=hub) as kernel,
        ):
            limited = kernel.execute(
                "thread",
                """responses = [mcp_call("demo__add", {"a": i, "b": 1}) for i in range(3)]
responses[2]["error"]["type"]
""",
            )
            next_call = kernel.execute(
                "thread", 'mcp_call("demo__add", {"a": 1, "b": 1})["ok"]'
            )

        self.assertEqual(limited["result"], "McpCallLimitExceeded")
        self.assertIs(next_call["result"], True)

    def test_arguments_and_results_have_hard_limits(self) -> None:
        """Reject large arguments and trim large tool responses."""
        config = McpHubConfig(max_argument_bytes=20, max_result_bytes=600)
        with McpHub(
            [McpServerConfig.in_process("demo", self.make_server())],
            config,
        ) as hub:
            rejected = hub.call("demo__add", {"unused": "x" * 100})
            trimmed = hub.call("demo__large_text", {"size": 10_000})

        self.assertEqual(rejected["error"]["type"], "McpArgumentsTooLarge")
        self.assertTrue(trimmed["truncated"])
        size = len(json.dumps(trimmed, separators=(",", ":")).encode())
        self.assertLessEqual(size, config.max_result_bytes)

    def test_total_call_time_is_limited_per_kernel_operation(self) -> None:
        """Bound the sum of MCP waits and reset the deadline next time."""
        config = McpHubConfig(
            call_timeout_seconds=1,
            max_total_call_seconds_per_operation=0.05,
        )
        with (
            McpHub(
                [McpServerConfig.in_process("demo", self.make_server())],
                config,
            ) as hub,
            PersistentMontyKernel(mcp_hub=hub) as kernel,
        ):
            limited = kernel.execute(
                "thread",
                """mcp_call("demo__slow", {"seconds": 0.03})
second = mcp_call("demo__slow", {"seconds": 0.03})
second["error"]["type"]
""",
            )
            next_operation = kernel.execute(
                "thread", 'mcp_call("demo__add", {"a": 1, "b": 2})["ok"]'
            )

        self.assertEqual(limited["result"], "McpCallTimeout")
        self.assertIs(next_operation["result"], True)

    def test_search_is_capped_and_unknown_names_are_structured(self) -> None:
        """Apply discovery caps and return errors without ending the REPL."""
        config = McpHubConfig(max_search_results=1)
        with McpHub(
            [McpServerConfig.in_process("demo", self.make_server())],
            config,
        ) as hub:
            search = hub.search("", limit=20)
            unknown = hub.call("missing", {})
            refreshed = hub.refresh("missing")

        self.assertEqual(search["count"], 1)
        self.assertTrue(search["limit_was_capped"])
        self.assertEqual(unknown["error"]["type"], "UnknownMcpTool")
        self.assertEqual(refreshed["error"]["type"], "UnknownMcpServer")

    def test_short_names_must_be_unambiguous(self) -> None:
        """Require a qualified name when two servers use one tool name."""
        with McpHub(
            [
                McpServerConfig.in_process("one", self.make_server("one")),
                McpServerConfig.in_process("two", self.make_server("two")),
            ]
        ) as hub:
            ambiguous = hub.call("add", {"a": 1, "b": 2})
            qualified = hub.call("two__add", {"a": 1, "b": 2})

        self.assertEqual(ambiguous["error"]["type"], "AmbiguousMcpTool")
        self.assertTrue(qualified["ok"])

    def test_type_check_stubs_cover_mcp_functions(self) -> None:
        """Let Monty type-check calls to MCP host functions."""
        with (
            McpHub([McpServerConfig.in_process("demo", self.make_server())]) as hub,
            PersistentMontyKernel(
                config=KernelConfig(type_check=True),
                mcp_hub=hub,
            ) as kernel,
        ):
            result = kernel.execute("typed", "mcp_search('add')['ok']")

        self.assertTrue(result["ok"])
        self.assertIs(result["result"], True)

    def test_stdio_server_works_from_the_repl(self) -> None:
        """Connect a configured standard-input server and call it from Monty."""
        fixture = Path("tests/fixtures/mcp_server.py").resolve()
        server = McpServerConfig.stdio(
            "local",
            sys.executable,
            args=[str(fixture)],
            cwd=Path.cwd(),
        )
        with (
            McpHub([server]) as hub,
            PersistentMontyKernel(mcp_hub=hub) as kernel,
        ):
            result = kernel.execute(
                "stdio",
                'mcp_call("local__multiply", {"a": 6, "b": 7})',
            )
            refreshed = hub.refresh("local")
            after_refresh = kernel.execute(
                "stdio",
                'mcp_call("local__multiply", {"a": 3, "b": 5})',
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["result"]["ok"])
        self.assertIn('"product": 42', result["result"]["content"][0]["text"])
        self.assertTrue(refreshed["servers"][0]["connected"])
        self.assertIn('"product": 15', after_refresh["result"]["content"][0]["text"])

    def test_langchain_repl_description_explains_mcp_functions(self) -> None:
        """Tell a LangChain agent how to discover and call configured tools."""
        with (
            McpHub([McpServerConfig.in_process("demo", self.make_server())]) as hub,
            PersistentMontyKernel(mcp_hub=hub) as kernel,
        ):
            tools = create_python_repl_tools(kernel)

        repl = next(tool for tool in tools if tool.name == "python_repl")
        self.assertIn("mcp_search(query)", repl.description)
        self.assertIn("mcp_call(name, arguments)", repl.description)

    def test_promoted_tool_uses_restart_for_a_saved_typed_session(self) -> None:
        """Restart a saved typed session before its first promoted tool call."""

        with (
            McpHub([McpServerConfig.in_process("demo", self.make_server())]) as hub,
            PersistentMontyKernel(
                config=KernelConfig(type_check=True),
                mcp_hub=hub,
            ) as kernel,
        ):
            promoted = kernel.execute("thread", 'mcp_promote("demo__add")')
            report = kernel.capabilities()
            unresolved = kernel.execute("thread", "demo__add(a=20, b=22)")
            called = kernel.execute(
                "thread",
                "demo__add(a=20, b=22)",
                restart=True,
            )

        self.assertTrue(promoted["result"]["ok"])
        self.assertEqual(promoted["result"]["available"], "next_execution")
        self.assertIn("demo__add", report["hosted_tools"]["functions"])
        self.assertIn("def demo__add", report["hosted_tools"]["catalog"])
        self.assertEqual(unresolved["error"]["code"], "unresolved_reference")
        self.assertEqual(called["result"]["value"], {"sum": 42})
        self.assertEqual(called["host_calls"]["count"], 1)
        self.assertTrue(called["session_restarted"])

    def test_configuration_and_name_collisions_are_rejected(self) -> None:
        """Reject unsafe ambiguity before a Monty operation starts."""
        remote = McpServerConfig.http(
            "remote",
            "https://example.com/mcp?token=secret",
            headers={"Authorization": "Bearer secret"},
        )
        self.assertNotIn("secret", repr(remote))
        with self.assertRaises(ValueError):
            McpServerConfig.http("bad__name", "https://example.com/mcp")
        with self.assertRaises(ValueError):
            McpHub(
                [
                    McpServerConfig.in_process("same", self.make_server()),
                    McpServerConfig.in_process("same", self.make_server()),
                ]
            )
        hub = McpHub([McpServerConfig.in_process("demo", self.make_server())])

        def reserved_name() -> None:
            """Provide a conflicting host function for validation."""

        with self.assertRaises(ValueError):
            PersistentMontyKernel(
                external_lookup={"mcp_call": reserved_name},
                mcp_hub=hub,
            )
        hub.close()


if __name__ == "__main__":
    unittest.main()
