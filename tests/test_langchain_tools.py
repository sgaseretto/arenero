import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic_monty import MemoryFile, OSAccess

from arenero import (
    AsyncPersistentMontyKernel,
    PersistentMontyKernel,
    SQLiteSessionStore,
    create_async_python_repl_tools,
    create_python_repl_tools,
)


class LangChainToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        store = SQLiteSessionStore(Path(self.directory.name) / "sessions.db")
        filesystem = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "def double(value):\n    return value * 2\n",
                )
            ]
        )
        self.kernel = PersistentMontyKernel(store, os_handler=filesystem)
        self.addCleanup(self.kernel.close)
        self.tools = create_python_repl_tools(self.kernel)

        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode(self.tools))
        builder.set_entry_point("tools")
        builder.add_edge("tools", END)
        self.graph = builder.compile()

    def call_tool(
        self,
        name: str,
        arguments: dict,
        thread_id: str = "thread-1",
    ) -> dict:
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": arguments,
                            "id": "test-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
        output = self.graph.invoke(
            state,
            {"configurable": {"thread_id": thread_id}},
        )
        return json.loads(output["messages"][-1].content)

    def test_schema_exposes_limit_but_hides_runtime(self) -> None:
        schema = self.tools[0].tool_call_schema.model_json_schema()
        repl = self.tools[0]

        self.assertIn("code", schema["properties"])
        self.assertIn("output_limit_chars", schema["properties"])
        self.assertIn("output_file", schema["properties"])
        self.assertIn("restart", schema["properties"])
        self.assertNotIn("runtime", schema["properties"])
        self.assertIn("! or start a multiline shell cell", repl.description)
        self.assertIn("same session", repl.description)
        self.assertIn("mode=shell", repl.description)
        self.assertIn("output_file", repl.description)

    def test_python_repl_runs_notebook_shell_cells(self) -> None:
        """Use only the REPL tool for Python, shell, and shared virtual files."""
        assigned = self.call_tool("python_repl", {"code": "value = 41"})
        calculated = self.call_tool(
            "python_repl",
            {"code": "!python -c 'value + 1'"},
        )
        shell = self.call_tool(
            "python_repl",
            {"code": ("%%shell\nmkdir -p /work\ncd /work\necho hello > note.txt\npwd")},
        )
        read = self.call_tool(
            "python_repl",
            {"code": "from pathlib import Path\nPath('/work/note.txt').read_text()"},
        )
        invalid = self.call_tool(
            "python_repl",
            {"code": "!echo one\necho two"},
        )

        self.assertEqual(assigned["mode"], "python")
        self.assertEqual(calculated["mode"], "shell")
        self.assertEqual(calculated["output"], "42")
        self.assertIn("truncation", calculated)
        self.assertIn("retryable", calculated)
        self.assertEqual(shell["output"], "/work\n")
        self.assertEqual(read["result"], "hello\n")
        self.assertEqual(invalid["exit_code"], 2)
        self.assertIn("Use %%shell", invalid["output"])

    def test_execute_file_schema_exposes_file_and_follow_up_code(self) -> None:
        """Expose file inputs to the model but keep runtime hidden."""
        execute_file = next(
            tool for tool in self.tools if tool.name == "python_repl_execute_file"
        )
        schema = execute_file.tool_call_schema.model_json_schema()

        self.assertIn("file_path", schema["properties"])
        self.assertIn("code", schema["properties"])
        self.assertIn("output_limit_chars", schema["properties"])
        self.assertIn("output_file", schema["properties"])
        self.assertNotIn("runtime", schema["properties"])

    def test_virtual_shell_tool_uses_the_thread_session(self) -> None:
        """Expose the virtual shell and keep its directory by thread ID."""
        shell = next(tool for tool in self.tools if tool.name == "virtual_shell")
        schema = shell.tool_call_schema.model_json_schema()

        self.assertIn("command", schema["properties"])
        self.assertIn("output_limit_chars", schema["properties"])
        self.assertIn("output_file", schema["properties"])
        self.assertNotIn("runtime", schema["properties"])
        self.assertIn("not an operating-system shell", shell.description)
        self.assertIn("grep uses fixed text unless -E is set", shell.description)
        self.assertIn("Unquoted *, ?, and ** expand", shell.description)
        self.assertIn("source FILE", shell.description)
        self.assertIn("full Bash scripts", shell.description)
        self.assertIn("Monty's limited Python", shell.description)
        self.assertIn("allows public HTTP and HTTPS URLs by", shell.description)
        self.assertIn("Private, local, and reserved", shell.description)
        self.assertIn("Check truncated", shell.description)

        made = self.call_tool("virtual_shell", {"command": "mkdir -p /work"})
        changed = self.call_tool("virtual_shell", {"command": "cd /work"})
        written = self.call_tool(
            "virtual_shell",
            {"command": "echo hello > note.txt"},
        )
        read = self.call_tool("virtual_shell", {"command": "cat note.txt"})
        other = self.call_tool(
            "virtual_shell",
            {"command": "pwd"},
            "thread-2",
        )

        self.assertEqual(made["exit_code"], 0)
        self.assertEqual(changed["exit_code"], 0)
        self.assertEqual(written["exit_code"], 0)
        self.assertEqual(read["output"], "hello\n")
        self.assertEqual(other["output"], "/\n")

    def test_adapter_exposes_all_filesystem_tools(self) -> None:
        """Create each bounded filesystem tool with hidden runtime data."""
        expected = {
            "read_file",
            "write_file",
            "edit_file",
            "delete_file",
            "list_files",
            "glob_files",
            "grep_files",
        }
        tools = {tool.name: tool for tool in self.tools}

        self.assertLessEqual(expected, tools.keys())
        for name in expected:
            schema = tools[name].tool_call_schema.model_json_schema()
            self.assertNotIn("runtime", schema["properties"])

    def test_tool_uses_thread_id_for_persistent_state(self) -> None:
        first = self.call_tool("python_repl", {"code": "x = 41"})
        second = self.call_tool(
            "python_repl",
            {"code": "x + 1", "output_limit_chars": 100},
        )

        self.assertTrue(first["ok"])
        self.assertEqual(second["result"], 42)
        self.assertTrue(second["state_restored"])

    def test_agent_can_execute_file_and_follow_up_code_in_one_call(self) -> None:
        """Load a helper, use it, and keep it in the agent thread."""
        loaded = self.call_tool(
            "python_repl_execute_file",
            {
                "file_path": "/helper.py",
                "code": "double(21)",
                "output_limit_chars": 100,
            },
        )
        reused = self.call_tool("python_repl", {"code": "double(5)"})

        self.assertTrue(loaded["ok"])
        self.assertEqual(loaded["result"], 42)
        self.assertEqual(reused["result"], 10)

    def test_agent_can_execute_a_static_virtual_module_graph(self) -> None:
        """Resolve a virtual import and follow-up code in one agent call."""

        self.call_tool(
            "write_file",
            {"path": "/main.py", "content": "from helper import double\n"},
        )

        loaded = self.call_tool(
            "python_repl_execute_module",
            {"entry_path": "/main.py", "code": "double(21)"},
        )
        reused = self.call_tool("python_repl", {"code": "double(5)"})

        self.assertEqual(loaded["result"], 42)
        self.assertEqual(reused["result"], 10)

    def test_repl_can_compose_a_supplied_langchain_tool(self) -> None:
        """Run a normal LangChain tool from code with the current runtime."""

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""

            return a * b

        tools = create_python_repl_tools(self.kernel, hosted_tools=(multiply,))
        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode(tools))
        builder.set_entry_point("tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        output = graph.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "python_repl",
                                "args": {"code": "multiply(a=6, b=7)"},
                                "id": "hosted-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            {"configurable": {"thread_id": "hosted-thread"}},
        )
        result = json.loads(output["messages"][-1].content)

        self.assertEqual(result["result"], 42)
        self.assertEqual(result["host_calls"]["count"], 1)

    def test_agent_filesystem_tools_share_files_between_threads(self) -> None:
        """Use every filesystem tool across two agent thread IDs."""
        written = self.call_tool(
            "write_file",
            {
                "path": "/notes.txt",
                "content": "Alpha\nbeta\nAlpha\n",
            },
            "one",
        )
        read = self.call_tool(
            "read_file",
            {"path": "/notes.txt", "limit": 100},
            "two",
        )
        edited = self.call_tool(
            "edit_file",
            {
                "path": "/notes.txt",
                "old_text": "beta",
                "new_text": "gamma",
            },
            "two",
        )
        listed = self.call_tool("list_files", {"path": "/"}, "one")
        globbed = self.call_tool(
            "glob_files",
            {"pattern": "*.txt", "root": "/"},
            "one",
        )
        grepped = self.call_tool(
            "grep_files",
            {
                "query": "alpha",
                "root": "/",
                "pattern": "*.txt",
                "case_sensitive": False,
            },
            "two",
        )
        deleted = self.call_tool(
            "delete_file",
            {"path": "/notes.txt"},
            "one",
        )
        missing = self.call_tool(
            "read_file",
            {"path": "/notes.txt"},
            "two",
        )

        self.assertEqual(written["result"], 17)
        self.assertEqual(read["result"], "Alpha\nbeta\nAlpha\n")
        self.assertEqual(edited["result"]["replacements"], 1)
        self.assertIn(
            {"path": "/notes.txt", "type": "file"},
            listed["result"]["entries"],
        )
        self.assertEqual(globbed["result"]["matches"], ["/notes.txt"])
        self.assertEqual(
            [match["line"] for match in grepped["result"]["matches"]],
            [1, 3],
        )
        self.assertEqual(deleted["result"], 1)
        self.assertFalse(missing["ok"])

    def test_reset_tool_resets_only_the_current_thread(self) -> None:
        self.call_tool("python_repl", {"code": "x = 1"}, "one")
        self.call_tool("python_repl", {"code": "x = 2"}, "two")

        reset = self.call_tool("reset_python_repl", {}, "one")
        missing = self.call_tool("python_repl", {"code": "x"}, "one")
        present = self.call_tool("python_repl", {"code": "x"}, "two")

        self.assertTrue(reset["ok"])
        self.assertFalse(missing["ok"])
        self.assertEqual(present["result"], 2)

    def test_agent_can_offload_trimmed_output(self) -> None:
        """Let the model preserve a larger bounded result in a virtual file."""
        result = self.call_tool(
            "python_repl",
            {
                "code": '"abcdefgh"',
                "output_limit_chars": 3,
                "output_file": "/outputs/result.json",
            },
        )
        saved = self.call_tool(
            "read_file",
            {"path": "/outputs/result.json", "limit": 1_000},
        )

        self.assertEqual(result["output_file"], "/outputs/result.json")
        self.assertTrue(result["truncation"]["result"])
        self.assertIn("abcdefgh", saved["result"])


class AsyncLangChainToolTests(unittest.IsolatedAsyncioTestCase):
    """Test the small native async LangChain adapter."""

    async def asyncSetUp(self) -> None:
        """Create one open async kernel and graph."""
        self.kernel = AsyncPersistentMontyKernel()
        await self.kernel.open()
        self.tools = create_async_python_repl_tools(self.kernel)
        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode(self.tools))
        builder.set_entry_point("tools")
        builder.add_edge("tools", END)
        self.graph = builder.compile()

    async def asyncTearDown(self) -> None:
        """Close the async worker pool."""
        await self.kernel.close()

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        thread_id: str = "thread-1",
    ) -> dict:
        """Invoke one async tool through a LangGraph tool node.

        Args:
            name: Tool name.
            arguments: Model-visible tool arguments.
            thread_id: Persistent agent session key.

        Returns:
            Parsed JSON tool response.
        """
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": arguments,
                            "id": "test-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
        output = await self.graph.ainvoke(
            state,
            {"configurable": {"thread_id": thread_id}},
        )
        return json.loads(output["messages"][-1].content)

    async def test_async_tools_persist_separate_agent_sessions(self) -> None:
        """Run two agent sessions in batch and restore each value."""
        first, second = await asyncio.gather(
            self.call_tool("python_repl", {"code": "value = 20"}, "one"),
            self.call_tool("python_repl", {"code": "value = 40"}, "two"),
        )
        one, two = await asyncio.gather(
            self.call_tool("python_repl", {"code": "value + 1"}, "one"),
            self.call_tool("python_repl", {"code": "value + 2"}, "two"),
        )

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(one["result"], 21)
        self.assertEqual(two["result"], 42)

    async def test_async_tool_scope_is_explicit(self) -> None:
        """Expose Python, file, and module execution without shell support."""
        tools = {tool.name: tool for tool in self.tools}

        self.assertEqual(
            set(tools),
            {
                "python_repl",
                "python_repl_execute_file",
                "python_repl_execute_module",
                "reset_python_repl",
            },
        )
        self.assertIn(
            "native asynchronous worker pool", tools["python_repl"].description
        )
        self.assertIn(
            "!COMMAND and %%shell cells are not supported",
            tools["python_repl"].description,
        )
        module_schema = tools["python_repl_execute_module"].tool_call_schema
        self.assertIn("entry_path", module_schema.model_json_schema()["properties"])


if __name__ == "__main__":
    unittest.main()
