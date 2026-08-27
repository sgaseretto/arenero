import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic_monty import MemoryFile, OSAccess

from webmonty import (
    PersistentMontyKernel,
    SQLiteSessionStore,
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

        self.assertIn("code", schema["properties"])
        self.assertIn("output_limit_chars", schema["properties"])
        self.assertNotIn("runtime", schema["properties"])

    def test_execute_file_schema_exposes_file_and_follow_up_code(self) -> None:
        """Expose file inputs to the model but keep runtime hidden."""
        execute_file = next(
            tool for tool in self.tools if tool.name == "python_repl_execute_file"
        )
        schema = execute_file.tool_call_schema.model_json_schema()

        self.assertIn("file_path", schema["properties"])
        self.assertIn("code", schema["properties"])
        self.assertIn("output_limit_chars", schema["properties"])
        self.assertNotIn("runtime", schema["properties"])

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


if __name__ == "__main__":
    unittest.main()
