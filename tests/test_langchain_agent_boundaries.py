"""Model-driven agent tests for files, MCP, errors, and recovery."""

import base64
import tempfile
import unittest
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent
from pydantic_monty import OSAccess

from arenero import (
    PersistentMontyKernel,
    SQLiteSessionStore,
    create_python_repl_tools,
)
from arenero.mcp import McpHub, McpServerConfig
from tests.test_langchain_agent import (
    ToolCallingFakeChatModel,
    named_tool_call,
    python_tool_call,
    tool_results,
)


def _config(thread_id: str) -> RunnableConfig:
    """Build one LangGraph configuration with a persistent thread ID.

    Args:
        thread_id: Agent and REPL session identity.

    Returns:
        Runnable configuration for an agent invocation.
    """

    return {"configurable": {"thread_id": thread_id}}


class AgentOutputRecoveryTests(unittest.TestCase):
    """Check that an agent can recover complete content from trimmed output."""

    def test_agent_reads_full_result_from_output_file_after_truncation(self) -> None:
        """Offload a trimmed result and read it in the same model-driven loop."""

        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call(
                        "large-output",
                        '"abcdefgh"',
                        output_limit_chars=3,
                        output_file="/outputs/result.json",
                    ),
                    named_tool_call(
                        "read-output",
                        "read_file",
                        {"path": "/outputs/result.json", "limit": 10_000},
                    ),
                    AIMessage(content="I recovered the complete result."),
                ]
            )
        )

        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            agent = create_agent(
                model,
                create_python_repl_tools(kernel),
                checkpointer=InMemorySaver(),
            )
            result = agent.invoke(
                {"messages": [HumanMessage(content="Produce the large result.")]},
                _config("output-agent"),
            )

        generated, recovered = tool_results(result)
        self.assertTrue(generated["truncation"]["result"])
        self.assertEqual(len(generated["result"]), 3)
        self.assertEqual(generated["output_file"], "/outputs/result.json")
        self.assertIn("abcdefgh", recovered["result"])
        self.assertEqual(
            result["messages"][-1].content,
            "I recovered the complete result.",
        )


class AgentFilesystemWorkflowTests(unittest.TestCase):
    """Check shell, Python, and file-tool composition across human turns."""

    def test_agent_shares_virtual_files_across_tools_and_turns(self) -> None:
        """Create, read, edit, and read one virtual file through three APIs."""

        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    named_tool_call(
                        "shell-write",
                        "virtual_shell",
                        {
                            "command": (
                                "mkdir -p /work && cd /work && "
                                "echo first > note.txt && pwd"
                            )
                        },
                    ),
                    AIMessage(content="Created."),
                    python_tool_call(
                        "python-read-first",
                        "from pathlib import Path\nPath('/work/note.txt').read_text()",
                    ),
                    AIMessage(content="Read first version."),
                    named_tool_call(
                        "edit-file",
                        "edit_file",
                        {
                            "path": "/work/note.txt",
                            "old_text": "first",
                            "new_text": "second",
                        },
                    ),
                    AIMessage(content="Edited."),
                    python_tool_call(
                        "python-read-second",
                        "from pathlib import Path\nPath('/work/note.txt').read_text()",
                    ),
                    AIMessage(content="Read second version."),
                ]
            )
        )

        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            agent = create_agent(
                model,
                create_python_repl_tools(kernel),
                checkpointer=InMemorySaver(),
            )
            config = _config("filesystem-agent")
            created = agent.invoke(
                {"messages": [HumanMessage(content="Create a note.")]}, config
            )
            first = agent.invoke(
                {"messages": [HumanMessage(content="Read the note in Python.")]},
                config,
            )
            edited = agent.invoke(
                {"messages": [HumanMessage(content="Edit the note.")]}, config
            )
            second = agent.invoke(
                {"messages": [HumanMessage(content="Read it again.")]}, config
            )

        self.assertEqual(tool_results(created)[0]["output"], "/work\n")
        self.assertEqual(tool_results(first)[-1]["result"], "first\n")
        self.assertEqual(tool_results(edited)[-1]["result"]["replacements"], 1)
        self.assertEqual(tool_results(second)[-1]["result"], "second\n")


class AgentMcpWorkflowTests(unittest.TestCase):
    """Check in-process MCP discovery, composition, and binary artifacts."""

    def test_agent_composes_mock_mcp_calls_and_materializes_image(self) -> None:
        """Discover and call MCP tools from Monty without external services."""

        server = MCPServer("agent-demo")

        @server.tool(description="Add two integers")
        def add(a: int, b: int) -> dict[str, int]:
            """Return the sum of two integers.

            Args:
                a: First integer.
                b: Second integer.

            Returns:
                Mapping with the integer sum.
            """

            return {"sum": a + b}

        @server.tool(description="Return a small image")
        def image() -> ImageContent:
            """Return deterministic PNG-shaped bytes as MCP image content.

            Returns:
                Base64 MCP image block.
            """

            return ImageContent(
                data=base64.b64encode(b"mock-image").decode("ascii"),
                mime_type="image/png",
            )

        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call(
                        "mcp-add",
                        "search = mcp_search('add integers')\n"
                        "response = mcp_call('demo__add', {'a': 3, 'b': 4})\n"
                        "{'tool': search['tools'][0]['name'], "
                        "'text': response['content'][0]['text']}",
                    ),
                    python_tool_call(
                        "mcp-image",
                        "mcp_call('demo__image', {})",
                    ),
                    AIMessage(content="MCP work completed."),
                ]
            )
        )

        with (
            McpHub([McpServerConfig.in_process("demo", server)]) as hub,
            PersistentMontyKernel(
                os_handler=OSAccess(),
                mcp_hub=hub,
                mcp_artifact_directory="/artifacts",
            ) as kernel,
        ):
            agent = create_agent(
                model,
                create_python_repl_tools(kernel),
                checkpointer=InMemorySaver(),
            )
            result = agent.invoke(
                {"messages": [HumanMessage(content="Use the MCP tools.")]},
                _config("mcp-agent"),
            )
            composed, media = tool_results(result)
            artifact = kernel.download_file(media["mcp_artifacts"][0])

        self.assertEqual(composed["result"]["tool"], "demo__add")
        self.assertIn('"sum": 7', composed["result"]["text"])
        self.assertEqual(len(media["mcp_artifacts"]), 1)
        self.assertEqual(artifact, b"mock-image")
        self.assertEqual(
            media["result"]["content"][0]["virtual_path"],
            media["mcp_artifacts"][0],
        )


class AgentCorrectionAndSnapshotTests(unittest.TestCase):
    """Check model correction and structured snapshot recovery paths."""

    def test_agent_corrects_unsupported_code_in_one_loop(self) -> None:
        """Read the Monty error and submit supported code before answering."""

        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("unsupported", "import requests"),
                    python_tool_call("corrected", "21 * 2"),
                    AIMessage(content="The corrected result is 42."),
                ]
            )
        )

        with PersistentMontyKernel() as kernel:
            agent = create_agent(model, create_python_repl_tools(kernel))
            result = agent.invoke(
                {"messages": [HumanMessage(content="Calculate the result.")]},
                _config("correction-agent"),
            )

        unsupported, corrected = tool_results(result)
        self.assertFalse(unsupported["ok"])
        self.assertEqual(unsupported["error"]["code"], "unsupported_import")
        self.assertIn("packages are unavailable", unsupported["error"]["hint"])
        self.assertEqual(corrected["result"], 42)
        self.assertEqual(
            result["messages"][-1].content,
            "The corrected result is 42.",
        )

    def test_agent_resets_incompatible_snapshot_and_retries(self) -> None:
        """Recover from a wrong Monty version through normal agent tools."""

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bad-session.db"
            store = SQLiteSessionStore(database)
            store.save(
                "snapshot-agent",
                b"incompatible-state",
                "0.0.0",
                expected_revision=None,
            )
            model = ToolCallingFakeChatModel(
                messages=iter(
                    [
                        python_tool_call("version-error", "1 + 1"),
                        named_tool_call("reset", "reset_python_repl", {}),
                        python_tool_call("retry", "1 + 1"),
                        AIMessage(content="The reset retry succeeded."),
                    ]
                )
            )

            with PersistentMontyKernel(store) as kernel:
                agent = create_agent(
                    model,
                    create_python_repl_tools(kernel),
                    checkpointer=InMemorySaver(),
                )
                result = agent.invoke(
                    {"messages": [HumanMessage(content="Recover and calculate.")]},
                    _config("snapshot-agent"),
                )

        incompatible, reset, retried = tool_results(result)
        self.assertFalse(incompatible["ok"])
        self.assertEqual(incompatible["error"]["type"], "SessionVersionError")
        self.assertIn("reset", incompatible["error"]["hint"].lower())
        self.assertTrue(reset["ok"])
        self.assertTrue(reset["reset"])
        self.assertEqual(retried["result"], 2)
        self.assertFalse(retried["state_restored"])


if __name__ == "__main__":
    unittest.main()
