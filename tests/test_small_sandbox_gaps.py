"""Validate the ten small sandbox changes with real Monty and mocked agents."""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent
from pydantic_monty import Monty, OSAccess

from arenero import (
    AreneroSandbox,
    AsyncAreneroSandbox,
    AsyncPersistentMontyKernel,
    FilesystemConfig,
    InMemorySessionStore,
    KernelConfig,
    PersistentMontyKernel,
    SQLiteSessionStore,
    decode_value,
    open_async_sandbox,
    open_sandbox,
)
from arenero.deepagents import AreneroInterpreterMiddleware
from arenero.mcp import McpHub, McpServerConfig
from tests.test_langchain_agent import (
    ToolCallingFakeChatModel,
    named_tool_call,
    tool_results,
)


def _blob(payload: bytes, kind: str = "image") -> dict[str, str]:
    """Build a normalized MCP binary content block.

    Args:
        payload: Raw bytes to encode.
        kind: Image or audio content type.

    Returns:
        Plain MCP content data with Base64 bytes and a media type.
    """
    return {
        "type": kind,
        "data": base64.b64encode(payload).decode(),
        "mimeType": "image/png" if kind == "image" else "audio/wav",
    }


class SmallSandboxTests(unittest.TestCase):
    """Check sync behavior, common helpers, and public wrappers."""

    def test_compound_module_exports_and_metadata(self) -> None:
        """Export only bound module-scope names, not function/class local names."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            kernel.upload_file(
                "/helper.py",
                b"A, (B, *REST) = (1, (2, 3, 4))\nif True:\n    C = 5\nif False:\n    ABSENT = 99\nfor I in [6]:\n    D = I\ntry:\n    E = 7\nexcept ValueError:\n    UNUSED = 1\ndef local():\n    LOCAL = 1\n    return LOCAL\nclass Holder:\n    CLASS_LOCAL = 1\nCOMP = [item for item in [8]]\n",
            )
            kernel.upload_file(
                "/main.py",
                b"from helper import A, B, REST, C, D, E\nimport helper\nRESULT = [A, B, REST, C, D, E]\n",
            )
            result = kernel.execute_module(
                "agent",
                "/main.py",
                "[RESULT, helper.__file__, helper.__name__, main.__file__, hasattr(helper, 'ABSENT'), hasattr(helper, 'LOCAL'), hasattr(helper, 'CLASS_LOCAL'), hasattr(helper, 'item')]",
            )
            self.assertTrue(result["ok"], result["error"])
            self.assertEqual(
                result["result"],
                [
                    [1, 2, [3, 4], 5, 6, 7],
                    "/helper.py",
                    "helper",
                    "/main.py",
                    False,
                    False,
                    False,
                    False,
                ],
            )
            self.assertEqual(
                kernel.execute("agent", "main.RESULT")["result"],
                [1, 2, [3, 4], 5, 6, 7],
            )

    def test_flat_file_metadata_is_native_and_checked(self) -> None:
        """Bind file metadata through Monty inputs without changing source syntax."""
        with PersistentMontyKernel(
            os_handler=OSAccess(), config=KernelConfig(type_check=True)
        ) as kernel:
            kernel.upload_file(
                "/dir/helper.py",
                b"from pathlib import Path\ndef nearby():\n    return Path(__file__).parent.joinpath('data.bin').read_bytes()\n",
            )
            kernel.upload_file("/dir/data.bin", b"\x00\xff")
            kernel.execute("agent", "import os\nos.chdir('/dir')")
            result = kernel.execute_file(
                "agent", "helper.py", "[nearby(), __file__, __name__]"
            )
            self.assertTrue(result["ok"], result["error"])
            self.assertEqual(
                decode_value(result["result"]),
                [b"\x00\xff", "/dir/helper.py", "__main__"],
            )
            self.assertEqual(
                kernel.execute("agent", "__file__")["result"], "/dir/helper.py"
            )

    def test_grep_options_reach_facades_and_agent_tools(self) -> None:
        """Keep literal defaults and reuse bounded native regex/case selection."""
        with open_sandbox() as sandbox:
            sandbox.write("notes.txt", "Alpha\nALPHA\nBeta\n")
            result = sandbox.grep("a.*a", regex=True, case_sensitive=False, limit=1)
            self.assertTrue(result.ok, result.error)
            self.assertEqual(len(result.value), 1)
            self.assertTrue(result.truncated)
            self.assertEqual(sandbox.grep("a.*a").value, [])
            facade = AreneroSandbox(sandbox.kernel, sandbox.session_id)
            self.assertEqual(
                len(facade.grep("alpha", "/workspace", case_sensitive=False).matches), 2
            )
            self.assertIsNotNone(facade.grep("[", "/workspace", regex=True).error)
            tools = sandbox.langchain_tools(include={"files"})
            grep = next(tool for tool in tools if tool.name == "grep_files")
            result = grep.invoke(
                {"query": "alpha", "case_sensitive": False, "limit": 1}
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["truncated"])

    def test_inspection_is_bounded_and_does_not_save(self) -> None:
        """Select public values without bootstrap, callbacks, or changed revisions."""
        calls = []
        with PersistentMontyKernel(
            bootstrap_code="ping()", external_lookup={"ping": lambda: calls.append(1)}
        ) as kernel:
            session = kernel.session("agent")
            session.execute("number = 42\nblob = b'OK'\nlarge = 'x' * 100\n_secret = 1")
            before = kernel.store.load("agent")
            result = session.inspect_values(
                ["number", "blob", "missing", "fs_write", "ping"]
            )
            self.assertEqual(
                decode_value(result["values"]), {"number": 42, "blob": b"OK"}
            )
            limited = session.inspect_values(["large"], 5)
            self.assertTrue(limited["truncated"])
            self.assertLessEqual(len(limited["values"]), 5)
            self.assertEqual(before, kernel.store.load("agent"))
            self.assertEqual(calls, [1])
            for names in [["number + 1"], ["_secret"], ["number"] * 101]:
                with self.subTest(names=names[:1]), self.assertRaises(ValueError):
                    session.inspect_values(names)
            self.assertEqual(kernel.inspect_values("missing", ["number"])["values"], {})

    def test_decode_value_round_trip_is_explicit_and_strict(self) -> None:
        """Decode exact records only, reject malformed payloads, and preserve input."""
        value = {
            "bytes": {"encoding": "base64", "content": "T0s="},
            "complex": {"type": "complex", "real": 1, "imag": 2},
            "ordinary": {"encoding": "base64", "content": "T0s=", "note": "keep"},
        }
        decoded = decode_value(value)
        self.assertEqual(decoded["bytes"], b"OK")
        self.assertEqual(decoded["complex"], 1 + 2j)
        self.assertEqual(decoded["ordinary"], value["ordinary"])
        self.assertEqual(value["bytes"]["content"], "T0s=")
        for record in [
            {"encoding": "base64", "content": "!"},
            {"type": "complex", "real": True, "imag": 2},
        ]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                decode_value(record)
        with PersistentMontyKernel() as kernel:
            result = kernel.execute(
                "agent",
                "[blob == b'OK', number == 1 + 2j]",
                inputs={"blob": decoded["bytes"], "number": decoded["complex"]},
            )
            self.assertEqual(result["result"], [True, True])

    def test_python_only_calls_overlap_and_deny_direct_files(self) -> None:
        """Skip the shared file lock only when mounts/callbacks are detached."""
        barrier = threading.Barrier(2)

        def rendezvous() -> int:
            """Meet the other execution within a bounded wait.

            Returns:
                The integer 42 when both worker calls overlap.
            """
            barrier.wait(timeout=3)
            return 42

        config = KernelConfig(min_worker_processes=2, max_worker_processes=2)
        with PersistentMontyKernel(
            config=config,
            os_handler=OSAccess(),
            external_lookup={"rendezvous": rendezvous},
            mcp_artifact_directory="/artifacts",
        ) as kernel:
            kernel.upload_file("/secret.txt", b"secret")
            with ThreadPoolExecutor(2) as executor:
                futures = [
                    executor.submit(
                        kernel.session(name).execute, "rendezvous()", filesystem=False
                    )
                    for name in ("a", "b")
                ]
                self.assertEqual(
                    [future.result(timeout=5)["result"] for future in futures], [42, 42]
                )
            for source in ["open('/secret.txt').read()", "fs_read('/secret.txt')"]:
                with self.subTest(source=source):
                    self.assertFalse(
                        kernel.execute("a", source, filesystem=False)["ok"]
                    )
            result = kernel.execute(
                "a",
                "'x' * 100",
                filesystem=False,
                output_limit_chars=1,
                output_file="/forbidden.json",
            )
            self.assertTrue(result["ok"])
            self.assertIsNotNone(result["output_file_error"])
            result = kernel.execute(
                "a", "blob", inputs={"blob": _blob(b"OK")}, filesystem=False
            )
            self.assertEqual(result["mcp_artifacts"], [])
            self.assertIn("data", result["result"])

    def test_managed_python_only_mode_skips_file_work(self) -> None:
        """Forward the flag without quota scans or indirect path inspection."""
        with open_sandbox() as sandbox:
            with patch.object(
                sandbox, "_guard_workspace", side_effect=AssertionError("must not scan")
            ):
                self.assertEqual(
                    sandbox.run_python("answer = 42\nanswer", filesystem=False).value,
                    42,
                )
            self.assertEqual(sandbox.inspect_values(["answer"]).value, {"answer": 42})
            self.assertEqual(sandbox.session_policy().value["status"], "match")
            facade = AreneroSandbox(sandbox.kernel, sandbox.session_id)
            self.assertEqual(
                facade.execute_python("answer", filesystem=False).exit_code, 0
            )

    def test_policy_survives_sqlite_fork_and_rejects_drift(self) -> None:
        """Keep provenance in snapshots without a schema change or implicit reset."""
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "state.db")
            with PersistentMontyKernel(store) as kernel:
                session = kernel.session("agent")
                session.execute("answer = 42")
                self.assertEqual(session.policy()["status"], "match")
                self.assertEqual(
                    session.fork("child").policy()["saved"], session.policy()["saved"]
                )
            config = replace(KernelConfig(), max_memory_bytes=32 * 1024 * 1024)
            with PersistentMontyKernel(store, config=config) as kernel:
                before = store.load("agent")
                report = kernel.session("agent").policy()
                self.assertEqual(report["status"], "changed")
                self.assertTrue(report["reset_required"])
                result = kernel.execute("agent", "answer = 0")
                self.assertEqual(result["error"]["code"], "session_policy_changed")
                self.assertEqual(result["error"]["recovery"], "restart_session")
                self.assertEqual(store.load("agent"), before)
                self.assertEqual(
                    kernel.execute("agent", "answer = 7\nanswer", restart=True)[
                        "result"
                    ],
                    7,
                )
                self.assertEqual(kernel.session("agent").policy()["status"], "match")

    def test_legacy_policy_stays_unknown_and_host_caps_can_change(self) -> None:
        """Do not invent provenance for old dumps or block output-limit changes."""
        store = InMemorySessionStore()
        with Monty() as pool, pool.checkout() as session:
            session.feed_run("answer = 42")
            state = session.dump()
        with PersistentMontyKernel(store) as kernel:
            store.save("legacy", state, kernel.monty_version, None)
            self.assertEqual(kernel.session_policy("legacy")["status"], "unknown")
            self.assertEqual(kernel.execute("legacy", "answer")["result"], 42)
            self.assertEqual(kernel.session_policy("legacy")["status"], "unknown")
            kernel.execute("new", "answer = 42")
        with PersistentMontyKernel(
            store, config=replace(KernelConfig(), default_output_limit_chars=1)
        ) as kernel:
            self.assertEqual(kernel.session_policy("new")["status"], "match")
            self.assertTrue(kernel.execute("new", "answer")["ok"])

    def test_all_interpreter_policy_changes_require_reset(self) -> None:
        """Detect limits, OS policy, checking mode, and stable stub changes."""
        original = KernelConfig(type_check=True, type_check_stubs="value: int")
        variants = [
            replace(original, max_feed_duration_seconds=1),
            replace(original, os_policy={"random_start": {"seed": 7}}),
            replace(original, type_check=False),
            replace(original, type_check_stubs="value: float"),
        ]
        for config in variants:
            with self.subTest(config=config):
                store = InMemorySessionStore()
                with PersistentMontyKernel(store, config=original) as kernel:
                    self.assertTrue(kernel.execute("agent", "value = 42")["ok"])
                before = store.load("agent")
                with PersistentMontyKernel(store, config=config) as kernel:
                    failed = kernel.execute("agent", "value = 0")
                    self.assertEqual(failed["error"]["code"], "session_policy_changed")
                    self.assertEqual(failed["session_revision"], before.revision)
                    self.assertEqual(store.load("agent"), before)

    def test_python_only_bootstrap_skips_files_until_reset(self) -> None:
        """Run startup code but never add skipped file bootstrap on restore."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            bootstrap_code="marker = 1",
            bootstrap_files=("/startup.py",),
        ) as kernel:
            kernel.upload_file("/startup.py", b"seed = 42")
            result = kernel.execute(
                "agent", "[marker, 'seed' in locals()]", filesystem=False
            )
            self.assertEqual(result["result"], [1, False])
            self.assertFalse(kernel.execute("agent", "seed")["ok"])
            self.assertEqual(
                kernel.execute("agent", "seed", restart=True)["result"], 42
            )


class AsyncSmallSandboxTests(unittest.IsolatedAsyncioTestCase):
    """Check native async utilities, artifacts, and the complete agent graph."""

    async def test_actual_mcp_image_call_materializes_on_native_async(self) -> None:
        """Drive the real MCP client bridge, decoder, and one-worker file writer."""
        server = MCPServer("media")

        @server.tool()
        def image() -> list[ImageContent]:
            """Return deterministic image bytes through MCP.

            Returns:
                One image content block with a Base64 payload.
            """
            return [ImageContent(type="image", data="AP8=", mimeType="image/png")]

        with McpHub([McpServerConfig.in_process("media", server)]) as hub:
            async with AsyncPersistentMontyKernel(
                mcp_hub=hub,
                os_handler=OSAccess(),
                mcp_artifact_directory="/artifacts",
                config=KernelConfig(min_worker_processes=1, max_worker_processes=1),
            ) as kernel:
                result = await asyncio.wait_for(
                    kernel.execute(
                        "agent", "await mcp_call('media__image')", output_limit_chars=0
                    ),
                    timeout=3,
                )
                self.assertTrue(result["ok"], result["error"])
                self.assertTrue(result["mcp_artifacts"])
                for path in result["mcp_artifacts"]:
                    self.assertEqual(await kernel.download_file(path), b"\x00\xff")

    async def test_async_artifact_write_holds_session_lock(self) -> None:
        """Keep session ordering until its predecessor saves artifacts."""
        async with AsyncPersistentMontyKernel(
            os_handler=OSAccess(),
            mcp_artifact_directory="/artifacts",
            config=KernelConfig(min_worker_processes=1, max_worker_processes=1),
        ) as kernel:
            started, release = asyncio.Event(), asyncio.Event()
            original = kernel.upload_file

            async def delayed_write(path: str, payload: bytes) -> int:
                """Wait at the artifact boundary, then use the real writer.

                Args:
                    path: Generated virtual destination.
                    payload: Decoded artifact bytes.

                Returns:
                    The real upload byte count.
                """
                started.set()
                await release.wait()
                return await original(path, payload)

            with patch.object(kernel, "upload_file", delayed_write):
                first = asyncio.create_task(
                    kernel.execute(
                        "agent", "answer = 42\nblock", inputs={"block": _blob(b"OK")}
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                second = asyncio.create_task(
                    kernel.execute("agent", "answer", filesystem=False)
                )
                await asyncio.sleep(0)
                self.assertFalse(second.done())
                release.set()
                self.assertTrue((await asyncio.wait_for(first, timeout=2))["ok"])
                self.assertEqual(
                    (await asyncio.wait_for(second, timeout=2))["result"], 42
                )

    async def test_native_async_utilities_and_partial_batches(self) -> None:
        """Reuse file helpers and ordered transfer results with per-file limits."""
        async with AsyncPersistentMontyKernel(
            os_handler=OSAccess(),
            filesystem_config=FilesystemConfig(max_transfer_bytes=3),
        ) as kernel:
            uploaded = await kernel.upload_files(
                [
                    ("/one.bin", b"\x00\xff"),
                    ("relative", b"x"),
                    ("/large.bin", b"1234"),
                    ("/two.bin", b"OK"),
                ]
            )
            self.assertEqual(
                [result.error is None for result in uploaded],
                [True, False, False, True],
            )
            downloaded = await kernel.download_files(
                ["/one.bin", "/missing", "/two.bin"]
            )
            self.assertEqual(downloaded[0].content, b"\x00\xff")
            self.assertIsNotNone(downloaded[1].error)
            self.assertEqual(downloaded[2].content, b"OK")
            result = await kernel.call_filesystem(
                "agent", "write", {"path": "/text.txt", "content": "Alpha\nALPHA"}
            )
            self.assertTrue(result["ok"], result["error"])
            result = await kernel.call_filesystem(
                "agent",
                "grep",
                {"query": "alpha", "case_sensitive": False, "pattern": "**/*.txt"},
            )
            self.assertTrue(result["ok"], result["error"])
            self.assertEqual(len(result["result"]["matches"]), 2)
            with patch("arenero.store.time.time", return_value=10**12):
                self.assertEqual(await kernel.delete_expired(), 1)
            self.assertEqual(await kernel.list_sessions(), [])

    async def test_native_metadata_inspection_policy_and_decode(self) -> None:
        """Keep all common data APIs consistent on native async workers."""
        store = InMemorySessionStore()
        async with AsyncPersistentMontyKernel(store, os_handler=OSAccess()) as kernel:
            await kernel.upload_file("/helper.py", b"value = b'OK'\nPATH = __file__\n")
            session = kernel.session("agent")
            result = await session.execute_file("/helper.py", "[PATH, __name__, value]")
            self.assertEqual(
                decode_value(result["result"]), ["/helper.py", "__main__", b"OK"]
            )
            before = store.load("agent")
            self.assertEqual(
                decode_value((await session.inspect_values(["value"]))["values"]),
                {"value": b"OK"},
            )
            self.assertEqual((await session.policy())["status"], "match")
            self.assertEqual(before, store.load("agent"))
            self.assertTrue((await session.execute("1 + 1", filesystem=False))["ok"])
        async with AsyncPersistentMontyKernel(
            store, config=replace(KernelConfig(), max_recursion_depth=50)
        ) as kernel:
            self.assertEqual(
                (await kernel.session_policy("agent"))["status"], "changed"
            )
            result = await kernel.execute("agent", "value")
            self.assertEqual(result["error"]["code"], "session_policy_changed")
            self.assertEqual(
                (await kernel.execute("agent", "42", restart=True))["result"], 42
            )

    async def test_async_artifacts_use_raw_results_after_checkout(self) -> None:
        """Save final images/audio/resources with one worker and tiny output caps."""
        config = KernelConfig(min_worker_processes=1, max_worker_processes=1)
        async with AsyncPersistentMontyKernel(
            config=config, os_handler=OSAccess(), mcp_artifact_directory="/artifacts"
        ) as kernel:
            value = [
                _blob(b"\x00\xff"),
                _blob(b"audio", "audio"),
                {
                    "type": "resource",
                    "resource": {
                        "blob": "T0s=",
                        "mimeType": "application/octet-stream",
                    },
                },
            ]
            result = await asyncio.wait_for(
                kernel.execute(
                    "agent",
                    "answer = 42\nblocks",
                    inputs={"blocks": value},
                    output_limit_chars=0,
                    output_file="/full.json",
                ),
                timeout=3,
            )
            self.assertTrue(result["ok"], result["error"])
            self.assertEqual(len(result["mcp_artifacts"]), 3)
            self.assertEqual(
                [await kernel.download_file(path) for path in result["mcp_artifacts"]],
                [b"\x00\xff", b"audio", b"OK"],
            )
            full = json.loads(await kernel.download_file("/full.json"))
            self.assertNotIn("data", full["result"][0])
            self.assertIn("virtual_path", full["result"][0])
            self.assertIn("data", value[0])
            self.assertEqual((await kernel.execute("agent", "answer"))["result"], 42)

    async def test_async_artifact_failures_and_cancellation_preserve_state(
        self,
    ) -> None:
        """Keep failed payloads and propagate cancellation after Python commit."""
        async with AsyncPersistentMontyKernel(
            os_handler=OSAccess(),
            mcp_artifact_directory="/artifacts",
            filesystem_config=FilesystemConfig(max_transfer_bytes=1),
        ) as kernel:
            result = await kernel.execute(
                "agent", "answer = 42\nblock", inputs={"block": _blob(b"OK")}
            )
            self.assertTrue(result["ok"])
            self.assertIn("artifact_error", result["result"])
            self.assertIn("data", result["result"])
            result = await kernel.execute("agent", "block", filesystem=False)
            self.assertEqual(result["mcp_artifacts"], [])
            copied, paths = await kernel.materialize_mcp_artifacts(
                {"type": "image", "data": "!"}
            )
            self.assertEqual(paths, [])
            self.assertIn("artifact_error", copied)
            with (
                patch.object(
                    kernel, "upload_file", AsyncMock(side_effect=asyncio.CancelledError)
                ),
                self.assertRaises(asyncio.CancelledError),
            ):
                await kernel.execute("agent", "answer = 7\nblock")
            self.assertEqual((await kernel.execute("agent", "answer"))["result"], 7)

    async def test_threaded_wrappers_forward_grep_and_python_only_flags(self) -> None:
        """Reuse sync behavior in the complete async managed/facade surface."""
        async with open_async_sandbox() as sandbox:
            await sandbox.call("write", "notes.txt", "Alpha\nALPHA")
            result = await sandbox.call(
                "grep", "alpha", regex=True, case_sensitive=False, limit=1
            )
            self.assertTrue(result.truncated)
            self.assertEqual(
                (
                    await sandbox.run_python("answer = 42\nanswer", filesystem=False)
                ).value,
                42,
            )
            self.assertEqual(
                (await sandbox.call("inspect_values", ["answer"])).value, {"answer": 42}
            )
            facade = AsyncAreneroSandbox(sandbox._sandbox.kernel, sandbox.id)
            self.assertEqual(
                len(
                    (
                        await facade.grep("alpha", "/workspace", case_sensitive=False)
                    ).matches
                ),
                2,
            )
            self.assertEqual(
                (await facade.execute_python("42", filesystem=False)).exit_code, 0
            )

    async def test_deepagents_output_files_across_two_human_turns(self) -> None:
        """Use stable middleware tool names and persistent state in a mocked graph."""
        async with AsyncPersistentMontyKernel(os_handler=OSAccess()) as kernel:
            await kernel.upload_file("/helper.py", b"A, B = (20, 22)\nTOTAL = A + B\n")
            middleware = AreneroInterpreterMiddleware(kernel=kernel)
            model = ToolCallingFakeChatModel(
                messages=iter(
                    [
                        named_tool_call(
                            "first",
                            "python_interpreter",
                            {
                                "code": "answer = 40\n'x' * 100",
                                "output_limit_chars": 1,
                                "output_file": "/first.json",
                            },
                        ),
                        AIMessage(content="First turn done."),
                        named_tool_call(
                            "second",
                            "python_interpreter_module",
                            {
                                "entry_path": "/helper.py",
                                "code": "answer += 2\n[answer, TOTAL, 'x' * 100]",
                                "output_limit_chars": 1,
                                "output_file": "/second.json",
                            },
                        ),
                        AIMessage(content="The answer is 42."),
                    ]
                )
            )
            agent = create_agent(
                model, middleware=[middleware], checkpointer=InMemorySaver()
            )
            config = {"configurable": {"thread_id": "conversation"}}
            first = await agent.ainvoke(
                {"messages": [HumanMessage(content="Start a calculation.")]}, config
            )
            second = await agent.ainvoke(
                {"messages": [HumanMessage(content="Continue and load the helper.")]},
                config,
            )
            self.assertEqual(tool_results(first)[0]["output_file"], "/first.json")
            self.assertEqual(tool_results(second)[-1]["output_file"], "/second.json")
            full = json.loads(await kernel.download_file("/second.json"))
            self.assertEqual(full["result"][:2], [42, 42])
            self.assertEqual(len(await kernel.list_sessions()), 1)
            self.assertEqual(
                [tool.name for tool in middleware.tools],
                ["python_interpreter", "python_interpreter_module"],
            )
