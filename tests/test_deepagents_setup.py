"""Root-aware agent setup and honest file metadata with no real model calls."""

import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from deepagents import FilesystemPermission, create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic_monty import CallbackFile, MemoryFile, MountDir, OSAccess

from arenero import AreneroSandbox, KernelConfig, PersistentMontyKernel, WorkspaceConfig
from arenero.deepagents import CodeInterpreterMiddleware
from arenero.deepagents_backend import AreneroBackend, AreneroFilesBackend
from arenero.deepagents_setup import check_deepagents_setup, deepagents_settings
from tests.test_deepagents_backends import ObserveRequests, _messages
from tests.test_deepagents_compat import _model
from tests.test_langchain_agent import named_tool_call


class SetupTests(unittest.IsolatedAsyncioTestCase):
    """Use the setup helper with real workers and scripted agent graphs."""

    async def test_rooted_capture_and_readback_in_agent(self) -> None:
        """Keep native capture paths inside a non-root backend without rerunning."""
        with (
            tempfile.TemporaryDirectory() as directory,
            PersistentMontyKernel(
                workspace=WorkspaceConfig(directory),
                config=KernelConfig(default_output_limit_chars=40),
            ) as kernel,
        ):
            backend = AreneroBackend(AreneroSandbox(kernel))
            backend.write("/workspace/source", "start\n" + "x" * 1000 + "\nend")
            settings = deepagents_settings(backend, capture_output=True)
            self.assertIs(settings["backend"].default, backend)
            self.assertEqual(settings["backend"].routes, {})
            self.assertTrue(
                check_deepagents_setup(backend, artifacts_root="/workspace")["ok"]
            )
            capture = "/workspace/large_tool_results/run"
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "run",
                        "execute",
                        {"command": "cat source; echo once >> runs; false"},
                    ),
                    named_tool_call(
                        "read",
                        "read_file",
                        {"file_path": capture, "offset": 2, "limit": 1},
                    ),
                    AIMessage(content="done"),
                ),
                **settings,
            )
            state = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "run"}]}
            )
            messages = _messages(state)
            self.assertIn(capture, messages[0].content)
            self.assertIn("exit code 1", messages[0].content)
            self.assertIn("end", messages[1].content)
            self.assertEqual(Path(directory, "runs").read_bytes(), b"once\n")
            self.assertEqual(
                backend.read(capture).file_data["content"],
                "start\n" + "x" * 1000 + "\nend",
            )
            # Settings do not own resources or change an existing opt-in flag.
            deepagents_settings(backend)
            self.assertTrue(backend.enable_capture_offload)
            self.assertEqual(kernel.execute("host", "6 * 7")["result"], 42)

    async def test_permission_aware_interpreter_across_two_human_turns(self) -> None:
        """Preserve Python state and enforce file rules with the helper settings."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel), root="/work")
            backend.write("/work/public", "41")
            rules = [
                FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")
            ]
            async with CodeInterpreterMiddleware(
                ptc=("read_file", "write_file")
            ) as interpreter:
                settings = deepagents_settings(
                    backend, interpreter=interpreter, permissions=rules
                )
                self.assertIs(settings["permissions"], rules)
                report = check_deepagents_setup(
                    backend,
                    artifacts_root=settings["backend"].artifacts_root,
                    interpreter=interpreter,
                    permissions=rules,
                )
                self.assertTrue(report["ok"])
                self.assertTrue(all(check["ok"] for check in report["checks"]))
                observer = ObserveRequests()
                settings["middleware"].append(observer)
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            "first",
                            "eval",
                            {
                                "code": "value=41\nawait tools.read_file(file_path='/work/public')"
                            },
                        ),
                        AIMessage(content="stored"),
                        named_tool_call(
                            "second",
                            "eval",
                            {
                                "code": "value += 1\nblocked=await tools.write_file(file_path='/work/blocked', content=str(value))\n[value, blocked]"
                            },
                        ),
                        AIMessage(content="done"),
                    ),
                    checkpointer=InMemorySaver(),
                    **settings,
                )
                config = {"configurable": {"thread_id": "setup-two-turns"}}
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "remember"}]}, config
                )
                state = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "continue"}]}, config
                )
                value, blocked = _messages(state)[-1].artifact["result"]
                self.assertEqual(value, 42)
                self.assertIn("denied", str(blocked).lower())
                self.assertEqual(
                    backend.download_files(["/work/blocked"])[0].error, "file_not_found"
                )
                self.assertNotIn(
                    "execute", [tool.name for tool in observer.requests[0].tools]
                )

    async def test_combined_backend_avoids_duplicate_middleware(self) -> None:
        """Keep a direct file grant working and report its permission boundary."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel), root="/work")
            rules = [
                FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")
            ]
            async with CodeInterpreterMiddleware(backend=backend) as interpreter:
                settings = deepagents_settings(
                    backend, interpreter=interpreter, permissions=rules
                )
                self.assertEqual(settings["middleware"], [interpreter])
                report = check_deepagents_setup(
                    backend,
                    artifacts_root="/work",
                    interpreter=interpreter,
                    permissions=rules,
                )
                self.assertTrue(report["ok"])
                warning = next(
                    c
                    for c in report["checks"]
                    if c["name"] == "deepagents:direct_files"
                )
                self.assertFalse(warning["ok"])
                self.assertFalse(warning["required"])
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            "write",
                            "eval",
                            {
                                "code": "await fs_write(file_path='/work/direct', content='yes')"
                            },
                        ),
                        AIMessage(content="done"),
                    ),
                    **settings,
                )
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "write"}]}
                )
                self.assertEqual(
                    backend.download_files(["/work/direct"])[0].content, b"yes"
                )

    async def test_checks_are_read_only_and_report_invalid_profiles(self) -> None:
        """Detect bad roots, execution permissions, and separate direct grants."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel), root="/work")
            rules = [
                FilesystemPermission(operations=["read"], paths=["/**"], mode="deny")
            ]
            with patch.object(
                kernel, "execute", side_effect=AssertionError("no probe")
            ):
                report = check_deepagents_setup(backend, permissions=rules)
            self.assertFalse(report["ok"])
            self.assertEqual([c["ok"] for c in report["checks"]], [False, False])
            self.assertIn("limits", report["capabilities"])
            other = AreneroFilesBackend(AreneroSandbox(kernel), root="/other")
            async with CodeInterpreterMiddleware(backend=other) as interpreter:
                report = check_deepagents_setup(
                    backend, artifacts_root="/work", interpreter=interpreter
                )
                self.assertFalse(report["checks"][-1]["ok"])
            async with CodeInterpreterMiddleware(os_handler=OSAccess()) as interpreter:
                report = check_deepagents_setup(
                    other,
                    artifacts_root="/other",
                    interpreter=interpreter,
                    permissions=rules,
                )
                self.assertFalse(
                    next(
                        c
                        for c in report["checks"]
                        if c["name"] == "deepagents:direct_files"
                    )["ok"]
                )
            with self.assertRaisesRegex(ValueError, "AreneroBackend"):
                deepagents_settings(other, capture_output=True)
            self.assertFalse(backend.enable_capture_offload)
            self.assertEqual(kernel.list_sessions(), [])

    async def test_setup_check_does_not_open_configured_mcp_hubs(self) -> None:
        """Defer the normal capability probe when it could connect MCP servers."""
        with PersistentMontyKernel() as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            with (
                patch.object(kernel, "mcp_hub", Mock()),
                patch.object(
                    kernel, "capabilities", side_effect=AssertionError("no MCP probe")
                ),
            ):
                report = check_deepagents_setup(backend)
            self.assertTrue(report["ok"])
            self.assertIsNone(report["capabilities"])
            self.assertTrue(report["capabilities_deferred"])


class MetadataTests(unittest.TestCase):
    """Read native stat values without synthetic dates or extra file authority."""

    def test_mounted_metadata_tracks_real_updates_without_python_state(self) -> None:
        """Return byte sizes and UTC mtime, but never label ctime as creation."""
        with (
            tempfile.TemporaryDirectory() as directory,
            PersistentMontyKernel(workspace=WorkspaceConfig(directory)) as kernel,
        ):
            backend = AreneroBackend(AreneroSandbox(kernel))
            source = Path(directory, "data.txt")
            backend.write("/workspace/data.txt", "é🙂")
            os.utime(source, (1_700_000_000, 1_700_000_000))
            expected = datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat()
            for entries in (
                backend.ls("/workspace").entries,
                backend.glob("*.txt").matches,
            ):
                self.assertEqual(entries[0]["size"], len("é🙂".encode()))
                self.assertEqual(entries[0]["modified_at"], expected)
            data = backend.read("/workspace/data.txt").file_data
            self.assertEqual(data["modified_at"], expected)
            self.assertNotIn("created_at", data)
            self.assertEqual(kernel.list_sessions(), [])
            self.assertEqual(backend.execute("echo next > data.txt").exit_code, 0)
            self.assertEqual(backend.glob("*.txt").matches[0]["size"], 5)
            self.assertEqual(
                kernel.execute(
                    "python",
                    "from pathlib import Path\nPath('data.txt').write_bytes(b'new')",
                )["result"],
                3,
            )
            self.assertEqual(backend.ls("/workspace").entries[0]["size"], 3)
            self.assertEqual(
                kernel.execute("python", "'metadata' in locals()")["result"], False
            )

    def test_memory_sizes_have_no_dates_and_listing_does_not_read_callbacks(
        self,
    ) -> None:
        """Omit unsupported metadata without calling a lazy file reader."""
        read = Mock(return_value=b"callback data")
        handler = OSAccess(
            [
                MemoryFile("/text", "é"),
                MemoryFile("/binary", b"\x00\xff"),
                CallbackFile("/callback", read=read, write=Mock()),
            ]
        )
        with PersistentMontyKernel(os_handler=handler) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            entries = {e["path"]: e for e in backend.ls("/").entries}
            self.assertEqual(entries["/text"]["size"], 2)
            self.assertEqual(entries["/binary"]["size"], 2)
            self.assertNotIn("size", entries["/callback"])
            self.assertTrue(all("modified_at" not in e for e in entries.values()))
            read.assert_not_called()
            for path in ("/text", "/binary"):
                data = backend.read(path).file_data
                self.assertNotIn("created_at", data)
                self.assertNotIn("modified_at", data)
            self.assertEqual(
                backend.read("/missing", limit=0).file_data,
                {"content": "", "encoding": "utf-8"},
            )

    def test_metadata_failure_is_optional_and_symlinks_stay_confined(self) -> None:
        """Do not fail a listing or read when a stat fails, or expose host files."""
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as outside,
        ):
            target = Path(outside, "secret")
            target.write_bytes(b"secret" * 100)
            Path(directory, "link").symlink_to(target)
            with PersistentMontyKernel(workspace=WorkspaceConfig(directory)) as kernel:
                backend = AreneroFilesBackend(AreneroSandbox(kernel))
                backend.write("/workspace/public", "ok")
                link = backend._metadata(["/workspace/link"]).get("/workspace/link", {})
                self.assertNotEqual(link.get("size"), target.stat().st_size)
                self.assertIsNotNone(backend.read("/workspace/link").error)
                Path(directory, "link").unlink()
                run = kernel._run_isolated

                def fail_stat(code: str, **kwargs: object) -> object:
                    """Fail metadata only; retain normal read and scan execution.

                    Args:
                        code: Trusted file operation source.
                        **kwargs: Original isolated-call arguments.

                    Returns:
                        Normal result for non-metadata operations.
                    """
                    if "metadata = {}" in code:
                        raise RuntimeError("stat unavailable")
                    return run(code, **kwargs)

                with patch.object(kernel, "_run_isolated", side_effect=fail_stat):
                    self.assertIsNone(backend.ls("/workspace").error)
                    self.assertIsNone(backend.glob("*").error)
                    self.assertEqual(
                        backend.read("/workspace/public").file_data["content"], "ok"
                    )

    def test_metadata_uses_active_memory_entries_not_stale_registration(self) -> None:
        """Do not read a callback that replaces or renames over a memory file."""
        read = Mock(return_value="lazy content")
        handler = OSAccess(
            [
                MemoryFile("/source", "old"),
                CallbackFile("/source", read=read, write=Mock()),
                MemoryFile("/target", "old"),
            ]
        )
        with PersistentMontyKernel(os_handler=handler) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            self.assertNotIn("/source", backend._metadata(["/source"]))
            result = kernel.execute(
                "rename", "from pathlib import Path\nPath('/source').rename('/target')"
            )
            self.assertTrue(result["ok"])
            self.assertNotIn("/target", backend._metadata(["/target"]))
            backend.ls("/")
            read.assert_not_called()

    def test_overlay_omits_dates_and_sync_agent_helper_works(self) -> None:
        """Keep overlay times unknown and run the helper through sync invoke."""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "file").write_bytes(b"data")
            with (
                MountDir(
                    host_path=directory, virtual_path="/work", mode="overlay"
                ) as mount,
                PersistentMontyKernel(mounts=mount) as kernel,
            ):
                backend = AreneroBackend(AreneroSandbox(kernel), root="/work")
                self.assertEqual(backend.glob("*").matches[0]["size"], 4)
                self.assertNotIn("modified_at", backend.read("/work/file").file_data)
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            "sync", "execute", {"command": "cat /work/file"}
                        ),
                        AIMessage(content="done"),
                    ),
                    **deepagents_settings(backend, capture_output=True),
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": "read"}]}
                )
                self.assertIn("data", _messages(result)[0].content)
