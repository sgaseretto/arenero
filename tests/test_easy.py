"""Validate the high-level managed sandbox API with the real Monty runtime."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from typing import Any, cast

from arenero import (
    KernelConfig,
    SandboxConfig,
    SandboxManager,
    open_async_sandbox_manager,
    open_sandbox,
)


class SandboxConfigTests(unittest.TestCase):
    """Validate simple managed-sandbox presets."""

    def test_presets_select_expected_lifetimes_and_workers(self) -> None:
        """Presets set storage and worker fields without hidden behavior."""

        self.assertEqual(SandboxConfig.ephemeral().persistence, "stateless")
        self.assertEqual(SandboxConfig.conversation().filesystem, "memory")
        batch = SandboxConfig.batch(3)
        self.assertEqual(batch.kernel.min_worker_processes, 3)
        self.assertEqual(batch.filesystem, "none")
        with tempfile.TemporaryDirectory() as directory:
            durable = SandboxConfig.durable(directory)
            self.assertEqual(durable.persistence, "durable")
            self.assertEqual(durable.filesystem, "workspace")

    def test_invalid_related_settings_are_rejected(self) -> None:
        """Configuration rejects missing durable paths and unsafe roots."""

        with self.assertRaises(ValueError):
            SandboxConfig(persistence="durable")
        with self.assertRaises(ValueError):
            SandboxConfig(workspace_root="relative")
        with self.assertRaises(ValueError):
            SandboxConfig(max_workspace_files=0)
        with self.assertRaises(ValueError):
            SandboxConfig(network=cast(Any, "invalid"))
        with self.assertRaises(ValueError):
            SandboxConfig(bootstrap_files=("/outside.py",))


class ManagedSandboxTests(unittest.TestCase):
    """Validate bound execution, files, quotas, and checkpoints."""

    def test_bound_session_persists_and_relative_file_tools_work(self) -> None:
        """One handle reuses Python state and a relative working directory."""

        with open_sandbox("agent-one") as sandbox:
            self.assertTrue(sandbox.run_python("total = 40").ok)
            self.assertEqual(sandbox.run_python("total + 2").value, 42)
            self.assertTrue(sandbox.mkdir("notes").ok)
            self.assertTrue(sandbox.chdir("notes").ok)
            written = sandbox.write("value.txt", "forty two")
            self.assertTrue(written.ok)
            self.assertEqual(sandbox.read("value.txt").value, "forty two")
            self.assertEqual(sandbox.run_shell("cat value.txt").value, "forty two")
            self.assertTrue(sandbox.run_cell("%limits").ok)
            escaped = sandbox.read("../../outside.txt")
            assert escaped.error is not None
            self.assertEqual(escaped.error.code, "file_read_failed")

    def test_trusted_bootstrap_runs_for_each_new_session(self) -> None:
        """Managed configuration forwards trusted startup code to the kernel."""

        config = SandboxConfig.conversation(bootstrap_code="base = 40")
        with SandboxManager(config) as manager:
            self.assertEqual(manager.open("one").run_python("base + 2").value, 42)
            self.assertEqual(manager.open("two").run_python("base + 3").value, 43)

    def test_ephemeral_state_is_fresh_but_runtime_files_remain(self) -> None:
        """The ephemeral preset discards Python state, not scratch files."""

        with open_sandbox("ephemeral", config=SandboxConfig.ephemeral()) as sandbox:
            sandbox.run_python("temporary = 42")
            self.assertFalse(sandbox.run_python("temporary").ok)
            sandbox.write("scratch.txt", "kept")
            self.assertEqual(sandbox.read("scratch.txt").value, "kept")

    def test_python_only_batch_checkpoint_and_tools_match_capabilities(self) -> None:
        """Python-only mode restores state and omits unusable agent tools."""

        with open_sandbox("python-only", config=SandboxConfig.batch(2)) as sandbox:
            sandbox.run_python("value = 1")
            checkpoint = sandbox.checkpoint().value
            sandbox.run_python("value = 2")
            self.assertTrue(sandbox.restore(checkpoint).ok)
            self.assertEqual(sandbox.run_python("value").value, 1)
            self.assertEqual(
                [tool.name for tool in sandbox.agent_bundle().tools],
                ["python_repl"],
            )
            self.assertIn("No virtual filesystem", sandbox.instructions())
            with self.assertRaises(ValueError):
                sandbox.langchain_tools({"files"})

    def test_file_revisions_prevent_stale_edits(self) -> None:
        """A stale revision returns one stable conflict error."""

        with open_sandbox("revision") as sandbox:
            first = sandbox.write("draft.txt", "first")
            revision = first.stats["revision"]
            sandbox.write("draft.txt", "second")
            stale = sandbox.edit(
                "draft.txt",
                "second",
                "third",
                expected_revision=str(revision),
            )
            self.assertFalse(stale.ok)
            assert stale.error is not None
            self.assertEqual(stale.error.code, "file_revision_conflict")

    def test_workspace_quota_covers_python_and_direct_writes(self) -> None:
        """Quota failures reject direct writes and roll back code-created files."""

        config = SandboxConfig.conversation(
            max_workspace_files=2,
            max_workspace_bytes=5,
        )
        with open_sandbox("quota", config=config) as sandbox:
            result = sandbox.run_python("fs_write('/workspace/large.txt', '123456')")
            self.assertFalse(result.ok)
            assert result.error is not None
            self.assertEqual(result.error.code, "workspace_quota_exceeded")
            self.assertFalse(sandbox.exists("large.txt").value)
            self.assertTrue(sandbox.write("small.txt", "12345").ok)
            self.assertFalse(sandbox.write("other.txt", "1").ok)

    def test_checkpoint_restores_state_files_and_working_directory(self) -> None:
        """One checkpoint restores the two separate persistence planes."""

        with open_sandbox("checkpoint") as sandbox:
            sandbox.run_python("answer = 42")
            sandbox.mkdir("data")
            sandbox.chdir("data")
            sandbox.write("answer.txt", "42")
            checkpoint = sandbox.checkpoint()
            self.assertTrue(checkpoint.ok)
            sandbox.run_python("answer = 0")
            sandbox.write("answer.txt", "0")
            self.assertTrue(sandbox.restore(checkpoint.value).ok)
            self.assertEqual(sandbox.run_python("answer").value, 42)
            self.assertEqual(sandbox.read("answer.txt").value, "42")
            self.assertEqual(sandbox.cwd, "/workspace/data")

    def test_checkpoint_moves_between_memory_and_durable_presets(self) -> None:
        """Portable checkpoints do not depend on the selected storage backend."""

        with open_sandbox("source") as source:
            source.run_python("saved = 'yes'")
            source.write("saved.txt", "yes")
            checkpoint = source.checkpoint().value
        with (
            tempfile.TemporaryDirectory() as directory,
            open_sandbox(
                "target",
                config=SandboxConfig.durable(directory),
            ) as target,
        ):
            self.assertTrue(target.restore(checkpoint).ok)
            self.assertEqual(target.run_python("saved").value, "yes")
            self.assertEqual(target.read("saved.txt").value, "yes")

    def test_checkpoint_rejects_another_execution_policy(self) -> None:
        """Restore rejects state created with different execution limits."""

        with open_sandbox("source-policy") as source:
            checkpoint = source.checkpoint().value
        config = SandboxConfig.conversation(kernel=KernelConfig(max_duration_seconds=5))
        with open_sandbox("target-policy", config=config) as target:
            result = target.restore(checkpoint)
            self.assertFalse(result.ok)
            assert result.error is not None
            self.assertEqual(result.error.code, "checkpoint_restore_failed")

    def test_tree_archives_are_deterministic_and_reject_unsafe_paths(self) -> None:
        """Tree transfer uses bounded safe ZIP paths and stable output."""

        with open_sandbox("archive") as sandbox:
            sandbox.write("b.txt", "b")
            sandbox.write("a.txt", "a")
            first = sandbox.export_tree().value
            second = sandbox.export_tree().value
            self.assertEqual(first, second)
            sandbox.delete("a.txt")
            sandbox.delete("b.txt")
            self.assertTrue(sandbox.import_tree(first).ok)
            self.assertEqual(sandbox.read("a.txt").value, "a")

            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as output:
                output.writestr("../escape.txt", "bad")
            rejected = sandbox.import_tree(stream.getvalue())
            self.assertFalse(rejected.ok)
            assert rejected.error is not None
            self.assertEqual(rejected.error.code, "tree_import_failed")

    def test_host_tree_transfer_round_trips_binary_files(self) -> None:
        """Host tree upload and download preserve binary content."""

        with tempfile.TemporaryDirectory() as source_directory:
            source = Path(source_directory)
            (source / "nested").mkdir()
            (source / "nested" / "image.bin").write_bytes(b"\x00\xffimage")
            with open_sandbox("transfer") as sandbox:
                uploaded = sandbox.upload_tree(source)
                self.assertTrue(uploaded.ok)
                with tempfile.TemporaryDirectory() as destination:
                    downloaded = sandbox.download_tree(destination)
                    self.assertTrue(downloaded.ok)
                    self.assertEqual(
                        (Path(destination) / "nested" / "image.bin").read_bytes(),
                        b"\x00\xffimage",
                    )

    def test_package_workflow_uses_relative_paths(self) -> None:
        """The managed package workflow creates, tests, and exports source."""

        with open_sandbox("package") as sandbox:
            self.assertTrue(sandbox.package_init("mathkit").ok)
            checked = sandbox.package_check("mathkit")
            self.assertTrue(checked.ok)
            tested = sandbox.package_test("mathkit", checked.value["revision"])
            self.assertTrue(tested.ok)
            exported = sandbox.package_export("mathkit", checked.value["revision"])
            self.assertTrue(exported.ok)
            self.assertTrue(exported.artifacts[0].endswith("-arenero.zip"))

    def test_common_result_serializes_binary_values(self) -> None:
        """Binary archive results use an explicit Base64 JSON record."""

        with open_sandbox("json-result") as sandbox:
            sandbox.write("value.bin", b"\x00\xff")
            value = sandbox.export_tree().to_dict()
            serialized = json.dumps(value)
            self.assertIn('"encoding": "base64"', serialized)


class SandboxManagerTests(unittest.TestCase):
    """Validate shared runtime ownership and async access."""

    def test_manager_isolates_state_shares_files_and_bounds_handles(self) -> None:
        """Manager sessions isolate Python and share one configured workspace."""

        config = SandboxConfig.conversation(max_active_sandboxes=2)
        with SandboxManager(config) as manager:
            first = manager.open("first")
            second = manager.open("second")
            first.run_python("private = 1")
            self.assertFalse(second.run_python("private").ok)
            first.write("shared.txt", "visible")
            self.assertEqual(second.read("shared.txt").value, "visible")
            with self.assertRaises(RuntimeError):
                manager.open("third")

    def test_fork_copies_python_state_and_keeps_shared_files(self) -> None:
        """A managed fork gets copied Python state and the same files."""

        with open_sandbox("fork-source") as source:
            source.run_python("value = 41")
            source.write("shared.txt", "before")
            forked = source.fork("fork-child")
            self.assertTrue(forked.ok)
            child = forked.value
            self.assertEqual(child.run_python("value + 1").value, 42)
            child.write("shared.txt", "after")
            self.assertEqual(source.read("shared.txt").value, "after")

    def test_cleanup_removes_only_expired_saved_handles(self) -> None:
        """Cleanup keeps an opened handle that has no saved Python state."""

        with SandboxManager() as manager:
            manager.open("empty")
            manager.open("saved").run_python("x = 1")
            time.sleep(0.001)
            self.assertEqual(manager.cleanup_expired(0), 1)
            self.assertEqual(manager.list(), ["empty"])

    def test_durable_preset_survives_runtime_restart(self) -> None:
        """SQLite state and mounted files survive a new owned runtime."""

        with tempfile.TemporaryDirectory() as directory:
            config = SandboxConfig.durable(directory)
            with open_sandbox("durable", config=config) as first:
                first.run_python("value = 7")
                first.write("value.txt", "7")
            with open_sandbox("durable", config=config) as second:
                self.assertEqual(second.run_python("value").value, 7)
                self.assertEqual(second.read("value.txt").value, "7")

    def test_async_manager_runs_distinct_sessions_and_file_calls(self) -> None:
        """Async handles use the shared worker pool without blocking callers."""

        async def scenario() -> None:
            """Run two independent async sessions."""

            config = SandboxConfig.batch(
                2,
                kernel=KernelConfig(request_timeout_seconds=5),
                filesystem="memory",
            )
            async with open_async_sandbox_manager(config) as manager:
                first, second = await asyncio.gather(
                    manager.open("async-first"),
                    manager.open("async-second"),
                )
                results = await asyncio.gather(
                    first.run_python("40 + 2"),
                    second.run_python("20 + 2"),
                )
                self.assertEqual([item.value for item in results], [42, 22])
                written = await first.call("write", "shared.txt", "async")
                self.assertTrue(written.ok)
                read = await second.call("read", "shared.txt")
                self.assertEqual(read.value, "async")
                self.assertEqual(await manager.list(), ["async-first", "async-second"])

        asyncio.run(scenario())

    def test_default_async_batch_runs_python_sessions_concurrently(self) -> None:
        """Two Python-only sessions can enter host calls at the same time."""

        barrier = threading.Barrier(2)

        def meet(value: int) -> int:
            """Wait until both Monty workers call the host.

            Args:
                value: Value returned after both calls arrive.

            Returns:
                Original value.
            """

            barrier.wait(timeout=2)
            return value

        async def scenario() -> None:
            """Run two calls that require real worker overlap."""

            async with open_async_sandbox_manager(
                SandboxConfig.batch(2),
                external_lookup={"meet": meet},
            ) as manager:
                first, second = await asyncio.gather(
                    manager.open("parallel-first"),
                    manager.open("parallel-second"),
                )
                results = await asyncio.gather(
                    first.run_python("meet(1)"),
                    second.run_python("meet(2)"),
                )
                self.assertEqual([item.value for item in results], [1, 2])

        asyncio.run(scenario())


class ManagedLangChainToolTests(unittest.TestCase):
    """Validate the combined fixed-session agent tool bundle."""

    def test_tools_share_one_session_and_common_result_contract(self) -> None:
        """Python, shell, file, and package tools bind to one sandbox."""

        with open_sandbox("tools") as sandbox:
            bundle = sandbox.agent_bundle()
            tools = {tool.name: tool for tool in bundle.tools}
            self.assertIn("Sandbox ID: tools", bundle.instructions)
            self.assertEqual(
                set(tools),
                {
                    "delete_file",
                    "edit_file",
                    "execute",
                    "glob_files",
                    "grep_files",
                    "list_files",
                    "package_check",
                    "package_export",
                    "package_init",
                    "package_test",
                    "python_repl",
                    "read_file",
                    "write_file",
                },
            )
            tools["python_repl"].invoke({"code": "value = 41"})
            result = tools["python_repl"].invoke({"code": "value + 1"})
            self.assertEqual(result["value"], 42)
            tools["write_file"].invoke({"path": "agent.txt", "content": "ok"})
            shell = tools["execute"].invoke({"command": "cat agent.txt"})
            self.assertEqual(shell["value"], "ok")
            self.assertIn("not an operating-system shell", tools["execute"].description)


if __name__ == "__main__":
    unittest.main()
