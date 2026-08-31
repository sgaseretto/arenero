import concurrent.futures
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from pydantic_monty import MemoryFile, MountDir, OSAccess

from arenero import (
    FilesystemConfig,
    InMemorySessionStore,
    KernelConfig,
    PersistentMontyKernel,
    SQLiteSessionStore,
    StatelessSessionStore,
    WorkspaceConfig,
    execute_repl_cell,
)


class KernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "sessions.db"

    def make_kernel(
        self,
        config: KernelConfig | None = None,
        external_lookup: dict | None = None,
    ) -> PersistentMontyKernel:
        return PersistentMontyKernel(
            SQLiteSessionStore(self.database),
            config=config,
            external_lookup=external_lookup,
        )

    def test_state_survives_kernel_restart(self) -> None:
        with self.make_kernel() as kernel:
            self.assertTrue(kernel.execute("thread-1", "x = 21")["ok"])

        with self.make_kernel() as kernel:
            execution = kernel.execute("thread-1", "x * 2")

        self.assertEqual(execution["result"], 42)
        self.assertTrue(execution["state_restored"])

    def test_default_store_keeps_state_between_agent_turns(self) -> None:
        with PersistentMontyKernel() as kernel:
            kernel.execute("conversation", "values = [10, 20, 30]")
            first_turn = kernel.execute("conversation", "sum(values)")
            second_turn = kernel.execute("conversation", "sum(values) / len(values)")

        self.assertEqual(first_turn["result"], 60)
        self.assertEqual(second_turn["result"], 20.0)
        self.assertTrue(second_turn["state_restored"])

    def test_shared_memory_store_survives_kernel_replacement(self) -> None:
        store = InMemorySessionStore()
        with PersistentMontyKernel(store) as kernel:
            kernel.execute("conversation", "x = 41")

        with PersistentMontyKernel(store) as kernel:
            execution = kernel.execute("conversation", "x + 1")

        self.assertEqual(execution["result"], 42)

    def test_new_memory_store_has_no_previous_state(self) -> None:
        with PersistentMontyKernel() as kernel:
            kernel.execute("conversation", "x = 41")

        with PersistentMontyKernel() as kernel:
            execution = kernel.execute("conversation", "x")

        self.assertFalse(execution["ok"])
        self.assertFalse(execution["state_restored"])

    def test_stateless_store_starts_clean_for_every_call(self) -> None:
        config = KernelConfig(max_state_bytes=1)
        with PersistentMontyKernel(StatelessSessionStore(), config) as kernel:
            first = kernel.execute("conversation", "x = 41")
            second = kernel.execute("conversation", "x")

        self.assertTrue(first["ok"])
        self.assertFalse(first["state_saved"])
        self.assertFalse(second["ok"])
        self.assertFalse(second["state_restored"])

    def test_sessions_are_isolated(self) -> None:
        with self.make_kernel() as kernel:
            kernel.execute("one", "x = 1")
            execution = kernel.execute("two", "x")

        self.assertFalse(execution["ok"])
        self.assertIn("NameError", execution["error"]["message"])

    def test_error_keeps_changes_made_before_the_error(self) -> None:
        with self.make_kernel() as kernel:
            failed = kernel.execute("thread", "x = 7\n1 / 0")
            recovered = kernel.execute("thread", "x")

        self.assertFalse(failed["ok"])
        self.assertTrue(failed["state_saved"])
        self.assertEqual(recovered["result"], 7)

    def test_output_limit_can_be_requested_and_is_hard_capped(self) -> None:
        config = KernelConfig(
            default_output_limit_chars=4,
            max_output_limit_chars=6,
        )
        with self.make_kernel(config) as kernel:
            requested = kernel.execute(
                "thread",
                'print("abcdefghij")\n"klmnopqrst"',
                output_limit_chars=5,
            )
            capped = kernel.execute(
                "thread",
                '"abcdefghij"',
                output_limit_chars=100,
            )

        self.assertEqual(requested["stdout"], "abcde")
        self.assertEqual(requested["result"], '"klmn')
        self.assertTrue(requested["truncated"]["stdout"])
        self.assertEqual(capped["output_limit_chars"], 6)
        self.assertTrue(capped["limit_was_capped"])

    def test_reset_removes_state(self) -> None:
        with self.make_kernel() as kernel:
            kernel.execute("thread", "x = 1")
            self.assertTrue(kernel.reset("thread"))
            execution = kernel.execute("thread", "x")

        self.assertFalse(execution["ok"])
        self.assertFalse(execution["state_restored"])

    def test_expired_session_starts_again(self) -> None:
        config = KernelConfig(session_ttl_seconds=0)
        with self.make_kernel(config) as kernel:
            kernel.execute("thread", "x = 1")
            execution = kernel.execute("thread", "x")

        self.assertFalse(execution["ok"])
        self.assertFalse(execution["state_restored"])

    def test_large_state_keeps_the_previous_checkpoint(self) -> None:
        config = KernelConfig(max_state_bytes=1)
        with self.make_kernel(config) as kernel:
            execution = kernel.execute("thread", "x = 1")

        self.assertEqual(execution["error"]["type"], "SessionStateTooLarge")
        self.assertFalse(execution["state_saved"])

    def test_execution_time_is_limited(self) -> None:
        config = KernelConfig(
            max_duration_seconds=0.01,
            request_timeout_seconds=1,
        )
        with self.make_kernel(config) as kernel:
            execution = kernel.execute("thread", "while True:\n    pass")

        self.assertFalse(execution["ok"])
        self.assertIn("TimeoutError", execution["error"]["message"])

    def test_captured_stream_size_is_limited(self) -> None:
        config = KernelConfig(max_stream_bytes=100)
        with self.make_kernel(config) as kernel:
            execution = kernel.execute("thread", 'print("x" * 1_000)')

        self.assertFalse(execution["ok"])
        self.assertIn("MemoryError", execution["error"]["message"])

    def test_wrong_monty_version_requires_reset(self) -> None:
        store = SQLiteSessionStore(self.database)
        store.save("thread", b"state", "0.0.0", expected_revision=None)

        with PersistentMontyKernel(store) as kernel:
            execution = kernel.execute("thread", "1 + 1")

        self.assertEqual(execution["error"]["type"], "SessionVersionError")

    def test_approved_host_function_is_available(self) -> None:
        with self.make_kernel(
            external_lookup={"host_add": lambda a, b: a + b}
        ) as kernel:
            execution = kernel.execute("thread", "host_add(20, 22)")

        self.assertEqual(execution["result"], 42)

    def test_filesystem_access_is_denied_by_default(self) -> None:
        with self.make_kernel() as kernel:
            execution = kernel.execute("thread", "open('/notes.txt', 'w')")

        self.assertFalse(execution["ok"])
        self.assertIn("PermissionError", execution["error"]["message"])

    def test_memory_files_remain_available_between_calls(self) -> None:
        os_handler = OSAccess(files=[MemoryFile("/data/input.txt", "21")])
        with PersistentMontyKernel(os_handler=os_handler) as kernel:
            written = kernel.execute(
                "thread",
                "from pathlib import Path\n"
                "value = int(Path('/data/input.txt').read_text())\n"
                "Path('/data/output.txt').write_text(str(value * 2))",
            )
            read = kernel.execute("thread", "open('/data/output.txt').read()")

        self.assertTrue(written["ok"])
        self.assertEqual(read["result"], "42")

    def test_execute_file_can_run_follow_up_code_in_one_operation(self) -> None:
        """Load a virtual helper and use it before the operation returns."""
        os_handler = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "print('helper loaded')\n"
                    "VALUE = 21\n"
                    "def double(value):\n"
                    "    return value * 2\n",
                )
            ]
        )
        with PersistentMontyKernel(os_handler=os_handler) as kernel:
            execution = kernel.execute_file(
                "thread",
                "/helper.py",
                "double(VALUE)",
            )
            later = kernel.execute("thread", "double(5)")

        self.assertTrue(execution["ok"])
        self.assertEqual(execution["result"], 42)
        self.assertEqual(execution["stdout"], "helper loaded\n")
        self.assertEqual(execution["session_revision"], 1)
        self.assertEqual(later["result"], 10)

    def test_execute_file_does_not_run_follow_up_after_file_error(self) -> None:
        """Stop before follow-up code when the virtual file fails."""
        os_handler = OSAccess(
            files=[MemoryFile("/broken.py", "before_error = 1\n1 / 0\n")]
        )
        with PersistentMontyKernel(os_handler=os_handler) as kernel:
            failed = kernel.execute_file(
                "thread",
                "/broken.py",
                "follow_up_ran = True",
            )
            before_error = kernel.execute("thread", "before_error")
            follow_up = kernel.execute("thread", "follow_up_ran")

        self.assertFalse(failed["ok"])
        self.assertEqual(before_error["result"], 1)
        self.assertFalse(follow_up["ok"])

    def test_execute_file_reports_a_missing_virtual_file(self) -> None:
        """Return a normal execution error when the virtual file is absent."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            execution = kernel.execute_file("thread", "/missing.py")

        self.assertFalse(execution["ok"])
        self.assertIn("FileNotFoundError", execution["error"]["message"])

    def test_executed_file_definitions_survive_kernel_restart(self) -> None:
        """Save loaded helper definitions in the normal Monty session dump."""
        store = SQLiteSessionStore(self.database)
        filesystem = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "def double(value):\n    return value * 2\n",
                )
            ]
        )
        with PersistentMontyKernel(store, os_handler=filesystem) as kernel:
            loaded = kernel.execute_file("thread", "/helper.py")

        with PersistentMontyKernel(store) as kernel:
            reused = kernel.execute("thread", "double(21)")

        self.assertTrue(loaded["ok"])
        self.assertEqual(reused["result"], 42)

    def test_read_only_mount_supports_multiple_directories(self) -> None:
        first = Path(self.directory.name) / "first"
        second = Path(self.directory.name) / "second"
        first.mkdir()
        second.mkdir()
        (first / "a.txt").write_text("A")
        (second / "b.txt").write_text("B")

        with (
            MountDir(
                host_path=first,
                virtual_path="/first",
                mode="read-only",
            ) as first_mount,
            MountDir(
                host_path=second,
                virtual_path="/second",
                mode="read-only",
            ) as second_mount,
            PersistentMontyKernel(mounts=[first_mount, second_mount]) as kernel,
        ):
            read = kernel.execute(
                "thread",
                "open('/first/a.txt').read() + open('/second/b.txt').read()",
            )
            write = kernel.execute(
                "thread",
                "open('/first/new.txt', 'w').write('no')",
            )

        self.assertEqual(read["result"], "AB")
        self.assertFalse(write["ok"])
        self.assertIn("PermissionError", write["error"]["message"])

    def test_read_write_mount_persists_files_between_calls(self) -> None:
        workspace = Path(self.directory.name) / "workspace"
        workspace.mkdir()

        with (
            MountDir(
                host_path=workspace,
                virtual_path="/workspace",
                mode="read-write",
                write_bytes_limit=100,
            ) as mount,
            PersistentMontyKernel(mounts=mount) as kernel,
        ):
            kernel.execute(
                "thread",
                "open('/workspace/result.txt', 'w').write('saved')",
            )
            read = kernel.execute(
                "thread",
                "open('/workspace/result.txt').read()",
            )

        self.assertEqual(read["result"], "saved")
        self.assertEqual((workspace / "result.txt").read_text(), "saved")

    def test_overlay_mount_discards_writes_after_each_call(self) -> None:
        workspace = Path(self.directory.name) / "overlay"
        workspace.mkdir()

        with (
            MountDir(
                host_path=workspace,
                virtual_path="/workspace",
                mode="overlay",
            ) as mount,
            PersistentMontyKernel(mounts=mount) as kernel,
        ):
            written = kernel.execute(
                "thread",
                "open('/workspace/temporary.txt', 'w').write('temporary')",
            )
            read = kernel.execute(
                "thread",
                "open('/workspace/temporary.txt').read()",
            )

        self.assertTrue(written["ok"])
        self.assertFalse(read["ok"])
        self.assertFalse((workspace / "temporary.txt").exists())

    def test_mount_write_limit_is_enforced(self) -> None:
        workspace = Path(self.directory.name) / "limited"
        workspace.mkdir()

        with (
            MountDir(
                host_path=workspace,
                virtual_path="/workspace",
                mode="read-write",
                write_bytes_limit=3,
            ) as mount,
            PersistentMontyKernel(mounts=mount) as kernel,
        ):
            execution = kernel.execute(
                "thread",
                "open('/workspace/result.txt', 'w').write('four')",
            )

        self.assertFalse(execution["ok"])

    def test_filesystem_helpers_are_available_inside_the_repl(self) -> None:
        """Use every high-level filesystem operation from normal REPL code."""
        workspace = Path(self.directory.name) / "helpers"
        workspace.mkdir()

        with (
            MountDir(
                host_path=workspace,
                virtual_path="/workspace",
                mode="read-write",
            ) as mount,
            PersistentMontyKernel(mounts=mount) as kernel,
        ):
            execution = kernel.execute(
                "thread",
                "fs_mkdir('/workspace/src', parents=True, exist_ok=True)\n"
                "fs_write('/workspace/src/a.py', 'Alpha\\nbeta\\nAlpha\\n')\n"
                "fs_write('/workspace/src/b.txt', 'other')\n"
                "edited = fs_edit('/workspace/src/a.py', 'beta', 'gamma')\n"
                "listing = fs_ls('/workspace', recursive=True)\n"
                "globbed = fs_glob('**/*.py', root='/workspace')\n"
                "grepped = fs_grep('alpha', root='/workspace', "
                "pattern='**/*.py', case_sensitive=False)\n"
                "regex_grep = fs_grep('^A.*a$', root='/workspace', "
                "pattern='**/*.py', regex=True)\n"
                "[fs_read('/workspace/src/a.py'), edited, listing, globbed, "
                "grepped, regex_grep]",
            )
            deleted = kernel.execute(
                "thread",
                "fs_delete('/workspace/src', recursive=True)",
            )

        read, edited, listing, globbed, grepped, regex_grep = execution["result"]
        self.assertEqual(read, "Alpha\ngamma\nAlpha\n")
        self.assertEqual(edited["replacements"], 1)
        self.assertEqual(
            {entry["path"] for entry in listing["entries"]},
            {
                "/workspace/src",
                "/workspace/src/a.py",
                "/workspace/src/b.txt",
            },
        )
        self.assertEqual(globbed["matches"], ["/workspace/src/a.py"])
        self.assertEqual([match["line"] for match in grepped["matches"]], [1, 3])
        self.assertEqual(
            [match["line"] for match in regex_grep["matches"]],
            [1, 3],
        )
        self.assertEqual(deleted["result"], 3)
        self.assertFalse((workspace / "src").exists())

    def test_filesystem_limits_are_hard_capped(self) -> None:
        """Apply configured hard caps to LLM-selected filesystem limits."""
        workspace = Path(self.directory.name) / "limits"
        workspace.mkdir()
        (workspace / "large.txt").write_text("abcdef")
        config = FilesystemConfig(
            default_read_limit_chars=3,
            max_read_limit_chars=5,
            max_write_chars=5,
            default_result_limit=1,
            max_result_limit=2,
            max_scan_entries=10,
            max_grep_chars=10,
            max_depth=2,
        )

        with (
            MountDir(
                host_path=workspace,
                virtual_path="/workspace",
                mode="read-write",
            ) as mount,
            PersistentMontyKernel(mounts=mount, filesystem_config=config) as kernel,
        ):
            default_read = kernel.execute(
                "thread",
                "fs_read('/workspace/large.txt')",
            )
            capped_read = kernel.execute(
                "thread",
                "fs_read('/workspace/large.txt', limit=100)",
            )
            rejected_write = kernel.execute(
                "thread",
                "fs_write('/workspace/new.txt', '123456')",
            )

        self.assertEqual(default_read["result"], "abc")
        self.assertEqual(capped_read["result"], "abcde")
        self.assertFalse(rejected_write["ok"])
        self.assertFalse((workspace / "new.txt").exists())

    def test_filesystem_call_quotes_values_and_rejects_unknown_arguments(self) -> None:
        """Keep tool values as data and restrict generated keyword names."""
        content = "value = 'quoted'\nfs_delete('/')"
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            written = kernel.call_filesystem(
                "thread",
                "write",
                {
                    "path": "/quoted.txt",
                    "content": content,
                    "append": False,
                    "create_parents": False,
                },
            )
            read = kernel.call_filesystem(
                "thread",
                "read",
                {"path": "/quoted.txt", "offset": 0, "limit": 100},
            )
            with self.assertRaises(TypeError):
                kernel.call_filesystem(
                    "thread",
                    "read",
                    {"path": "/quoted.txt", "unexpected": True},
                )

        self.assertTrue(written["ok"])
        self.assertEqual(read["result"], content)

    def test_filesystem_edit_requires_one_exact_match_by_default(self) -> None:
        """Prevent an ambiguous edit unless replace_all is explicit."""
        workspace = Path(self.directory.name) / "edit"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("old old")

        with (
            MountDir(
                host_path=workspace,
                virtual_path="/workspace",
                mode="read-write",
            ) as mount,
            PersistentMontyKernel(mounts=mount) as kernel,
        ):
            ambiguous = kernel.execute(
                "thread",
                "fs_edit('/workspace/notes.txt', 'old', 'new')",
            )
            replaced = kernel.execute(
                "thread",
                "fs_edit('/workspace/notes.txt', 'old', 'new', replace_all=True)",
            )

        self.assertFalse(ambiguous["ok"])
        self.assertEqual(replaced["result"]["replacements"], 2)
        self.assertEqual((workspace / "notes.txt").read_text(), "new new")

    def test_calls_for_one_session_are_serialized(self) -> None:
        with self.make_kernel() as kernel:
            kernel.execute("thread", "counter = 0")
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                executions = list(
                    executor.map(
                        lambda _: kernel.execute("thread", "counter += 1"),
                        range(12),
                    )
                )
            final = kernel.execute("thread", "counter")

        self.assertTrue(all(item["ok"] for item in executions))
        self.assertEqual(final["result"], 12)

    def test_worker_checkout_timeout_returns_retryable_result(self) -> None:
        """Return bounded backpressure when the sync worker pool is full."""
        started = threading.Event()
        release = threading.Event()

        def hold() -> int:
            """Own the only worker until the test releases it."""
            started.set()
            release.wait(timeout=1)
            return 1

        config = KernelConfig(
            max_worker_processes=1,
            worker_checkout_timeout_seconds=0.02,
        )
        with (
            PersistentMontyKernel(
                config=config,
                external_lookup={"hold": hold},
            ) as kernel,
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
        ):
            first = executor.submit(kernel.execute, "one", "hold()")
            self.assertTrue(started.wait(timeout=1))
            busy = kernel.execute("two", "2")
            release.set()
            first.result(timeout=1)

        self.assertEqual(busy["error"]["type"], "WorkerCheckoutTimeout")
        self.assertTrue(busy["retryable"])

    def test_workspace_preserves_state_and_files_across_restarts(self) -> None:
        """Restore SQLite state and host-backed files as one workspace."""
        workspace = Path(self.directory.name) / "workspace"
        workspace_config = WorkspaceConfig(workspace)
        store = SQLiteSessionStore(self.database)

        with PersistentMontyKernel(store, workspace=workspace_config) as kernel:
            first = kernel.execute(
                "thread",
                "value = 40\nopen('/workspace/value.txt', 'w').write('2')",
            )

        with PersistentMontyKernel(store, workspace=workspace_config) as kernel:
            restored = kernel.execute(
                "thread",
                "value + int(open('/workspace/value.txt').read())",
            )
            shell = execute_repl_cell(kernel, "new-thread", "!pwd")

        self.assertTrue(first["ok"])
        self.assertEqual(restored["result"], 42)
        self.assertTrue(restored["state_restored"])
        self.assertEqual(shell["output"], "/workspace\n")
        self.assertEqual((workspace / "value.txt").read_text(), "2")

    def test_type_check_rejects_code_before_execution(self) -> None:
        """Use Monty checking with filesystem and host-function declarations."""
        config = KernelConfig(
            type_check=True,
            type_check_stubs="def host_add(a: int, b: int) -> int: ...",
        )
        workspace = WorkspaceConfig(Path(self.directory.name) / "typed")
        with PersistentMontyKernel(
            config=config,
            external_lookup={"host_add": lambda a, b: a + b},
            workspace=workspace,
        ) as kernel:
            valid = kernel.execute(
                "thread",
                "x: int = host_add(20, 22)\nfs_write('/workspace/value.txt', str(x))",
            )
            restored = kernel.execute("thread", "x + 1")
            invalid = kernel.execute("thread", 'bad: int = "text"')
            missing = kernel.execute("thread", "bad")

        self.assertTrue(valid["ok"])
        self.assertEqual(restored["result"], 43)
        self.assertEqual(invalid["error"]["type"], "MontyTypingError")
        self.assertIn("not assignable", invalid["error"]["message"])
        self.assertFalse(missing["ok"])

    def test_trimmed_python_and_shell_output_can_be_offloaded(self) -> None:
        """Save larger bounded results when model-facing output is shortened."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            python = execute_repl_cell(
                kernel,
                "thread",
                '"abcdefgh"',
                3,
                output_file="/python.json",
            )
            shell = execute_repl_cell(
                kernel,
                "thread",
                "!echo abcdefgh",
                3,
                output_file="/shell.txt",
            )
            python_file = json.loads(kernel.download_file("/python.json"))
            shell_file = kernel.download_file("/shell.txt").decode()

        self.assertEqual(python["mode"], "python")
        self.assertEqual(python["output_file"], "/python.json")
        self.assertTrue(python["truncation"]["result"])
        self.assertEqual(python_file["result"], "abcdefgh")
        self.assertEqual(shell["mode"], "shell")
        self.assertEqual(shell["output"], "abc")
        self.assertEqual(shell["output_file"], "/shell.txt")
        self.assertTrue(shell["truncation"]["output"])
        self.assertEqual(shell_file, "abcdefgh\n")

    def test_shared_filesystem_serializes_different_sessions(self) -> None:
        """Prevent same-kernel filesystem work from running at the same time."""
        state_lock = threading.Lock()
        active = 0
        maximum_active = 0

        def slow_host_call() -> int:
            """Record concurrent host calls during one Monty feed."""
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return 1

        with (
            PersistentMontyKernel(
                external_lookup={"slow_host_call": slow_host_call},
                os_handler=OSAccess(),
            ) as kernel,
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(
                executor.map(
                    lambda session: kernel.execute(session, "slow_host_call()"),
                    ("one", "two"),
                )
            )

        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual(maximum_active, 1)

    def test_kernel_lists_safe_session_metadata(self) -> None:
        """Inspect sessions without exposing their snapshot bytes."""
        with self.make_kernel() as kernel:
            kernel.execute("two", "value = 2")
            kernel.execute("one", "value = 1")
            sessions = kernel.list_sessions()
            one = kernel.get_session_info("one")

        self.assertEqual([info.session_id for info in sessions], ["one", "two"])
        self.assertEqual(one.session_id, "one")
        self.assertGreater(one.state_size_bytes, 0)
        self.assertEqual(one.revision, 1)


if __name__ == "__main__":
    unittest.main()
