import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from pydantic_monty import OSAccess

from arenero import (
    AreneroSandbox,
    AsyncAreneroSandbox,
    FilesystemConfig,
    KernelConfig,
    PersistentMontyKernel,
    SQLiteSessionStore,
    read_media_file,
)


class SandboxTests(unittest.TestCase):
    """Test the framework-neutral sandbox utilities."""

    def make_sandbox(
        self,
        *,
        kernel_config: KernelConfig | None = None,
        filesystem_config: FilesystemConfig | None = None,
    ) -> tuple[PersistentMontyKernel, AreneroSandbox]:
        """Create one sandbox backed by a shared memory filesystem.

        Args:
            kernel_config: Optional execution limits.
            filesystem_config: Optional filesystem limits.

        Returns:
            Open kernel and its sandbox facade.
        """
        kernel = PersistentMontyKernel(
            config=kernel_config,
            os_handler=OSAccess(),
            filesystem_config=filesystem_config,
        )
        self.addCleanup(kernel.close)
        return kernel, AreneroSandbox(kernel, session_id="sandbox-test")

    def test_execute_runs_python_and_rejects_real_commands(self) -> None:
        """Run Monty Python but reject operating-system programs."""
        _, sandbox = self.make_sandbox()
        uploaded = sandbox.upload_files(
            [("/work/helper.py", b"def double(value):\n    return value * 2\n")]
        )

        sandbox.execute("cd /work")
        loaded = sandbox.execute("python helper.py")
        calculated = sandbox.execute("python -c 'double(21)'")
        rejected = sandbox.execute("git status")

        self.assertIsNone(uploaded[0].error)
        self.assertEqual(loaded.exit_code, 0)
        self.assertEqual(calculated.output, "42")
        self.assertEqual(rejected.exit_code, 127)
        self.assertIn("Unsupported command", rejected.output)

    def test_virtual_shell_uses_relative_paths_and_file_commands(self) -> None:
        """Use the persistent directory with common virtual file commands."""
        _, sandbox = self.make_sandbox()

        made = sandbox.execute("mkdir -p /workspace/data")
        changed = sandbox.execute("cd /workspace")
        written = sandbox.execute("echo Alpha > notes.txt")
        appended = sandbox.execute("echo beta >> notes.txt")
        copied = sandbox.execute("cp notes.txt copy.txt")
        moved = sandbox.execute("mv copy.txt moved.txt")
        listed = sandbox.execute("ls .")
        found = sandbox.execute("find . '*.txt'")
        removed = sandbox.execute("rm moved.txt")
        current = sandbox.execute("pwd")
        content = sandbox.execute("cat notes.txt")

        for result in (made, changed, written, appended, copied, moved, removed):
            self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(current.output, "/workspace\n")
        self.assertEqual(content.output, "Alpha\nbeta\n")
        self.assertIn("/workspace/notes.txt", listed.output)
        self.assertIn("/workspace/moved.txt", found.output)
        self.assertNotIn("/workspace/moved.txt", sandbox.execute("ls .").output)

    def test_virtual_shell_pipes_and_redirects_bounded_text(self) -> None:
        """Pass bounded text through pipes and virtual file redirection."""
        _, sandbox = self.make_sandbox()
        sandbox.write("/input.txt", "alpha\nBeta\nALPHA\nomega\n")

        piped = sandbox.execute(
            "cat /input.txt | grep -in alpha | head -n 1 > /result.txt"
        )
        result = sandbox.execute("cat /result.txt")
        counted = sandbox.execute("wc -l < /input.txt")
        tail = sandbox.execute("tail -n 2 /input.txt")

        self.assertEqual(piped.exit_code, 0, piped.output)
        self.assertEqual(piped.output, "")
        self.assertEqual(result.output, "1:alpha\n")
        self.assertEqual(counted.output, "4\n")
        self.assertEqual(tail.output, "ALPHA\nomega\n")

    def test_virtual_shell_output_limit_and_rejected_syntax(self) -> None:
        """Apply caller output limits and reject unsupported shell syntax."""
        config = KernelConfig(
            default_output_limit_chars=6,
            max_output_limit_chars=8,
            max_shell_command_chars=20,
            max_shell_commands=2,
        )
        _, sandbox = self.make_sandbox(kernel_config=config)

        limited = sandbox.execute("echo 123456789", output_limit_chars=5)
        capped = sandbox.execute("echo 123456789", output_limit_chars=100)
        chained = sandbox.execute("echo one; echo two", output_limit_chars=8)
        background = sandbox.execute("echo one &")
        too_long = sandbox.execute("echo 12345678901234567890")
        too_many = sandbox.execute("echo one | cat | cat")
        too_many_chained = sandbox.execute("echo a;echo b;echo c")

        self.assertEqual(limited.output, "12345")
        self.assertTrue(limited.truncated)
        self.assertEqual(capped.output, "12345678")
        self.assertTrue(capped.truncated)
        self.assertEqual(chained.exit_code, 0)
        self.assertEqual(chained.output, "one\ntwo\n")
        self.assertEqual(background.exit_code, 2)
        self.assertIn("length limit", too_long.output)
        self.assertIn("command limit", too_many.output)
        self.assertIn("command limit", too_many_chained.output)

    def test_virtual_shell_chains_commands_by_exit_code(self) -> None:
        """Run sequential and conditional command lists with shell exit codes."""
        _, sandbox = self.make_sandbox()
        sandbox.write("/input.txt", "alpha\nbeta\n")

        always = sandbox.execute("echo one; echo two")
        on_success = sandbox.execute("grep alpha /input.txt && echo found")
        on_failure = sandbox.execute("grep missing /input.txt || echo fallback")
        skipped = sandbox.execute("grep alpha /input.txt || echo hidden")

        self.assertEqual(always.output, "one\ntwo\n")
        self.assertEqual(on_success.output, "alpha\nfound\n")
        self.assertEqual(on_failure.output, "fallback\n")
        self.assertEqual(skipped.output, "alpha\n")
        self.assertEqual(skipped.exit_code, 0)

    def test_virtual_shell_help_which_and_allowlist(self) -> None:
        """Discover enabled commands and reject commands outside an allowlist."""
        config = KernelConfig(
            allowed_shell_commands=frozenset({"echo", "help", "which"})
        )
        _, sandbox = self.make_sandbox(kernel_config=config)

        help_result = sandbox.execute("help")
        usage = sandbox.execute("help echo")
        found = sandbox.execute("which echo")
        missing = sandbox.execute("which grep")
        rejected = sandbox.execute("python -c '1 + 1'")

        self.assertIn("Commands: echo help which", help_result.output)
        self.assertEqual(usage.output, "echo [-n] [TEXT ...]\n")
        self.assertEqual(found.output, "echo\n")
        self.assertEqual(missing.exit_code, 1)
        self.assertEqual(missing.output, "grep: not found\n")
        self.assertEqual(rejected.exit_code, 127)

    def test_virtual_shell_small_text_commands_and_tee(self) -> None:
        """Use the small line and path commands with virtual files."""
        _, sandbox = self.make_sandbox()
        sandbox.write("/input.txt", "beta\nalpha\nalpha\n")

        piped = sandbox.execute("sort /input.txt | uniq -c | tee /counts.txt")
        reversed_text = sandbox.execute("echo abc | rev")
        basename = sandbox.execute("basename /work/file.txt")
        dirname = sandbox.execute("dirname /work/file.txt")

        self.assertEqual(piped.output, "      2 alpha\n      1 beta\n")
        self.assertEqual(sandbox.execute("cat /counts.txt").output, piped.output)
        self.assertEqual(reversed_text.output, "cba\n")
        self.assertEqual(basename.output, "file.txt\n")
        self.assertEqual(dirname.output, "/work\n")

    def test_virtual_shell_binary_encoding_and_hashing(self) -> None:
        """Encode and hash binary virtual files without text decoding."""
        _, sandbox = self.make_sandbox()
        content = b"\x00\xffarenero"
        sandbox.upload_files([("/data.bin", content)])

        encoded = sandbox.execute("base64 /data.bin")
        hashed = sandbox.execute("sha256sum /data.bin")

        self.assertEqual(base64.b64decode(encoded.output), content)
        self.assertEqual(
            hashed.output,
            "e64f3faad3530bea9adbda8808456bcc43cf4a417a7e734eee79e1f1f5dd7d52"
            "  /data.bin\n",
        )

    def test_virtual_shell_copies_binary_files(self) -> None:
        """Preserve binary bytes when shell commands copy or move files."""
        _, sandbox = self.make_sandbox()
        source = b"\x00\xff\x10"
        sandbox.upload_files([("/source.bin", source)])

        copied = sandbox.execute("cp /source.bin /copy.bin")
        moved = sandbox.execute("mv /copy.bin /moved.bin")
        files = sandbox.download_files(["/source.bin", "/moved.bin"])

        self.assertEqual(copied.exit_code, 0, copied.output)
        self.assertEqual(moved.exit_code, 0, moved.output)
        self.assertEqual([file.content for file in files], [source, source])
        self.assertIsNotNone(sandbox.read("/copy.bin").error)

    def test_virtual_shell_grep_uses_filesystem_limits(self) -> None:
        """Apply the configured match cap to host-side shell grep."""
        filesystem_config = FilesystemConfig(
            default_result_limit=1,
            max_result_limit=2,
        )
        _, sandbox = self.make_sandbox(filesystem_config=filesystem_config)
        sandbox.write("/input.txt", "Alpha\nAlpha\n")

        result = sandbox.execute("grep Alpha /input.txt")

        self.assertEqual(result.output, "Alpha\n")
        self.assertTrue(result.truncated)

    def test_virtual_shell_working_directory_is_session_state(self) -> None:
        """Persist the directory in SQLite and isolate it by session ID."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = Path(directory.name) / "sessions.db"
        filesystem = OSAccess()

        first_kernel = PersistentMontyKernel(
            SQLiteSessionStore(database),
            os_handler=filesystem,
        )
        first = AreneroSandbox(first_kernel, session_id="one")
        other = AreneroSandbox(first_kernel, session_id="two")
        self.assertEqual(first.execute("mkdir -p /work").exit_code, 0)
        self.assertEqual(first.execute("cd /work").exit_code, 0)
        self.assertEqual(other.execute("pwd").output, "/\n")
        first_kernel.close()

        second_kernel = PersistentMontyKernel(
            SQLiteSessionStore(database),
            os_handler=filesystem,
        )
        self.addCleanup(second_kernel.close)
        restored = AreneroSandbox(second_kernel, session_id="one")

        self.assertEqual(restored.execute("pwd").output, "/work\n")

    def test_per_call_execute_timeout_is_explicitly_unsupported(self) -> None:
        """Reject a timeout that the persistent kernel cannot apply per call."""
        _, sandbox = self.make_sandbox()

        result = sandbox.execute("python -c '1 + 1'", timeout=1)

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Per-call timeouts", result.output)

    def test_file_utilities_share_text_and_binary_files(self) -> None:
        """Use the file operation and transfer planes on one filesystem."""
        _, sandbox = self.make_sandbox()

        written = sandbox.write("/notes.txt", "one\ntwo\nthree\n")
        read = sandbox.read("/notes.txt", offset=1, limit=1)
        edited = sandbox.edit("/notes.txt", "two", "TWO")
        listed = sandbox.ls("/")
        globbed = sandbox.glob("*.txt", "/")
        grepped = sandbox.grep("TWO", "/", "*.txt")
        uploaded = sandbox.upload_files([("/image.bin", b"\x00\x01\xff")])
        binary_read = sandbox.read("/image.bin")
        downloaded = sandbox.download_files(["/image.bin"])
        deleted = sandbox.delete("/notes.txt")

        self.assertEqual(written.path, "/notes.txt")
        self.assertEqual(read.file_data, {"content": "two\n", "encoding": "utf-8"})
        self.assertEqual(read.total_lines, 3)
        self.assertEqual(read.start_line, 2)
        self.assertEqual(read.end_line, 2)
        self.assertEqual(read.next_offset, 2)
        self.assertEqual(edited.occurrences, 1)
        self.assertIn({"path": "/notes.txt", "is_dir": False}, listed.matches)
        self.assertIn({"path": "/notes.txt"}, globbed.matches)
        self.assertEqual(grepped.matches[0]["line"], 2)
        self.assertIsNone(uploaded[0].error)
        self.assertEqual(binary_read.file_data["encoding"], "base64")
        self.assertEqual(
            base64.b64decode(binary_read.file_data["content"]),
            b"\x00\x01\xff",
        )
        self.assertEqual(downloaded[0].content, b"\x00\x01\xff")
        self.assertEqual(deleted.path, "/notes.txt")

    def test_transfer_limit_is_a_hard_cap(self) -> None:
        """Reject binary uploads larger than the configured transfer cap."""
        config = FilesystemConfig(max_transfer_bytes=2)
        _, sandbox = self.make_sandbox(filesystem_config=config)

        result = sandbox.upload_files([("/large.bin", b"123")])

        self.assertIn("transfer limit", result[0].error)

    def test_media_bridge_builds_a_langchain_image_block(self) -> None:
        """Read the repository image from virtual storage for an LLM message."""
        kernel, sandbox = self.make_sandbox()
        source = Path("ant-image.png").read_bytes()
        uploaded = sandbox.upload_files([("/images/ant-image.png", source)])

        media = read_media_file(kernel, "/images/ant-image.png")
        block = media.as_langchain_image(detail="low")

        self.assertIsNone(uploaded[0].error)
        self.assertEqual(media.content, source)
        self.assertEqual(media.media_type, "image/png")
        self.assertEqual(block["type"], "image_url")
        self.assertEqual(block["image_url"]["detail"], "low")
        self.assertTrue(block["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_terminal_time_limit_resets_saved_session(self) -> None:
        """Discard persistent state after Monty reports a terminal timeout."""
        config = KernelConfig(
            max_duration_seconds=0.01,
            request_timeout_seconds=1,
        )
        kernel, _ = self.make_sandbox(kernel_config=config)
        kernel.execute("limited", "value = 42")

        failed = kernel.execute("limited", "while True:\n    pass")
        after = kernel.execute("limited", "value")

        self.assertFalse(failed["ok"])
        self.assertFalse(failed["state_saved"])
        self.assertFalse(after["ok"])
        self.assertFalse(after["state_restored"])

    def test_async_sandbox_wraps_execution_and_files(self) -> None:
        """Use the sandbox from an async application without blocking calls."""
        kernel, _ = self.make_sandbox()
        sandbox = AsyncAreneroSandbox(kernel, session_id="async-test")

        async def run() -> tuple[object, object, object, object]:
            """Run representative async sandbox operations."""
            written = await sandbox.write("/note.txt", "hello")
            read = await sandbox.read("/note.txt")
            python = await sandbox.execute_python("value = 41")
            shell = await sandbox.execute("python -c 'value + 1'")
            return written, read, python, shell

        written, read, python, shell = asyncio.run(run())

        self.assertEqual(written.path, "/note.txt")
        self.assertEqual(read.file_data["content"], "hello")
        self.assertEqual(python.exit_code, 0)
        self.assertEqual(shell.output, "42")


if __name__ == "__main__":
    unittest.main()
