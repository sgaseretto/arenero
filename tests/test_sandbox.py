import base64
import unittest
from pathlib import Path

from pydantic_monty import OSAccess

from webmonty import (
    FilesystemConfig,
    KernelConfig,
    PersistentMontyKernel,
    WebMontySandbox,
    read_media_file,
)


class SandboxTests(unittest.TestCase):
    """Test the framework-neutral sandbox utilities."""

    def make_sandbox(
        self,
        *,
        kernel_config: KernelConfig | None = None,
        filesystem_config: FilesystemConfig | None = None,
    ) -> tuple[PersistentMontyKernel, WebMontySandbox]:
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
        return kernel, WebMontySandbox(kernel, session_id="sandbox-test")

    def test_limited_execute_runs_code_and_virtual_files(self) -> None:
        """Run only the two documented command forms."""
        _, sandbox = self.make_sandbox()
        uploaded = sandbox.upload_files(
            [("/helper.py", b"def double(value):\n    return value * 2\n")]
        )

        loaded = sandbox.execute("python /helper.py")
        calculated = sandbox.execute("python -c 'double(21)'")
        rejected = sandbox.execute("ls -la | grep py")

        self.assertIsNone(uploaded[0].error)
        self.assertEqual(loaded.exit_code, 0)
        self.assertEqual(calculated.output, "42")
        self.assertEqual(rejected.exit_code, 127)
        self.assertIn("Unsupported command", rejected.output)

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


if __name__ == "__main__":
    unittest.main()
