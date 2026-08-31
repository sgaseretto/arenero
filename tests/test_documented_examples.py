import unittest

from pydantic_monty import MemoryFile, OSAccess

from arenero import AreneroSandbox, PersistentMontyKernel, execute_repl_cell


class DocumentedExamplesTests(unittest.TestCase):
    """Keep the central documentation examples aligned with real behavior."""

    def test_persistent_calculation_example(self) -> None:
        """Reuse a value in a later call with the same session ID."""
        with PersistentMontyKernel() as kernel:
            first = kernel.execute("conversation-42", "values = [2, 4, 6]\nsum(values)")
            later = kernel.execute(
                "conversation-42",
                "sum(value * value for value in values)",
            )

        self.assertEqual(first["result"], 12)
        self.assertEqual(later["result"], 56)

    def test_virtual_helper_example(self) -> None:
        """Run a helper file and follow-up code in one operation."""
        filesystem = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "def double(value):\n    return value * 2\n",
                )
            ]
        )
        with PersistentMontyKernel(os_handler=filesystem) as kernel:
            result = kernel.execute_file(
                "conversation-42",
                "/helper.py",
                code="double(21)",
            )
            later = kernel.execute("conversation-42", "double(5)")
            imported = kernel.execute("conversation-42", "import helper")

        self.assertEqual(result["result"], 42)
        self.assertEqual(later["result"], 10)
        self.assertFalse(imported["ok"])

    def test_notebook_shell_example(self) -> None:
        """Run a shell cell through the shared notebook cell API."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            execute_repl_cell(kernel, "thread-1", "total = 21 * 2")
            cell = execute_repl_cell(
                kernel,
                "thread-1",
                "%%shell\necho Alpha > /notes.txt\ngrep Alpha /notes.txt",
            )
            python = execute_repl_cell(kernel, "thread-1", "total")

        self.assertEqual(cell["mode"], "shell")
        self.assertEqual(cell["exit_code"], 0)
        self.assertEqual(cell["output"].strip(), "Alpha")
        self.assertEqual(python["result"], 42)

    def test_documented_shell_and_python_limits(self) -> None:
        """Reject packages and real commands while keeping shell text literal."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            unsupported_import = kernel.execute("thread-1", "import numpy")
            sandbox = AreneroSandbox(kernel, session_id="thread-1")
            git = sandbox.execute("git status")
            environment = sandbox.execute("echo $HOME")
            regex = sandbox.execute("echo Alpha | grep 'A.*a'")

        self.assertFalse(unsupported_import["ok"])
        self.assertEqual(git.exit_code, 127)
        self.assertEqual(environment.output.strip(), "$HOME")
        self.assertEqual(regex.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
