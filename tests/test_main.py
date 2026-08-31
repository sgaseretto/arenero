import io
import tempfile
import unittest
from pathlib import Path

from arenero import PersistentMontyKernel, SQLiteSessionStore
from arenero.cli import run_repl


class ReplTests(unittest.TestCase):
    def run_session(self, lines: list[str]) -> tuple[str, str]:
        inputs = iter(lines)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.db")
            with PersistentMontyKernel(store) as kernel:
                run_repl(
                    kernel,
                    input_fn=lambda _prompt: next(inputs),
                    stdout=stdout,
                    stderr=stderr,
                )

        return stdout.getvalue(), stderr.getvalue()

    def test_state_and_expression_output(self) -> None:
        stdout, stderr = self.run_session(["x = 21", "x * 2", "exit"])

        self.assertIn("Out[2]: 42", stdout)
        self.assertEqual(stderr, "")

    def test_multiline_input(self) -> None:
        stdout, stderr = self.run_session(
            ["def double(value):", "    return value * 2", "", "double(5)", "quit"]
        )

        self.assertIn("Out[2]: 10", stdout)
        self.assertEqual(stderr, "")

    def test_runtime_error_does_not_end_session(self) -> None:
        stdout, stderr = self.run_session(["1 / 0", "6 * 7", "exit()"])

        self.assertIn("ZeroDivisionError: division by zero", stderr)
        self.assertIn("Out[2]: 42", stdout)

    def test_notebook_shell_cells_share_the_python_session(self) -> None:
        """Run single and multiline shell cells from the terminal REPL."""
        stdout, stderr = self.run_session(
            [
                "x = 41",
                "!python -c 'x + 1'",
                "%%shell",
                "echo one",
                "echo two",
                "",
                "exit",
            ]
        )

        self.assertIn("42", stdout)
        self.assertIn("one\ntwo\n", stdout)
        self.assertEqual(stderr, "")


if __name__ == "__main__":
    unittest.main()
