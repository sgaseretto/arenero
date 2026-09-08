import base64
import tempfile
import unittest
from pathlib import Path

from pydantic_monty import OSAccess

from arenero import (
    AreneroSandbox,
    KernelConfig,
    PersistentMontyKernel,
    SQLiteSessionStore,
    StatelessSessionStore,
)


class VirtualBashTests(unittest.TestCase):
    """Test the supported Bash-like virtual shell subset."""

    def make_sandbox(
        self,
        config: KernelConfig | None = None,
    ) -> AreneroSandbox:
        """Create one in-memory virtual shell.

        Args:
            config: Optional kernel and shell limits.

        Returns:
            Sandbox with persistent Monty state and memory files.
        """

        kernel = PersistentMontyKernel(config=config, os_handler=OSAccess())
        self.addCleanup(kernel.close)
        return AreneroSandbox(kernel, session_id="virtual-bash")

    def test_variables_quotes_status_and_unset_persist(self) -> None:
        """Expand persistent variables with Bash-like quote and status rules."""
        sandbox = self.make_sandbox()

        self.assertEqual(sandbox.execute("NAME=alpha").exit_code, 0)
        quoted = sandbox.execute("echo \"$NAME\" '$NAME'")
        sandbox.execute("false")
        status = sandbox.execute("echo $?")
        listed = sandbox.execute("export COUNT=2; env")
        sandbox.execute("unset NAME")
        missing = sandbox.execute("echo x${NAME}x")

        self.assertEqual(quoted.output, "alpha $NAME\n")
        self.assertEqual(status.output, "1\n")
        self.assertIn("COUNT=2\n", listed.output)
        self.assertEqual(missing.output, "xx\n")

    def test_home_oldpwd_comments_and_continuations(self) -> None:
        """Use virtual home paths, previous directory, comments, and continuations."""
        sandbox = self.make_sandbox()
        sandbox.execute("mkdir -p /home /other")
        sandbox.execute("HOME=/home")

        home = sandbox.execute("cd; pwd")
        sandbox.execute("cd /other")
        previous = sandbox.execute("cd -")
        comment = sandbox.execute("echo visible # ignored")
        continued = sandbox.execute("echo hello\\\nworld")

        self.assertEqual(home.output, "/home\n")
        self.assertEqual(previous.output, "/home\n")
        self.assertEqual(comment.output, "visible\n")
        self.assertEqual(continued.output, "helloworld\n")

    def test_bounded_glob_expansion_honors_quotes(self) -> None:
        """Expand unquoted virtual wildcards and keep quoted wildcards literal."""
        sandbox = self.make_sandbox()
        sandbox.execute("mkdir -p /work")
        sandbox.execute("touch /work/a.txt /work/b.txt /work/c.md")

        expanded = sandbox.execute("echo /work/*.txt")
        literal = sandbox.execute("echo '/work/*.txt'")
        unmatched = sandbox.execute("echo /work/*.csv")

        self.assertEqual(expanded.output, "/work/a.txt /work/b.txt\n")
        self.assertEqual(literal.output, "/work/*.txt\n")
        self.assertEqual(unmatched.output, "/work/*.csv\n")

    def test_here_documents_expand_only_unquoted_delimiters(self) -> None:
        """Use bounded here documents with explicit variable expansion rules."""
        sandbox = self.make_sandbox()
        sandbox.execute("NAME=Arenero")

        expanded = sandbox.execute("cat <<EOF\nhello $NAME\nEOF\n")
        literal = sandbox.execute("cat <<'EOF'\nhello $NAME\nEOF\n")
        saved = sandbox.execute("cat > /message.txt <<EOF\n$NAME works\nEOF\n")

        self.assertEqual(expanded.output, "hello Arenero\n")
        self.assertEqual(literal.output, "hello $NAME\n")
        self.assertEqual(saved.exit_code, 0, saved.output)
        self.assertEqual(sandbox.execute("cat /message.txt").output, "Arenero works\n")

    def test_core_builtins_and_printf(self) -> None:
        """Run true, false, test, bracket, touch, and the small printf subset."""
        sandbox = self.make_sandbox()

        created = sandbox.execute("touch /empty.txt")
        file_test = sandbox.execute("test -f /empty.txt && echo yes")
        bracket = sandbox.execute("[ 4 -gt 2 ] && printf '%s\\n' larger")
        empty = sandbox.execute("test -s /empty.txt || echo empty")

        self.assertEqual(created.exit_code, 0)
        self.assertEqual(file_test.output, "yes\n")
        self.assertEqual(bracket.output, "larger\n")
        self.assertEqual(empty.output, "empty\n")

    def test_extended_grep_flags_and_recursive_search(self) -> None:
        """Use regex, invert, count, quiet, file-name, and recursive grep modes."""
        sandbox = self.make_sandbox()
        sandbox.execute("mkdir -p /logs/nested")
        sandbox.write("/logs/a.txt", "INFO one\nERROR two\n")
        sandbox.write("/logs/nested/b.txt", "ERROR three\n")

        regex = sandbox.execute("grep -En 'ERROR (two|three)' /logs/a.txt")
        inverted = sandbox.execute("grep -v ERROR /logs/a.txt")
        counted = sandbox.execute("grep -rc ERROR /logs")
        files = sandbox.execute("grep -rl ERROR /logs")
        quiet = sandbox.execute("grep -rq missing /logs")

        self.assertEqual(regex.output, "2:ERROR two\n")
        self.assertEqual(inverted.output, "INFO one\n")
        self.assertIn("/logs/a.txt:1\n", counted.output)
        self.assertIn("/logs/nested/b.txt:1\n", counted.output)
        self.assertEqual(
            files.output,
            "/logs/a.txt\n/logs/nested/b.txt\n",
        )
        self.assertEqual(quiet.exit_code, 1)
        self.assertEqual(quiet.output, "")

    def test_recursive_copy_move_and_long_listing(self) -> None:
        """Copy and move bounded directory trees and list their entry types."""
        sandbox = self.make_sandbox()
        sandbox.execute("mkdir -p /source/nested")
        sandbox.write("/source/a.txt", "a")
        sandbox.write("/source/nested/b.txt", "b")

        copied = sandbox.execute("cp -r /source /copy")
        moved = sandbox.execute("mv /copy /moved")
        sandbox.execute("mkdir /selected")
        selected = sandbox.execute("cp /source/a.txt /source/nested/b.txt /selected")
        listed = sandbox.execute("ls -lR /moved")

        self.assertEqual(copied.exit_code, 0, copied.output)
        self.assertEqual(moved.exit_code, 0, moved.output)
        self.assertEqual(selected.exit_code, 0, selected.output)
        self.assertEqual(sandbox.execute("cat /moved/nested/b.txt").output, "b")
        self.assertEqual(
            sandbox.execute("cat /selected/a.txt /selected/b.txt").output, "ab"
        )
        self.assertIn("d /moved/nested/\n", listed.output)
        self.assertIn("- /moved/a.txt\n", listed.output)

    def test_base64_decode_is_text_only(self) -> None:
        """Decode UTF-8 Base64 and reject decoded binary pipeline output."""
        sandbox = self.make_sandbox()
        encoded = base64.b64encode(b"hello\n").decode()
        binary = base64.b64encode(b"\xff").decode()

        decoded = sandbox.execute(f"echo {encoded} | base64 -d")
        rejected = sandbox.execute(f"echo {binary} | base64 -d")

        self.assertEqual(decoded.output, "hello\n")
        self.assertEqual(rejected.exit_code, 1)
        self.assertIn("not UTF-8", rejected.output)

    def test_source_uses_current_state_and_shared_command_budget(self) -> None:
        """Run straight-line scripts and enforce one shared nested command budget."""
        sandbox = self.make_sandbox(KernelConfig(max_shell_commands=4))
        sandbox.write(
            "/setup.sh",
            "VALUE=42\nmkdir -p /data\necho $VALUE > /data/value.txt\n",
        )
        sandbox.write("/too-many.sh", "true\ntrue\ntrue\ntrue\n")

        sourced = sandbox.execute("source /setup.sh")
        reused = sandbox.execute("echo $VALUE; cat /data/value.txt")
        limited = sandbox.execute("source /too-many.sh")

        self.assertEqual(sourced.exit_code, 0, sourced.output)
        self.assertEqual(reused.output, "42\n42\n")
        self.assertEqual(limited.exit_code, 2)
        self.assertIn("command limit", limited.output)

    def test_source_nesting_has_a_separate_limit(self) -> None:
        """Reject nested source files above the configured depth."""
        sandbox = self.make_sandbox(KernelConfig(max_shell_script_depth=1))
        sandbox.write("/inner.sh", "echo inner\n")
        sandbox.write("/outer.sh", "source /inner.sh\n")

        result = sandbox.execute("source /outer.sh")

        self.assertEqual(result.exit_code, 2)
        self.assertIn("script depth", result.output)

    def test_expansion_and_variable_limits_return_clear_errors(self) -> None:
        """Reject wildcard and variable growth above configured hard limits."""
        config = KernelConfig(max_shell_expanded_args=2, max_shell_variables=1)
        sandbox = self.make_sandbox(config)
        sandbox.write("/a.txt", "a")
        sandbox.write("/b.txt", "b")
        sandbox.write("/c.txt", "c")

        expanded = sandbox.execute("echo /*.txt")
        self.assertEqual(sandbox.execute("A=1").exit_code, 0)
        variables = sandbox.execute("B=2")

        self.assertEqual(expanded.exit_code, 2)
        self.assertIn("argument expansion", expanded.output)
        self.assertEqual(variables.exit_code, 2)
        self.assertIn("variable count", variables.output)

    def test_multiline_repl_shell_cell_preserves_here_document(self) -> None:
        """Keep multiline shell syntax intact through the notebook cell bridge."""
        sandbox = self.make_sandbox()

        result = sandbox.execute_cell(
            "%%shell\nNAME=cell\ncat <<EOF > /from-cell.txt\n$NAME works\nEOF\n"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            sandbox.read("/from-cell.txt").file_data["content"], "cell works\n"
        )

    def test_shell_state_survives_sqlite_kernel_restart(self) -> None:
        """Restore variables, status, and old directory from a SQLite snapshot."""
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            filesystem = OSAccess()
            with PersistentMontyKernel(
                SQLiteSessionStore(database), os_handler=filesystem
            ) as first_kernel:
                first = AreneroSandbox(first_kernel, session_id="durable-shell")
                first.execute("mkdir -p /one /two")
                first.execute("cd /one; cd /two; VALUE=kept; false")

            with PersistentMontyKernel(
                SQLiteSessionStore(database), os_handler=filesystem
            ) as second_kernel:
                restored = AreneroSandbox(second_kernel, session_id="durable-shell")
                state = restored.execute('printf \'%s|%s|%s\' "$VALUE" "$?" "$OLDPWD"')

        self.assertEqual(state.output, "kept|1|/one")

    def test_stateless_directory_changes_apply_inside_one_call(self) -> None:
        """Use a new directory in one call but reset it before the next call."""
        kernel = PersistentMontyKernel(StatelessSessionStore(), os_handler=OSAccess())
        self.addCleanup(kernel.close)
        sandbox = AreneroSandbox(kernel, session_id="stateless-shell")
        sandbox.execute("mkdir -p /work")

        same_call = sandbox.execute("cd /work; pwd")
        later_call = sandbox.execute("pwd")

        self.assertEqual(same_call.output, "/work\n")
        self.assertEqual(later_call.output, "/\n")

    def test_parser_rejects_unsupported_or_incomplete_syntax(self) -> None:
        """Return clear errors for command substitution, quotes, and here documents."""
        sandbox = self.make_sandbox()

        substitution = sandbox.execute("echo $(pwd)")
        quote = sandbox.execute("echo 'open")
        document = sandbox.execute("cat <<EOF\nmissing end\n")

        self.assertEqual(substitution.exit_code, 2)
        self.assertIn("command substitution", substitution.output)
        self.assertEqual(quote.exit_code, 2)
        self.assertIn("quote is not closed", quote.output)
        self.assertEqual(document.exit_code, 2)
        self.assertIn("missing terminator", document.output)

    def test_explicit_standard_input_marker_works_for_text_commands(self) -> None:
        """Use a dash as explicit pipeline input for common text commands."""
        sandbox = self.make_sandbox()

        result = sandbox.execute("printf 'b\\na\\n' | sort - | cat - | head -n 1 -")
        digest = sandbox.execute("printf value | sha256sum -")
        encoded = sandbox.execute("printf value | base64 -")

        self.assertEqual(result.output, "a\n")
        self.assertTrue(digest.output.endswith("  -\n"))
        self.assertEqual(encoded.output, "dmFsdWU=\n")

    def test_kernel_rejects_nonpositive_new_shell_limits(self) -> None:
        """Validate wildcard, variable, and source-depth configuration limits."""
        for field in (
            "max_shell_expanded_args",
            "max_shell_variables",
            "max_shell_script_depth",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "shell limits"),
            ):
                KernelConfig(**{field: 0})


if __name__ == "__main__":
    unittest.main()
