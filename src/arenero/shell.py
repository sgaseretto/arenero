import base64
import binascii
import hashlib
import json
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_monty import MontyError

from ._deadline import remaining
from .events import emit_event
from .hosted_tools import HostedToolSet
from .kernel import ExecutionResult, PersistentMontyKernel, _output_limit
from .network import NetworkRequest, NetworkResponse, fetch_url
from .values import error_text as _error_text
from .values import trim_text as _trim

_STATE_NAME = "__arenero_shell_state"
_HEREDOC_PREFIX = "__arenero_heredoc_"
_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PYTHON_COMMANDS = {"monty", "python", "python3"}
_SHELL_ERRORS = (MontyError, ValueError, TypeError, RuntimeError)
_COMMAND_USAGE = {
    ".": ". FILE",
    "[": "[ EXPRESSION ]",
    "base64": "base64 [-d] [FILE]",
    "basename": "basename PATH",
    "cat": "cat FILE...",
    "cd": "cd [PATH|-]",
    "cp": "cp [-r] SOURCE... DESTINATION",
    "curl": (
        "curl [-fLsiI] [-X METHOD] [-H 'NAME: VALUE'] [-d DATA] "
        "[-o FILE] [--max-time SECONDS] URL"
    ),
    "dirname": "dirname PATH",
    "echo": "echo [-n] [TEXT ...]",
    "env": "env",
    "export": "export [NAME[=VALUE] ...]",
    "find": "find [PATH] [PATTERN]",
    "glob": "glob PATTERN [PATH]",
    "grep": "grep [-EFinvcqlrR] PATTERN [FILE ...]",
    "head": "head [-n COUNT] [FILE]",
    "help": "help [COMMAND]",
    "ls": "ls [-aA1ldR] [PATH ...]",
    "mkdir": "mkdir [-p] PATH...",
    "monty": "monty -c CODE or monty FILE.py",
    "mv": "mv SOURCE... DESTINATION",
    "pwd": "pwd",
    "printf": "printf FORMAT [ARGUMENT ...]",
    "python": "python -c CODE or python FILE.py",
    "python3": "python3 -c CODE or python3 FILE.py",
    "rev": "rev [FILE]",
    "rm": "rm [-r] [-f] PATH...",
    "sha256sum": "sha256sum [FILE ...]",
    "source": "source FILE",
    "sort": "sort [-r] [FILE]",
    "tail": "tail [-n COUNT] [FILE]",
    "tee": "tee [-a] FILE...",
    "test": "test EXPRESSION",
    "touch": "touch FILE...",
    "true": "true",
    "false": "false",
    "uniq": "uniq [-c] [FILE]",
    "unset": "unset NAME...",
    "wc": "wc [-lwc] [FILE]",
    "which": "which COMMAND...",
}


@dataclass(frozen=True)
class ShellResult:
    """Describe one virtual shell execution.

    Attributes:
        output: Command output or error text.
        exit_code: Zero for success and nonzero for failure.
        truncated: Whether an output limit shortened the result.
        output_file: Virtual path that received larger bounded output.
        output_file_error: Error that prevented output offloading.
    """

    output: str
    exit_code: int
    truncated: bool = False
    output_file: str | None = None
    output_file_error: str | None = None


@dataclass(frozen=True)
class _WordPart:
    """Store one word fragment and its quote mode.

    Attributes:
        text: Literal source text without quote markers.
        quote: ``none``, ``single``, or ``double``.
    """

    text: str
    quote: str = "none"


@dataclass(frozen=True)
class _Word:
    """Store one shell word before variable and path expansion.

    Attributes:
        parts: Ordered literal and quoted fragments.
    """

    parts: tuple[_WordPart, ...]

    def literal(self) -> str:
        """Return the word text before expansion.

        Returns:
            Joined word fragments.
        """

        return "".join(part.text for part in self.parts)


@dataclass(frozen=True)
class _HereDocument:
    """Store one extracted here-document body.

    Attributes:
        content: Text between the header and terminator.
        expand: Whether virtual variables expand in the body.
    """

    content: str
    expand: bool


@dataclass(frozen=True)
class _Command:
    """Store one parsed command and its redirections."""

    arguments: list[_Word]
    input_path: _Word | None = None
    output_path: _Word | None = None
    here_document: _HereDocument | None = None
    append: bool = False


@dataclass(frozen=True)
class _Pipeline:
    """Store one pipeline and the operator that precedes it.

    Attributes:
        commands: Commands that pass text from left to right.
        operator: ``;``, ``&&``, or ``||`` before this pipeline.
    """

    commands: list[_Command]
    operator: str | None = None


@dataclass(frozen=True)
class _CurlOptions:
    """Store the supported curl request options."""

    url: str
    method: str
    headers: dict[str, str]
    data: bytes | None
    output_path: str | None
    follow_redirects: bool
    include_headers: bool
    fail_on_http_error: bool
    timeout_seconds: float | None


class _UsageError(ValueError):
    """Report invalid virtual shell syntax or command arguments."""


class VirtualShell:
    """Run a small command allowlist over an Arenero virtual filesystem."""

    def __init__(self, kernel: PersistentMontyKernel, session_id: str) -> None:
        """Create a virtual shell for one persistent session.

        Args:
            kernel: Kernel that owns the virtual filesystem and Monty sessions.
            session_id: Session that stores Python state and the working directory.
        """
        self.kernel = kernel
        self.session_id = session_id
        self._variables: dict[str, str] = {}
        self._oldpwd = self.kernel.default_working_directory
        self._last_status = 0
        self._current_cwd = self.kernel.default_working_directory
        self._command_budget = self.kernel.config.max_shell_commands
        self._script_depth = 0
        self._commands: dict[
            str, Callable[[list[str], str | None, str], ShellResult]
        ] = {
            ".": self._source,
            "[": self._bracket,
            "base64": self._base64,
            "basename": self._basename,
            "cat": self._cat,
            "cd": self._cd,
            "cp": self._cp,
            "dirname": self._dirname,
            "echo": self._echo,
            "env": self._env,
            "export": self._export,
            "false": self._false,
            "find": self._find,
            "glob": self._glob,
            "grep": self._grep,
            "head": self._head,
            "help": self._help,
            "ls": self._ls,
            "mkdir": self._mkdir,
            "mv": self._mv,
            "pwd": self._pwd,
            "printf": self._printf,
            "rev": self._rev,
            "rm": self._rm,
            "sha256sum": self._sha256sum,
            "sort": self._sort,
            "source": self._source,
            "tail": self._tail,
            "tee": self._tee,
            "test": self._test,
            "touch": self._touch,
            "true": self._true,
            "uniq": self._uniq,
            "unset": self._unset,
            "wc": self._wc,
            "which": self._which,
        }
        if self.kernel.network_config.enabled:
            self._commands["curl"] = self._curl
        allowed = self.kernel.config.allowed_shell_commands
        if allowed is not None:
            self._commands = {
                name: handler
                for name, handler in self._commands.items()
                if name in allowed
            }

    def available_commands(self) -> list[str]:
        """Return sorted command names enabled by the active policy.

        Returns:
            Enabled virtual command names.
        """
        return self._available_commands()

    def execute(
        self,
        command: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
    ) -> ShellResult:
        """Execute one bounded virtual shell command list.

        Args:
            command: Text in the documented virtual shell grammar.
            output_limit_chars: Optional final output limit below the hard maximum.
            output_file: Optional absolute virtual path used when output is trimmed.

        Returns:
            Command output, exit code, and truncation state.
        """
        emit_event(
            self.kernel.event_sink,
            "execution",
            "shell_started",
            session_id=self.session_id,
            command_chars=len(command),
        )
        with self.kernel.filesystem_guard():
            remaining()
            result = self._execute_locked(command, output_limit_chars, output_file)
            remaining()
        emit_event(
            self.kernel.event_sink,
            "execution",
            "shell_finished",
            session_id=self.session_id,
            exit_code=result.exit_code,
            output_chars=len(result.output),
        )
        return result

    def _execute_locked(
        self,
        command: str,
        output_limit_chars: int | None,
        output_file: str | None,
    ) -> ShellResult:
        """Execute while this kernel owns its shared filesystem lock.

        Args:
            command: Text in the documented virtual shell grammar.
            output_limit_chars: Optional final output limit.
            output_file: Optional virtual path for larger bounded output.

        Returns:
            Command output, exit code, and truncation state.
        """
        try:
            if len(command) > self.kernel.config.max_shell_command_chars:
                raise _UsageError("command exceeds the virtual shell length limit")
            limit = self._output_limit(output_limit_chars)
        except _UsageError as exc:
            return ShellResult(str(exc), 2)

        state_result = self._load_shell_state()
        if isinstance(state_result, ShellResult):
            return state_result
        self._variables, self._oldpwd, self._last_status, cwd_result = state_result
        self._current_cwd = cwd_result
        self._command_budget = self.kernel.config.max_shell_commands

        usage_error = False
        try:
            result, _ = self._execute_program(command, cwd_result)
        except _UsageError as exc:
            result = ShellResult(str(exc), 2)
            usage_error = True
        self._last_status = result.exit_code
        saved = self._save_shell_state()
        if saved is not None and result.exit_code == 0:
            result = saved
        if usage_error:
            return result

        output = result.output
        last_result = result
        truncated = result.truncated

        returned_output, shortened = _trim(output, limit)
        was_truncated = truncated or shortened
        saved_output_file = None
        output_file_error = None
        if output_file is not None and was_truncated:
            try:
                self.kernel.upload_file(output_file, output.encode("utf-8"))
            except _SHELL_ERRORS as exc:
                output_file_error = str(exc)
            else:
                saved_output_file = output_file
        return ShellResult(
            returned_output,
            last_result.exit_code,
            was_truncated,
            saved_output_file,
            output_file_error,
        )

    def _execute_program(self, command: str, cwd: str) -> tuple[ShellResult, str]:
        """Parse and run one top-level command list or sourced script.

        Args:
            command: Virtual shell source.
            cwd: Current absolute virtual directory.

        Returns:
            Combined result and final working directory.

        Raises:
            _UsageError: If syntax or a shared execution limit is invalid.
        """

        if len(command) > self.kernel.config.max_shell_command_chars:
            raise _UsageError("script exceeds the virtual shell length limit")
        pipelines = _parse(command)
        command_count = sum(len(pipeline.commands) for pipeline in pipelines)
        if command_count > self._command_budget:
            raise _UsageError("command list exceeds the virtual shell command limit")
        self._command_budget -= command_count

        output = ""
        last_result = ShellResult("", 0)
        truncated = False
        for pipeline in pipelines:
            remaining()
            if pipeline.operator == "&&" and last_result.exit_code != 0:
                continue
            if pipeline.operator == "||" and last_result.exit_code == 0:
                continue
            last_result, cwd = self._execute_pipeline(pipeline.commands, cwd)
            remaining()
            self._last_status = last_result.exit_code
            output, shortened = _trim(
                output + last_result.output,
                self.kernel.config.max_output_limit_chars,
            )
            truncated = truncated or last_result.truncated or shortened
        return ShellResult(output, last_result.exit_code, truncated), cwd

    def _execute_pipeline(
        self,
        commands: list[_Command],
        cwd: str,
    ) -> tuple[ShellResult, str]:
        """Execute one parsed pipeline.

        Args:
            commands: Parsed commands that pass text from left to right.
            cwd: Current absolute virtual directory.

        Returns:
            Pipeline result and the updated working directory.
        """
        pipeline_input: str | None = None
        truncated = False
        for parsed in commands:
            remaining()
            arguments = self._expand_words(parsed.arguments, cwd)
            input_path = self._expanded_path(parsed.input_path, cwd)
            output_path = self._expanded_path(parsed.output_path, cwd)
            if parsed.here_document is not None:
                content = parsed.here_document.content
                pipeline_input = (
                    self._expand_parameters(content, cwd)
                    if parsed.here_document.expand
                    else content
                )
            if input_path is not None:
                read = self._read_text(input_path)
                if isinstance(read, ShellResult):
                    return read, cwd
                pipeline_input = read

            assignment = (
                _split_assignment(arguments[0]) if len(arguments) == 1 else None
            )
            if assignment is not None:
                self._set_variable(*assignment)
                result = ShellResult("", 0)
            else:
                result = self._run(arguments, pipeline_input, cwd)
            remaining()
            truncated = truncated or result.truncated
            if result.exit_code != 0:
                return ShellResult(result.output, result.exit_code, truncated), cwd

            pipeline_input = result.output
            if arguments[0] in {".", "cd", "source"}:
                cwd = self._current_cwd

            if output_path is not None:
                written = self._write_text(
                    output_path,
                    pipeline_input,
                    append=parsed.append,
                )
                if written is not None:
                    return written, cwd
                pipeline_input = ""

            pipeline_input, shortened = _trim(
                pipeline_input,
                self.kernel.config.max_output_limit_chars,
            )
            truncated = truncated or shortened

        return ShellResult(pipeline_input or "", 0, truncated), cwd

    def _run(self, arguments: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Run one parsed allowlisted command.

        Args:
            arguments: Command name and arguments.
            stdin: Text from a pipe or input redirection.
            cwd: Current absolute virtual directory.

        Returns:
            Command result.
        """
        name = arguments[0]
        if name in _PYTHON_COMMANDS and self._is_available(name):
            return self._python(arguments, stdin, cwd)
        handler = self._commands.get(name)
        if handler is None:
            return ShellResult(f"Unsupported command: {name}", 127)
        try:
            return handler(arguments[1:], stdin, cwd)
        except _UsageError as exc:
            return ShellResult(str(exc), 2)
        except _SHELL_ERRORS as exc:
            return ShellResult(str(exc), 1)

    def _expand_words(self, words: list[_Word], cwd: str) -> list[str]:
        """Expand variables, tilde paths, and bounded virtual globs.

        Args:
            words: Parsed words with quote information.
            cwd: Current absolute virtual directory.

        Returns:
            Expanded command arguments.

        Raises:
            _UsageError: If expansion exceeds a hard limit.
        """

        arguments: list[str] = []
        for index, word in enumerate(words):
            arguments.extend(self._expand_word(word, cwd, allow_glob=index > 0))
            if len(arguments) > self.kernel.config.max_shell_expanded_args:
                raise _UsageError("shell argument expansion exceeds its limit")
        return arguments

    def _expand_word(
        self,
        word: _Word,
        cwd: str,
        *,
        allow_glob: bool,
    ) -> list[str]:
        """Expand one parsed word while honoring its quote modes.

        Args:
            word: Parsed word fragments.
            cwd: Current absolute virtual directory.
            allow_glob: Whether unquoted wildcard text can expand.

        Returns:
            One literal argument or bounded matching virtual paths.
        """

        output: list[str] = []
        can_glob = False
        for index, part in enumerate(word.parts):
            text = (
                part.text
                if part.quote == "single"
                else self._expand_parameters(part.text, cwd)
            )
            if (
                index == 0
                and part.quote == "none"
                and (text == "~" or text.startswith("~/"))
            ):
                home = self._variables.get(
                    "HOME", self.kernel.default_working_directory
                ).rstrip("/")
                text = home + text[1:]
                if not text:
                    text = "/"
            if part.quote == "none" and any(mark in text for mark in "*?"):
                can_glob = True
            output.append(text)
        value = "".join(output)
        if allow_glob and can_glob:
            return self._expand_glob(value, cwd)
        return [value]

    def _expand_glob(self, value: str, cwd: str) -> list[str]:
        """Expand one wildcard argument through the bounded virtual filesystem.

        Args:
            value: Expanded argument containing ``*`` or ``?``.
            cwd: Current absolute virtual directory.

        Returns:
            Absolute matches, or the literal value when no path matches.

        Raises:
            _UsageError: If the filesystem scan fails or is incomplete.
        """

        root = "/" if value.startswith("/") else cwd
        pattern = value.lstrip("/") if value.startswith("/") else value
        result = self.kernel.call_filesystem(
            self.session_id,
            "glob",
            {
                "pattern": pattern,
                "root": root,
                "limit": self.kernel.filesystem_config.max_result_limit,
            },
        )
        if not result["ok"]:
            raise _UsageError(_error_text(result))
        matches = result["result"]
        if matches["truncated"]:
            raise _UsageError("wildcard expansion exceeds the filesystem limits")
        return matches["matches"] or [value]

    def _expanded_path(self, word: _Word | None, cwd: str) -> str | None:
        """Expand one redirection word into exactly one virtual path.

        Args:
            word: Optional parsed path word.
            cwd: Current absolute virtual directory.

        Returns:
            Absolute path or ``None`` when no redirection exists.

        Raises:
            _UsageError: If a wildcard produces more than one path.
        """

        if word is None:
            return None
        values = self._expand_word(word, cwd, allow_glob=True)
        if len(values) != 1:
            raise _UsageError("redirection path expands to more than one file")
        return self._path(values[0], cwd)

    def _expand_parameters(self, text: str, cwd: str) -> str:
        """Expand the supported virtual shell parameter forms.

        Args:
            text: Text from an unquoted or double-quoted fragment.
            cwd: Current absolute virtual directory.

        Returns:
            Text with ``$NAME``, ``${NAME}``, and ``$?`` expanded.

        Raises:
            _UsageError: If parameter or command-substitution syntax is invalid.
        """

        if "$(" in text or "`" in text:
            raise _UsageError("command substitution is not supported")
        variables = dict(self._variables)
        variables.setdefault("HOME", self.kernel.default_working_directory)
        variables["PWD"] = cwd
        variables["OLDPWD"] = self._oldpwd
        output: list[str] = []
        index = 0
        while index < len(text):
            if text[index] != "$":
                output.append(text[index])
                index += 1
                continue
            if index + 1 >= len(text):
                output.append("$")
                break
            next_character = text[index + 1]
            if next_character == "?":
                output.append(str(self._last_status))
                index += 2
                continue
            if next_character == "{":
                end = text.find("}", index + 2)
                if end < 0:
                    raise _UsageError("parameter expansion is missing }")
                name = text[index + 2 : end]
                if _VARIABLE_NAME.fullmatch(name) is None:
                    raise _UsageError(f"unsupported parameter expansion: ${{{name}}}")
                output.append(variables.get(name, ""))
                index = end + 1
                continue
            end = index + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            if end == index + 1 or text[index + 1].isdigit():
                output.append("$")
                index += 1
                continue
            name = text[index + 1 : end]
            output.append(variables.get(name, ""))
            index = end
        return "".join(output)

    def _set_variable(self, name: str, value: str) -> None:
        """Set one bounded persistent virtual shell variable.

        Args:
            name: Valid shell variable name.
            value: Text value.

        Raises:
            _UsageError: If the name, value, or variable count is invalid.
        """

        if _VARIABLE_NAME.fullmatch(name) is None:
            raise _UsageError(f"invalid shell variable name: {name}")
        if name in {"OLDPWD", "PWD"}:
            raise _UsageError(f"{name} is managed by the virtual shell")
        if len(value) > self.kernel.config.max_shell_command_chars:
            raise _UsageError("shell variable value exceeds the command length limit")
        if (
            name not in self._variables
            and len(self._variables) >= self.kernel.config.max_shell_variables
        ):
            raise _UsageError("virtual shell variable count exceeds its limit")
        self._variables[name] = value

    def _is_available(self, name: str) -> bool:
        """Return whether one command is enabled by the allowlist.

        Args:
            name: Virtual command name.

        Returns:
            Whether the command is enabled.
        """
        allowed = self.kernel.config.allowed_shell_commands
        return allowed is None or name in allowed

    def _available_commands(self) -> list[str]:
        """Return sorted command names enabled for this shell.

        Returns:
            Enabled virtual command names.
        """
        python_commands = {
            name for name in _PYTHON_COMMANDS if self._is_available(name)
        }
        return sorted(set(self._commands) | python_commands)

    def _python(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Run Monty code or one virtual Python file.

        Args:
            args: Python command and its arguments.
            stdin: Pipeline input, which Python commands do not accept.
            cwd: Current absolute virtual directory.

        Returns:
            Monty execution result.
        """
        if stdin is not None:
            return ShellResult("python does not accept shell input", 2)
        if len(args) == 3 and args[1] == "-c":
            result = self.kernel.execute(
                self.session_id,
                args[2],
                self.kernel.config.max_output_limit_chars,
            )
            return shell_result_from_execution(result)
        if len(args) == 2 and args[1].endswith(".py"):
            result = self.kernel.execute_file(
                self.session_id,
                self._path(args[1], cwd),
                output_limit_chars=self.kernel.config.max_output_limit_chars,
            )
            return shell_result_from_execution(result)
        return ShellResult(
            "Usage: python -c CODE or python FILE.py",
            2,
        )

    def _help(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """List enabled commands or show one command usage.

        Args:
            args: Optional command name.
            stdin: Pipeline input, which this command rejects.
            cwd: Current virtual directory. This command does not use it.

        Returns:
            Enabled commands or one usage line.
        """
        _expect_no_input(stdin, "help")
        _expect_count(args, 0, 1, "help [COMMAND]")
        available = self._available_commands()
        if not args:
            return ShellResult(
                "Commands: "
                + " ".join(available)
                + "\nOperators: | ; newline && || < > >> <<\n",
                0,
            )
        name = args[0]
        if name not in available:
            return ShellResult(f"Unsupported command: {name}", 127)
        return ShellResult(_COMMAND_USAGE[name] + "\n", 0)

    def _which(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Report which command names are enabled in the virtual shell.

        Args:
            args: Command names to check.
            stdin: Pipeline input, which this command rejects.
            cwd: Current virtual directory. This command does not use it.

        Returns:
            Command names and a nonzero code if one is not enabled.
        """
        _expect_no_input(stdin, "which")
        if not args:
            raise _UsageError("Usage: which COMMAND...")
        available = set(self._available_commands())
        lines = [name if name in available else f"{name}: not found" for name in args]
        missing = any(name not in available for name in args)
        return ShellResult("".join(f"{line}\n" for line in lines), int(missing))

    def _basename(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Return the last part of one POSIX path.

        Args:
            args: One path to inspect.
            stdin: Pipeline input, which this command rejects.
            cwd: Current virtual directory. This command does not use it.

        Returns:
            Last path part.
        """
        _expect_no_input(stdin, "basename")
        _expect_count(args, 1, 1, "basename PATH")
        trimmed = args[0].rstrip("/")
        return ShellResult((posixpath.basename(trimmed) if trimmed else "/") + "\n", 0)

    def _dirname(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Return the directory part of one POSIX path.

        Args:
            args: One path to inspect.
            stdin: Pipeline input, which this command rejects.
            cwd: Current virtual directory. This command does not use it.

        Returns:
            Directory path part.
        """
        _expect_no_input(stdin, "dirname")
        _expect_count(args, 1, 1, "dirname PATH")
        trimmed = args[0].rstrip("/")
        directory = "/" if not trimmed else posixpath.dirname(trimmed) or "."
        return ShellResult(directory + "\n", 0)

    def _pwd(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Print the current virtual directory."""
        _expect_no_input(stdin, "pwd")
        _expect_count(args, 0, 0, "pwd")
        return ShellResult(cwd + "\n", 0)

    def _cd(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Change the persistent virtual directory."""
        _expect_no_input(stdin, "cd")
        _expect_count(args, 0, 1, "cd [PATH]")
        show_target = bool(args and args[0] == "-")
        requested = (
            self._oldpwd
            if show_target
            else args[0]
            if args
            else self._variables.get("HOME", self.kernel.default_working_directory)
        )
        target = self._path(requested, cwd)
        listed = self.kernel.call_filesystem(
            self.session_id,
            "ls",
            {"path": target, "recursive": False, "limit": 1},
        )
        if not listed["ok"]:
            return ShellResult(_error_text(listed), 1)
        self._oldpwd = cwd
        result = self.kernel.execute(
            self.session_id,
            f"import os as _arenero_os\n_arenero_os.chdir({target!r})",
            self.kernel.config.max_output_limit_chars,
        )
        changed = shell_result_from_execution(result)
        if changed.exit_code == 0:
            self._current_cwd = target
        if changed.exit_code == 0 and show_target:
            return ShellResult(target + "\n", 0)
        return changed

    def _ls(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """List bounded virtual paths with common Bash-style flags."""
        _expect_no_input(stdin, "ls")
        flags: set[str] = set()
        while args and args[0].startswith("-") and args[0] != "-":
            option = args.pop(0)
            if option == "--":
                break
            short = option[1:]
            if not short or any(flag not in "aA1ldR" for flag in short):
                raise _UsageError("Usage: ls [-aA1ldR] [PATH ...]")
            flags.update(short)
        paths = args or ["."]
        output: list[str] = []
        truncated = False
        for path in paths:
            target = self._path(path, cwd)
            kind = self._path_kind(target)
            if kind is None:
                return ShellResult(f"ls: path does not exist: {path}", 1, truncated)
            if kind == "file" or "d" in flags:
                output.append(_ls_line(target, kind, "l" in flags))
                continue
            result = self.kernel.call_filesystem(
                self.session_id,
                "ls",
                {
                    "path": target,
                    "recursive": "R" in flags,
                    "limit": self.kernel.filesystem_config.max_result_limit,
                },
            )
            if not result["ok"]:
                return ShellResult(_error_text(result), 1, truncated)
            value = result["result"]
            output.extend(
                _ls_line(entry["path"], entry["type"], "l" in flags)
                for entry in sorted(value["entries"], key=lambda item: item["path"])
            )
            truncated = truncated or value["truncated"]
        return ShellResult("".join(f"{line}\n" for line in output), 0, truncated)

    def _cat(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Concatenate UTF-8 virtual files or pipeline input."""
        if not args:
            if stdin is None:
                raise _UsageError("Usage: cat FILE... or pipe text into cat")
            return ShellResult(stdin, 0)
        output = ""
        for path in args:
            read = (
                stdin
                if path == "-" and stdin is not None
                else ShellResult("cat: standard input is not available", 2)
                if path == "-"
                else self._read_text(self._path(path, cwd))
            )
            if isinstance(read, ShellResult):
                return read
            output, truncated = _trim(
                output + read,
                self.kernel.config.max_output_limit_chars,
            )
            if truncated:
                return ShellResult(output, 0, True)
        return ShellResult(output, 0)

    def _curl(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Fetch one allowlisted URL and return or save its bounded response.

        Args:
            args: Supported curl flags and one absolute URL.
            stdin: Pipeline input, which curl does not accept.
            cwd: Current absolute virtual directory.

        Returns:
            Response text, an empty successful file-download result, or an error.
        """
        _expect_no_input(stdin, "curl")
        options = _parse_curl_arguments(args)
        config = self.kernel.network_config
        emit_event(
            self.kernel.event_sink,
            "network",
            "request_started",
            session_id=self.session_id,
            method=options.method,
            request_bytes=len(options.data or b""),
        )
        response = fetch_url(
            NetworkRequest(
                url=options.url,
                method=options.method,
                headers=tuple(options.headers.items()),
                body=options.data,
            ),
            config,
            follow_redirects=options.follow_redirects,
            timeout_seconds=min(
                options.timeout_seconds or config.timeout_seconds,
                remaining() or config.timeout_seconds,
            ),
        )
        remaining()
        emit_event(
            self.kernel.event_sink,
            "network",
            "request_finished",
            session_id=self.session_id,
            status=response.status,
            response_bytes=len(response.body),
        )
        if options.fail_on_http_error and response.status >= 400:
            return ShellResult(
                f"curl: HTTP {response.status} {response.reason}".rstrip(),
                22,
            )

        header_text = _curl_header_text(response) if options.include_headers else ""
        if options.output_path is not None:
            content = header_text.encode("utf-8") + response.body
            self.kernel.upload_file(self._path(options.output_path, cwd), content)
            return ShellResult("", 0)
        try:
            body_text = response.body.decode("utf-8")
        except UnicodeDecodeError:
            return ShellResult(
                "curl response is not UTF-8 text; use -o FILE to save its bytes",
                1,
            )
        return ShellResult(header_text + body_text, 0)

    def _base64(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Encode or decode one virtual file or pipeline input.

        Args:
            args: Optional virtual file path.
            stdin: Optional UTF-8 pipeline input.
            cwd: Current absolute virtual directory.

        Returns:
            Base64 text or an error result.
        """
        decode = bool(args and args[0] in {"-d", "--decode"})
        paths = args[1:] if decode else args
        _expect_count(paths, 0, 1, "base64 [-d] [FILE]")
        if paths and paths[0] == "-":
            if stdin is None:
                return ShellResult("base64: standard input is not available", 2)
            content = stdin.encode("utf-8")
        elif paths:
            try:
                content = self.kernel.download_file(self._path(paths[0], cwd))
            except _SHELL_ERRORS as exc:
                return ShellResult(str(exc), 1)
        elif stdin is not None:
            content = stdin.encode("utf-8")
        else:
            return ShellResult("base64 needs a file or pipeline input", 2)
        if decode:
            try:
                decoded = base64.b64decode(b"".join(content.split()), validate=True)
            except (binascii.Error, ValueError) as exc:
                return ShellResult(f"base64: invalid input: {exc}", 1)
            try:
                return ShellResult(decoded.decode("utf-8"), 0)
            except UnicodeDecodeError:
                return ShellResult(
                    "base64 decoded bytes are not UTF-8 and cannot enter a text pipe",
                    1,
                )
        encoded = base64.b64encode(content).decode("ascii") + "\n"
        if len(encoded) > self.kernel.config.max_output_limit_chars:
            return ShellResult("base64 output exceeds the shell output limit", 1)
        return ShellResult(encoded, 0)

    def _sha256sum(
        self,
        args: list[str],
        stdin: str | None,
        cwd: str,
    ) -> ShellResult:
        """Calculate SHA-256 hashes for virtual files or pipeline input.

        Args:
            args: Optional virtual file paths.
            stdin: Optional UTF-8 pipeline input.
            cwd: Current absolute virtual directory.

        Returns:
            Hash lines or an error result.
        """
        values: list[tuple[str, bytes]] = []
        if args:
            for path in args:
                if path == "-":
                    if stdin is None:
                        return ShellResult(
                            "sha256sum: standard input is not available", 2
                        )
                    values.append(("-", stdin.encode("utf-8")))
                    continue
                target = self._path(path, cwd)
                try:
                    values.append((path, self.kernel.download_file(target)))
                except _SHELL_ERRORS as exc:
                    return ShellResult(str(exc), 1)
        elif stdin is not None:
            values.append(("-", stdin.encode("utf-8")))
        else:
            return ShellResult("sha256sum needs a file or pipeline input", 2)
        output = "".join(
            f"{hashlib.sha256(content).hexdigest()}  {name}\n"
            for name, content in values
        )
        return ShellResult(output, 0)

    def _head(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Return the first lines of a UTF-8 file or pipeline input."""
        count, paths = _line_arguments(args, "head")
        text = self._command_text(paths, stdin, cwd, "head")
        if isinstance(text, ShellResult):
            return text
        return ShellResult("".join(text.splitlines(keepends=True)[:count]), 0)

    def _tail(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Return the last lines of a UTF-8 file or pipeline input."""
        count, paths = _line_arguments(args, "tail")
        text = self._command_text(paths, stdin, cwd, "tail")
        if isinstance(text, ShellResult):
            return text
        if count == 0:
            return ShellResult("", 0)
        return ShellResult("".join(text.splitlines(keepends=True)[-count:]), 0)

    def _grep(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Search bounded UTF-8 input with common Bash-style grep flags."""
        flags: set[str] = set()
        while args and args[0].startswith("-") and args[0] != "-":
            option = args.pop(0)
            if option == "--":
                break
            short = option[1:]
            if not short or any(flag not in "EFinvcqlrR" for flag in short):
                raise _UsageError("Usage: grep [-EFinvcqlr] PATTERN [FILE ...]")
            flags.update(short)
        if not args:
            raise _UsageError("Usage: grep [-EFinvcqlr] PATTERN [FILE ...]")
        if "E" in flags and "F" in flags:
            raise _UsageError("grep -E and -F cannot be used together")
        pattern, paths = args[0], args[1:]
        sources_result = self._grep_sources(paths, stdin, cwd, bool(flags & {"r", "R"}))
        if isinstance(sources_result, ShellResult):
            return sources_result
        sources = sources_result
        multiple = len(sources) > 1
        expression = pattern if "E" in flags else re.escape(pattern)
        try:
            matcher = re.compile(expression, re.IGNORECASE if "i" in flags else 0)
        except re.error as exc:
            return ShellResult(f"grep: invalid regular expression: {exc}", 2)
        output: list[str] = []
        selected_total = 0
        characters_scanned = 0
        truncated = False
        for path, text in sources:
            remaining = (
                self.kernel.filesystem_config.max_grep_chars - characters_scanned
            )
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining]
                truncated = True
            characters_scanned += len(text)
            file_count = 0
            for number, line in enumerate(text.splitlines(), start=1):
                selected = matcher.search(line) is not None
                if "v" in flags:
                    selected = not selected
                if not selected:
                    continue
                file_count += 1
                selected_total += 1
                if "q" in flags:
                    return ShellResult("", 0, truncated)
                if "l" in flags:
                    if (
                        len(output)
                        >= self.kernel.filesystem_config.default_result_limit
                    ):
                        truncated = True
                        break
                    output.append((path or "-") + "\n")
                    break
                if "c" in flags:
                    continue
                if len(output) >= self.kernel.filesystem_config.default_result_limit:
                    truncated = True
                    break
                prefix = f"{path}:" if multiple and path is not None else ""
                if "n" in flags:
                    prefix += f"{number}:"
                output.append(f"{prefix}{line}\n")
            if "c" in flags:
                if len(output) >= self.kernel.filesystem_config.default_result_limit:
                    truncated = True
                    break
                prefix = f"{path}:" if multiple and path is not None else ""
                output.append(f"{prefix}{file_count}\n")
            if truncated:
                break
        return ShellResult(
            "".join(output),
            0 if selected_total else 1,
            truncated,
        )

    def _grep_sources(
        self,
        paths: list[str],
        stdin: str | None,
        cwd: str,
        recursive: bool,
    ) -> list[tuple[str | None, str]] | ShellResult:
        """Collect bounded UTF-8 grep sources.

        Args:
            paths: File or directory arguments.
            stdin: Optional pipeline input.
            cwd: Current absolute virtual directory.
            recursive: Whether directory arguments are accepted.

        Returns:
            Source labels and text, or a shell error.
        """

        if not paths:
            if stdin is None:
                return ShellResult("grep needs a file or pipeline input", 2)
            return [(None, stdin)]
        sources: list[tuple[str | None, str]] = []
        for path in paths:
            if path == "-":
                if stdin is None:
                    return ShellResult("grep: standard input is not available", 2)
                sources.append((None, stdin))
                continue
            target = self._path(path, cwd)
            kind = self._path_kind(target)
            if kind == "directory":
                if not recursive:
                    return ShellResult(f"grep: {path} is a directory; use -r", 1)
                listed = self.kernel.call_filesystem(
                    self.session_id,
                    "ls",
                    {
                        "path": target,
                        "recursive": True,
                        "limit": self.kernel.filesystem_config.max_result_limit,
                    },
                )
                if not listed["ok"]:
                    return ShellResult(_error_text(listed), 1)
                value = listed["result"]
                if value["truncated"]:
                    return ShellResult(
                        "grep recursive file scan exceeds filesystem limits",
                        1,
                        True,
                    )
                file_paths = sorted(
                    entry["path"]
                    for entry in value["entries"]
                    if entry["type"] == "file"
                )
            elif kind == "file":
                file_paths = [target]
            else:
                return ShellResult(f"grep: path does not exist: {path}", 1)
            for file_path in file_paths:
                read = self._read_text(file_path)
                if isinstance(read, ShellResult):
                    return read
                sources.append((file_path, read))
        return sources

    def _find(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Find virtual paths below a root with a glob pattern."""
        _expect_no_input(stdin, "find")
        _expect_count(args, 0, 2, "find [PATH] [PATTERN]")
        root = self._path(args[0] if args else ".", cwd)
        pattern = args[1] if len(args) == 2 else "**/*"
        return self._glob_result(root, pattern)

    def _glob(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Find virtual paths that match a glob pattern."""
        _expect_no_input(stdin, "glob")
        _expect_count(args, 1, 2, "glob PATTERN [PATH]")
        root = self._path(args[1] if len(args) == 2 else ".", cwd)
        return self._glob_result(root, args[0])

    def _glob_result(self, root: str, pattern: str) -> ShellResult:
        """Run one bounded virtual glob operation."""
        result = self.kernel.call_filesystem(
            self.session_id,
            "glob",
            {"pattern": pattern, "root": root, "limit": None},
        )
        if not result["ok"]:
            return ShellResult(_error_text(result), 1)
        value = result["result"]
        return ShellResult(
            "".join(f"{path}\n" for path in value["matches"]),
            0,
            value["truncated"],
        )

    def _echo(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Print literal arguments."""
        _expect_no_input(stdin, "echo")
        newline = not args or args[0] != "-n"
        values = args if newline else args[1:]
        return ShellResult(" ".join(values) + ("\n" if newline else ""), 0)

    def _env(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Print the persistent virtual shell variables.

        Args:
            args: No arguments are supported.
            stdin: Pipeline input, which this command rejects.
            cwd: Current absolute virtual directory.

        Returns:
            Sorted ``NAME=VALUE`` lines.
        """

        _expect_no_input(stdin, "env")
        _expect_count(args, 0, 0, "env")
        variables = dict(self._variables)
        variables.setdefault("HOME", self.kernel.default_working_directory)
        variables["OLDPWD"] = self._oldpwd
        variables["PWD"] = cwd
        return ShellResult(
            "".join(f"{name}={variables[name]}\n" for name in sorted(variables)),
            0,
        )

    def _export(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Set or list persistent virtual shell variables.

        Args:
            args: Variable names or ``NAME=VALUE`` assignments.
            stdin: Pipeline input, which this command rejects.
            cwd: Current directory used only when listing variables.

        Returns:
            Empty output after changes, or the current variable list.
        """

        _expect_no_input(stdin, "export")
        if not args:
            return self._env([], None, cwd)
        for item in args:
            assignment = _split_assignment(item)
            if assignment is None:
                self._set_variable(item, self._variables.get(item, ""))
            else:
                self._set_variable(*assignment)
        return ShellResult("", 0)

    def _unset(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Remove persistent virtual shell variables.

        Args:
            args: Variable names to remove.
            stdin: Pipeline input, which this command rejects.
            cwd: Current directory, which this command does not use.

        Returns:
            Empty output on success.
        """

        _expect_no_input(stdin, "unset")
        if not args:
            raise _UsageError("Usage: unset NAME...")
        for name in args:
            if _VARIABLE_NAME.fullmatch(name) is None:
                raise _UsageError(f"invalid shell variable name: {name}")
            if name in {"OLDPWD", "PWD"}:
                raise _UsageError(f"{name} is managed by the virtual shell")
            self._variables.pop(name, None)
        return ShellResult("", 0)

    def _true(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Return shell success without output."""

        _expect_no_input(stdin, "true")
        _expect_count(args, 0, 0, "true")
        return ShellResult("", 0)

    def _false(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Return shell failure without output."""

        _expect_no_input(stdin, "false")
        _expect_count(args, 0, 0, "false")
        return ShellResult("", 1)

    def _test(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Evaluate one small Bash-like file, string, or integer expression.

        Args:
            args: Supported ``test`` expression tokens.
            stdin: Pipeline input, which this command rejects.
            cwd: Current absolute virtual directory.

        Returns:
            Exit code zero when the expression is true, otherwise one.
        """

        _expect_no_input(stdin, "test")
        return ShellResult("", int(not self._test_value(args, cwd)))

    def _bracket(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Evaluate a bracketed alias of the virtual ``test`` command.

        Args:
            args: Expression tokens followed by ``]``.
            stdin: Pipeline input, which this command rejects.
            cwd: Current absolute virtual directory.

        Returns:
            Exit code zero when the expression is true, otherwise one.
        """

        _expect_no_input(stdin, "[")
        if not args or args[-1] != "]":
            raise _UsageError("Usage: [ EXPRESSION ]")
        return ShellResult("", int(not self._test_value(args[:-1], cwd)))

    def _test_value(self, args: list[str], cwd: str) -> bool:
        """Return the value of one supported test expression.

        Args:
            args: Expression tokens without a closing bracket.
            cwd: Current absolute virtual directory.

        Returns:
            Boolean expression result.
        """

        if not args:
            return False
        if len(args) == 1:
            return bool(args[0])
        if len(args) == 2:
            operator, value = args
            if operator == "!":
                return not bool(value)
            if operator == "-n":
                return bool(value)
            if operator == "-z":
                return not value
            if operator in {"-d", "-e", "-f", "-s"}:
                target = self._path(value, cwd)
                kind = self._path_kind(target)
                if operator == "-e":
                    return kind is not None
                if operator == "-d":
                    return kind == "directory"
                if operator == "-f":
                    return kind == "file"
                if kind != "file":
                    return False
                try:
                    return bool(self.kernel.download_file(target))
                except _SHELL_ERRORS:
                    return False
        if len(args) == 3:
            left, operator, right = args
            if operator in {"=", "=="}:
                return left == right
            if operator == "!=":
                return left != right
            if operator in {"-eq", "-ne", "-lt", "-le", "-gt", "-ge"}:
                try:
                    first, second = int(left), int(right)
                except ValueError as exc:
                    raise _UsageError("integer test values must be integers") from exc
                comparisons = {
                    "-eq": first == second,
                    "-ne": first != second,
                    "-lt": first < second,
                    "-le": first <= second,
                    "-gt": first > second,
                    "-ge": first >= second,
                }
                return comparisons[operator]
        raise _UsageError(
            "test supports one string, -n, -z, -e, -f, -d, -s, string "
            "comparison, or integer comparison"
        )

    def _touch(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Create missing regular files without changing existing content.

        Args:
            args: One or more virtual file paths.
            stdin: Pipeline input, which this command rejects.
            cwd: Current absolute virtual directory.

        Returns:
            Empty output or the first file error.
        """

        _expect_no_input(stdin, "touch")
        if args and args[0] == "--":
            args = args[1:]
        if not args:
            raise _UsageError("Usage: touch FILE...")
        for path in args:
            target = self._path(path, cwd)
            kind = self._path_kind(target)
            if kind == "directory":
                return ShellResult(f"touch: {path} is a directory", 1)
            if kind is None:
                try:
                    self.kernel.upload_file(target, b"")
                except _SHELL_ERRORS as exc:
                    return ShellResult(str(exc), 1)
        return ShellResult("", 0)

    def _printf(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Format text with a small Bash-like ``printf`` subset.

        Args:
            args: Format string followed by values.
            stdin: Pipeline input, which this command rejects.
            cwd: Current directory, which this command does not use.

        Returns:
            Formatted UTF-8 text.
        """

        _expect_no_input(stdin, "printf")
        if not args:
            raise _UsageError("Usage: printf FORMAT [ARGUMENT ...]")
        return ShellResult(_format_printf(args[0], args[1:]), 0)

    def _source(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Run one straight-line virtual shell script in the current state.

        Args:
            args: One virtual script path.
            stdin: Pipeline input, which this command rejects.
            cwd: Current absolute virtual directory.

        Returns:
            Combined script output and final exit code.
        """

        _expect_no_input(stdin, "source")
        _expect_count(args, 1, 1, "source FILE")
        if self._script_depth >= self.kernel.config.max_shell_script_depth:
            return ShellResult("source nesting exceeds the shell script depth limit", 2)
        script = self._read_text(self._path(args[0], cwd))
        if isinstance(script, ShellResult):
            return script
        self._script_depth += 1
        try:
            result, _ = self._execute_program(script, cwd)
        finally:
            self._script_depth -= 1
        return result

    def _wc(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Count lines, words, and UTF-8 bytes."""
        flags = "lwc"
        if args and args[0].startswith("-"):
            flags = args.pop(0)[1:]
            if not flags or any(flag not in "lwc" for flag in flags):
                raise _UsageError("Usage: wc [-lwc] [FILE]")
        _expect_count(args, 0, 1, "wc [-lwc] [FILE]")
        text = self._command_text(args, stdin, cwd, "wc")
        if isinstance(text, ShellResult):
            return text
        values = {
            "l": len(text.splitlines()),
            "w": len(text.split()),
            "c": len(text.encode("utf-8")),
        }
        return ShellResult(" ".join(str(values[flag]) for flag in flags) + "\n", 0)

    def _sort(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Sort UTF-8 input lines.

        Args:
            args: Optional ``-r`` flag and virtual file path.
            stdin: Optional UTF-8 pipeline input.
            cwd: Current absolute virtual directory.

        Returns:
            Sorted lines or an error result.
        """
        reverse = bool(args and args[0] == "-r")
        paths = args[1:] if reverse else args
        _expect_count(paths, 0, 1, "sort [-r] [FILE]")
        text = self._command_text(paths, stdin, cwd, "sort")
        if isinstance(text, ShellResult):
            return text
        lines = sorted(text.splitlines(), reverse=reverse)
        return ShellResult("".join(f"{line}\n" for line in lines), 0)

    def _uniq(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Remove adjacent duplicate UTF-8 input lines.

        Args:
            args: Optional ``-c`` flag and virtual file path.
            stdin: Optional UTF-8 pipeline input.
            cwd: Current absolute virtual directory.

        Returns:
            Adjacent unique lines or an error result.
        """
        count = bool(args and args[0] == "-c")
        paths = args[1:] if count else args
        _expect_count(paths, 0, 1, "uniq [-c] [FILE]")
        text = self._command_text(paths, stdin, cwd, "uniq")
        if isinstance(text, ShellResult):
            return text
        groups: list[tuple[str, int]] = []
        for line in text.splitlines():
            if groups and groups[-1][0] == line:
                previous, total = groups[-1]
                groups[-1] = previous, total + 1
            else:
                groups.append((line, 1))
        output = "".join(
            f"{total:7} {line}\n" if count else f"{line}\n" for line, total in groups
        )
        return ShellResult(output, 0)

    def _rev(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Reverse the characters in each UTF-8 input line.

        Args:
            args: Optional virtual file path.
            stdin: Optional UTF-8 pipeline input.
            cwd: Current absolute virtual directory.

        Returns:
            Reversed lines or an error result.
        """
        _expect_count(args, 0, 1, "rev [FILE]")
        text = self._command_text(args, stdin, cwd, "rev")
        if isinstance(text, ShellResult):
            return text
        output = ""
        for line in text.splitlines(keepends=True):
            ending = ""
            if line.endswith("\n"):
                line, ending = line[:-1], "\n"
            if line.endswith("\r"):
                line, ending = line[:-1], "\r" + ending
            output += line[::-1] + ending
        return ShellResult(output, 0)

    def _tee(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Copy pipeline input to virtual files and command output.

        Args:
            args: Optional ``-a`` flag and one or more virtual file paths.
            stdin: Required UTF-8 pipeline input.
            cwd: Current absolute virtual directory.

        Returns:
            Unchanged pipeline input or an error result.
        """
        append = bool(args and args[0] == "-a")
        paths = args[1:] if append else args
        if not paths:
            raise _UsageError("Usage: tee [-a] FILE...")
        if stdin is None:
            raise _UsageError("tee needs pipeline input")
        for path in paths:
            written = self._write_text(
                self._path(path, cwd),
                stdin,
                append=append,
            )
            if written is not None:
                return written
        return ShellResult(stdin, 0)

    def _mkdir(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Create one or more virtual directories."""
        _expect_no_input(stdin, "mkdir")
        parents = bool(args and args[0] == "-p")
        paths = args[1:] if parents else args
        if not paths:
            raise _UsageError("Usage: mkdir [-p] PATH...")
        for path in paths:
            result = self.kernel.call_filesystem(
                self.session_id,
                "mkdir",
                {
                    "path": self._path(path, cwd),
                    "parents": parents,
                    "exist_ok": parents,
                },
            )
            if not result["ok"]:
                return ShellResult(_error_text(result), 1)
        return ShellResult("", 0)

    def _rm(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Delete virtual files or bounded directory trees."""
        _expect_no_input(stdin, "rm")
        recursive = False
        force = False
        while args and args[0].startswith("-"):
            flags = args.pop(0)[1:]
            if not flags or any(flag not in "rf" for flag in flags):
                raise _UsageError("Usage: rm [-r] [-f] PATH...")
            recursive = recursive or "r" in flags
            force = force or "f" in flags
        if not args:
            raise _UsageError("Usage: rm [-r] [-f] PATH...")
        for path in args:
            result = self.kernel.call_filesystem(
                self.session_id,
                "delete",
                {"path": self._path(path, cwd), "recursive": recursive},
            )
            if not result["ok"]:
                error = _error_text(result)
                if not force or "FileNotFoundError" not in error:
                    return ShellResult(error, 1)
        return ShellResult("", 0)

    def _cp(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Copy bounded virtual files or directory trees."""
        _expect_no_input(stdin, "cp")
        recursive, sources, destination = _copy_arguments(args, "cp")
        target = self._path(destination, cwd)
        target_kind = self._path_kind(target)
        if len(sources) > 1 and target_kind != "directory":
            return ShellResult("cp: multiple sources need a directory destination", 2)
        for item in sources:
            source = self._path(item, cwd)
            final = (
                posixpath.join(target, posixpath.basename(source))
                if target_kind == "directory"
                else target
            )
            copied = self._copy_path(source, final, recursive)
            if copied is not None:
                return copied
        return ShellResult("", 0)

    def _mv(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Move bounded virtual files or directory trees with copy then delete."""
        _expect_no_input(stdin, "mv")
        _, sources, destination = _copy_arguments(args, "mv")
        recursive = True
        target = self._path(destination, cwd)
        target_kind = self._path_kind(target)
        if len(sources) > 1 and target_kind != "directory":
            return ShellResult("mv: multiple sources need a directory destination", 2)
        for item in sources:
            source = self._path(item, cwd)
            source_kind = self._path_kind(source)
            if source_kind is None:
                return ShellResult(f"mv: path does not exist: {item}", 1)
            final = (
                posixpath.join(target, posixpath.basename(source))
                if target_kind == "directory"
                else target
            )
            copied = self._copy_path(source, final, recursive)
            if copied is not None:
                return copied
            result = self.kernel.call_filesystem(
                self.session_id,
                "delete",
                {"path": source, "recursive": source_kind == "directory"},
            )
            if not result["ok"]:
                return ShellResult(_error_text(result), 1)
        return ShellResult("", 0)

    def _copy_path(
        self,
        source: str,
        destination: str,
        recursive: bool,
    ) -> ShellResult | None:
        """Copy one regular file or bounded directory tree.

        Args:
            source: Absolute virtual source path.
            destination: Absolute virtual destination path.
            recursive: Whether directory trees are allowed.

        Returns:
            ``None`` on success or the first shell error.
        """

        kind = self._path_kind(source)
        if kind is None:
            return ShellResult(f"cp: path does not exist: {source}", 1)
        if source == destination:
            return ShellResult("source and destination must differ", 2)
        if kind == "file":
            try:
                self.kernel.upload_file(destination, self.kernel.download_file(source))
            except _SHELL_ERRORS as exc:
                return ShellResult(str(exc), 1)
            return None
        if not recursive:
            return ShellResult(f"cp: {source} is a directory; use -r", 1)
        prefix = source.rstrip("/") + "/"
        if destination.startswith(prefix):
            return ShellResult("cp: cannot copy a directory inside itself", 1)
        listed = self.kernel.call_filesystem(
            self.session_id,
            "ls",
            {
                "path": source,
                "recursive": True,
                "limit": self.kernel.filesystem_config.max_result_limit,
            },
        )
        if not listed["ok"]:
            return ShellResult(_error_text(listed), 1)
        value = listed["result"]
        if value["truncated"]:
            return ShellResult("cp: directory scan exceeds filesystem limits", 1, True)
        made = self.kernel.call_filesystem(
            self.session_id,
            "mkdir",
            {"path": destination, "parents": True, "exist_ok": True},
        )
        if not made["ok"]:
            return ShellResult(_error_text(made), 1)
        entries = sorted(
            value["entries"],
            key=lambda entry: (entry["type"] != "directory", entry["path"]),
        )
        transferred = 0
        for entry in entries:
            relative = entry["path"][len(prefix) :]
            target = posixpath.join(destination, relative)
            if entry["type"] == "directory":
                result = self.kernel.call_filesystem(
                    self.session_id,
                    "mkdir",
                    {"path": target, "parents": True, "exist_ok": True},
                )
                if not result["ok"]:
                    return ShellResult(_error_text(result), 1)
                continue
            if entry["type"] != "file":
                return ShellResult("cp: symbolic links are not supported", 1)
            try:
                content = self.kernel.download_file(entry["path"])
                transferred += len(content)
                if transferred > self.kernel.filesystem_config.max_transfer_bytes:
                    return ShellResult("cp: directory transfer exceeds byte limit", 1)
                self.kernel.upload_file(target, content)
            except _SHELL_ERRORS as exc:
                return ShellResult(str(exc), 1)
        return None

    def _command_text(
        self,
        paths: list[str],
        stdin: str | None,
        cwd: str,
        command: str,
    ) -> str | ShellResult:
        """Read one command input from a file or pipeline.

        Args:
            paths: Zero or one virtual file path.
            stdin: Optional pipeline input.
            cwd: Current absolute virtual directory.
            command: Command name used in error text.

        Returns:
            Input text or an error result.
        """
        _expect_count(paths, 0, 1, f"{command} [-n COUNT] [FILE]")
        if paths and paths[0] == "-":
            if stdin is None:
                return ShellResult(f"{command}: standard input is not available", 2)
            return stdin
        if paths:
            return self._read_text(self._path(paths[0], cwd))
        if stdin is None:
            return ShellResult(f"{command} needs a file or pipeline input", 2)
        return stdin

    def _read_text(self, path: str) -> str | ShellResult:
        """Read one bounded UTF-8 virtual file."""
        try:
            return self.kernel.download_file(path).decode("utf-8")
        except UnicodeDecodeError:
            return ShellResult(f"{path} is not a UTF-8 text file", 1)
        except _SHELL_ERRORS as exc:
            return ShellResult(str(exc), 1)

    def _write_text(
        self,
        path: str,
        content: str,
        *,
        append: bool,
    ) -> ShellResult | None:
        """Write redirected text through the bounded filesystem helper."""
        result = self.kernel.call_filesystem(
            self.session_id,
            "write",
            {
                "path": path,
                "content": content,
                "append": append,
                "create_parents": True,
            },
        )
        return None if result["ok"] else ShellResult(_error_text(result), 1)

    def _load_shell_state(self) -> tuple[dict[str, str], str, int, str] | ShellResult:
        """Read shell state and Monty's native virtual directory in one call.

        Returns:
            Variables, previous directory, exit status, and directory, or an error.
        """

        captured: dict[str, Any] = {}

        def _arenero_capture_shell_state(value: dict[str, Any], cwd: str) -> bool:
            """Copy internal shell state across the Monty host boundary.

            Args:
                value: Persisted virtual shell state.
                cwd: Monty's current virtual directory.

            Returns:
                Always ``True`` after the value is captured.
            """

            captured["value"] = value
            captured["cwd"] = cwd
            return True

        code = (
            "import os as _arenero_os\n"
            f"try:\n    {_STATE_NAME}\n"
            f"except NameError:\n    {_STATE_NAME} = "
            "{'variables': {}, 'oldpwd': _arenero_os.getcwd(), 'status': 0}\n"
            f"_arenero_capture_shell_state(value={_STATE_NAME}, cwd=_arenero_os.getcwd())"
        )
        result = self.kernel.execute(
            self.session_id,
            code,
            self.kernel.config.max_output_limit_chars,
            hosted_tools=HostedToolSet.from_callables([_arenero_capture_shell_state]),
        )
        if not result["ok"]:
            return ShellResult(_error_text(result), 1, _was_truncated(result))
        value = captured.get("value")
        if not isinstance(value, dict):
            return ShellResult("The virtual shell variable state is invalid", 1)
        variables = value.get("variables")
        oldpwd = value.get("oldpwd")
        status = value.get("status")
        if (
            not isinstance(variables, dict)
            or len(variables) > self.kernel.config.max_shell_variables
            or any(
                not isinstance(name, str)
                or _VARIABLE_NAME.fullmatch(name) is None
                or name in {"OLDPWD", "PWD"}
                or not isinstance(item, str)
                or len(item) > self.kernel.config.max_shell_command_chars
                for name, item in variables.items()
            )
            or not isinstance(oldpwd, str)
            or not oldpwd.startswith("/")
            or not isinstance(status, int)
        ):
            return ShellResult("The virtual shell variable state is invalid", 1)
        return dict(variables), oldpwd, status, captured["cwd"]

    def _save_shell_state(self) -> ShellResult | None:
        """Save current virtual variables, previous directory, and exit status.

        Returns:
            ``None`` on success or an error result.
        """

        state = {
            "variables": self._variables,
            "oldpwd": self._oldpwd,
            "status": self._last_status,
        }
        result = self.kernel.execute(
            self.session_id,
            f"{_STATE_NAME} = {state!r}",
            self.kernel.config.max_output_limit_chars,
        )
        if result["ok"]:
            return None
        return ShellResult(_error_text(result), 1, _was_truncated(result))

    def _path_kind(self, path: str) -> str | None:
        """Return whether one exact virtual path is a file or directory.

        Args:
            path: Absolute normalized virtual path.

        Returns:
            ``file``, ``directory``, or ``None`` when the path does not exist.
        """

        if path == "/":
            return "directory"
        parent = posixpath.dirname(path) or "/"
        name = posixpath.basename(path)
        found = self.kernel.call_filesystem(
            self.session_id,
            "glob",
            {"pattern": name, "root": parent, "limit": 2},
        )
        if not found["ok"] or path not in found["result"]["matches"]:
            return None
        listed = self.kernel.call_filesystem(
            self.session_id,
            "ls",
            {"path": path, "recursive": False, "limit": 1},
        )
        return "directory" if listed["ok"] else "file"

    def _path(self, path: str, cwd: str) -> str:
        """Resolve one path against the virtual working directory."""
        if not path:
            raise _UsageError("virtual paths must not be empty")
        joined = path if path.startswith("/") else posixpath.join(cwd, path)
        normalized = posixpath.normpath(joined)
        return "/" if normalized == "/" else "/" + normalized.lstrip("/")

    def _output_limit(self, requested: int | None) -> int:
        """Validate a shell limit, then use the shared output-limit calculation.

        Args:
            requested: Nonnegative caller limit, or None for the kernel default.

        Returns:
            Effective character limit under the kernel hard maximum.

        Raises:
            _UsageError: If the caller supplies a negative limit.
        """
        if requested is not None and requested < 0:
            raise _UsageError("output_limit_chars must be at least 0")
        return _output_limit(self.kernel.config, requested)[0]


def shell_result_from_execution(result: ExecutionResult) -> ShellResult:
    """Convert one kernel result to a virtual shell result.

    Args:
        result: Structured kernel execution result.

    Returns:
        Combined output, exit code, and truncation state.
    """
    parts = [result["stdout"], result["stderr"]]
    if result["ok"] and result["result"] is not None:
        value = result["result"]
        parts.append(value if isinstance(value, str) else json.dumps(value))
    elif not result["ok"]:
        parts.append(_error_text(result))
    return ShellResult(
        output="".join(parts),
        exit_code=0 if result["ok"] else 1,
        truncated=_was_truncated(result),
        output_file=result["output_file"],
        output_file_error=result["output_file_error"],
    )


def _parse(command: str) -> list[_Pipeline]:
    """Parse the Bash-like virtual shell grammar.

    Args:
        command: Raw command text.

    Returns:
        Parsed pipelines and their control operators.

    Raises:
        _UsageError: If syntax is empty, incomplete, or unsupported.
    """
    prepared, heredocs = _extract_heredocs(command)
    tokens = _lex(prepared)
    if not tokens:
        raise _UsageError("command must not be empty")

    pipelines: list[_Pipeline] = []
    commands: list[_Command] = []
    pipeline_operator: str | None = None
    arguments: list[_Word] = []
    input_path: _Word | None = None
    output_path: _Word | None = None
    append = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, str) and token in {"|", ";", "&&", "||", "\n"}:
            separator = ";" if token == "\n" else token
            if not arguments:
                if token == "\n" or (commands and separator == "|"):
                    index += 1
                    continue
                raise _UsageError(f"{separator} needs a command on each side")
            commands.append(
                _parsed_command(
                    arguments,
                    input_path,
                    output_path,
                    append,
                    heredocs,
                )
            )
            arguments, input_path, output_path, append = [], None, None, False
            if separator != "|":
                pipelines.append(_Pipeline(commands, pipeline_operator))
                commands = []
                pipeline_operator = separator
        elif isinstance(token, str) and token in {"<", ">", ">>"}:
            if index + 1 >= len(tokens) or not isinstance(tokens[index + 1], _Word):
                raise _UsageError(f"{token} needs a virtual file path")
            path = tokens[index + 1]
            assert isinstance(path, _Word)
            index += 1
            if token == "<":
                if input_path is not None:
                    raise _UsageError("a command can have only one input file")
                input_path = path
            else:
                if output_path is not None:
                    raise _UsageError("a command can have only one output file")
                output_path = path
                append = token == ">>"
        elif _is_operator(token):
            raise _UsageError(f"unsupported shell operator: {token}")
        else:
            assert isinstance(token, _Word)
            arguments.append(token)
        index += 1

    if not arguments:
        if pipelines and pipeline_operator == ";" and not commands:
            return pipelines
        raise _UsageError("an operator needs a command on each side")
    commands.append(
        _parsed_command(arguments, input_path, output_path, append, heredocs)
    )
    pipelines.append(_Pipeline(commands, pipeline_operator))
    return pipelines


def _lex(command: str) -> list[_Word | str]:
    """Tokenize shell source while preserving quote modes.

    Args:
        command: Source with extracted here-document bodies.

    Returns:
        Words, operators, and newline separators.

    Raises:
        _UsageError: If a quote or escape is incomplete.
    """

    tokens: list[_Word | str] = []
    parts: list[_WordPart] = []
    quote: str | None = None
    word_started = False
    index = 0
    while index < len(command):
        character = command[index]
        if quote == "single":
            if character == "'":
                quote = None
            else:
                _append_word_part(parts, character, "single")
            index += 1
            continue
        if quote == "double":
            if character == '"':
                quote = None
            elif character == "\\":
                if index + 1 >= len(command):
                    raise _UsageError("shell escape is incomplete")
                if command[index + 1] == "\n":
                    index += 2
                    continue
                if command[index + 1] in {'"', "$", "`", "\\"}:
                    _append_word_part(parts, command[index + 1], "single")
                    index += 2
                    continue
                _append_word_part(parts, "\\", "double")
            else:
                _append_word_part(parts, character, "double")
            index += 1
            continue

        if character in {"'", '"'}:
            quote = "single" if character == "'" else "double"
            word_started = True
            _append_word_part(parts, "", quote)
        elif character == "\\":
            if index + 1 >= len(command):
                raise _UsageError("shell escape is incomplete")
            if command[index + 1] == "\n":
                index += 2
                continue
            word_started = True
            _append_word_part(parts, command[index + 1], "single")
            index += 2
            continue
        elif character == "#" and not word_started:
            while index < len(command) and command[index] != "\n":
                index += 1
            continue
        elif character.isspace():
            if word_started:
                tokens.append(_Word(tuple(parts)))
                parts = []
                word_started = False
            if character == "\n":
                tokens.append("\n")
        elif character in "|<>&;":
            if word_started:
                tokens.append(_Word(tuple(parts)))
                parts = []
                word_started = False
            pair = command[index : index + 2]
            if pair in {"&&", "||", ">>", "<<"}:
                tokens.append(pair)
                index += 2
                continue
            tokens.append(character)
        else:
            word_started = True
            _append_word_part(parts, character, "none")
        index += 1
    if quote is not None:
        raise _UsageError(f"shell {quote} quote is not closed")
    if word_started:
        tokens.append(_Word(tuple(parts)))
    return tokens


def _append_word_part(parts: list[_WordPart], text: str, quote: str) -> None:
    """Append text and merge adjacent fragments with the same quote mode.

    Args:
        parts: Word fragments to update.
        text: Text to append. Empty text preserves empty quoted words.
        quote: ``none``, ``single``, or ``double``.
    """

    if parts and parts[-1].quote == quote and text:
        previous = parts[-1]
        parts[-1] = _WordPart(previous.text + text, quote)
    elif text or not parts or parts[-1].quote != quote:
        parts.append(_WordPart(text, quote))


def _extract_heredocs(command: str) -> tuple[str, dict[str, _HereDocument]]:
    """Replace supported here documents with internal input placeholders.

    Args:
        command: Raw multiline shell source.

    Returns:
        Source without bodies and the extracted document mapping.

    Raises:
        _UsageError: If a terminator is missing.
    """

    lines = command.splitlines(keepends=True)
    output: list[str] = []
    documents: dict[str, _HereDocument] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = _heredoc_marker(line)
        if marker is None:
            output.append(line)
            index += 1
            continue
        start, end, delimiter, expand = marker
        placeholder = f"{_HEREDOC_PREFIX}{len(documents)}__"
        output.append(line[:start] + "< " + placeholder + line[end:])
        body: list[str] = []
        index += 1
        while index < len(lines) and lines[index].rstrip("\r\n") != delimiter:
            body.append(lines[index])
            index += 1
        if index >= len(lines):
            raise _UsageError(f"here document is missing terminator: {delimiter}")
        documents[placeholder] = _HereDocument("".join(body), expand)
        index += 1
    return "".join(output), documents


def _heredoc_marker(line: str) -> tuple[int, int, str, bool] | None:
    """Find one unquoted ``<<WORD`` marker in a command line.

    Args:
        line: One source line including its optional newline.

    Returns:
        Marker bounds, delimiter, and expansion flag, or ``None``.

    Raises:
        _UsageError: If the delimiter is empty or not closed.
    """

    quote: str | None = None
    index = 0
    while index + 1 < len(line):
        character = line[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character == "\\" and quote == '"':
                index += 1
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        if character == "\\":
            index += 2
            continue
        if line[index : index + 2] != "<<" or line[index : index + 3] == "<<<":
            index += 1
            continue
        start = index
        index += 2
        while index < len(line) and line[index] in " \t":
            index += 1
        if index >= len(line) or line[index] in "\r\n|<>&;":
            raise _UsageError("<< needs a here-document delimiter")
        quoted = line[index] in {"'", '"'}
        if quoted:
            delimiter_quote = line[index]
            delimiter_start = index + 1
            delimiter_end = line.find(delimiter_quote, delimiter_start)
            if delimiter_end < 0:
                raise _UsageError("here-document delimiter quote is not closed")
            delimiter = line[delimiter_start:delimiter_end]
            end = delimiter_end + 1
        else:
            delimiter_start = index
            while (
                index < len(line)
                and not line[index].isspace()
                and line[index] not in "|<>&;"
            ):
                index += 1
            delimiter = line[delimiter_start:index]
            end = index
        if not delimiter:
            raise _UsageError("here-document delimiter must not be empty")
        return start, end, delimiter, not quoted
    return None


def _parsed_command(
    arguments: list[_Word],
    input_path: _Word | None,
    output_path: _Word | None,
    append: bool,
    heredocs: dict[str, _HereDocument],
) -> _Command:
    """Build one parsed command and attach an extracted here document.

    Args:
        arguments: Unexpanded command words.
        input_path: Optional input path or internal here-document placeholder.
        output_path: Optional output path.
        append: Whether output redirection appends.
        heredocs: Extracted documents by placeholder.

    Returns:
        Parsed immutable command.
    """

    document = None
    if input_path is not None and input_path.literal() in heredocs:
        document = heredocs[input_path.literal()]
        input_path = None
    return _Command(arguments, input_path, output_path, document, append)


def _is_operator(token: _Word | str) -> bool:
    """Return whether one token contains shell punctuation.

    Args:
        token: Parsed word or punctuation token.

    Returns:
        Whether the token is unsupported or handled shell punctuation.
    """

    return (
        isinstance(token, str)
        and bool(token)
        and all(character in "|<>&;" for character in token)
    )


def _split_assignment(value: str) -> tuple[str, str] | None:
    """Split one valid ``NAME=VALUE`` shell assignment.

    Args:
        value: Expanded command word.

    Returns:
        Name and value, or ``None`` when the word is not an assignment.
    """

    name, separator, content = value.partition("=")
    if not separator or _VARIABLE_NAME.fullmatch(name) is None:
        return None
    return name, content


def _copy_arguments(args: list[str], command: str) -> tuple[bool, list[str], str]:
    """Parse bounded ``cp`` or ``mv`` source and destination arguments.

    Args:
        args: Expanded command arguments.
        command: ``cp`` or ``mv`` for usage text and flag rules.

    Returns:
        Recursive flag, source paths, and destination path.

    Raises:
        _UsageError: If flags or the path count are invalid.
    """

    recursive = False
    values = list(args)
    while values and values[0].startswith("-") and values[0] != "-":
        option = values.pop(0)
        if option == "--":
            break
        flags = option[1:]
        if command != "cp" or not flags or any(flag not in "rR" for flag in flags):
            usage = (
                "cp [-r] SOURCE... DESTINATION"
                if command == "cp"
                else "mv SOURCE... DESTINATION"
            )
            raise _UsageError(f"Usage: {usage}")
        recursive = True
    if len(values) < 2:
        usage = (
            "cp [-r] SOURCE... DESTINATION"
            if command == "cp"
            else "mv SOURCE... DESTINATION"
        )
        raise _UsageError(f"Usage: {usage}")
    return recursive, values[:-1], values[-1]


def _ls_line(path: str, kind: str, long_format: bool) -> str:
    """Format one virtual ``ls`` entry.

    Args:
        path: Absolute virtual path.
        kind: Filesystem entry type.
        long_format: Whether to add the small type marker.

    Returns:
        One output line without its final newline.
    """

    display = path + ("/" if kind == "directory" and path != "/" else "")
    if not long_format:
        return display
    return ("d " if kind == "directory" else "- ") + display


def _format_printf(format_text: str, arguments: list[str]) -> str:
    """Format text with the supported virtual ``printf`` conversions.

    Args:
        format_text: Format containing ``%s``, ``%d``, ``%f``, ``%b``, or ``%%``.
        arguments: Values consumed by conversions.

    Returns:
        Formatted text. The format repeats while arguments remain.

    Raises:
        _UsageError: If a conversion or numeric value is invalid.
    """

    decoded_format = _decode_backslash_escapes(format_text)
    output: list[str] = []
    argument_index = 0
    first_pass = True
    while first_pass or argument_index < len(arguments):
        first_pass = False
        index = 0
        consumed = False
        while index < len(decoded_format):
            if decoded_format[index] != "%":
                output.append(decoded_format[index])
                index += 1
                continue
            if index + 1 >= len(decoded_format):
                raise _UsageError("printf format ends with %")
            conversion = decoded_format[index + 1]
            index += 2
            if conversion == "%":
                output.append("%")
                continue
            if conversion not in {"b", "d", "f", "s"}:
                raise _UsageError(f"printf conversion is not supported: %{conversion}")
            value = arguments[argument_index] if argument_index < len(arguments) else ""
            if argument_index < len(arguments):
                argument_index += 1
            consumed = True
            if conversion == "s":
                output.append(value)
            elif conversion == "b":
                output.append(_decode_backslash_escapes(value))
            elif conversion == "d":
                try:
                    output.append(str(int(value or "0")))
                except ValueError as exc:
                    raise _UsageError(f"printf integer is invalid: {value}") from exc
            else:
                try:
                    output.append(str(float(value or "0")))
                except ValueError as exc:
                    raise _UsageError(f"printf number is invalid: {value}") from exc
        if not consumed:
            break
    return "".join(output)


def _decode_backslash_escapes(value: str) -> str:
    """Decode the small escape set supported by virtual ``printf``.

    Args:
        value: Format or ``%b`` argument text.

    Returns:
        Text with newline, tab, carriage-return, and backslash escapes decoded.
    """

    replacements = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
    output: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            next_character = value[index + 1]
            if next_character in replacements:
                output.append(replacements[next_character])
                index += 2
                continue
        output.append(value[index])
        index += 1
    return "".join(output)


def _line_arguments(args: list[str], command: str) -> tuple[int, list[str]]:
    """Parse the shared ``head`` and ``tail`` arguments.

    Args:
        args: Command arguments.
        command: Command name used in error text.

    Returns:
        Line count and remaining file paths.
    """
    count = 10
    if args and args[0] == "-n":
        if len(args) < 2:
            raise _UsageError(f"Usage: {command} [-n COUNT] [FILE]")
        try:
            count = int(args[1])
        except ValueError as exc:
            raise _UsageError("line count must be an integer") from exc
        args = args[2:]
    if count < 0:
        raise _UsageError("line count must be at least 0")
    return count, args


def _expect_count(
    arguments: list[str],
    minimum: int,
    maximum: int,
    usage: str,
) -> None:
    """Check one command argument count.

    Args:
        arguments: Command arguments to count.
        minimum: Smallest accepted count.
        maximum: Largest accepted count.
        usage: Usage text returned on failure.

    Raises:
        _UsageError: If the count is outside the accepted range.
    """
    if not minimum <= len(arguments) <= maximum:
        raise _UsageError(f"Usage: {usage}")


def _expect_no_input(stdin: str | None, command: str) -> None:
    """Reject pipeline input for commands that cannot consume it.

    Args:
        stdin: Optional pipeline input.
        command: Command name used in error text.

    Raises:
        _UsageError: If pipeline input is present.
    """
    if stdin is not None:
        raise _UsageError(f"{command} does not accept shell input")


def _parse_curl_arguments(args: list[str]) -> _CurlOptions:
    """Parse the small supported subset of curl arguments.

    Args:
        args: Raw curl arguments after the command name.

    Returns:
        Validated curl options.

    Raises:
        _UsageError: If an option or argument is invalid or unsupported.
    """
    method = "GET"
    explicit_method = False
    headers: dict[str, str] = {}
    data: bytes | None = None
    output_path: str | None = None
    follow_redirects = False
    include_headers = False
    fail_on_http_error = False
    timeout_seconds: float | None = None
    urls: list[str] = []
    index = 0
    options_done = False

    while index < len(args):
        argument = args[index]
        if argument == "--":
            options_done = True
        elif not options_done and argument in {"-f", "--fail"}:
            fail_on_http_error = True
        elif not options_done and argument in {"-L", "--location"}:
            follow_redirects = True
        elif not options_done and argument in {"-i", "--include"}:
            include_headers = True
        elif not options_done and argument in {"-I", "--head"}:
            method = "HEAD"
            explicit_method = True
        elif not options_done and argument in {
            "-s",
            "-S",
            "--silent",
            "--show-error",
        }:
            pass
        elif (
            not options_done
            and argument.startswith("-")
            and not argument.startswith("--")
            and len(argument) > 2
        ):
            if any(flag not in "fLsiIS" for flag in argument[1:]):
                raise _UsageError(f"Unsupported curl option: {argument}")
            fail_on_http_error = fail_on_http_error or "f" in argument
            follow_redirects = follow_redirects or "L" in argument
            include_headers = include_headers or "i" in argument
            if "I" in argument:
                method = "HEAD"
                explicit_method = True
        elif not options_done and argument in {
            "-X",
            "--request",
            "-H",
            "--header",
            "-d",
            "--data",
            "-o",
            "--output",
            "--max-time",
        }:
            if index + 1 >= len(args):
                raise _UsageError(f"curl option needs a value: {argument}")
            value = args[index + 1]
            index += 1
            if argument in {"-X", "--request"}:
                method = value.upper()
                if not method.isalpha():
                    raise _UsageError("curl method must contain only letters")
                explicit_method = True
            elif argument in {"-H", "--header"}:
                name, separator, header_value = value.partition(":")
                name = name.strip()
                if (
                    not separator
                    or not name
                    or any(character in value for character in "\r\n")
                    or name.lower()
                    in {
                        "connection",
                        "content-length",
                        "host",
                        "proxy-authorization",
                        "transfer-encoding",
                    }
                ):
                    raise _UsageError("curl header is invalid or controlled by Arenero")
                headers[name] = header_value.strip()
            elif argument in {"-d", "--data"}:
                if data is not None:
                    raise _UsageError("curl accepts only one data value")
                data = value.encode("utf-8")
                headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
            elif argument in {"-o", "--output"}:
                if output_path is not None:
                    raise _UsageError("curl accepts only one output file")
                output_path = value
            else:
                try:
                    timeout_seconds = float(value)
                except ValueError as exc:
                    raise _UsageError("curl timeout must be a number") from exc
                if timeout_seconds <= 0:
                    raise _UsageError("curl timeout must be greater than 0")
        elif not options_done and argument.startswith("-"):
            raise _UsageError(f"Unsupported curl option: {argument}")
        else:
            urls.append(argument)
        index += 1

    if len(urls) != 1:
        raise _UsageError(f"Usage: {_COMMAND_USAGE['curl']}")
    if data is not None and not explicit_method:
        method = "POST"
    if method == "HEAD" and data is not None:
        raise _UsageError("curl HEAD requests cannot include data")
    return _CurlOptions(
        url=urls[0],
        method=method,
        headers=headers,
        data=data,
        output_path=output_path,
        follow_redirects=follow_redirects,
        include_headers=include_headers,
        fail_on_http_error=fail_on_http_error,
        timeout_seconds=timeout_seconds,
    )


def _curl_header_text(response: NetworkResponse) -> str:
    """Render HTTP status and headers for the curl include option.

    Args:
        response: Completed HTTP response.

    Returns:
        UTF-8 status and header block.
    """
    status = f"HTTP {response.status} {response.reason}".rstrip()
    headers = "".join(f"{name}: {value}\n" for name, value in response.headers)
    return f"{status}\n{headers}\n"


def _was_truncated(result: ExecutionResult) -> bool:
    """Return whether the kernel shortened any result field."""
    return any(result["truncated"].values())
