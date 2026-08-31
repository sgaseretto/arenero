import base64
import hashlib
import json
import posixpath
import shlex
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_monty import MontyError

from .kernel import ExecutionResult, PersistentMontyKernel
from .network import NetworkRequest, NetworkResponse, fetch_url

_CWD_NAME = "__arenero_shell_cwd"
_PYTHON_COMMANDS = {"monty", "python", "python3"}
_SHELL_ERRORS = (MontyError, ValueError, TypeError, RuntimeError)
_COMMAND_USAGE = {
    "base64": "base64 [FILE]",
    "basename": "basename PATH",
    "cat": "cat FILE...",
    "cd": "cd [PATH]",
    "cp": "cp SOURCE DESTINATION",
    "curl": (
        "curl [-fLsiI] [-X METHOD] [-H 'NAME: VALUE'] [-d DATA] "
        "[-o FILE] [--max-time SECONDS] URL"
    ),
    "dirname": "dirname PATH",
    "echo": "echo [-n] [TEXT ...]",
    "find": "find [PATH] [PATTERN]",
    "glob": "glob PATTERN [PATH]",
    "grep": "grep [-i] [-n] PATTERN [FILE ...]",
    "head": "head [-n COUNT] [FILE]",
    "help": "help [COMMAND]",
    "ls": "ls [PATH ...]",
    "mkdir": "mkdir [-p] PATH...",
    "monty": "monty -c CODE or monty FILE.py",
    "mv": "mv SOURCE DESTINATION",
    "pwd": "pwd",
    "python": "python -c CODE or python FILE.py",
    "python3": "python3 -c CODE or python3 FILE.py",
    "rev": "rev [FILE]",
    "rm": "rm [-r] [-f] PATH...",
    "sha256sum": "sha256sum [FILE ...]",
    "sort": "sort [-r] [FILE]",
    "tail": "tail [-n COUNT] [FILE]",
    "tee": "tee [-a] FILE...",
    "uniq": "uniq [-c] [FILE]",
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
class _Command:
    """Store one parsed command and its redirections."""

    arguments: list[str]
    input_path: str | None = None
    output_path: str | None = None
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
        self._commands: dict[
            str, Callable[[list[str], str | None, str], ShellResult]
        ] = {
            "base64": self._base64,
            "basename": self._basename,
            "cat": self._cat,
            "cd": self._cd,
            "cp": self._cp,
            "dirname": self._dirname,
            "echo": self._echo,
            "find": self._find,
            "glob": self._glob,
            "grep": self._grep,
            "head": self._head,
            "help": self._help,
            "ls": self._ls,
            "mkdir": self._mkdir,
            "mv": self._mv,
            "pwd": self._pwd,
            "rev": self._rev,
            "rm": self._rm,
            "sha256sum": self._sha256sum,
            "sort": self._sort,
            "tail": self._tail,
            "tee": self._tee,
            "uniq": self._uniq,
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
        with self.kernel.filesystem_guard():
            return self._execute_locked(command, output_limit_chars, output_file)

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
            pipelines = _parse(command)
            command_count = sum(len(pipeline.commands) for pipeline in pipelines)
            if command_count > self.kernel.config.max_shell_commands:
                raise _UsageError(
                    "command list exceeds the virtual shell command limit"
                )
            limit = self._output_limit(output_limit_chars)
        except _UsageError as exc:
            return ShellResult(str(exc), 2)

        cwd_result = self._load_cwd()
        if isinstance(cwd_result, ShellResult):
            return cwd_result
        cwd = cwd_result
        output = ""
        last_result = ShellResult("", 0)
        truncated = False

        for pipeline in pipelines:
            if pipeline.operator == "&&" and last_result.exit_code != 0:
                continue
            if pipeline.operator == "||" and last_result.exit_code == 0:
                continue
            last_result, cwd = self._execute_pipeline(pipeline.commands, cwd)
            output, shortened = _trim(
                output + last_result.output,
                self.kernel.config.max_output_limit_chars,
            )
            truncated = truncated or last_result.truncated or shortened

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
            input_path = (
                self._path(parsed.input_path, cwd)
                if parsed.input_path is not None
                else None
            )
            output_path = (
                self._path(parsed.output_path, cwd)
                if parsed.output_path is not None
                else None
            )
            if input_path is not None:
                read = self._read_text(input_path)
                if isinstance(read, ShellResult):
                    return read, cwd
                pipeline_input = read

            result = self._run(parsed.arguments, pipeline_input, cwd)
            truncated = truncated or result.truncated
            if result.exit_code != 0:
                return ShellResult(result.output, result.exit_code, truncated), cwd

            pipeline_input = result.output
            if parsed.arguments[0] == "cd":
                loaded = self._load_cwd()
                if isinstance(loaded, ShellResult):
                    return loaded, cwd
                cwd = loaded

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
                "Commands: " + " ".join(available) + "\nOperators: | ; && || < > >>\n",
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
        target = self._path(args[0] if args else "/", cwd)
        listed = self.kernel.call_filesystem(
            self.session_id,
            "ls",
            {"path": target, "recursive": False, "limit": 1},
        )
        if not listed["ok"]:
            return ShellResult(_error_text(listed), 1)
        result = self.kernel.execute(
            self.session_id,
            f"{_CWD_NAME} = {target!r}",
            self.kernel.config.max_output_limit_chars,
        )
        return shell_result_from_execution(result)

    def _ls(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """List direct entries in one or more virtual directories."""
        _expect_no_input(stdin, "ls")
        paths = args or ["."]
        output: list[str] = []
        truncated = False
        for path in paths:
            target = self._path(path, cwd)
            result = self.kernel.call_filesystem(
                self.session_id,
                "ls",
                {"path": target, "recursive": False, "limit": None},
            )
            if not result["ok"]:
                return ShellResult(_error_text(result), 1, truncated)
            value = result["result"]
            output.extend(
                entry["path"] + ("/" if entry["type"] == "directory" else "")
                for entry in value["entries"]
            )
            truncated = truncated or value["truncated"]
        return ShellResult("".join(f"{path}\n" for path in output), 0, truncated)

    def _cat(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Concatenate UTF-8 virtual files or pipeline input."""
        if not args:
            if stdin is None:
                raise _UsageError("Usage: cat FILE... or pipe text into cat")
            return ShellResult(stdin, 0)
        output = ""
        for path in args:
            read = self._read_text(self._path(path, cwd))
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
        response = fetch_url(
            NetworkRequest(
                url=options.url,
                method=options.method,
                headers=tuple(options.headers.items()),
                body=options.data,
            ),
            config,
            follow_redirects=options.follow_redirects,
            timeout_seconds=options.timeout_seconds,
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
        """Encode one virtual file or pipeline input as Base64 text.

        Args:
            args: Optional virtual file path.
            stdin: Optional UTF-8 pipeline input.
            cwd: Current absolute virtual directory.

        Returns:
            Base64 text or an error result.
        """
        _expect_count(args, 0, 1, "base64 [FILE]")
        if args:
            try:
                content = self.kernel.download_file(self._path(args[0], cwd))
            except _SHELL_ERRORS as exc:
                return ShellResult(str(exc), 1)
        elif stdin is not None:
            content = stdin.encode("utf-8")
        else:
            return ShellResult("base64 needs a file or pipeline input", 2)
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
        """Search literal text in UTF-8 files or pipeline input."""
        ignore_case = False
        line_numbers = False
        while args and args[0].startswith("-") and args[0] != "-":
            flags = args.pop(0)[1:]
            if not flags or any(flag not in "in" for flag in flags):
                raise _UsageError("Usage: grep [-i] [-n] PATTERN [FILE ...]")
            ignore_case = ignore_case or "i" in flags
            line_numbers = line_numbers or "n" in flags
        if not args:
            raise _UsageError("Usage: grep [-i] [-n] PATTERN [FILE ...]")
        pattern, paths = args[0], args[1:]
        sources: list[tuple[str | None, str | None]] = []
        if paths:
            sources.extend((self._path(path, cwd), None) for path in paths)
        elif stdin is not None:
            sources.append((None, stdin))
        else:
            raise _UsageError("grep needs a file or pipeline input")

        needle = pattern.lower() if ignore_case else pattern
        multiple = len(sources) > 1
        matches: list[str] = []
        characters_scanned = 0
        truncated = False
        for path, text in sources:
            if text is None:
                assert path is not None
                read = self._read_text(path)
                if isinstance(read, ShellResult):
                    return read
                text = read
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
            for number, line in enumerate(text.splitlines(), start=1):
                candidate = line.lower() if ignore_case else line
                if needle not in candidate:
                    continue
                if len(matches) >= self.kernel.filesystem_config.default_result_limit:
                    truncated = True
                    break
                prefix = f"{path}:" if multiple and path is not None else ""
                if line_numbers:
                    prefix += f"{number}:"
                matches.append(f"{prefix}{line}\n")
            if truncated:
                break
        return ShellResult("".join(matches), 0 if matches else 1, truncated)

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
        """Copy one regular virtual file without changing its bytes."""
        _expect_no_input(stdin, "cp")
        _expect_count(args, 2, 2, "cp SOURCE DESTINATION")
        source = self._path(args[0], cwd)
        destination = self._path(args[1], cwd)
        try:
            content = self.kernel.download_file(source)
            self.kernel.upload_file(destination, content)
        except _SHELL_ERRORS as exc:
            return ShellResult(str(exc), 1)
        return ShellResult("", 0)

    def _mv(self, args: list[str], stdin: str | None, cwd: str) -> ShellResult:
        """Move one regular virtual file with copy then delete."""
        _expect_no_input(stdin, "mv")
        _expect_count(args, 2, 2, "mv SOURCE DESTINATION")
        if self._path(args[0], cwd) == self._path(args[1], cwd):
            return ShellResult("source and destination must differ", 2)
        copied = self._cp(args, stdin, cwd)
        if copied.exit_code != 0:
            return copied
        result = self.kernel.call_filesystem(
            self.session_id,
            "delete",
            {"path": self._path(args[0], cwd), "recursive": False},
        )
        if not result["ok"]:
            return ShellResult(_error_text(result), 1)
        return ShellResult("", 0)

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

    def _load_cwd(self) -> str | ShellResult:
        """Load or initialize the persistent virtual working directory."""
        code = (
            f"try:\n    {_CWD_NAME}\n"
            f"except NameError:\n    {_CWD_NAME} = "
            f"{self.kernel.default_working_directory!r}\n"
            f"{_CWD_NAME}"
        )
        result = self.kernel.execute(
            self.session_id,
            code,
            self.kernel.config.max_output_limit_chars,
        )
        if not result["ok"]:
            return ShellResult(_error_text(result), 1, _was_truncated(result))
        value = result["result"]
        if not isinstance(value, str) or not value.startswith("/"):
            return ShellResult("The virtual working directory state is invalid", 1)
        return value

    def _path(self, path: str, cwd: str) -> str:
        """Resolve one path against the virtual working directory."""
        if not path:
            raise _UsageError("virtual paths must not be empty")
        joined = path if path.startswith("/") else posixpath.join(cwd, path)
        normalized = posixpath.normpath(joined)
        return "/" if normalized == "/" else "/" + normalized.lstrip("/")

    def _output_limit(self, requested: int | None) -> int:
        """Return the requested final output limit under the hard maximum."""
        if requested is None:
            return self.kernel.config.default_output_limit_chars
        if requested < 0:
            raise _UsageError("output_limit_chars must be at least 0")
        return min(requested, self.kernel.config.max_output_limit_chars)


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
    """Parse the small virtual shell grammar.

    Args:
        command: Raw command text.

    Returns:
        Parsed pipelines and their control operators.

    Raises:
        _UsageError: If syntax is empty, incomplete, or unsupported.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|<>&;")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise _UsageError(str(exc)) from exc
    if not tokens:
        raise _UsageError("command must not be empty")

    pipelines: list[_Pipeline] = []
    commands: list[_Command] = []
    pipeline_operator: str | None = None
    arguments: list[str] = []
    input_path: str | None = None
    output_path: str | None = None
    append = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"|", ";", "&&", "||"}:
            if not arguments:
                raise _UsageError(f"{token} needs a command on each side")
            commands.append(_Command(arguments, input_path, output_path, append))
            arguments, input_path, output_path, append = [], None, None, False
            if token != "|":
                pipelines.append(_Pipeline(commands, pipeline_operator))
                commands = []
                pipeline_operator = token
        elif token in {"<", ">", ">>"}:
            if index + 1 >= len(tokens) or _is_operator(tokens[index + 1]):
                raise _UsageError(f"{token} needs a virtual file path")
            path = tokens[index + 1]
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
            arguments.append(token)
        index += 1

    if not arguments:
        raise _UsageError("an operator needs a command on each side")
    commands.append(_Command(arguments, input_path, output_path, append))
    pipelines.append(_Pipeline(commands, pipeline_operator))
    return pipelines


def _is_operator(token: str) -> bool:
    """Return whether one token contains shell punctuation."""
    return bool(token) and all(character in "|<>&;" for character in token)


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


def _trim(text: str, limit: int) -> tuple[str, bool]:
    """Shorten text with the same marker used by the kernel.

    Args:
        text: Text to check.
        limit: Maximum returned character count.

    Returns:
        Limited text and whether it was shortened.
    """
    if len(text) <= limit:
        return text, False
    if limit == 0:
        return "", True
    marker = "... <trimmed>"
    if len(marker) + 4 >= limit:
        return text[:limit], True
    return text[: limit - len(marker)] + marker, True


def _error_text(result: ExecutionResult) -> str:
    """Return error text from one failed kernel operation."""
    error = result["error"]
    return error["message"] if error is not None else "unknown Arenero error"


def _was_truncated(result: ExecutionResult) -> bool:
    """Return whether the kernel shortened any result field."""
    return any(result["truncated"].values())
