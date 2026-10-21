import asyncio
import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic_monty import MontyError

from ._deadline import shell_deadline
from .cell import execute_repl_cell
from .hosted_tools import HostedToolSet
from .kernel import ExecutionResult, PersistentMontyKernel, StatePolicy
from .modules import VirtualModuleConfig
from .shell import ShellResult, VirtualShell, shell_result_from_execution
from .values import error_text as _error_text

VIRTUAL_SHELL_TOOL_DESCRIPTION = """Run a small allowlisted shell over the
Arenero virtual filesystem. This is not an operating-system shell and it never
starts a process or arbitrary executable.

Supported commands and forms:
- help [COMMAND]; which COMMAND...; pwd; cd [PATH|-]; ls [-aA1ldR] [PATH...]
- env; export [NAME[=VALUE] ...]; unset NAME...; true; false;
  test EXPRESSION; [ EXPRESSION ]
- mkdir [-p] PATH...; touch FILE...; rm [-r] [-f] PATH...;
  cp [-r] SOURCE... DESTINATION; mv SOURCE... DESTINATION
- cat [FILE...]; head [-n COUNT] [FILE]; tail [-n COUNT] [FILE]
- grep [-EFinvcqlrR] PATTERN [FILE...]; sort [-r] [FILE];
  uniq [-c] [FILE]; rev [FILE]; wc [-lwc] [FILE]; echo [-n] [TEXT...];
  printf FORMAT [ARGUMENT ...]
- find [PATH] [GLOB]; glob GLOB [PATH]; basename PATH; dirname PATH
- tee [-a] FILE...; base64 [-d] [FILE]; sha256sum [FILE...]
- source FILE or . FILE for a straight-line virtual shell script
- curl [-fLsiI] [-X METHOD] [-H 'NAME: VALUE'] [-d DATA] [-o FILE]
  [--max-time SECONDS] URL
- python -c CODE or python FILE.py. python3 and monty are aliases.

Grammar and state:
- Single and double quotes, backslash escaping, # comments at word boundaries,
  newlines, and backslash-newline continuation are supported.
- | is a UTF-8 text pipe. ; or a newline always starts the next pipeline. && runs
  after success. || runs after failure. A failed command stops its pipeline.
- < reads a virtual UTF-8 file. > replaces a virtual text file. >> appends.
  <<WORD supplies one here document. A quoted delimiter disables $ expansion.
- NAME=VALUE and export set persistent virtual variables. $NAME, ${NAME}, $?,
  PWD, OLDPWD, HOME, ~, and ~/path are supported. cd - uses OLDPWD.
- Unquoted *, ?, and ** expand against virtual paths. Quoted wildcards stay
  literal. An unmatched wildcard also stays literal. Expansion is bounded.
- Paths use POSIX syntax. Relative paths use the session working directory.
- Shell variables, PWD, OLDPWD, and $? use the same session snapshot as Python
  state. SQLite makes that state durable across kernel restarts. Virtual files
  have their own lifetime and are not stored inside the snapshot.

Important limits: use help to discover commands because the application can
disable commands. grep uses fixed text unless -E is set; -E uses Python regular
expression syntax, not exact POSIX ERE syntax. find, glob, and argument expansion
support only *, ?, and **. Pipes and normal output are text-only. base64 -d can
put only UTF-8 output in a pipe. cp -r and mv are bounded, do not preserve file
metadata, reject symbolic links, and can leave partial output after an error. mv
is a copy-then-delete operation and is not atomic. ls -l reports only the entry
type and path. touch creates missing files but does not change timestamps. test
supports only the documented small expression subset. source runs the same small
grammar and shares the command, output, filesystem, and nesting limits.

Python runs Monty's limited Python subset, not CPython; it has no third-party
packages or normal file imports. Native eval/exec run Monty source and bypass
static checking. Python os.chdir() and shell cd share one virtual directory.
Interpreter duration is per feed, not per session lifetime.
curl allows public HTTP and HTTPS URLs by
default. NetworkConfig can restrict exact URL origins, path prefixes, and methods,
or disable curl. Each redirect is checked again. Private, local, and reserved
addresses are blocked by default. Use -o FILE for a binary response; normal curl
output is UTF-8 text.
--max-time lowers connection and between-read idle timeouts; it is not a strict
wall-clock limit for DNS or a response that keeps sending small data.

Not supported: full Bash scripts or control flow such as if, for, while, case,
functions, arrays, aliases, or traps; command substitution with $(...) or
backticks; arithmetic, brace, pathname class, field-splitting, process, or advanced
parameter expansion; assignment prefixes before a command; stderr or file-
descriptor redirection; here strings; multiple here documents on one command
line; background jobs; process control; binary pipes; interactive input;
subprocesses; Git; package managers; compilers; unlisted commands or flags;
network access from Monty code; unrestricted network access; or extra Python
script arguments. A shebang has no special meaning. Use a real container or
managed sandbox for those tasks.

Results have output, exit_code, and truncated fields. Check truncated before you
treat output as complete. Set output_file to an absolute virtual path to save
larger bounded output when the response is trimmed. The sandbox execute API
accepts a cooperative per-call timeout: expiry returns exit 124 and stops later
commands. It cannot preempt an active worker, host callback, DNS lookup, or
system sleep. Completed effects remain. None or zero adds no call deadline;
negative values fail with exit 2. Shell cells do not add a timeout argument."""

MONTY_SANDBOX_INSTRUCTIONS = (
    "## Arenero virtual shell\n\n" + VIRTUAL_SHELL_TOOL_DESCRIPTION
)

SandboxErrors = (MontyError, ValueError, TypeError, RuntimeError)


@dataclass(frozen=True)
class SandboxExecuteResult:
    """Describe one limited sandbox execution.

    Attributes:
        output: Combined value, standard output, standard error, or error text.
        exit_code: Zero for success and nonzero for failure.
        truncated: Whether Arenero shortened any output field.
        output_file: Virtual path that received larger bounded output.
        output_file_error: Error that prevented output offloading.
    """

    output: str
    exit_code: int
    truncated: bool = False
    output_file: str | None = None
    output_file_error: str | None = None


@dataclass(frozen=True)
class SandboxReadResult:
    """Describe a line-based text read or Base64 binary read.

    Attributes:
        error: Error text, or ``None`` on success.
        file_data: Content and encoding metadata.
        total_lines: Total text lines when content is UTF-8.
        start_line: First returned one-based line number.
        end_line: Last returned one-based line number.
        next_offset: Next zero-based line offset, if more text remains.
    """

    error: str | None = None
    file_data: dict[str, str] | None = None
    total_lines: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    next_offset: int | None = None


@dataclass(frozen=True)
class SandboxOperationResult:
    """Describe one filesystem mutation.

    Attributes:
        error: Error text, or ``None`` on success.
        path: Affected virtual path.
        occurrences: Replacement count for edit operations.
    """

    error: str | None = None
    path: str | None = None
    occurrences: int | None = None


@dataclass(frozen=True)
class SandboxSearchResult:
    """Describe a filesystem listing or search.

    Attributes:
        error: Error text, or ``None`` on success.
        matches: Structured file entries or line matches.
        truncated: Whether the configured search cap stopped the operation.
    """

    error: str | None = None
    matches: list[dict[str, Any]] | None = None
    truncated: bool = False


@dataclass(frozen=True)
class FileUploadResult:
    """Describe one application-side binary upload.

    Attributes:
        path: Requested virtual path.
        error: Error text, or ``None`` on success.
    """

    path: str
    error: str | None = None


@dataclass(frozen=True)
class FileDownloadResult:
    """Describe one application-side binary download.

    Attributes:
        path: Requested virtual path.
        content: File bytes, or ``None`` on failure.
        error: Error text, or ``None`` on success.
    """

    path: str
    content: bytes | None = None
    error: str | None = None


class AreneroSandbox:
    """Expose sandbox-style utilities over one persistent Monty kernel."""

    def __init__(
        self,
        kernel: PersistentMontyKernel,
        session_id: str = "sandbox",
        sandbox_id: str | None = None,
    ) -> None:
        """Create a limited sandbox facade.

        Args:
            kernel: Kernel that owns interpreter sessions and virtual files.
            session_id: Persistent interpreter session used for execution.
            sandbox_id: Optional public identifier.
        """
        self.kernel = kernel
        self.session_id = session_id
        self._id = sandbox_id or f"arenero:{session_id}"

    @property
    def id(self) -> str:
        """Return the stable sandbox identifier.

        Returns:
            Sandbox identifier.
        """
        return self._id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
        output_limit_chars: int | None = None,
        output_file: str | None = None,
    ) -> SandboxExecuteResult:
        """Execute one allowlisted virtual shell pipeline.

        Args:
            command: Text in the documented virtual shell grammar.
            timeout: Cooperative wall deadline in seconds; None or zero uses
                configured limits only. In-progress host work is not preempted.
            output_limit_chars: Optional final output limit below the hard maximum.
            output_file: Optional absolute virtual path used when output is trimmed.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return self._execute_shell(
            command,
            timeout=timeout,
            output_limit_chars=output_limit_chars,
            output_file=output_file,
        )

    def _execute_shell(
        self,
        command: str,
        *,
        timeout: int | None = None,
        output_limit_chars: int | None = None,
        output_file: str | None = None,
        capture: Callable[[ShellResult], None] | None = None,
    ) -> SandboxExecuteResult:
        """Use the same shell with an optional bounded pre-display capture.

        Args:
            command: Documented virtual shell source.
            timeout: Cooperative deadline, or None/zero for configured limits.
            output_limit_chars: Optional returned-text cap.
            output_file: Optional existing overflow destination.
            capture: Private observer of output before final display trimming.

        Returns:
            The unchanged public shell result.
        """
        try:
            with shell_deadline(timeout):
                shell = VirtualShell(self.kernel, self.session_id)
                shell._capture_output = capture
                return _shell_response(
                    shell.execute(
                        command,
                        output_limit_chars,
                        output_file=output_file,
                    )
                )
        except TimeoutError as exc:
            return SandboxExecuteResult(output=str(exc), exit_code=124)
        except ValueError as exc:
            return SandboxExecuteResult(output=str(exc), exit_code=2)

    def execute_python(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
        filesystem: bool = True,
    ) -> SandboxExecuteResult:
        """Execute Monty-compatible Python directly.

        Args:
            code: Monty-compatible Python source.
            output_limit_chars: Optional output limit below the hard maximum.
            output_file: Optional absolute virtual path used when output is trimmed.
            state_policy: State save policy for this operation.
            restart: Reset state before execution when true.
            hosted_tools: Optional tools for this operation only.
            inputs: Values bound once before the first user-code feed.
            filesystem: Allow direct files when true; false skips file bootstrap and the shared file lock.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return _execution_response(
            self.kernel.execute(
                self.session_id,
                code,
                output_limit_chars,
                output_file=output_file,
                state_policy=state_policy,
                inputs=inputs,
                restart=restart,
                hosted_tools=hosted_tools,
                filesystem=filesystem,
            )
        )

    def execute_cell(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> dict[str, Any]:
        """Execute one Python or notebook-style shell cell.

        Args:
            code: Monty Python, one ``!command``, or a ``%%shell`` cell.
            output_limit_chars: Optional result limit below the hard maximum.
            output_file: Optional absolute virtual path used when output is trimmed.
            state_policy: State save policy for Python and ``%run`` execution.
            restart: Reset state before an executable cell when true.
            hosted_tools: Optional tools for this operation only.
            inputs: Values bound once before the first user-code feed.

        Returns:
            A result with ``mode=python`` or ``mode=shell``.
        """
        return execute_repl_cell(
            self.kernel,
            self.session_id,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            inputs=inputs,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    def execute_file(
        self,
        path: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
    ) -> SandboxExecuteResult:
        """Execute one virtual Python file in the persistent session.

        Args:
            path: Absolute virtual Python file path.
            output_limit_chars: Optional output limit below the hard maximum.
            output_file: Optional absolute virtual path used when output is trimmed.
            state_policy: State save policy for this operation.
            inputs: Values bound once before the first user-code feed.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return _execution_response(
            self.kernel.execute_file(
                self.session_id,
                path,
                output_limit_chars=output_limit_chars,
                output_file=output_file,
                state_policy=state_policy,
                inputs=inputs,
            )
        )

    def execute_module(
        self,
        entry_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        module_config: VirtualModuleConfig | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
    ) -> SandboxExecuteResult:
        """Resolve and execute a static virtual module graph.

        Args:
            entry_path: Absolute virtual entry ``.py`` file.
            code: Optional follow-up code.
            output_limit_chars: Optional output limit below the hard maximum.
            module_config: Optional module roots, aliases, and limits.
            state_policy: State save policy for this operation.
            inputs: Values bound once before the first user-code feed.
            output_file: Optional virtual path for bounded untrimmed output.

        Returns:
            Combined output, exit code, and truncation state.
        """

        return _execution_response(
            self.kernel.execute_module(
                self.session_id,
                entry_path,
                code,
                output_limit_chars,
                module_config=module_config,
                state_policy=state_policy,
                output_file=output_file,
                inputs=inputs,
            )
        )

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2_000,
    ) -> SandboxReadResult:
        """Read text by lines or return binary content as Base64.

        Args:
            file_path: Absolute virtual file path.
            offset: Zero-based text line offset. Negative values become zero.
            limit: Maximum text lines. Nonpositive values return empty text.

        Returns:
            Read result with content and pagination data.
        """
        try:
            content = self.kernel.download_file(file_path)
        except SandboxErrors as exc:
            return SandboxReadResult(error=str(exc))
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return SandboxReadResult(
                file_data={
                    "content": base64.b64encode(content).decode("ascii"),
                    "encoding": "base64",
                }
            )
        start = max(0, offset)
        if limit <= 0:
            return SandboxReadResult(file_data={"content": "", "encoding": "utf-8"})
        lines = text.splitlines(keepends=True)
        selected = lines[start : start + limit]
        end = start + len(selected)
        return SandboxReadResult(
            file_data={"content": "".join(selected), "encoding": "utf-8"},
            total_lines=len(lines),
            start_line=start + 1 if selected else None,
            end_line=end if selected else None,
            next_offset=end if end < len(lines) else None,
        )

    def write(self, file_path: str, content: str) -> SandboxOperationResult:
        """Create or replace one UTF-8 text file.

        Args:
            file_path: Absolute virtual file path.
            content: Complete text content.

        Returns:
            Mutation result.
        """
        try:
            self.kernel.upload_file(file_path, content.encode("utf-8"))
        except SandboxErrors as exc:
            return SandboxOperationResult(error=str(exc))
        return SandboxOperationResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> SandboxOperationResult:
        """Replace exact text in one UTF-8 file.

        Args:
            file_path: Absolute virtual file path.
            old_string: Exact text to replace.
            new_string: Replacement text.
            replace_all: Replace every occurrence when true.

        Returns:
            Mutation result with the replacement count.
        """
        with self.kernel.filesystem_guard():
            return self._edit_locked(
                file_path,
                old_string,
                new_string,
                replace_all,
            )

    def _edit_locked(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
    ) -> SandboxOperationResult:
        """Edit one file while the shared filesystem lock is held.

        Args:
            file_path: Absolute virtual file path.
            old_string: Exact text to replace.
            new_string: Replacement text.
            replace_all: Replace every occurrence when true.

        Returns:
            Mutation result with the replacement count.
        """
        if old_string == new_string:
            return SandboxOperationResult(error="old_string and new_string must differ")
        read = self.read(file_path)
        if read.error is not None:
            return SandboxOperationResult(error=read.error)
        if read.file_data is None or read.file_data["encoding"] != "utf-8":
            return SandboxOperationResult(error="edit requires a UTF-8 text file")
        text = self.kernel.download_file(file_path).decode("utf-8")
        occurrences = text.count(old_string)
        if occurrences == 0:
            return SandboxOperationResult(error="old_string was not found")
        if not replace_all and occurrences != 1:
            return SandboxOperationResult(
                error="old_string must occur exactly once unless replace_all is true"
            )
        replacement_count = occurrences if replace_all else 1
        updated = text.replace(old_string, new_string, -1 if replace_all else 1)
        written = self.write(file_path, updated)
        if written.error is not None:
            return written
        return SandboxOperationResult(
            path=file_path,
            occurrences=replacement_count,
        )

    def delete(self, file_path: str) -> SandboxOperationResult:
        """Delete one virtual path recursively.

        Args:
            file_path: Absolute virtual file or directory path.

        Returns:
            Mutation result.
        """
        result = self.kernel.call_filesystem(
            self.session_id,
            "delete",
            {"path": file_path, "recursive": True},
        )
        if not result["ok"]:
            return SandboxOperationResult(error=_error_text(result))
        return SandboxOperationResult(path=file_path)

    def ls(self, path: str = "/") -> SandboxSearchResult:
        """List direct entries under one virtual directory.

        Args:
            path: Absolute virtual directory path.

        Returns:
            Structured file entries.
        """
        result = self.kernel.call_filesystem(
            self.session_id,
            "ls",
            {"path": path, "recursive": False, "limit": None},
        )
        if not result["ok"]:
            return SandboxSearchResult(error=_error_text(result))
        value = result["result"]
        entries = [
            {
                "path": entry["path"],
                "is_dir": entry["type"] == "directory",
            }
            for entry in value["entries"]
        ]
        return SandboxSearchResult(
            matches=entries,
            truncated=value["truncated"],
        )

    def glob(self, pattern: str, path: str = "/") -> SandboxSearchResult:
        """Find virtual paths with a bounded glob search.

        Args:
            pattern: POSIX pattern with ``*``, ``?``, or ``**``.
            path: Absolute virtual search root.

        Returns:
            Matching path entries.
        """
        result = self.kernel.call_filesystem(
            self.session_id,
            "glob",
            {"pattern": pattern, "root": path, "limit": None},
        )
        if not result["ok"]:
            return SandboxSearchResult(error=_error_text(result))
        value = result["result"]
        return SandboxSearchResult(
            matches=[{"path": match} for match in value["matches"]],
            truncated=value["truncated"],
        )

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str = "**/*",
        *,
        max_count: int | None = None,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> SandboxSearchResult:
        """Search bounded UTF-8 files with the existing Monty grep helper.

        Args:
            pattern: Literal text to find.
            path: Absolute virtual search root.
            glob: File-selection glob pattern.
            max_count: Optional requested match cap below the hard maximum.
            regex: Use Monty regular expressions when true; otherwise literal text.
            case_sensitive: Match case when true.

        Returns:
            Structured line matches.
        """
        result = self.kernel.call_filesystem(
            self.session_id,
            "grep",
            {
                "query": pattern,
                "root": path,
                "pattern": glob,
                "regex": regex,
                "case_sensitive": case_sensitive,
                "limit": max_count,
            },
        )
        if not result["ok"]:
            return SandboxSearchResult(error=_error_text(result))
        value = result["result"]
        return SandboxSearchResult(
            matches=value["matches"],
            truncated=value["truncated"],
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResult]:
        """Upload binary files through the application transfer plane.

        Args:
            files: Ordered ``(absolute_path, content)`` pairs.

        Returns:
            One result per input file in the same order.
        """
        results = []
        for path, content in files:
            try:
                self.kernel.upload_file(path, content)
            except SandboxErrors as exc:
                results.append(FileUploadResult(path=path, error=str(exc)))
            else:
                results.append(FileUploadResult(path=path))
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResult]:
        """Download binary files through the application transfer plane.

        Args:
            paths: Ordered absolute virtual file paths.

        Returns:
            One result per input path in the same order.
        """
        results = []
        for path in paths:
            try:
                content = self.kernel.download_file(path)
            except SandboxErrors as exc:
                results.append(FileDownloadResult(path=path, error=str(exc)))
            else:
                results.append(FileDownloadResult(path=path, content=content))
        return results


class AsyncAreneroSandbox:
    """Expose the synchronous sandbox through async worker-thread calls."""

    def __init__(
        self,
        kernel: PersistentMontyKernel,
        session_id: str = "sandbox",
        sandbox_id: str | None = None,
    ) -> None:
        """Create an asynchronous sandbox facade.

        Args:
            kernel: Kernel that owns interpreter sessions and virtual files.
            session_id: Persistent interpreter session used for execution.
            sandbox_id: Optional public identifier.
        """
        self.sync = AreneroSandbox(kernel, session_id, sandbox_id)

    @property
    def id(self) -> str:
        """Return the stable sandbox identifier.

        Returns:
            Sandbox identifier.
        """
        return self.sync.id

    async def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
        output_limit_chars: int | None = None,
        output_file: str | None = None,
    ) -> SandboxExecuteResult:
        """Execute one virtual shell command list without blocking the event loop.

        Args:
            command: Text in the documented virtual shell grammar.
            timeout: Unsupported per-call timeout.
            output_limit_chars: Optional final output limit.
            output_file: Optional virtual path used when output is trimmed.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return await asyncio.to_thread(
            self.sync.execute,
            command,
            timeout=timeout,
            output_limit_chars=output_limit_chars,
            output_file=output_file,
        )

    async def execute_python(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
        filesystem: bool = True,
    ) -> SandboxExecuteResult:
        """Execute Monty Python without blocking the event loop.

        Args:
            code: Monty-compatible Python source.
            output_limit_chars: Optional output limit.
            output_file: Optional virtual path used when output is trimmed.
            state_policy: State save policy for this operation.
            inputs: Values bound once before the first user-code feed.
            filesystem: Allow direct files when true; false skips file bootstrap and the shared file lock.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return await asyncio.to_thread(
            self.sync.execute_python,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            inputs=inputs,
            filesystem=filesystem,
        )

    async def execute_cell(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one Python or shell cell without blocking the event loop.

        Args:
            code: Monty Python, one ``!command``, or a ``%%shell`` cell.
            output_limit_chars: Optional result limit.
            output_file: Optional virtual path used when output is trimmed.
            state_policy: State save policy for Python and ``%run`` execution.
            inputs: Values bound once before the first user-code feed.

        Returns:
            Stable Python or shell cell result.
        """
        return await asyncio.to_thread(
            self.sync.execute_cell,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            inputs=inputs,
        )

    async def execute_file(
        self,
        path: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
    ) -> SandboxExecuteResult:
        """Execute one virtual Python file without blocking the event loop.

        Args:
            path: Absolute virtual Python file path.
            output_limit_chars: Optional output limit.
            output_file: Optional virtual path used when output is trimmed.
            state_policy: State save policy for this operation.
            inputs: Values bound once before the first user-code feed.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return await asyncio.to_thread(
            self.sync.execute_file,
            path,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            inputs=inputs,
        )

    async def execute_module(
        self,
        entry_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        module_config: VirtualModuleConfig | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
    ) -> SandboxExecuteResult:
        """Resolve and execute static virtual modules without blocking.

        Args:
            entry_path: Absolute virtual entry ``.py`` file.
            code: Optional follow-up code.
            output_limit_chars: Optional output limit.
            module_config: Optional module roots, aliases, and limits.
            state_policy: State save policy for this operation.
            inputs: Values bound once before the first user-code feed.
            output_file: Optional virtual path for bounded untrimmed output.

        Returns:
            Combined output, exit code, and truncation state.
        """

        return await asyncio.to_thread(
            self.sync.execute_module,
            entry_path,
            code,
            output_limit_chars,
            module_config=module_config,
            state_policy=state_policy,
            output_file=output_file,
            inputs=inputs,
        )

    async def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2_000,
    ) -> SandboxReadResult:
        """Read one virtual file without blocking the event loop.

        Args:
            file_path: Absolute virtual file path.
            offset: Zero-based text line offset.
            limit: Maximum text lines.

        Returns:
            Text or Base64 file result.
        """
        return await asyncio.to_thread(self.sync.read, file_path, offset, limit)

    async def write(self, file_path: str, content: str) -> SandboxOperationResult:
        """Write one virtual text file without blocking the event loop.

        Args:
            file_path: Absolute virtual file path.
            content: Complete UTF-8 text.

        Returns:
            Filesystem mutation result.
        """
        return await asyncio.to_thread(self.sync.write, file_path, content)

    async def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> SandboxOperationResult:
        """Edit one virtual text file without blocking the event loop.

        Args:
            file_path: Absolute virtual file path.
            old_string: Exact text to replace.
            new_string: Replacement text.
            replace_all: Replace every occurrence when true.

        Returns:
            Filesystem mutation result.
        """
        return await asyncio.to_thread(
            self.sync.edit,
            file_path,
            old_string,
            new_string,
            replace_all,
        )

    async def delete(self, file_path: str) -> SandboxOperationResult:
        """Delete one virtual path without blocking the event loop.

        Args:
            file_path: Absolute virtual file or directory path.

        Returns:
            Filesystem mutation result.
        """
        return await asyncio.to_thread(self.sync.delete, file_path)

    async def ls(self, path: str = "/") -> SandboxSearchResult:
        """List one virtual directory without blocking the event loop.

        Args:
            path: Absolute virtual directory path.

        Returns:
            Structured file entries.
        """
        return await asyncio.to_thread(self.sync.ls, path)

    async def glob(self, pattern: str, path: str = "/") -> SandboxSearchResult:
        """Find virtual paths without blocking the event loop.

        Args:
            pattern: POSIX pattern with ``*``, ``?``, or ``**``.
            path: Absolute virtual search root.

        Returns:
            Matching path entries.
        """
        return await asyncio.to_thread(self.sync.glob, pattern, path)

    async def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str = "**/*",
        *,
        max_count: int | None = None,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> SandboxSearchResult:
        """Search virtual text files without blocking the event loop.

        Args:
            pattern: Literal text to find.
            path: Absolute virtual search root.
            glob: File-selection glob pattern.
            max_count: Optional requested match cap.
            regex: Use Monty regular expressions when true.
            case_sensitive: Match case when true.

        Returns:
            Structured line matches.
        """
        return await asyncio.to_thread(
            self.sync.grep,
            pattern,
            path,
            glob,
            max_count=max_count,
            regex=regex,
            case_sensitive=case_sensitive,
        )

    async def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResult]:
        """Upload binary files without blocking the event loop.

        Args:
            files: Ordered ``(absolute_path, content)`` pairs.

        Returns:
            One result per input file.
        """
        return await asyncio.to_thread(self.sync.upload_files, files)

    async def download_files(self, paths: list[str]) -> list[FileDownloadResult]:
        """Download binary files without blocking the event loop.

        Args:
            paths: Ordered absolute virtual file paths.

        Returns:
            One result per input path.
        """
        return await asyncio.to_thread(self.sync.download_files, paths)


def _execution_response(result: ExecutionResult) -> SandboxExecuteResult:
    """Convert one kernel result to the limited sandbox response.

    Args:
        result: Structured kernel execution result.

    Returns:
        Combined output, exit code, and truncation state.
    """
    return _shell_response(shell_result_from_execution(result))


def _shell_response(result: ShellResult) -> SandboxExecuteResult:
    """Convert one internal virtual shell result to the public response.

    Args:
        result: Internal shell result.

    Returns:
        Public sandbox execution result.
    """
    return SandboxExecuteResult(
        output=result.output,
        exit_code=result.exit_code,
        truncated=result.truncated,
        output_file=result.output_file,
        output_file_error=result.output_file_error,
    )
