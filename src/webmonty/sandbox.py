import base64
import json
import shlex
from dataclasses import dataclass
from typing import Any

from pydantic_monty import MontyError

from .kernel import ExecutionResult, PersistentMontyKernel

MONTY_SANDBOX_INSTRUCTIONS = """## Limited WebMonty execution

The execute operation is not a shell. It accepts only these forms:

- `python -c '<Monty-compatible Python code>'`
- `python /absolute/virtual/file.py`

There are no shell operators, pipes, redirections, subprocesses, Git commands,
package installation, network commands, or third-party Python packages. Code runs
in Monty's documented Python subset. Use the dedicated read, write, edit, delete,
ls, glob, and grep operations for files. Virtual paths are absolute POSIX paths.
Files are shared when several agent sessions use the same kernel filesystem.
"""

SandboxErrors = (MontyError, ValueError, TypeError, RuntimeError)


@dataclass(frozen=True)
class SandboxExecuteResult:
    """Describe one limited sandbox execution.

    Attributes:
        output: Combined value, standard output, standard error, or error text.
        exit_code: Zero for success and nonzero for failure.
        truncated: Whether WebMonty shortened any output field.
    """

    output: str
    exit_code: int
    truncated: bool = False


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


class WebMontySandbox:
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
        self._id = sandbox_id or f"webmonty:{session_id}"

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
    ) -> SandboxExecuteResult:
        """Execute one command from the strict WebMonty command grammar.

        Args:
            command: ``python -c CODE`` or ``python /virtual/file.py``.
            timeout: Unsupported per-call timeout. Configure ``KernelConfig``.

        Returns:
            Combined output, exit code, and truncation state.
        """
        if timeout is not None:
            return SandboxExecuteResult(
                output=(
                    "Per-call timeouts are not supported. Configure "
                    "KernelConfig.max_duration_seconds and "
                    "request_timeout_seconds."
                ),
                exit_code=2,
            )
        try:
            arguments = shlex.split(command)
        except ValueError as exc:
            return SandboxExecuteResult(str(exc), 2)
        if (
            len(arguments) == 3
            and arguments[0] in {"python", "python3", "monty"}
            and arguments[1] == "-c"
        ):
            return self.execute_python(arguments[2])
        if (
            len(arguments) == 2
            and arguments[0] in {"python", "python3", "monty"}
            and arguments[1].startswith("/")
            and arguments[1].endswith(".py")
        ):
            return self.execute_file(arguments[1])
        return SandboxExecuteResult(
            output=(
                "Unsupported command. Use `python -c '<Monty code>'` or "
                "`python /absolute/virtual/file.py`."
            ),
            exit_code=127,
        )

    def execute_python(self, code: str) -> SandboxExecuteResult:
        """Execute Monty-compatible Python directly.

        Args:
            code: Monty-compatible Python source.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return _execution_response(self.kernel.execute(self.session_id, code))

    def execute_file(self, path: str) -> SandboxExecuteResult:
        """Execute one virtual Python file in the persistent session.

        Args:
            path: Absolute virtual Python file path.

        Returns:
            Combined output, exit code, and truncation state.
        """
        return _execution_response(self.kernel.execute_file(self.session_id, path))

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
    ) -> SandboxSearchResult:
        """Search for literal text in bounded virtual text files.

        Args:
            pattern: Literal text to find.
            path: Absolute virtual search root.
            glob: File-selection glob pattern.
            max_count: Optional requested match cap below the hard maximum.

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
                "regex": False,
                "case_sensitive": True,
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


def _execution_response(result: ExecutionResult) -> SandboxExecuteResult:
    """Convert one kernel result to the limited sandbox response.

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
    return SandboxExecuteResult(
        output="".join(parts),
        exit_code=0 if result["ok"] else 1,
        truncated=any(result["truncated"].values()),
    )


def _error_text(result: ExecutionResult) -> str:
    """Read the error text from a failed kernel result.

    Args:
        result: Structured kernel execution result.

    Returns:
        Error text or a generic fallback.
    """
    error = result["error"]
    return error["message"] if error is not None else "unknown WebMonty error"
