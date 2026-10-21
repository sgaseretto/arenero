"""Optional Deep Agents backend records over the existing Arenero sandbox."""

import asyncio
import posixpath
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    ExecuteOffloadResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.utils import compile_grep_include_glob, create_file_data
from deepagents.middleware import filesystem as _filesystem
from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import BaseTool

from .hosted_tools import HostedTool, HostedToolConfig, HostedToolSet
from .sandbox import MONTY_SANDBOX_INSTRUCTIONS, AreneroSandbox, SandboxErrors
from .shell import ShellResult

_EXECUTE_DESCRIPTIONS = {
    _filesystem.EXECUTE_TOOL_DESCRIPTION,
    _filesystem._EXECUTE_TOOL_DESCRIPTION_WITH_GREP_ONLY,
    _filesystem._EXECUTE_TOOL_DESCRIPTION_WITH_GLOB_ONLY,
    _filesystem._EXECUTE_TOOL_DESCRIPTION_WITHOUT_SEARCH,
}
_GREP_DESCRIPTIONS = {
    _filesystem.GREP_TOOL_DESCRIPTION,
    _filesystem._GREP_TOOL_DESCRIPTION_WITHOUT_EXECUTE,
}


class AreneroFilesBackend(BackendProtocol):
    """Expose existing files without granting shell execution.

    Args:
        sandbox: Sync sandbox whose kernel owns all files and shell state.
        root: Absolute virtual root for file operations. Defaults to the kernel
            workspace root. This is not a new shell confinement boundary.
    """

    # Keep native thread forwarding, even in the BaseSandbox subclass below.
    als = BackendProtocol.als
    aread = BackendProtocol.aread
    awrite = BackendProtocol.awrite
    aedit = BackendProtocol.aedit
    adelete = BackendProtocol.adelete
    aglob = BackendProtocol.aglob
    agrep = BackendProtocol.agrep
    aupload_files = BackendProtocol.aupload_files
    adownload_files = BackendProtocol.adownload_files

    def __init__(self, sandbox: AreneroSandbox, *, root: str | None = None) -> None:
        """Bind existing storage; inherited a* methods use worker threads."""
        self.sandbox = sandbox
        self.root = posixpath.normpath(
            sandbox.kernel.default_working_directory if root is None else root
        )
        if not self.root.startswith("/"):
            raise ValueError("root must be an absolute virtual path")

    @property
    def id(self) -> str:
        """Return the existing sandbox identity.

        Returns:
            Stable sandbox ID.
        """
        return self.sandbox.id

    def _path(self, path: str | None) -> str:
        """Check a path against the bound file root.

        Args:
            path: Absolute virtual path, or None for the root.

        Returns:
            Normalized absolute path.
        """
        path = self.root if path is None else path
        if not path.startswith("/") or ".." in path.split("/"):
            raise ValueError("invalid_path: use an absolute path without ..")
        path = posixpath.normpath(path)
        if (
            self.root != "/"
            and path != self.root
            and not path.startswith(self.root + "/")
        ):
            raise ValueError("permission_denied: path is outside the backend root")
        return path

    def instructions(self) -> str:
        """Describe the file-only surface for an agent.

        Returns:
            File root and permission boundaries.
        """
        return (
            f"Arenero file tools use the virtual root {self.root}. "
            "This backend does not provide execute or a shell. Use the registered "
            "file tools. Deep Agents permissions apply to those tools, not direct "
            "host transfers, mounts, MCP tools, or separately granted fs_* functions."
        )

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read raw text lines or Base64 bytes in native backend records.

        Args:
            file_path: Absolute virtual file.
            offset: Zero-based offset; negative values become zero.
            limit: Line count; nonpositive values do not inspect the file.

        Returns:
            File data and valid paging metadata, or an error.
        """
        if limit <= 0:
            return ReadResult(file_data=create_file_data(""), no_lines_requested=True)
        try:
            result = self.sandbox.read(self._path(file_path), offset, limit)
            data = asdict(result)
            if result.file_data is not None:
                data["file_data"] = create_file_data(**result.file_data)
            if result.start_line is None:
                data.update(total_lines=None, next_offset=None)
            return ReadResult(**data)
        except SandboxErrors as exc:
            return ReadResult(error=str(exc))

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create or replace one text file.

        Args:
            file_path: Absolute virtual destination.
            content: Complete UTF-8 text.

        Returns:
            Native write result.
        """
        try:
            result = self.sandbox.write(self._path(file_path), content)
            return WriteResult(error=result.error, path=result.path)
        except SandboxErrors as exc:
            return WriteResult(error=str(exc))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Apply an exact text replacement with the existing atomic file lock.

        Args:
            file_path: Absolute virtual file.
            old_string: Exact text that must exist.
            new_string: Different replacement text.
            replace_all: Replace all matches when true; otherwise require one.

        Returns:
            Native edit result with replacement count.
        """
        try:
            return EditResult(
                **asdict(
                    self.sandbox.edit(
                        self._path(file_path), old_string, new_string, replace_all
                    )
                )
            )
        except SandboxErrors as exc:
            return EditResult(error=str(exc))

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a path through a disposable file checkout.

        Args:
            file_path: Absolute file or directory; directories are recursive.

        Returns:
            Native deletion result.
        """
        try:
            path = self._path(file_path)
            self.sandbox.kernel._call_filesystem_raw(
                "delete", {"path": path, "recursive": True}
            )
            return DeleteResult(path=path)
        except SandboxErrors as exc:
            return DeleteResult(error=str(exc))

    def _scan(self, path: str, recursive: bool) -> dict[str, Any]:
        """Scan bounded paths without saving or entering interpreter state.

        Args:
            path: Absolute virtual directory.
            recursive: Visit subdirectories when true.

        Returns:
            Raw entries and a scan-truncation flag.
        """
        return self.sandbox.kernel._call_filesystem_raw(
            "ls",
            {
                "path": path,
                "recursive": recursive,
                "limit": self.sandbox.kernel.filesystem_config.max_result_limit,
            },
        )

    def ls(self, path: str) -> LsResult:
        """List direct file and directory entries.

        Args:
            path: Absolute virtual directory.

        Returns:
            Native listing, with an error if the bounded listing is incomplete.
        """
        try:
            result = self._scan(self._path(path), False)
            return LsResult(
                entries=[
                    {"path": entry["path"], "is_dir": entry["type"] == "directory"}
                    for entry in result["entries"]
                ],
                error="listing reached the filesystem scan limit"
                if result["truncated"]
                else None,
            )
        except SandboxErrors as exc:
            return LsResult(error=str(exc))

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find regular files with the shared Deep Agents glob matcher.

        Args:
            pattern: Basename or root-relative glob, with classes and braces.
            path: Absolute search root; None uses the backend root.

        Returns:
            Absolute file records and an honest truncation flag, or an error.
        """
        try:
            root = self._path(path)
            match = compile_grep_include_glob(pattern)
            result = self._scan(root, True)
            entries: list[FileInfo] = [
                {"path": entry["path"], "is_dir": False}
                for entry in result["entries"]
                if entry["type"] == "file"
                and match(posixpath.relpath(entry["path"], root))
            ]
            return GlobResult(matches=entries, truncated=result["truncated"])
        except SandboxErrors as exc:
            return GlobResult(error=str(exc))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search literal text with a total result and character budget.

        Args:
            pattern: Literal substring, not a regular expression.
            path: File or directory; None uses the backend root.
            glob: Optional Deep Agents include-glob.
            max_count: Total requested match cap; application caps still apply.

        Returns:
            One-based line records and truncation state, or an error.
        """
        try:
            root = self._path(path)
            match = compile_grep_include_glob("*" if glob is None else glob)
            try:
                scan = self._scan(root, True)
                paths = [e["path"] for e in scan["entries"] if e["type"] == "file"]
            except SandboxErrors:
                # A file root is valid too. Download below distinguishes missing paths.
                paths, scan = [root], {"truncated": False}
            config = self.sandbox.kernel.filesystem_config
            cap = min(
                config.max_result_limit,
                max(0, max_count) if max_count is not None else config.max_result_limit,
            )
            matches, chars = [], 0
            truncated = scan["truncated"]
            for file in paths:
                relative = (
                    posixpath.basename(file)
                    if file == root
                    else posixpath.relpath(file, root)
                )
                if not match(relative):
                    continue
                try:
                    text = self.sandbox.kernel.download_file(file).decode("utf-8")
                except UnicodeDecodeError:
                    continue
                available = config.max_grep_chars - chars
                chars += len(text)
                if len(text) > available:
                    return GrepResult(matches=matches, truncated=True)
                for number, line in enumerate(text.splitlines(), 1):
                    if pattern in line:
                        if len(matches) == cap:
                            return GrepResult(matches=matches, truncated=True)
                        matches.append({"path": file, "line": number, "text": line})
            return GrepResult(matches=matches, truncated=truncated)
        except SandboxErrors as exc:
            return GrepResult(error=str(exc))

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload bytes independently, preserving order and partial success.

        Args:
            files: Absolute path and bytes pairs.

        Returns:
            One native result per input, including normalized error codes.
        """
        results = []
        for path, content in files:
            try:
                self.sandbox.kernel.upload_file(self._path(path), content)
                results.append(FileUploadResponse(path=path))
            except SandboxErrors as exc:
                results.append(
                    FileUploadResponse(path=path, error=_transfer_error(exc))
                )
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download bytes independently, preserving order and partial success.

        Args:
            paths: Absolute virtual files.

        Returns:
            One native result per input, without interpreting bytes as text.
        """
        results = []
        for path in paths:
            try:
                results.append(
                    FileDownloadResponse(
                        path=path,
                        content=self.sandbox.kernel.download_file(self._path(path)),
                    )
                )
            except SandboxErrors as exc:
                results.append(
                    FileDownloadResponse(path=path, error=_transfer_error(exc))
                )
        return results

    def _hosted_tools(self, *, config: HostedToolConfig | None = None) -> HostedToolSet:
        """Expose this same file authority to a detached Python interpreter.

        Args:
            config: Optional operation budgets inherited from the interpreter.

        Returns:
            Private async-callable aliases. The interpreter adds public fs_*
            wrappers so the native kernel helpers cannot shadow the bridge.
        """
        return HostedToolSet(
            tuple(
                HostedTool.from_callable(
                    method, name="fs_" + name, python_name="_arenero_backend_fs_" + name
                )
                for name, method in (
                    ("read", self.read),
                    ("write", self.write),
                    ("edit", self.edit),
                    ("delete", self.delete),
                    ("ls", self.ls),
                    ("glob", self.glob),
                    ("grep", self.grep),
                )
            ),
            config or HostedToolConfig(),
        )

    def _read_source(self, path: str) -> bytes:
        """Read bounded source bytes through the existing backend root.

        Args:
            path: Absolute virtual source path.

        Returns:
            Unpaged bytes, without changing persistent Python state.
        """
        return self.sandbox.kernel.download_file(self._path(path))

    def _write_output(self, path: str, content: bytes) -> None:
        """Write bounded output through the same backend authority.

        Args:
            path: Absolute virtual destination path.
            content: Output JSON bytes, subject to the normal transfer cap.
        """
        self.sandbox.kernel.upload_file(self._path(path), content)


class AreneroBackend(AreneroFilesBackend, BaseSandbox):
    """Add the existing virtual shell and opt-in native output capture.

    Args:
        sandbox: Caller-owned sandbox; no host process is started.
        root: Absolute file root, or None for the kernel workspace root.

    Attributes:
        enable_capture_offload: False by default. Set True to let Deep Agents
            save large, bounded execute output before display trimming.
    """

    def instructions(self) -> str:
        """Describe the limited shell and file authority.

        Returns:
            Shell rules, file root, and cooperative timeout limits.
        """
        return (
            MONTY_SANDBOX_INSTRUCTIONS
            + f"\nFile tools use the virtual root {self.root}. "
            "execute is not Bash. Its per-call timeout is cooperative: exit 124 "
            "stops later commands, but cannot preempt an active host call, "
            "DNS lookup, worker operation, or system sleep. Completed effects remain."
        )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run the limited virtual shell, never a host process.

        Args:
            command: Documented Arenero shell grammar and allowlisted commands.
            timeout: Cooperative deadline; zero or None uses configured limits.

        Returns:
            Combined bounded output; exit 124 means deadline exceeded.
        """
        result = self.sandbox.execute(command, timeout=timeout)
        return ExecuteResponse(
            output=result.output, exit_code=result.exit_code, truncated=result.truncated
        )

    def execute_with_offload(
        self,
        command: str,
        capture_path: str,
        *,
        max_inline_bytes: int,
        max_capture_bytes: int | None = None,
        timeout: int | None = None,
    ) -> ExecuteOffloadResult:
        """Capture bounded virtual output without a Bash wrapper or rerun.

        Args:
            command: Normal virtual shell command.
            capture_path: Absolute destination inside the backend file root.
            max_inline_bytes: Nonnegative inline UTF-8 byte budget.
            max_capture_bytes: Optional nonnegative capture cap, further bounded
                by the kernel transfer limit. None uses that limit.
            timeout: Existing cooperative deadline.

        Returns:
            Native offload record. Truncation marks incomplete capture, not a
            shortened preview. A failed write returns no file pointer and does
            not repeat or roll back the command.
        """
        if not self.enable_capture_offload:
            return ExecuteOffloadResult(
                offloaded=False, response=self.execute(command, timeout=timeout)
            )
        path = self._path(capture_path)
        if max_inline_bytes < 0 or (
            max_capture_bytes is not None and max_capture_bytes < 0
        ):
            raise ValueError("capture and inline byte limits must be nonnegative")
        config = self.sandbox.kernel.config
        cap = self.sandbox.kernel.filesystem_config.max_transfer_bytes
        if max_capture_bytes is not None:
            cap = min(cap, max_capture_bytes)
        captured: list[ShellResult] = []
        result = self.sandbox._execute_shell(
            command, timeout=timeout, capture=captured.append
        )
        source = captured[-1] if captured and result.exit_code != 124 else result
        data = source.output.encode("utf-8")
        truncated = source.truncated or len(data) > cap
        data = data[:cap].decode("utf-8", errors="ignore").encode("utf-8")
        budget = min(max_inline_bytes, config.default_output_limit_chars)
        if len(data) <= budget:
            return ExecuteOffloadResult(
                offloaded=False,
                response=ExecuteResponse(
                    output=data.decode("utf-8"),
                    exit_code=result.exit_code,
                    truncated=truncated,
                ),
            )
        try:
            self._write_output(path, data)
        except SandboxErrors as exc:
            output = _preview(
                f"Output capture failed: {exc}. Command effects remain.\n".encode()
                + data,
                budget,
            )
            return ExecuteOffloadResult(
                offloaded=False,
                response=ExecuteResponse(
                    output=output, exit_code=result.exit_code, truncated=True
                ),
            )
        return ExecuteOffloadResult(
            offloaded=True,
            response=ExecuteResponse(
                output=_preview(data, budget),
                exit_code=result.exit_code,
                truncated=truncated,
            ),
        )

    async def aexecute_with_offload(
        self,
        command: str,
        capture_path: str,
        *,
        max_inline_bytes: int,
        max_capture_bytes: int | None = None,
        timeout: int | None = None,
    ) -> ExecuteOffloadResult:
        """Offload the same capture operation to a host thread.

        Args:
            command: Virtual shell command.
            capture_path: Absolute capture destination.
            max_inline_bytes: Nonnegative inline UTF-8 byte budget.
            max_capture_bytes: Optional bounded file byte cap.
            timeout: Cooperative shell deadline.

        Returns:
            Native offload record. Cancellation does not stop an active thread.
        """
        return await asyncio.to_thread(
            self.execute_with_offload,
            command,
            capture_path,
            max_inline_bytes=max_inline_bytes,
            max_capture_bytes=max_capture_bytes,
            timeout=timeout,
        )


def _preview(data: bytes, limit: int) -> str:
    """Return a UTF-8 head/tail preview within a byte limit.

    Args:
        data: Bounded UTF-8 capture.
        limit: Nonnegative byte budget.

    Returns:
        Text with incomplete edge characters omitted.
    """
    if len(data) <= limit:
        return data.decode("utf-8")
    marker = b"\n... <trimmed> ...\n"
    if limit <= len(marker):
        return data[:limit].decode("utf-8", errors="ignore")
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return (
        data[:head].decode("utf-8", errors="ignore")
        + marker.decode()
        + data[-tail:].decode("utf-8", errors="ignore")
    )


def _backend_request(
    request: ModelRequest[ContextT], backend: AreneroFilesBackend
) -> ModelRequest[ContextT]:
    """Adjust stock tool descriptions without changing tools or filters.

    Args:
        request: Current filtered model request.
        backend: Explicit Arenero backend for this graph.

    Returns:
        Copied request with accurate standard descriptions. Custom descriptions
        and the registered tool schemas and callbacks are unchanged.
    """
    execution = isinstance(backend, AreneroBackend) and any(
        (t.name if isinstance(t, BaseTool) else t.get("name")) == "execute"
        for t in request.tools
    )
    tools = []
    for tool in request.tools:
        name = tool.name if isinstance(tool, BaseTool) else tool.get("name")
        description = (
            tool.description
            if isinstance(tool, BaseTool)
            else tool.get("description", "")
        )
        updated = description
        if name == "execute" and execution and description in _EXECUTE_DESCRIPTIONS:
            updated = (
                "Run Arenero's limited virtual shell, not Bash or CPython. "
                "Use help for enabled commands and flags. Text pipes, redirects, "
                "and Monty python -c/file.py are supported. No rg, Git, pip, "
                "arbitrary executables, or full Bash scripts. Prefer the registered "
                "file tools for files and searches. timeout is cooperative; exit "
                "124 does not undo completed effects. Check truncation notices."
            )
        elif name == "grep" and description in _GREP_DESCRIPTIONS:
            updated = (
                "Search for a LITERAL case-sensitive text pattern, not regex. "
                "Use separate calls for alternatives; foo|bar matches that exact "
                "text. Results use one-based lines and bounded output."
            )
            if execution:
                updated += " For regex, execute can use virtual grep -E with Python regex syntax, if enabled; rg is unavailable."
        if updated != description:
            tool = (
                tool.model_copy(update={"description": updated})
                if isinstance(tool, BaseTool)
                else {**tool, "description": updated}
            )
        tools.append(tool)
    return request.override(tools=tools)


class AreneroBackendMiddleware(AgentMiddleware):
    """Add Arenero backend instructions and accurate stock tool descriptions.

    Args:
        backend: Caller-owned file-only or virtual-shell backend used by the graph.
    """

    def __init__(self, backend: AreneroFilesBackend) -> None:
        """Bind a backend without opening or closing its resources."""
        self.backend = backend

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        """Build one model request using only the currently visible tools.

        Args:
            request: Filtered model request.

        Returns:
            Request with backend instructions and unchanged tool authority.
        """
        request = _backend_request(request, self.backend)
        return request.override(
            system_message=append_to_system_message(
                request.system_message, self.backend.instructions()
            )
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Prepare a synchronous model call.

        Args:
            request: Current model request.
            handler: Next middleware or model.

        Returns:
            Model response.
        """
        return handler(self._prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Prepare an asynchronous model call.

        Args:
            request: Current model request.
            handler: Next middleware or model.

        Returns:
            Awaited model response.
        """
        return await handler(self._prepare(request))


def _transfer_error(exc: BaseException) -> str:
    """Map common file failures to the backend transfer vocabulary.

    Args:
        exc: File transfer or path validation error.

    Returns:
        Standard error code, or the original error for other failures.
    """
    text = str(exc)
    for needle, code in (
        ("FileNotFound", "file_not_found"),
        ("not found", "file_not_found"),
        ("IsADirectory", "is_directory"),
        ("Permission", "permission_denied"),
        ("permission_denied", "permission_denied"),
        ("invalid_path", "invalid_path"),
    ):
        if needle in text:
            return code
    return text
