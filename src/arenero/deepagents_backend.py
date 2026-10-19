"""Optional Deep Agents backend records over the existing Arenero sandbox."""

import posixpath
from dataclasses import asdict
from typing import Any

from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import compile_grep_include_glob, create_file_data

from .hosted_tools import HostedTool, HostedToolConfig, HostedToolSet
from .sandbox import MONTY_SANDBOX_INSTRUCTIONS, AreneroSandbox, SandboxErrors


class AreneroBackend(SandboxBackendProtocol):
    """Adapt a caller-owned sandbox without shell-based file wrappers.

    Args:
        sandbox: Sync sandbox whose kernel owns all files and shell state.
        root: Absolute virtual root for file operations. Defaults to the kernel
            workspace root. This is not a new shell confinement boundary.
    """

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
        """Describe the limited shell and shared file authority for an agent.

        Returns:
            Current shell rules, file root, and cooperative timeout limits.
        """
        return (
            MONTY_SANDBOX_INSTRUCTIONS
            + f"\nFile tools use the virtual root {self.root}. "
            "execute is not Bash. Its per-call timeout is cooperative: exit 124 "
            "stops later commands, but cannot preempt an active host call, "
            "DNS lookup, worker operation, or system sleep. Completed effects remain."
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
