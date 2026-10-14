"""High-level owned sandbox APIs for common Arenero use cases."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import io
import json
import posixpath
import re
import shlex
import stat
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic_monty import MontyError, OSAccess

from .cell import execute_repl_cell
from .filesystem import FilesystemConfig
from .hosted_tools import HostedToolSet
from .kernel import KernelConfig, PersistentMontyKernel, StatePolicy
from .modules import VirtualModuleConfig
from .network import NetworkConfig
from .packages import MontyPackageManager, PackageConfig
from .runtime import AreneroRuntime
from .shell import VirtualShell
from .store import (
    InMemorySessionStore,
    SQLiteSessionStore,
    StatelessSessionStore,
)
from .values import json_value as _json_value
from .workspace import WorkspaceConfig

PersistenceMode = Literal["stateless", "memory", "durable"]
FilesystemMode = Literal["none", "memory", "workspace"]
NetworkMode = Literal["public", "off"]
SandboxFeature = Literal["python", "files", "shell", "packages"]

_CHECKPOINT_FORMAT = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MANAGED_ERRORS = (MontyError, TypeError, ValueError, RuntimeError)


@dataclass(frozen=True, slots=True)
class SandboxError:
    """Describe one stable managed-sandbox failure.

    Attributes:
        code: Machine-readable failure code.
        message: Human-readable failure text.
        hint: Suggested next action.
        retryable: Whether retrying without a code change can help.
    """

    code: str
    message: str
    hint: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Return one consistent result from a managed sandbox operation.

    Attributes:
        ok: Whether the operation succeeded.
        value: Primary result value.
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Shell-style status when applicable.
        error: Structured failure data.
        truncated: Whether any model-facing output was shortened.
        artifacts: Virtual output paths created by the operation.
        stats: Execution or operation statistics.
    """

    ok: bool
    value: Any = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    error: SandboxError | None = None
    truncated: bool = False
    artifacts: tuple[str, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert this result to a JSON-compatible mapping.

        Returns:
            Result fields with a nested error mapping.
        """

        return {
            "ok": self.ok,
            "value": _json_value(self.value),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "error": asdict(self.error) if self.error is not None else None,
            "truncated": self.truncated,
            "artifacts": list(self.artifacts),
            "stats": _json_value(dict(self.stats)),
        }


@dataclass(frozen=True, slots=True)
class SandboxAgentBundle:
    """Keep one fixed-session tool list with its generated instructions.

    Attributes:
        tools: Selected LangChain tools for one managed sandbox.
        instructions: Prompt-ready active sandbox instructions.
    """

    tools: tuple[Any, ...]
    instructions: str


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Configure the high-level sandbox factory and its presets.

    Attributes:
        persistence: Python state storage mode.
        filesystem: Virtual filesystem mode.
        directory: Host directory for durable state or workspace files.
        workspace_root: Confined virtual root used by the high-level handle.
        network: Public, disabled, or custom virtual curl policy.
        kernel: Existing kernel limits and worker settings.
        filesystem_limits: Existing per-operation filesystem limits.
        bootstrap_code: Trusted Python run once for each new session.
        bootstrap_files: Existing absolute virtual files run for each new session.
        mcp_artifact_directory: Optional absolute virtual MCP output directory.
        max_workspace_files: Maximum files managed below ``workspace_root``.
        max_workspace_bytes: Maximum combined workspace bytes.
        max_active_sandboxes: Maximum handles opened by one manager.
        allow_binary_assets: Allow non-Python package files under ``assets/`` when true.
    """

    persistence: PersistenceMode = "memory"
    filesystem: FilesystemMode = "memory"
    directory: str | Path | None = None
    workspace_root: str = "/workspace"
    network: NetworkMode | NetworkConfig = "public"
    kernel: KernelConfig = field(default_factory=KernelConfig)
    filesystem_limits: FilesystemConfig = field(default_factory=FilesystemConfig)
    bootstrap_code: str | None = None
    bootstrap_files: tuple[str, ...] = ()
    mcp_artifact_directory: str | None = None
    max_workspace_files: int = 1_000
    max_workspace_bytes: int = 10_000_000
    max_active_sandboxes: int = 100
    allow_binary_assets: bool = False

    def __post_init__(self) -> None:
        """Validate related persistence, path, and quota settings."""

        root = posixpath.normpath(self.workspace_root)
        if (
            root != self.workspace_root
            or not root.startswith("/")
            or root.startswith("//")
            or root == "/"
        ):
            raise ValueError("workspace_root must be a normalized absolute path")
        if self.persistence not in {"stateless", "memory", "durable"}:
            raise ValueError("persistence must be stateless, memory, or durable")
        if self.filesystem not in {"none", "memory", "workspace"}:
            raise ValueError("filesystem must be none, memory, or workspace")
        if isinstance(self.network, str) and self.network not in {"public", "off"}:
            raise ValueError("network must be public, off, or NetworkConfig")
        confined = (*self.bootstrap_files, self.mcp_artifact_directory)
        if any(
            path is not None
            and (
                posixpath.normpath(path) != path
                or (
                    path != self.workspace_root
                    and not path.startswith(self.workspace_root + "/")
                )
            )
            for path in confined
        ):
            raise ValueError(
                "bootstrap and MCP artifact paths must stay in workspace_root"
            )
        if self.filesystem == "none" and (
            self.bootstrap_files or self.mcp_artifact_directory is not None
        ):
            raise ValueError("bootstrap files and MCP artifacts require a filesystem")
        if self.persistence == "durable" and self.directory is None:
            raise ValueError("durable persistence requires directory")
        if self.filesystem == "workspace" and self.directory is None:
            raise ValueError("workspace filesystem requires directory")
        limits = (
            self.max_workspace_files,
            self.max_workspace_bytes,
            self.max_active_sandboxes,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("managed sandbox limits must be greater than 0")

    @classmethod
    def ephemeral(cls, **overrides: Any) -> Self:
        """Create a stateless preset with in-memory files.

        Args:
            **overrides: Explicit ``SandboxConfig`` field overrides.

        Returns:
            Stateless preset configuration.
        """

        return cls(persistence="stateless", filesystem="memory", **overrides)

    @classmethod
    def conversation(cls, **overrides: Any) -> Self:
        """Create a process-lifetime state and file preset.

        Args:
            **overrides: Explicit ``SandboxConfig`` field overrides.

        Returns:
            In-memory conversation preset configuration.
        """

        return cls(persistence="memory", filesystem="memory", **overrides)

    @classmethod
    def durable(cls, directory: str | Path, **overrides: Any) -> Self:
        """Create a SQLite state and host-workspace preset.

        Args:
            directory: Dedicated host directory for state and files.
            **overrides: Explicit ``SandboxConfig`` field overrides.

        Returns:
            Durable preset configuration.
        """

        return cls(
            persistence="durable",
            filesystem="workspace",
            directory=directory,
            **overrides,
        )

    @classmethod
    def batch(cls, workers: int = 4, **overrides: Any) -> Self:
        """Create an in-memory preset for concurrent agent sessions.

        Args:
            workers: Fixed Monty worker pool size.
            **overrides: Explicit ``SandboxConfig`` field overrides.

        Returns:
            In-memory state and Python-only workers for true session overlap.
        """

        if workers <= 0:
            raise ValueError("workers must be greater than 0")
        kernel = overrides.pop("kernel", KernelConfig())
        filesystem = overrides.pop("filesystem", "none")
        if not isinstance(kernel, KernelConfig):
            raise TypeError("kernel must be KernelConfig")
        return cls(
            persistence="memory",
            filesystem=filesystem,
            kernel=replace(
                kernel,
                min_worker_processes=workers,
                max_worker_processes=workers,
            ),
            **overrides,
        )


class ManagedSandbox:
    """Own or borrow one configured sandbox session and workspace."""

    def __init__(
        self,
        runtime: AreneroRuntime,
        sandbox_id: str,
        config: SandboxConfig,
        *,
        owns_runtime: bool,
    ) -> None:
        """Create a managed sandbox handle.

        Args:
            runtime: Runtime that owns the configured kernel.
            sandbox_id: Stable session and public sandbox ID.
            config: High-level sandbox configuration.
            owns_runtime: Close the runtime when this handle closes.
        """

        _validate_id(sandbox_id)
        self.runtime = runtime
        self.kernel = runtime.kernel
        self.sandbox_id = sandbox_id
        self.session_id = sandbox_id
        self.config = config
        self.root = config.workspace_root
        self._owns_runtime = owns_runtime
        self._closed = False
        self._ensure_root()

    @property
    def cwd(self) -> str:
        """Read the last saved Monty directory shared by all sandbox APIs.

        Returns:
            Native virtual directory, or the configured root for a new session.
        """
        return self.kernel.get_working_directory(self.session_id)

    @property
    def id(self) -> str:
        """Return the stable sandbox ID.

        Returns:
            Sandbox ID.
        """

        return self.sandbox_id

    def __enter__(self) -> Self:
        """Return this open managed sandbox.

        Returns:
            This sandbox.
        """

        return self

    def __exit__(self, *_: object) -> None:
        """Close resources owned by this handle."""

        self.close()

    def close(self) -> None:
        """Close this handle and any runtime it owns."""

        if self._closed or not self._owns_runtime:
            return
        self._closed = True
        self.runtime.close()

    def resolve_path(self, path: str) -> str:
        """Resolve one relative or absolute path inside the workspace root.

        Args:
            path: Relative path, or an absolute path inside the root.

        Returns:
            Normalized absolute virtual path.

        Raises:
            ValueError: If the path is invalid or escapes the workspace.
        """

        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError("path must be a non-empty string")
        candidate = path if path.startswith("/") else posixpath.join(self.cwd, path)
        normalized = posixpath.normpath(candidate)
        if normalized != self.root and not normalized.startswith(self.root + "/"):
            raise ValueError(f"path must stay below {self.root}")
        return normalized

    def chdir(self, path: str) -> SandboxResult:
        """Change the working directory used by relative high-level paths.

        Args:
            path: Existing relative or absolute virtual directory.

        Returns:
            Result containing the new directory.
        """

        try:
            resolved = self.resolve_path(path)
            changed = self.kernel.execute(
                self.session_id,
                f"import os as _arenero_os\n_arenero_os.chdir({resolved!r})",
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "invalid_path", str(exc), "Choose a directory in the workspace."
            )
        if not changed["ok"]:
            return _from_error_mapping(changed.get("error"), "directory_not_found")
        return SandboxResult(True, resolved)

    def run_python(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> SandboxResult:
        """Run one Monty Python feed in this persistent session.

        Args:
            code: Monty-compatible Python source.
            output_limit_chars: Optional output limit below the host maximum.
            output_file: Optional relative or absolute overflow file.
            state_policy: Python state save policy.
            restart: Reset Python state before execution when true.
            hosted_tools: Optional operation-only host tools.
            inputs: Optional values bound before this user feed, after bootstrap.

        Returns:
            Common managed sandbox result.
        """

        def operation() -> SandboxResult:
            """Run the configured kernel operation."""

            try:
                overflow = self.resolve_path(output_file) if output_file else None
                result = self.kernel.execute(
                    self.session_id,
                    code,
                    output_limit_chars,
                    output_file=overflow,
                    state_policy=state_policy,
                    restart=restart,
                    hosted_tools=hosted_tools,
                    inputs=inputs,
                )
            except _MANAGED_ERRORS as exc:
                return _failure(
                    "python_execution_failed",
                    str(exc),
                    "Revise the code or configuration.",
                )
            return _from_execution(result)

        return self._guard_workspace(operation)

    def run_shell(
        self,
        command: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
    ) -> SandboxResult:
        """Run one allowlisted virtual-shell command list.

        Args:
            command: Virtual-shell text, not an operating-system command.
            output_limit_chars: Optional output limit below the host maximum.
            output_file: Optional relative or absolute overflow file.

        Returns:
            Common managed sandbox result.
        """

        def operation() -> SandboxResult:
            """Run and normalize the virtual-shell operation."""

            try:
                overflow = self.resolve_path(output_file) if output_file else None
                result = VirtualShell(self.kernel, self.session_id).execute(
                    command,
                    output_limit_chars,
                    output_file=overflow,
                )
            except _MANAGED_ERRORS as exc:
                return _failure(
                    "shell_execution_failed",
                    str(exc),
                    "Use the documented virtual shell grammar.",
                )
            if result.exit_code:
                return SandboxResult(
                    False,
                    result.output,
                    exit_code=result.exit_code,
                    error=SandboxError(
                        "shell_command_failed",
                        result.output.rstrip() or "virtual shell command failed",
                        "Inspect the command output and use only enabled shell forms.",
                    ),
                    truncated=result.truncated,
                    artifacts=(result.output_file,) if result.output_file else (),
                )
            return SandboxResult(
                True,
                result.output,
                exit_code=0,
                truncated=result.truncated,
                artifacts=(result.output_file,) if result.output_file else (),
            )

        return self._guard_workspace(operation)

    def run_cell(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
        restart: bool = False,
    ) -> SandboxResult:
        """Run a Python, ``!command``, or ``%%shell`` cell.

        Args:
            code: Notebook-style cell source.
            output_limit_chars: Optional output limit below the host maximum.
            output_file: Optional relative or absolute overflow file.
            state_policy: Python state save policy.
            restart: Reset Python state before a Python cell when true.
            inputs: Values bound once before the first user-code feed.

        Returns:
            Common managed sandbox result.
        """

        stripped = code.lstrip()
        if inputs and stripped.startswith(("!", "%%")):
            return _failure(
                "invalid_cell_inputs",
                "Shell cells do not accept inputs.",
                "Use a Python cell or %run for inputs.",
            )
        if stripped.startswith("%%shell"):
            _, _, command = stripped.partition("\n")
            return self.run_shell(command, output_limit_chars, output_file=output_file)
        if stripped.startswith("!"):
            return self.run_shell(
                stripped[1:], output_limit_chars, output_file=output_file
            )
        if stripped.startswith("%run"):
            try:
                arguments = shlex.split(stripped[1:])
            except ValueError as exc:
                return _failure(
                    "invalid_repl_command", str(exc), "Check the command quoting."
                )
            if len(arguments) == 2 and arguments[0] == "run":
                return self.run_file(
                    arguments[1],
                    output_limit_chars=output_limit_chars,
                    output_file=output_file,
                    state_policy=state_policy,
                    inputs=inputs,
                    restart=restart,
                )
        if stripped.startswith("%"):
            try:
                result = execute_repl_cell(
                    self.kernel,
                    self.session_id,
                    stripped,
                    output_limit_chars,
                    output_file=(
                        self.resolve_path(output_file) if output_file else None
                    ),
                    state_policy=state_policy,
                    inputs=inputs,
                    restart=restart,
                )
            except _MANAGED_ERRORS as exc:
                return _failure(
                    "repl_command_failed", str(exc), "Use a supported percent command."
                )
            return _from_execution(result)
        return self.run_python(
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            inputs=inputs,
            restart=restart,
        )

    def run_file(
        self,
        path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
        restart: bool = False,
    ) -> SandboxResult:
        """Run one virtual Python file and optional follow-up code.

        Args:
            path: Relative or absolute virtual Python file.
            code: Optional follow-up code in the same operation.
            output_limit_chars: Optional public output limit.
            output_file: Relative or absolute overflow file within the workspace.
            state_policy: Python state save policy.
            inputs: Values bound once before the first user-code feed.
            restart: Reset Python state before this file when true.

        Returns:
            Common managed sandbox result.
        """

        def operation() -> SandboxResult:
            """Run and normalize the virtual file operation."""

            try:
                result = self.kernel.execute_file(
                    self.session_id,
                    self.resolve_path(path),
                    code=code,
                    output_limit_chars=output_limit_chars,
                    output_file=self.resolve_path(output_file) if output_file else None,
                    state_policy=state_policy,
                    inputs=inputs,
                    restart=restart,
                )
            except _MANAGED_ERRORS as exc:
                return _failure(
                    "file_execution_failed",
                    str(exc),
                    "Choose a readable Monty Python file.",
                )
            return _from_execution(result)

        return self._guard_workspace(operation)

    def run_module(
        self,
        path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        module_config: VirtualModuleConfig | None = None,
        state_policy: StatePolicy = "always",
        inputs: Mapping[str, Any] | None = None,
    ) -> SandboxResult:
        """Run one bounded static virtual-module graph.

        Args:
            path: Relative or absolute entry Python file.
            code: Optional follow-up code.
            output_limit_chars: Optional public output limit.
            output_file: Relative or absolute overflow file within the workspace.
            module_config: Optional explicit module rules.
            state_policy: Python state save policy.
            inputs: Values bound once before the first user-code feed.

        Returns:
            Common managed sandbox result.
        """

        def operation() -> SandboxResult:
            """Run and normalize the virtual module operation."""

            try:
                result = self.kernel.execute_module(
                    self.session_id,
                    self.resolve_path(path),
                    code,
                    output_limit_chars,
                    output_file=self.resolve_path(output_file) if output_file else None,
                    module_config=module_config
                    or VirtualModuleConfig(roots=(self.root,)),
                    state_policy=state_policy,
                    inputs=inputs,
                )
            except _MANAGED_ERRORS as exc:
                return _failure(
                    "module_execution_failed",
                    str(exc),
                    "Use a supported static module graph.",
                )
            return _from_execution(result)

        return self._guard_workspace(operation)

    def package_init(
        self,
        name: str,
        root: str | None = None,
        version: str = "0.1.0",
    ) -> SandboxResult:
        """Create one dependency-free Monty package scaffold.

        Args:
            name: Python package identifier.
            root: Optional relative or absolute project directory.
            version: Simple source-package version.

        Returns:
            Package initialization result.
        """

        def operation() -> SandboxResult:
            """Create and normalize the package scaffold."""

            target = self.resolve_path(root or name)
            return _from_package(self._packages().init(name, target, version))

        return self._guard_workspace(operation)

    def package_check(
        self,
        root: str,
        expected_revision: str | None = None,
    ) -> SandboxResult:
        """Validate one dependency-free Monty package.

        Args:
            root: Relative or absolute package project directory.
            expected_revision: Optional revision from an earlier check.

        Returns:
            Package validation result.
        """

        try:
            return _from_package(
                self._packages().check(
                    self.resolve_path(root),
                    expected_revision,
                )
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "package_check_failed", str(exc), "Check the package path and limits."
            )

    def package_test(
        self,
        root: str,
        expected_revision: str | None = None,
    ) -> SandboxResult:
        """Run one package verification file in a fresh Monty session.

        Args:
            root: Relative or absolute package project directory.
            expected_revision: Optional revision from an earlier check.

        Returns:
            Package verification result.
        """

        try:
            return _from_package(
                self._packages().test(
                    self.resolve_path(root),
                    expected_revision,
                )
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "package_test_failed",
                str(exc),
                "Check the package source and verify.py.",
            )

    def package_export(
        self,
        root: str,
        expected_revision: str | None = None,
    ) -> SandboxResult:
        """Create one deterministic Arenero source ZIP.

        Args:
            root: Relative or absolute package project directory.
            expected_revision: Optional revision from an earlier check.

        Returns:
            Package export result with the virtual archive path.
        """

        def operation() -> SandboxResult:
            """Export and normalize the package archive."""

            return _from_package(
                self._packages().export(
                    self.resolve_path(root),
                    expected_revision,
                )
            )

        return self._guard_workspace(operation)

    def read(self, path: str, *, encoding: str | None = "utf-8") -> SandboxResult:
        """Read one relative or absolute virtual file.

        Args:
            path: Virtual file path.
            encoding: Text encoding, or ``None`` to return bytes.

        Returns:
            File content, revision, path, and size.
        """

        try:
            resolved = self.resolve_path(path)
            content = self.kernel.download_file(resolved)
            value: Any = content if encoding is None else content.decode(encoding)
        except UnicodeDecodeError as exc:
            return _failure(
                "file_decode_failed",
                str(exc),
                "Read the file as bytes with encoding=None.",
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "file_read_failed", str(exc), "Check the path and transfer limit."
            )
        return SandboxResult(
            True,
            value,
            stats={
                "path": resolved,
                "size_bytes": len(content),
                "revision": _revision(content),
            },
        )

    def write(
        self,
        path: str,
        content: str | bytes,
        *,
        expected_revision: str | None = None,
    ) -> SandboxResult:
        """Create or replace one file with an optional revision check.

        Args:
            path: Relative or absolute virtual file path.
            content: UTF-8 text or bytes.
            expected_revision: Optional revision returned by ``read``.

        Returns:
            New path, size, and revision.
        """

        try:
            resolved = self.resolve_path(path)
            data = content.encode() if isinstance(content, str) else content
            if expected_revision is not None:
                current = self.kernel.download_file(resolved)
                if _revision(current) != expected_revision:
                    return _failure(
                        "file_revision_conflict",
                        "file changed after it was read",
                        "Read the file again and apply the edit to the new revision.",
                    )
            self._require_projected_quota(resolved, data)
            self.kernel.upload_file(resolved, data)
        except _MANAGED_ERRORS as exc:
            return _failure(
                "file_write_failed", str(exc), "Check the path and workspace limits."
            )
        return SandboxResult(
            True,
            resolved,
            stats={
                "path": resolved,
                "size_bytes": len(data),
                "revision": _revision(data),
            },
        )

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        expected_revision: str | None = None,
    ) -> SandboxResult:
        """Replace exact text with an optional file revision check.

        Args:
            path: Relative or absolute UTF-8 file path.
            old: Exact source text.
            new: Replacement text.
            replace_all: Replace all occurrences when true.
            expected_revision: Optional revision returned by ``read``.

        Returns:
            New revision and replacement count.
        """

        read = self.read(path)
        if not read.ok:
            return read
        if (
            expected_revision is not None
            and read.stats["revision"] != expected_revision
        ):
            return _failure(
                "file_revision_conflict",
                "file changed after it was read",
                "Read the file again and apply the edit to the new revision.",
            )
        text = read.value
        if not isinstance(text, str):
            return _failure(
                "file_decode_failed",
                "edit requires UTF-8 text",
                "Use write for binary files.",
            )
        count = text.count(old)
        if count == 0:
            return _failure(
                "text_not_found",
                "old text was not found",
                "Read the current file before editing.",
            )
        if not replace_all and count != 1:
            return _failure(
                "ambiguous_edit",
                "old text occurs more than once",
                "Use a larger exact match or set replace_all=True.",
            )
        updated = text.replace(old, new, -1 if replace_all else 1)
        written = self.write(
            path, updated, expected_revision=str(read.stats["revision"])
        )
        if not written.ok:
            return written
        return replace(
            written, stats={**written.stats, "occurrences": count if replace_all else 1}
        )

    def mkdir(self, path: str, *, parents: bool = True) -> SandboxResult:
        """Create one virtual directory.

        Args:
            path: Relative or absolute directory path.
            parents: Create missing parents when true.

        Returns:
            Created directory path.
        """

        try:
            resolved = self.resolve_path(path)
            result = self.kernel.call_filesystem(
                self.session_id,
                "mkdir",
                {"path": resolved, "parents": parents, "exist_ok": parents},
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "directory_create_failed", str(exc), "Check the workspace path."
            )
        if not result["ok"]:
            return _from_error_mapping(result.get("error"), "directory_create_failed")
        return SandboxResult(True, resolved)

    def delete(self, path: str, *, recursive: bool = False) -> SandboxResult:
        """Delete one virtual path.

        Args:
            path: Relative or absolute path below the workspace root.
            recursive: Delete a non-empty directory when true.

        Returns:
            Deleted path.
        """

        try:
            resolved = self.resolve_path(path)
            if resolved == self.root:
                raise ValueError("the workspace root cannot be deleted")
            result = self.kernel.call_filesystem(
                self.session_id,
                "delete",
                {"path": resolved, "recursive": recursive},
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "file_delete_failed",
                str(exc),
                "Choose a path below the workspace root.",
            )
        if not result["ok"]:
            return _from_error_mapping(result.get("error"), "file_delete_failed")
        return SandboxResult(True, resolved)

    def copy(self, source: str, destination: str) -> SandboxResult:
        """Copy one regular virtual file.

        Args:
            source: Relative or absolute source file.
            destination: Relative or absolute destination file.

        Returns:
            Destination path and revision.
        """

        read = self.read(source, encoding=None)
        if not read.ok:
            return read
        return self.write(destination, read.value)

    def move(self, source: str, destination: str) -> SandboxResult:
        """Move one regular virtual file.

        Args:
            source: Relative or absolute source file.
            destination: Relative or absolute destination file.

        Returns:
            Destination path and revision.
        """

        try:
            if self.resolve_path(source) == self.resolve_path(destination):
                return _failure(
                    "same_move_path",
                    "source and destination are the same file",
                    "Choose another destination path.",
                )
        except ValueError as exc:
            return _failure("invalid_path", str(exc), "Choose paths in the workspace.")
        copied = self.copy(source, destination)
        if not copied.ok:
            return copied
        removed = self.delete(source)
        return copied if removed.ok else removed

    def ls(self, path: str = ".", *, recursive: bool = False) -> SandboxResult:
        """List bounded virtual entries.

        Args:
            path: Relative or absolute directory.
            recursive: Include nested entries when true.

        Returns:
            Entry mappings and truncation status.
        """

        try:
            resolved = self.resolve_path(path)
            result = self.kernel.call_filesystem(
                self.session_id,
                "ls",
                {"path": resolved, "recursive": recursive, "limit": None},
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "directory_list_failed", str(exc), "Check the directory path."
            )
        if not result["ok"]:
            return _from_error_mapping(result.get("error"), "directory_list_failed")
        value = result["result"]
        return SandboxResult(
            True,
            value["entries"],
            truncated=bool(value["truncated"]),
        )

    def glob(self, pattern: str, path: str = ".") -> SandboxResult:
        """Find paths with the bounded virtual glob rules.

        Args:
            pattern: Pattern with ``*``, ``?``, or ``**``.
            path: Relative or absolute search root.

        Returns:
            Matching paths and truncation status.
        """

        try:
            result = self.kernel.call_filesystem(
                self.session_id,
                "glob",
                {"pattern": pattern, "root": self.resolve_path(path), "limit": None},
            )
        except _MANAGED_ERRORS as exc:
            return _failure("glob_failed", str(exc), "Check the root and glob pattern.")
        if not result["ok"]:
            return _from_error_mapping(result.get("error"), "glob_failed")
        value = result["result"]
        return SandboxResult(True, value["matches"], truncated=bool(value["truncated"]))

    def grep(self, query: str, path: str = ".", pattern: str = "**/*") -> SandboxResult:
        """Find literal text in bounded virtual UTF-8 files.

        Args:
            query: Literal text to find.
            path: Relative or absolute search root.
            pattern: File-selection glob.

        Returns:
            Matching line records and truncation status.
        """

        try:
            result = self.kernel.call_filesystem(
                self.session_id,
                "grep",
                {
                    "query": query,
                    "root": self.resolve_path(path),
                    "pattern": pattern,
                    "regex": False,
                    "case_sensitive": True,
                    "limit": None,
                },
            )
        except _MANAGED_ERRORS as exc:
            return _failure(
                "grep_failed", str(exc), "Use literal text and a supported glob."
            )
        if not result["ok"]:
            return _from_error_mapping(result.get("error"), "grep_failed")
        value = result["result"]
        return SandboxResult(True, value["matches"], truncated=bool(value["truncated"]))

    def exists(self, path: str) -> SandboxResult:
        """Check whether one virtual path exists.

        Args:
            path: Relative or absolute virtual path.

        Returns:
            Boolean existence value.
        """

        try:
            resolved = self.resolve_path(path)
        except ValueError as exc:
            return _failure("invalid_path", str(exc), "Choose a path in the workspace.")
        if resolved == self.root:
            return SandboxResult(True, self.config.filesystem != "none")
        parent = posixpath.dirname(resolved)
        listed = self.ls(parent)
        if not listed.ok:
            return SandboxResult(True, False)
        return SandboxResult(
            True,
            any(entry.get("path") == resolved for entry in listed.value),
        )

    def stat(self, path: str) -> SandboxResult:
        """Return simple metadata for one virtual path.

        Args:
            path: Relative or absolute virtual path.

        Returns:
            Path, type, and file size and revision when applicable.
        """

        try:
            resolved = self.resolve_path(path)
        except ValueError as exc:
            return _failure("invalid_path", str(exc), "Choose a path in the workspace.")
        if resolved == self.root and self.config.filesystem != "none":
            return SandboxResult(True, {"path": resolved, "type": "directory"})
        parent = posixpath.dirname(resolved)
        listed = self.ls(parent)
        if not listed.ok:
            return listed
        entry = next(
            (item for item in listed.value if item.get("path") == resolved), None
        )
        if entry is None:
            return _failure(
                "path_not_found", f"path does not exist: {resolved}", "Check the path."
            )
        if entry.get("type") == "directory":
            return SandboxResult(True, {"path": resolved, "type": "directory"})
        read = self.read(resolved, encoding=None)
        if not read.ok:
            return read
        return SandboxResult(
            True,
            {
                "path": resolved,
                "type": "file",
                "size_bytes": read.stats["size_bytes"],
                "revision": read.stats["revision"],
            },
        )

    def upload_tree(self, source: str | Path, destination: str = ".") -> SandboxResult:
        """Upload one local directory tree through the host transfer plane.

        Args:
            source: Local host directory.
            destination: Virtual destination directory.

        Returns:
            Uploaded virtual paths.
        """

        host_root = Path(source)
        if not host_root.is_dir():
            return _failure(
                "upload_source_invalid",
                "source must be a directory",
                "Choose an existing local directory.",
            )
        entries = sorted(host_root.rglob("*"))
        if any(path.is_symlink() for path in entries):
            return _failure(
                "upload_symlink_rejected",
                "upload trees cannot contain symlinks",
                "Upload regular files only.",
            )
        files = [path for path in entries if path.is_file()]
        try:
            projected = {
                posixpath.relpath(
                    self.resolve_path(
                        posixpath.join(
                            destination, path.relative_to(host_root).as_posix()
                        )
                    ),
                    self.root,
                ): path.read_bytes()
                for path in files
            }
            self._require_tree_quota(projected)
        except (OSError, ValueError) as exc:
            return _failure(
                "upload_tree_failed", str(exc), "Check the source and workspace limits."
            )
        uploaded: list[str] = []
        for relative, content in projected.items():
            target = f"{self.root}/{relative}"
            result = self.write(target, content)
            if not result.ok:
                return result
            uploaded.append(target)
        return SandboxResult(True, uploaded)

    def download_tree(
        self,
        destination: str | Path,
        source: str = ".",
        *,
        overwrite: bool = False,
    ) -> SandboxResult:
        """Download one virtual tree into a local host directory.

        Args:
            destination: Local host directory.
            source: Virtual source directory.
            overwrite: Replace existing local files when true.

        Returns:
            Written local paths.
        """

        root = self.resolve_path(source)
        try:
            files = self._capture_files(root=root)
            local_root = Path(destination)
            written: list[str] = []
            for relative, content in files.items():
                target = local_root / relative
                if target.exists() and not overwrite:
                    raise FileExistsError(f"local file already exists: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                written.append(str(target))
        except (MontyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            return _failure(
                "download_tree_failed",
                str(exc),
                "Check the local destination and virtual source.",
            )
        return SandboxResult(True, written)

    def export_tree(self, source: str = ".") -> SandboxResult:
        """Create a deterministic ZIP of virtual workspace files.

        Args:
            source: Relative or absolute virtual source directory.

        Returns:
            ZIP bytes and file statistics.
        """

        try:
            files = self._capture_files(root=self.resolve_path(source))
            data = _zip_files(files)
        except _MANAGED_ERRORS as exc:
            return _failure(
                "tree_export_failed", str(exc), "Reduce the tree or check its paths."
            )
        return SandboxResult(
            True, data, stats={"file_count": len(files), "size_bytes": len(data)}
        )

    def import_tree(self, archive: bytes, destination: str = ".") -> SandboxResult:
        """Import a bounded ZIP into the virtual workspace.

        Args:
            archive: ZIP bytes from a trusted application boundary.
            destination: Relative or absolute virtual destination.

        Returns:
            Imported virtual paths.
        """

        try:
            files = _read_zip_files(
                archive,
                self.config.max_workspace_files,
                self.config.max_workspace_bytes,
            )
            projected = {
                posixpath.relpath(
                    self.resolve_path(posixpath.join(destination, relative)),
                    self.root,
                ): content
                for relative, content in files.items()
            }
            self._require_tree_quota(projected)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            return _failure(
                "tree_import_failed",
                str(exc),
                "Use a bounded ZIP with safe relative file paths.",
            )
        imported: list[str] = []
        for relative, content in projected.items():
            result = self.write(f"{self.root}/{relative}", content)
            if not result.ok:
                return result
            imported.append(str(result.value))
        return SandboxResult(True, imported)

    def checkpoint(self) -> SandboxResult:
        """Create a portable checkpoint with Python state and workspace files.

        Returns:
            Checkpoint ZIP bytes and safe metadata.
        """

        try:
            files = self._capture_files()
            stored = self.kernel.store.load(self.session_id)
            metadata = {
                "format": _CHECKPOINT_FORMAT,
                "sandbox_id": self.sandbox_id,
                "monty_version": self.kernel.monty_version,
                "workspace_root": self.root,
                "cwd": self.cwd,
                "policy_digest": self.policy_digest(),
                "created_at": time.time(),
                "has_state": stored is not None,
            }
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as output:
                _write_zip_entry(
                    output,
                    "metadata.json",
                    json.dumps(metadata, sort_keys=True).encode(),
                )
                if stored is not None:
                    _write_zip_entry(output, "state.bin", stored.state)
                for relative, content in files.items():
                    _write_zip_entry(output, f"files/{relative}", content)
            data = stream.getvalue()
        except _MANAGED_ERRORS as exc:
            return _failure(
                "checkpoint_failed",
                str(exc),
                "Check session state and workspace limits.",
            )
        return SandboxResult(
            True,
            data,
            stats={
                "file_count": len(files),
                "size_bytes": len(data),
                "has_state": stored is not None,
            },
        )

    def restore(self, checkpoint: bytes) -> SandboxResult:
        """Replace Python state and workspace files from one checkpoint.

        Args:
            checkpoint: Bytes returned by ``checkpoint``.

        Returns:
            Restored checkpoint metadata.
        """

        try:
            with zipfile.ZipFile(io.BytesIO(checkpoint)) as source:
                metadata_info = source.getinfo("metadata.json")
                if metadata_info.file_size > 64_000:
                    raise ValueError("checkpoint metadata is too large")
                metadata = json.loads(source.read(metadata_info))
                if not isinstance(metadata, dict):
                    raise TypeError("checkpoint metadata must be an object")
                if metadata.get("format") != _CHECKPOINT_FORMAT:
                    raise ValueError("unsupported checkpoint format")
                if metadata.get("monty_version") != self.kernel.monty_version:
                    raise ValueError("checkpoint uses another Monty version")
                if metadata.get("workspace_root") != self.root:
                    raise ValueError("checkpoint uses another workspace root")
                if metadata.get("policy_digest") != self.policy_digest():
                    raise ValueError("checkpoint uses another sandbox policy")
                files = _read_checkpoint_files(
                    source,
                    self.config.max_workspace_files,
                    self.config.max_workspace_bytes,
                )
                state = None
                if metadata.get("has_state"):
                    state_info = source.getinfo("state.bin")
                    if state_info.file_size > self.config.kernel.max_state_bytes:
                        raise ValueError("checkpoint state exceeds max_state_bytes")
                    state = source.read(state_info)
            with self.kernel.filesystem_guard():
                previous_files = self._capture_files()
                previous_state = self.kernel.store.load(self.session_id)
                try:
                    self._replace_files(files)
                    self.kernel.store.delete(self.session_id)
                    if state is not None:
                        self.kernel.store.save(
                            self.session_id,
                            state,
                            self.kernel.monty_version,
                            None,
                        )
                except Exception:
                    self._replace_files(previous_files)
                    self.kernel.store.delete(self.session_id)
                    if previous_state is not None:
                        self.kernel.store.save(
                            self.session_id,
                            previous_state.state,
                            previous_state.monty_version,
                            None,
                        )
                    raise
        except (
            KeyError,
            MontyError,
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
            zipfile.BadZipFile,
        ) as exc:
            return _failure(
                "checkpoint_restore_failed",
                str(exc),
                "Use a checkpoint from the same active sandbox policy and Monty version.",
            )
        return SandboxResult(True, metadata)

    def reset(self) -> SandboxResult:
        """Delete Python state for this sandbox without deleting files.

        Returns:
            Whether saved state existed.
        """

        return SandboxResult(True, self.kernel.reset(self.session_id))

    def fork(self, sandbox_id: str) -> SandboxResult:
        """Fork Python state while sharing the configured files.

        Args:
            sandbox_id: New session identifier.

        Returns:
            New borrowed managed sandbox handle.
        """

        try:
            _validate_id(sandbox_id)
            info = self.kernel.fork_session(self.session_id, sandbox_id)
        except _MANAGED_ERRORS as exc:
            return _failure(
                "sandbox_fork_failed", str(exc), "Choose a new valid sandbox ID."
            )
        child = ManagedSandbox(
            self.runtime,
            sandbox_id,
            self.config,
            owns_runtime=False,
        )
        return SandboxResult(True, child, stats={"session": asdict(info)})

    def info(self) -> SandboxResult:
        """Return active identity, persistence, capabilities, and workspace use.

        Returns:
            Safe sandbox metadata.
        """

        files = self._capture_files()
        sessions = {item.session_id: item for item in self.kernel.store.list_sessions()}
        current = sessions.get(self.session_id)
        return SandboxResult(
            True,
            {
                "id": self.id,
                "session_id": self.session_id,
                "root": self.root,
                "cwd": self.cwd,
                "persistence": self.config.persistence,
                "filesystem": self.config.filesystem,
                "allow_binary_assets": self.config.allow_binary_assets,
                "policy_digest": self.policy_digest(),
                "workspace": {
                    "file_count": len(files),
                    "total_bytes": sum(len(value) for value in files.values()),
                    "max_files": self.config.max_workspace_files,
                    "max_bytes": self.config.max_workspace_bytes,
                },
                "session": asdict(current) if current is not None else None,
                "capabilities": self.kernel.capabilities(),
            },
        )

    def instructions(self) -> str:
        """Return prompt-ready instructions for the managed sandbox.

        Returns:
            Active kernel instructions plus relative-path and checkpoint rules.
        """

        workspace = (
            "No virtual filesystem is configured. Use Python tools only.\n"
            if self.config.filesystem == "none"
            else (
                "File-tool paths can be relative to the current directory. "
                "Python file APIs still follow Monty's own path rules.\n"
                f"Workspace limits: {self.config.max_workspace_files} files and "
                f"{self.config.max_workspace_bytes} bytes.\n"
            )
        )
        return (
            self.kernel.instructions()
            + "\n\n## Managed sandbox\n"
            + f"Sandbox ID: {self.id}\n"
            + f"Workspace root: {self.root}\n"
            + f"Current directory: {self.cwd}\n"
            + workspace
            + f"Package binary assets under assets/: {self.config.allow_binary_assets}.\n"
            + "Python state and files have separate lifecycles unless the host uses "
            + "checkpoint and restore. Forked sessions share files.\n"
        )

    def policy_digest(self) -> str:
        """Return a stable digest of restore-sensitive sandbox settings.

        Returns:
            SHA-256 configuration digest.
        """

        capabilities = self.kernel.capabilities()
        mcp_policy = dict(capabilities["mcp"])
        mcp_policy.pop("servers", None)
        data = {
            "filesystem_available": self.config.filesystem != "none",
            "workspace_root": self.root,
            "network": self.config.network,
            "kernel": asdict(self.config.kernel),
            "filesystem_limits": asdict(self.config.filesystem_limits),
            "max_workspace_files": self.config.max_workspace_files,
            "max_workspace_bytes": self.config.max_workspace_bytes,
            "hosted_tools": capabilities["hosted_tools"],
            "mcp": mcp_policy,
            "bootstrap": capabilities["bootstrap"],
            "external_lookup_names": sorted(self.kernel.external_lookup),
        }
        if self.config.allow_binary_assets:
            data["allow_binary_assets"] = True
        encoded = json.dumps(_policy_value(data), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def langchain_tools(
        self,
        include: Iterable[SandboxFeature] | None = None,
    ) -> list[Any]:
        """Create fixed-session LangChain tools for this sandbox.

        Args:
            include: Feature groups to expose. By default, use all active groups.

        Returns:
            LangChain tools that use this managed handle.
        """

        from .langchain_managed_tools import create_managed_sandbox_tools

        selected = frozenset(
            include
            if include is not None
            else (
                ("python",)
                if self.config.filesystem == "none"
                else ("python", "files", "shell", "packages")
            )
        )
        unavailable = (
            selected - {"python"} if self.config.filesystem == "none" else set()
        )
        if unavailable:
            raise ValueError(
                f"tool groups require a configured filesystem: {sorted(unavailable)}"
            )
        return create_managed_sandbox_tools(self, selected)

    def agent_bundle(
        self,
        include: Iterable[SandboxFeature] | None = None,
    ) -> SandboxAgentBundle:
        """Build selected LangChain tools with matching model instructions.

        Args:
            include: Feature groups to expose. By default, use all active groups.

        Returns:
            Fixed-session tools and current prompt-ready instructions.
        """

        return SandboxAgentBundle(
            tuple(self.langchain_tools(include)),
            self.instructions(),
        )

    def _ensure_root(self) -> None:
        """Create the virtual root without running bootstrap or saving state."""

        if self.config.filesystem == "none":
            return
        self.kernel._run_isolated(
            f"from pathlib import Path\nPath({self.root!r}).mkdir(parents=True, exist_ok=True)"
        )

    def _capture_files(self, *, root: str | None = None) -> dict[str, bytes]:
        """Read a bounded virtual tree for quota and checkpoint operations.

        Args:
            root: Optional subtree root. Defaults to the workspace root.

        Returns:
            Mapping of relative POSIX paths to bytes.
        """

        source_root = root or self.root
        if self.config.filesystem == "none":
            return {}
        value = self.kernel._call_filesystem_raw(
            "ls",
            {
                "path": source_root,
                "recursive": True,
                "limit": self.config.filesystem_limits.max_scan_entries,
            },
        )
        if value["truncated"]:
            raise ValueError("workspace exceeds the filesystem scan limit")
        files: dict[str, bytes] = {}
        total = 0
        for entry in value["entries"]:
            if entry["type"] != "file":
                continue
            if len(files) >= self.config.max_workspace_files:
                raise ValueError("workspace exceeds max_workspace_files")
            path = entry["path"]
            content = self.kernel.download_file(path)
            total += len(content)
            if total > self.config.max_workspace_bytes:
                raise ValueError("workspace exceeds max_workspace_bytes")
            files[posixpath.relpath(path, source_root)] = content
        return files

    def _require_projected_quota(self, path: str, content: bytes) -> None:
        """Check workspace quotas for one direct file replacement.

        Args:
            path: Absolute target path.
            content: Replacement bytes.

        Raises:
            ValueError: If the projected workspace exceeds a hard quota.
        """

        files = self._capture_files()
        relative = posixpath.relpath(path, self.root)
        if relative not in files and len(files) >= self.config.max_workspace_files:
            raise ValueError("workspace exceeds max_workspace_files")
        projected = sum(len(value) for value in files.values())
        projected -= len(files.get(relative, b""))
        projected += len(content)
        if projected > self.config.max_workspace_bytes:
            raise ValueError("workspace exceeds max_workspace_bytes")

    def _require_tree_quota(self, replacements: Mapping[str, bytes]) -> None:
        """Check quotas after a set of relative file replacements.

        Args:
            replacements: Workspace-relative paths and replacement bytes.

        Raises:
            ValueError: If the projected workspace exceeds a hard quota.
        """

        files = self._capture_files()
        if any(
            len(content) > self.config.filesystem_limits.max_transfer_bytes
            for content in replacements.values()
        ):
            raise ValueError("one file exceeds max_transfer_bytes")
        files.update(replacements)
        if len(files) > self.config.max_workspace_files:
            raise ValueError("workspace exceeds max_workspace_files")
        if (
            sum(len(value) for value in files.values())
            > self.config.max_workspace_bytes
        ):
            raise ValueError("workspace exceeds max_workspace_bytes")

    def _guard_workspace(self, operation: Callable[[], SandboxResult]) -> SandboxResult:
        """Roll back file changes when one execution exceeds workspace quotas.

        Args:
            operation: Execution operation that can change virtual files.

        Returns:
            Original result, or a quota failure after file rollback.
        """

        if self.config.filesystem == "none":
            try:
                return operation()
            except _MANAGED_ERRORS as exc:
                return _failure(
                    "sandbox_operation_failed",
                    str(exc),
                    "Check the active sandbox capabilities and operation inputs.",
                )
        with self.kernel.filesystem_guard():
            try:
                before = self._capture_files()
            except _MANAGED_ERRORS as exc:
                return _failure(
                    "workspace_quota_exceeded",
                    str(exc),
                    "Reduce the existing workspace before execution.",
                )
            try:
                result = operation()
            except _MANAGED_ERRORS as exc:
                try:
                    self._replace_files(before)
                except _MANAGED_ERRORS as rollback:
                    return _failure(
                        "workspace_rollback_failed",
                        str(rollback),
                        "Discard this sandbox because its file state is uncertain.",
                    )
                return _failure(
                    "sandbox_operation_failed",
                    str(exc),
                    "Check the active sandbox capabilities and operation inputs.",
                )
            try:
                self._capture_files()
            except _MANAGED_ERRORS as exc:
                try:
                    self._replace_files(before)
                except _MANAGED_ERRORS as rollback:
                    return _failure(
                        "workspace_rollback_failed",
                        str(rollback),
                        "Discard this sandbox because its file state is uncertain.",
                    )
                return _failure(
                    "workspace_quota_exceeded",
                    str(exc),
                    "Reduce file output. File changes were rolled back; Python state may remain changed.",
                )
        return result

    def _packages(self) -> MontyPackageManager:
        """Create a package manager bounded by managed workspace limits.

        Returns:
            Package manager for this virtual workspace.
        """

        limits = self.config.filesystem_limits
        return MontyPackageManager(
            self.kernel,
            PackageConfig(
                allowed_root=self.root,
                max_files=min(100, self.config.max_workspace_files),
                max_file_bytes=min(256_000, limits.max_transfer_bytes),
                max_total_bytes=min(2_000_000, self.config.max_workspace_bytes),
                max_archive_bytes=min(2_500_000, self.config.max_workspace_bytes),
                allow_binary_assets=self.config.allow_binary_assets,
            ),
        )

    def _replace_files(self, files: Mapping[str, bytes]) -> None:
        """Replace workspace files while the filesystem lock is held.

        Args:
            files: Safe relative paths and replacement bytes.
        """

        if self.config.filesystem == "none":
            if files:
                raise ValueError(
                    "checkpoint contains files but no filesystem is configured"
                )
            return
        listed = self.kernel.call_filesystem(
            self.session_id,
            "ls",
            {"path": self.root, "recursive": False, "limit": None},
        )
        if not listed["ok"]:
            error = _error_info(listed.get("error"), "workspace_replace_failed")
            raise ValueError(error.message)
        for entry in listed["result"]["entries"]:
            deleted = self.kernel.call_filesystem(
                self.session_id,
                "delete",
                {"path": entry["path"], "recursive": True},
            )
            if not deleted["ok"]:
                error = _error_info(deleted.get("error"), "workspace_replace_failed")
                raise ValueError(error.message)
        for relative, content in files.items():
            self.kernel.upload_file(f"{self.root}/{relative}", content)


class AsyncManagedSandbox:
    """Expose all managed sandbox operations through non-blocking thread calls."""

    def __init__(self, sandbox: ManagedSandbox) -> None:
        """Create an async wrapper.

        Args:
            sandbox: Owned or borrowed synchronous managed sandbox.
        """

        self._sandbox = sandbox

    @property
    def id(self) -> str:
        """Return the stable sandbox ID.

        Returns:
            Sandbox ID.
        """

        return self._sandbox.id

    @property
    def cwd(self) -> str:
        """Return the current virtual directory.

        Returns:
            Absolute virtual working directory.
        """

        return self._sandbox.cwd

    async def __aenter__(self) -> Self:
        """Return this async handle.

        Returns:
            This handle.
        """

        return self

    async def __aexit__(self, *_: object) -> None:
        """Close owned resources without blocking the event loop."""

        await self.close()

    async def close(self) -> None:
        """Close the wrapped sandbox in a worker thread."""

        await asyncio.to_thread(self._sandbox.close)

    async def run_python(self, code: str, **kwargs: Any) -> SandboxResult:
        """Run Monty Python without blocking the event loop.

        Args:
            code: Monty-compatible Python source.
            **kwargs: Managed ``run_python`` options.

        Returns:
            Common managed sandbox result.
        """

        return await asyncio.to_thread(self._sandbox.run_python, code, **kwargs)

    async def run_shell(self, command: str, **kwargs: Any) -> SandboxResult:
        """Run the virtual shell without blocking the event loop.

        Args:
            command: Virtual-shell command text.
            **kwargs: Managed ``run_shell`` options.

        Returns:
            Common managed sandbox result.
        """

        return await asyncio.to_thread(self._sandbox.run_shell, command, **kwargs)

    async def run_cell(self, code: str, **kwargs: Any) -> SandboxResult:
        """Run one notebook-style cell without blocking the event loop.

        Args:
            code: Python or virtual-shell cell.
            **kwargs: Managed ``run_cell`` options.

        Returns:
            Common managed sandbox result.
        """

        return await asyncio.to_thread(self._sandbox.run_cell, code, **kwargs)

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call another managed method without blocking the event loop.

        Args:
            method: Public wrapped method name.
            *args: Positional method arguments.
            **kwargs: Keyword method arguments.

        Returns:
            Wrapped method result.
        """

        if method.startswith("_"):
            raise ValueError("private managed methods cannot be called")
        function = getattr(self._sandbox, method)
        if not callable(function):
            raise TypeError(f"managed attribute is not callable: {method}")
        return await asyncio.to_thread(function, *args, **kwargs)

    def instructions(self) -> str:
        """Return current model instructions.

        Returns:
            Prompt-ready managed instructions.
        """

        return self._sandbox.instructions()

    def langchain_tools(
        self,
        include: Iterable[SandboxFeature] | None = None,
    ) -> list[Any]:
        """Return the fixed-session managed LangChain tools.

        Args:
            include: Feature groups to expose. By default, use all active groups.

        Returns:
            Tools backed by the thread-safe synchronous worker pool.
        """

        return self._sandbox.langchain_tools(include)

    def agent_bundle(
        self,
        include: Iterable[SandboxFeature] | None = None,
    ) -> SandboxAgentBundle:
        """Return selected tools with current managed instructions.

        Args:
            include: Feature groups to expose. By default, use all active groups.

        Returns:
            Fixed-session tools and prompt-ready instructions.
        """

        return self._sandbox.agent_bundle(include)


class SandboxManager:
    """Own one runtime and open bounded sandbox handles over it."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        external_lookup: Mapping[str, Callable[..., Any]] | None = None,
        hosted_tools: HostedToolSet | None = None,
        mcp_hub: Any = None,
        event_sink: Any = None,
    ) -> None:
        """Create one shared managed runtime.

        Args:
            config: Optional high-level preset or custom configuration.
            external_lookup: Approved host values and synchronous functions.
            hosted_tools: Optional schema-aware host functions.
            mcp_hub: Optional caller-configured MCP hub.
            event_sink: Optional event callback.
        """

        self.config = config or SandboxConfig.conversation()
        self.runtime = _build_runtime(
            self.config,
            external_lookup=external_lookup,
            hosted_tools=hosted_tools,
            mcp_hub=mcp_hub,
            event_sink=event_sink,
        )
        self._handles: dict[str, ManagedSandbox] = {}
        self._closed = False

    def __enter__(self) -> Self:
        """Return this open manager.

        Returns:
            This manager.
        """

        return self

    def __exit__(self, *_: object) -> None:
        """Close all managed resources."""

        self.close()

    def open(self, sandbox_id: str) -> ManagedSandbox:
        """Open or return one fixed-ID sandbox handle.

        Args:
            sandbox_id: Stable session identifier.

        Returns:
            Borrowed managed sandbox handle.
        """

        if self._closed:
            raise RuntimeError("sandbox manager is closed")
        _validate_id(sandbox_id)
        existing = self._handles.get(sandbox_id)
        if existing is not None:
            return existing
        if len(self._handles) >= self.config.max_active_sandboxes:
            raise RuntimeError("max_active_sandboxes reached")
        handle = ManagedSandbox(
            self.runtime,
            sandbox_id,
            self.config,
            owns_runtime=False,
        )
        self._handles[sandbox_id] = handle
        return handle

    def list(self) -> builtins.list[str]:
        """List currently opened sandbox IDs.

        Returns:
            Sorted active handle IDs.
        """

        return sorted(self._handles)

    def delete(self, sandbox_id: str) -> bool:
        """Delete one sandbox's Python state and active handle.

        Args:
            sandbox_id: Sandbox to delete.

        Returns:
            Whether saved Python state existed.
        """

        self._handles.pop(sandbox_id, None)
        return self.runtime.kernel.reset(sandbox_id)

    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Delete expired Python sessions from the configured store.

        Args:
            max_age_seconds: Maximum session age.

        Returns:
            Deleted stored-session count.
        """

        before = {item.session_id for item in self.runtime.kernel.store.list_sessions()}
        deleted = self.runtime.kernel.store.delete_expired(max_age_seconds)
        after = {item.session_id for item in self.runtime.kernel.store.list_sessions()}
        for sandbox_id in before - after:
            self._handles.pop(sandbox_id, None)
        return deleted

    def close(self) -> None:
        """Close the shared runtime once."""

        if self._closed:
            return
        self._closed = True
        self._handles.clear()
        self.runtime.close()


class AsyncSandboxManager:
    """Provide asynchronous access to a shared synchronous SandboxManager."""

    def __init__(self, manager: SandboxManager) -> None:
        """Create an asynchronous manager wrapper.

        Args:
            manager: Owned synchronous manager.
        """

        self._manager = manager

    async def __aenter__(self) -> Self:
        """Return this open manager.

        Returns:
            This manager.
        """

        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the manager without blocking the event loop."""

        await self.close()

    async def open(self, sandbox_id: str) -> AsyncManagedSandbox:
        """Open one async managed sandbox.

        Args:
            sandbox_id: Stable session identifier.

        Returns:
            Async wrapper around a shared managed handle.
        """

        handle = await asyncio.to_thread(self._manager.open, sandbox_id)
        return AsyncManagedSandbox(handle)

    async def delete(self, sandbox_id: str) -> bool:
        """Delete one sandbox's Python state.

        Args:
            sandbox_id: Sandbox to delete.

        Returns:
            Whether saved state existed.
        """

        return await asyncio.to_thread(self._manager.delete, sandbox_id)

    async def list(self) -> builtins.list[str]:
        """List opened sandbox IDs without blocking the event loop.

        Returns:
            Sorted opened sandbox IDs.
        """

        return await asyncio.to_thread(self._manager.list)

    async def cleanup_expired(self, max_age_seconds: float) -> int:
        """Delete expired sessions without blocking the event loop.

        Args:
            max_age_seconds: Maximum session age.

        Returns:
            Deleted session count.
        """

        return await asyncio.to_thread(self._manager.cleanup_expired, max_age_seconds)

    async def close(self) -> None:
        """Close the shared manager without blocking the event loop."""

        await asyncio.to_thread(self._manager.close)


def open_sandbox(
    sandbox_id: str = "sandbox",
    *,
    config: SandboxConfig | None = None,
    external_lookup: Mapping[str, Callable[..., Any]] | None = None,
    hosted_tools: HostedToolSet | None = None,
    mcp_hub: Any = None,
    event_sink: Any = None,
) -> ManagedSandbox:
    """Create one ready-to-use sandbox that owns its complete runtime.

    Args:
        sandbox_id: Stable session and public sandbox ID.
        config: Optional high-level preset or custom configuration.
        external_lookup: Approved host values and synchronous functions.
        hosted_tools: Optional schema-aware host functions.
        mcp_hub: Optional caller-configured MCP hub.
        event_sink: Optional event callback.

    Returns:
        Owned managed sandbox handle.
    """

    active = config or SandboxConfig.conversation()
    runtime = _build_runtime(
        active,
        external_lookup=external_lookup,
        hosted_tools=hosted_tools,
        mcp_hub=mcp_hub,
        event_sink=event_sink,
    )
    try:
        return ManagedSandbox(runtime, sandbox_id, active, owns_runtime=True)
    except BaseException:
        runtime.close()
        raise


def open_async_sandbox(
    sandbox_id: str = "sandbox",
    *,
    config: SandboxConfig | None = None,
    **kwargs: Any,
) -> AsyncManagedSandbox:
    """Create a full-feature async wrapper around an owned sandbox.

    Args:
        sandbox_id: Stable session and public sandbox ID.
        config: Optional high-level preset. Use ``batch`` for several agents.
        **kwargs: Host functions, hosted tools, MCP hub, or event sink.

    Returns:
        Async managed sandbox backed by the subprocess Monty pool.
    """

    return AsyncManagedSandbox(open_sandbox(sandbox_id, config=config, **kwargs))


def open_sandbox_manager(
    config: SandboxConfig | None = None,
    **kwargs: Any,
) -> SandboxManager:
    """Create one manager over a shared subprocess worker pool.

    Args:
        config: Optional high-level sandbox configuration.
        **kwargs: Host functions, hosted tools, MCP hub, or event sink.

    Returns:
        Owned synchronous sandbox manager.
    """

    return SandboxManager(config, **kwargs)


def open_async_sandbox_manager(
    config: SandboxConfig | None = None,
    **kwargs: Any,
) -> AsyncSandboxManager:
    """Create an async manager over one shared subprocess worker pool.

    Args:
        config: Optional high-level preset. Use ``batch`` for concurrency.
        **kwargs: Host functions, hosted tools, MCP hub, or event sink.

    Returns:
        Owned asynchronous manager wrapper.
    """

    return AsyncSandboxManager(SandboxManager(config, **kwargs))


def _build_runtime(
    config: SandboxConfig,
    **kernel_options: Any,
) -> AreneroRuntime:
    """Build one existing runtime from a high-level configuration.

    Args:
        config: Validated managed sandbox configuration.
        **kernel_options: Host boundary values passed to the kernel.

    Returns:
        Runtime that owns the selected store, files, and kernel.
    """

    directory = Path(config.directory) if config.directory is not None else None
    if config.persistence == "stateless":
        store = StatelessSessionStore()
    elif config.persistence == "durable":
        assert directory is not None
        store = SQLiteSessionStore(directory / "sessions.db")
    else:
        store = InMemorySessionStore()

    os_handler = OSAccess() if config.filesystem == "memory" else None
    workspace = None
    if config.filesystem == "workspace":
        assert directory is not None
        workspace = WorkspaceConfig(
            directory / "workspace",
            virtual_path=config.workspace_root,
            write_bytes_limit=config.filesystem_limits.max_transfer_bytes,
            memory_usage_limit=config.max_workspace_bytes,
        )
    network = config.network
    network_config = (
        network
        if isinstance(network, NetworkConfig)
        else NetworkConfig(enabled=network == "public")
    )
    kernel = PersistentMontyKernel(
        store,
        config.kernel,
        os_handler=os_handler,
        filesystem_config=config.filesystem_limits,
        network_config=network_config,
        workspace=workspace,
        bootstrap_code=config.bootstrap_code,
        bootstrap_files=config.bootstrap_files,
        mcp_artifact_directory=config.mcp_artifact_directory,
        **kernel_options,
    )
    if config.filesystem != "none":
        kernel.default_working_directory = config.workspace_root
    return AreneroRuntime(kernel)


def _from_execution(result: Mapping[str, Any]) -> SandboxResult:
    """Convert one kernel result to the managed result contract.

    Args:
        result: Structured kernel execution result.

    Returns:
        Common managed sandbox result.
    """

    error = result.get("error")
    artifacts = tuple(
        item.get("path")
        for item in result.get("mcp_artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    )
    output_file = result.get("output_file")
    if isinstance(output_file, str):
        artifacts = (*artifacts, output_file)
    truncated_value = result.get("truncated", False)
    truncated = (
        any(bool(value) for value in truncated_value.values())
        if isinstance(truncated_value, dict)
        else bool(truncated_value)
    )
    if result.get("ok"):
        return SandboxResult(
            True,
            result.get("result"),
            str(result.get("stdout", "")),
            str(result.get("stderr", "")),
            exit_code=0,
            truncated=truncated,
            artifacts=artifacts,
            stats=result.get("stats") or {},
        )
    return SandboxResult(
        False,
        result.get("result"),
        str(result.get("stdout", "")),
        str(result.get("stderr", "")),
        exit_code=1,
        error=_error_info(error, "python_execution_failed"),
        truncated=truncated,
        artifacts=artifacts,
        stats=result.get("stats") or {},
    )


def _from_error_mapping(error: Any, fallback: str) -> SandboxResult:
    """Create a failed result from an existing error mapping.

    Args:
        error: Existing error mapping or text.
        fallback: Code used when the mapping has no code.

    Returns:
        Failed managed sandbox result.
    """

    return SandboxResult(False, error=_error_info(error, fallback), exit_code=1)


def _from_package(result: Mapping[str, Any]) -> SandboxResult:
    """Convert one package workflow result to the common contract.

    Args:
        result: Package manager result.

    Returns:
        Common managed sandbox result.
    """

    if result.get("ok"):
        archive = result.get("archive_path")
        return SandboxResult(
            True,
            dict(result),
            artifacts=(archive,) if isinstance(archive, str) else (),
        )
    return _from_error_mapping(result.get("error"), "package_operation_failed")


def _error_info(error: Any, fallback: str) -> SandboxError:
    """Normalize an existing error value.

    Args:
        error: Mapping, string, or absent error.
        fallback: Default stable code.

    Returns:
        Structured managed error.
    """

    if isinstance(error, Mapping):
        return SandboxError(
            str(error.get("code") or fallback),
            str(error.get("message") or error.get("type") or "operation failed"),
            str(
                error.get("hint")
                or "Inspect the operation inputs and active capabilities."
            ),
            bool(error.get("retryable", False)),
        )
    return SandboxError(
        fallback,
        str(error or "operation failed"),
        "Inspect the operation inputs and active capabilities.",
    )


def _failure(code: str, message: str, hint: str) -> SandboxResult:
    """Create one simple failed managed result.

    Args:
        code: Stable machine-readable code.
        message: Human-readable failure text.
        hint: Suggested next action.

    Returns:
        Failed managed sandbox result.
    """

    return SandboxResult(False, exit_code=1, error=SandboxError(code, message, hint))


def _revision(content: bytes) -> str:
    """Hash one file value.

    Args:
        content: File bytes.

    Returns:
        SHA-256 hexadecimal revision.
    """

    return hashlib.sha256(content).hexdigest()


def _validate_id(sandbox_id: str) -> None:
    """Validate one public sandbox and session identifier.

    Args:
        sandbox_id: Candidate identifier.

    Raises:
        ValueError: If the identifier is not a simple stable value.
    """

    if not isinstance(sandbox_id, str) or _SAFE_ID.fullmatch(sandbox_id) is None:
        raise ValueError(
            "sandbox_id must contain 1 to 128 letters, digits, dots, underscores, or hyphens"
        )


def _policy_value(value: Any) -> Any:
    """Normalize policy data for one stable digest.

    Args:
        value: Configuration value to normalize.

    Returns:
        Stable JSON-compatible policy data.
    """

    if hasattr(value, "__dataclass_fields__"):
        return _policy_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _policy_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(_policy_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_policy_value(item) for item in value]
    if callable(value):
        return {"configured_callable": True}
    if isinstance(value, bytes):
        return _json_value(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _zip_files(files: Mapping[str, bytes]) -> bytes:
    """Create deterministic ZIP bytes from safe relative files.

    Args:
        files: Relative paths and file bytes.

    Returns:
        Deterministic ZIP bytes.
    """

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as output:
        for path, content in sorted(files.items()):
            _write_zip_entry(output, path, content)
    return stream.getvalue()


def _write_zip_entry(output: zipfile.ZipFile, path: str, content: bytes) -> None:
    """Write one deterministic regular ZIP entry.

    Args:
        output: Open destination ZIP.
        path: Safe relative archive path.
        content: Entry bytes.
    """

    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    output.writestr(info, content)


def _read_zip_files(archive: bytes, max_files: int, max_bytes: int) -> dict[str, bytes]:
    """Read bounded regular files from a ZIP.

    Args:
        archive: Source ZIP bytes.
        max_files: Maximum extracted files.
        max_bytes: Maximum total uncompressed bytes.

    Returns:
        Safe relative paths and content.
    """

    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        files: dict[str, bytes] = {}
        total = 0
        for info in source.infolist():
            if info.is_dir():
                continue
            path = _safe_zip_path(info)
            relative = path.as_posix()
            if relative in files:
                raise ValueError("ZIP contains a duplicate path")
            if len(files) >= max_files:
                raise ValueError("ZIP exceeds max_workspace_files")
            total += info.file_size
            if total > max_bytes:
                raise ValueError("ZIP exceeds max_workspace_bytes")
            files[relative] = source.read(info)
    return files


def _read_checkpoint_files(
    source: zipfile.ZipFile,
    max_files: int,
    max_bytes: int,
) -> dict[str, bytes]:
    """Read the bounded file subtree from a checkpoint ZIP.

    Args:
        source: Open checkpoint ZIP.
        max_files: Maximum workspace files.
        max_bytes: Maximum total workspace bytes.

    Returns:
        Safe relative workspace paths and bytes.
    """

    files: dict[str, bytes] = {}
    total = 0
    for info in source.infolist():
        if not info.filename.startswith("files/") or info.is_dir():
            continue
        path = _safe_zip_path(
            info,
            name=info.filename.removeprefix("files/"),
        )
        relative = path.as_posix()
        if relative in files:
            raise ValueError("checkpoint contains a duplicate file path")
        if len(files) >= max_files:
            raise ValueError("checkpoint exceeds max_workspace_files")
        total += info.file_size
        if total > max_bytes:
            raise ValueError("checkpoint exceeds max_workspace_bytes")
        files[relative] = source.read(info)
    return files


def _safe_zip_path(
    info: zipfile.ZipInfo,
    *,
    name: str | None = None,
) -> PurePosixPath:
    """Validate one regular relative ZIP entry path.

    Args:
        info: ZIP entry metadata.
        name: Optional path after a checkpoint prefix is removed.

    Returns:
        Safe relative POSIX path.

    Raises:
        ValueError: If the entry is unsafe or not a regular file.
    """

    path = PurePosixPath(name if name is not None else info.filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("ZIP contains an unsafe path")
    kind = (info.external_attr >> 16) & 0o170000
    if kind not in {0, stat.S_IFREG}:
        raise ValueError("ZIP contains a non-regular file")
    return path
