from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, get_args

from pydantic_monty import (
    CollectStreams,
    Monty,
    MontyCrashedError,
    MontyDisconnectError,
    MontyError,
    MontySession,
    MontyShutdown,
    MountDir,
    OsHandler,
    OSPolicy,
    ResourceLimits,
    TypeCheckFormat,
)

from .events import EventSink, emit_event
from .filesystem import (
    FILESYSTEM_TYPE_CHECK_STUBS,
    FilesystemConfig,
    FilesystemOperation,
    FilesystemValue,
    build_filesystem_call,
    build_filesystem_prelude,
)
from .hosted_tools import HostedToolScope, HostedToolSet
from .mcp_artifacts import materialize_mcp_artifacts
from .modules import VirtualModuleConfig, VirtualModuleResolver
from .network import NetworkConfig
from .store import (
    ConcurrentSessionUpdate,
    InMemorySessionStore,
    SessionInfo,
    SessionStore,
)
from .values import json_value as _json_value
from .workspace import WorkspaceConfig

if TYPE_CHECKING:
    from .mcp import McpHub
    from .session import SessionHandle

StatePolicy = Literal["always", "success", "never"]
RecoveryAction = Literal[
    "none", "retry_same_call", "revise_code", "restart_session", "retry_later", "stop"
]


class ExecutionError(TypedDict):
    """Describe an execution error returned to a caller."""

    type: str
    code: str
    message: str
    hint: str
    retryable: bool
    recovery: RecoveryAction


class ExecutionStatistics(TypedDict):
    """Describe timing, state, and original output sizes."""

    execution_id: str
    wall_time_seconds: float
    checkout_wait_seconds: float
    state_before_bytes: int
    state_after_bytes: int
    original_output_chars: dict[str, int]


class TruncationInfo(TypedDict):
    """Identify result fields that were shortened."""

    result: bool
    stdout: bool
    stderr: bool
    error: bool


class ExecutionResult(TypedDict):
    """Describe the complete result of one kernel operation."""

    ok: bool
    result: Any
    stdout: str
    stderr: str
    error: ExecutionError | None
    truncated: TruncationInfo
    output_limit_chars: int
    limit_was_capped: bool
    state_saved: bool
    state_restored: bool
    session_revision: int | None
    retryable: bool
    output_file: str | None
    output_file_error: str | None
    state_policy: StatePolicy
    session_restarted: bool
    stats: ExecutionStatistics
    mcp_artifacts: list[str]
    host_calls: dict[str, Any]


@dataclass
class _ExecutionMetrics:
    """Collect mutable metrics while one kernel operation runs.

    Attributes:
        execution_id: Unique host-generated operation ID.
        started_at: Monotonic operation start time.
        checkout_wait_seconds: Time spent waiting for a Monty worker.
        state_before_bytes: Loaded snapshot size.
        state_after_bytes: New snapshot size, or zero when not dumped.
        session_restarted: Whether saved state was reset for this operation.
    """

    execution_id: str
    started_at: float
    checkout_wait_seconds: float = 0.0
    state_before_bytes: int = 0
    state_after_bytes: int = 0
    session_restarted: bool = False
    hosted_scope: HostedToolScope | None = None


@dataclass(frozen=True)
class KernelConfig:
    """Set execution and result limits for a kernel.

    Attributes:
        default_output_limit_chars: Output size used when no lower limit is given.
        max_output_limit_chars: Hard maximum for each output field.
        max_shell_command_chars: Maximum virtual shell command length.
        max_shell_commands: Maximum commands in one virtual shell command list.
        max_shell_expanded_args: Maximum arguments after wildcard expansion.
        max_shell_variables: Maximum persistent virtual shell variables.
        max_shell_script_depth: Maximum nested ``source`` script depth.
        allowed_shell_commands: Enabled virtual command names, or all commands.
        max_stream_bytes: Maximum collected standard stream bytes.
        max_state_bytes: Maximum saved Monty session size.
        max_feed_duration_seconds: Interpreter time limit for each code feed.
        max_turn_duration_seconds: Interpreter time between host answers, or None.
        max_suspensions: Maximum host/OS suspensions in one worker checkout.
        max_total_sleep_seconds: Maximum total system sleep in one checkout.
        os_policy: Monty clock, sleep, and initial random seed policy, or defaults.
        max_memory_bytes: Maximum tracked Monty allocations.
        max_recursion_depth: Maximum Monty call depth.
        request_timeout_seconds: Worker request wall-time limit.
        min_worker_processes: Minimum warm Monty worker count.
        max_worker_processes: Maximum Monty worker count, or ``None`` for the
            Monty default.
        worker_checkout_timeout_seconds: Maximum wait for a free worker, or
            ``None`` for no checkout deadline.
        max_checkouts_per_worker: Maximum uses before Monty replaces a worker,
            or ``None`` for the Monty default.
        session_ttl_seconds: Maximum saved-session age, or ``None`` for no expiry.
        type_check: Whether Monty type-checks each user feed before execution.
        type_check_stubs: Optional declarations for approved host functions.
        type_check_format: Monty type-check diagnostic format.
    """

    default_output_limit_chars: int = 8_000
    max_output_limit_chars: int = 50_000
    max_shell_command_chars: int = 20_000
    max_shell_commands: int = 8
    max_shell_expanded_args: int = 1_000
    max_shell_variables: int = 100
    max_shell_script_depth: int = 4
    allowed_shell_commands: frozenset[str] | None = None
    max_stream_bytes: int = 1_000_000
    max_state_bytes: int = 5_000_000
    max_feed_duration_seconds: float = 5.0
    max_turn_duration_seconds: float | None = None
    max_suspensions: int = 1_000
    max_total_sleep_seconds: float = 10.0
    os_policy: OSPolicy | None = None
    max_memory_bytes: int = 64 * 1024 * 1024
    max_recursion_depth: int = 100
    request_timeout_seconds: float = 10.0
    min_worker_processes: int = 1
    max_worker_processes: int | None = None
    worker_checkout_timeout_seconds: float | None = None
    max_checkouts_per_worker: int | None = None
    session_ttl_seconds: float | None = 24 * 60 * 60
    type_check: bool = False
    type_check_stubs: str | None = None
    type_check_format: TypeCheckFormat = "concise"

    def __post_init__(self) -> None:
        """Validate related output and byte limit values."""

        if self.default_output_limit_chars < 0:
            raise ValueError("default_output_limit_chars must be at least 0")
        if self.max_output_limit_chars < self.default_output_limit_chars:
            raise ValueError("max_output_limit_chars must not be less than the default")
        shell_limits = (
            self.max_shell_command_chars,
            self.max_shell_commands,
            self.max_shell_expanded_args,
            self.max_shell_variables,
            self.max_shell_script_depth,
        )
        if any(value <= 0 for value in shell_limits):
            raise ValueError("shell limits must be greater than 0")
        if self.allowed_shell_commands is not None and any(
            not name or name.strip() != name for name in self.allowed_shell_commands
        ):
            raise ValueError("allowed shell command names must not be empty or padded")
        if self.max_stream_bytes <= 0 or self.max_state_bytes <= 0:
            raise ValueError("byte limits must be greater than 0")
        if self.max_feed_duration_seconds <= 0 or (
            self.max_turn_duration_seconds is not None
            and self.max_turn_duration_seconds <= 0
        ):
            raise ValueError("interpreter duration limits must be greater than 0")
        if self.max_suspensions <= 0 or self.max_total_sleep_seconds < 0:
            raise ValueError(
                "suspensions must be positive and sleep must be nonnegative"
            )
        if self.min_worker_processes <= 0:
            raise ValueError("min_worker_processes must be greater than 0")
        if (
            self.max_worker_processes is not None
            and self.max_worker_processes < self.min_worker_processes
        ):
            raise ValueError(
                "max_worker_processes must not be less than min_worker_processes"
            )
        if (
            self.worker_checkout_timeout_seconds is not None
            and self.worker_checkout_timeout_seconds <= 0
        ):
            raise ValueError("worker_checkout_timeout_seconds must be greater than 0")
        if (
            self.max_checkouts_per_worker is not None
            and self.max_checkouts_per_worker <= 0
        ):
            raise ValueError("max_checkouts_per_worker must be greater than 0")

    def resource_limits(self) -> ResourceLimits:
        """Build the shared Monty limits for sync and async checkouts.

        Returns:
            Per-feed and per-turn time, memory, recursion, suspension, and sleep limits.
        """
        limits: ResourceLimits = {
            "max_feed_duration_secs": self.max_feed_duration_seconds,
            "max_memory": self.max_memory_bytes,
            "max_recursion_depth": self.max_recursion_depth,
            "max_suspensions": self.max_suspensions,
            "max_total_sleep_secs": self.max_total_sleep_seconds,
        }
        if self.max_turn_duration_seconds is not None:
            limits["max_turn_duration_secs"] = self.max_turn_duration_seconds
        return limits


InfrastructureErrors = (
    MontyCrashedError,
    MontyDisconnectError,
    MontyShutdown,
)


class PersistentMontyKernel:
    """Run code and virtual Python files in isolated Monty sessions."""

    def __init__(
        self,
        store: SessionStore | None = None,
        config: KernelConfig | None = None,
        external_lookup: Mapping[str, Callable[..., Any]] | None = None,
        os_handler: OsHandler | None = None,
        mounts: MountDir | list[MountDir] | None = None,
        filesystem_config: FilesystemConfig | None = None,
        network_config: NetworkConfig | None = None,
        workspace: WorkspaceConfig | None = None,
        mcp_hub: McpHub | None = None,
        hosted_tools: HostedToolSet | None = None,
        bootstrap_code: str | None = None,
        bootstrap_files: tuple[str, ...] = (),
        event_sink: EventSink | None = None,
        mcp_artifact_directory: str | None = None,
    ) -> None:
        """Create a persistent Monty kernel.

        Args:
            store: Backend that stores Python session dumps.
            config: Resource and output limits. Defaults to ``KernelConfig``.
            external_lookup: Approved host values and synchronous functions.
            os_handler: Optional virtual filesystem handler.
            mounts: Optional host directory mount or list of mounts.
            filesystem_config: Limits for the sandbox filesystem helpers.
            network_config: Optional shell network policy. The default allows
                bounded requests to public HTTP and HTTPS URLs.
            workspace: Optional persistent directory mounted at a virtual path.
            mcp_hub: Optional configured MCP client hub. The caller owns its
                lifecycle and must close it after the kernel.
            hosted_tools: Optional schema-aware host functions for Monty code.
            bootstrap_code: Trusted code run once when a new session starts.
            bootstrap_files: Virtual Python files run once for each new session.
            event_sink: Optional observer for safe execution and session metadata.
            mcp_artifact_directory: Optional virtual directory for binary MCP
                content returned as the final Python value.
        """

        self.store = store if store is not None else InMemorySessionStore()
        self.config = config or KernelConfig()
        self.external_lookup = dict(external_lookup or {})
        self.os_handler = os_handler
        self.filesystem_config = filesystem_config or FilesystemConfig()
        self.network_config = network_config or NetworkConfig()
        self.workspace = workspace
        self.mcp_hub = mcp_hub
        self.hosted_tools = hosted_tools or HostedToolSet()
        self.bootstrap_code = bootstrap_code
        self.bootstrap_files = tuple(bootstrap_files)
        self.event_sink = event_sink
        self.mcp_artifact_directory = mcp_artifact_directory
        if any(not path.startswith("/") for path in self.bootstrap_files):
            raise ValueError("bootstrap file paths must be absolute virtual paths")
        if mcp_artifact_directory is not None and not mcp_artifact_directory.startswith(
            "/"
        ):
            raise ValueError("mcp_artifact_directory must be an absolute virtual path")
        if mcp_hub is not None:
            collisions = self.external_lookup.keys() & mcp_hub.host_functions().keys()
            if collisions:
                names = ", ".join(sorted(collisions))
                raise ValueError(f"external_lookup uses reserved MCP names: {names}")
        hosted_names = {tool.alias for tool in self.hosted_tools.tools}
        collisions = self.external_lookup.keys() & hosted_names
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"external_lookup uses hosted tool names: {names}")
        self._owned_mount: MountDir | None = None
        configured_mounts = _mount_list(mounts)
        if workspace is not None:
            self._owned_mount = workspace.create_mount()
            configured_mounts.append(self._owned_mount.__enter__())
        self.mounts = configured_mounts or None
        self.default_working_directory = (
            workspace.virtual_path if workspace is not None else "/"
        )
        self._filesystem_prelude = build_filesystem_prelude(self.filesystem_config)
        self._filesystem_version = hashlib.sha256(
            self._filesystem_prelude.encode()
        ).hexdigest()
        host_stubs = self.config.type_check_stubs
        if mcp_hub is not None:
            host_stubs = "\n".join(
                value for value in (host_stubs, mcp_hub.type_check_stubs) if value
            )
        self._base_type_check_stubs = host_stubs
        self._type_check_stubs = self._operation_type_check_stubs(self.hosted_tools)
        self.monty_version = importlib.metadata.version("pydantic-monty")
        try:
            self._pool_context = Monty(
                min_processes=self.config.min_worker_processes,
                max_processes=self.config.max_worker_processes,
                checkout_timeout=self.config.worker_checkout_timeout_seconds,
                request_timeout=self.config.request_timeout_seconds,
                max_checkouts_per_worker=self.config.max_checkouts_per_worker,
            )
            self._pool = self._pool_context.__enter__()
        except BaseException:
            if self._owned_mount is not None:
                self._owned_mount.__exit__(None, None, None)
            raise
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._filesystem_lock = threading.RLock()
        self._filesystem_enabled = self.os_handler is not None or bool(self.mounts)
        self._closed = False

    def __enter__(self) -> Self:
        """Enter the kernel context.

        Returns:
            This kernel.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the kernel when its context ends.

        Args:
            *exc_info: Optional exception details supplied by Python.
        """
        self.close()

    def close(self) -> None:
        """Close the Monty worker pool once."""
        if not self._closed:
            try:
                self._pool_context.__exit__(None, None, None)
            finally:
                if self._owned_mount is not None:
                    self._owned_mount.__exit__(None, None, None)
                self._closed = True

    def reset(self, session_id: str) -> bool:
        """Delete the saved Python state for one session.

        Args:
            session_id: Session key to reset.

        Returns:
            ``True`` when saved state existed.
        """
        with self._lock_for(session_id), self.filesystem_guard():
            deleted = self.store.delete(session_id)
        emit_event(
            self.event_sink,
            "session",
            "reset",
            session_id=session_id,
            existed=deleted,
        )
        return deleted

    def session(self, session_id: str) -> SessionHandle:
        """Bind this kernel to one session ID.

        Args:
            session_id: Session key with 1 to 255 characters.

        Returns:
            A bound ``SessionHandle``.
        """
        from .session import SessionHandle

        return SessionHandle(self, session_id)

    def capabilities(self) -> dict[str, Any]:
        """Return active limits and integration capabilities.

        Returns:
            JSON-compatible capability data without secrets.
        """
        from .capabilities import capability_report

        return capability_report(self)

    def instructions(self) -> str:
        """Generate model instructions from the active configuration.

        Returns:
            Plain model instructions with the capability report.
        """
        from .capabilities import model_instructions

        return model_instructions(self)

    def fork_session(self, source_id: str, target_id: str) -> SessionInfo:
        """Copy one saved Python snapshot to a new session.

        Args:
            source_id: Existing session to copy.
            target_id: New session that receives the snapshot.

        Returns:
            Metadata for the new session.

        Raises:
            ValueError: If IDs are invalid, equal, missing, or already used.
            RuntimeError: If the configured store does not save state.
        """
        _validate_session_id(source_id)
        _validate_session_id(target_id)
        if source_id == target_id:
            raise ValueError("source and target session IDs must differ")
        if not self.store.stores_state:
            raise RuntimeError("the configured store does not save session state")
        with ExitStack() as stack:
            for session_id in sorted((source_id, target_id)):
                stack.enter_context(self._lock_for(session_id))
            source = self.store.load(
                source_id,
                max_age_seconds=self.config.session_ttl_seconds,
            )
            if source is None:
                raise ValueError(f"source session does not exist: {source_id}")
            if source.monty_version != self.monty_version:
                raise ValueError("source session uses a different Monty version")
            if self.store.load(target_id) is not None:
                raise ValueError(f"target session already exists: {target_id}")
            revision = self.store.save(
                target_id,
                source.state,
                source.monty_version,
                expected_revision=None,
            )
        assert revision is not None
        info = self.get_session_info(target_id)
        assert info is not None
        emit_event(
            self.event_sink,
            "session",
            "forked",
            source_session_id=source_id,
            target_session_id=target_id,
            state_size_bytes=len(source.state),
        )
        return info

    def materialize_mcp_artifacts(
        self,
        value: Any,
        directory: str = "/mcp-artifacts",
    ) -> tuple[Any, list[str]]:
        """Write binary MCP result blocks to the virtual filesystem.

        Args:
            value: Normalized MCP result or a value that contains one.
            directory: Absolute virtual destination directory.

        Returns:
            Copied value with virtual paths and the written path list.
        """
        materialized, paths = materialize_mcp_artifacts(
            value,
            self.upload_file,
            directory,
        )
        for path in paths:
            emit_event(self.event_sink, "filesystem", "artifact_written", path=path)
        return materialized, paths

    def delete_expired(self) -> int:
        """Delete sessions older than the configured lifetime.

        Returns:
            Number of deleted sessions, or zero when expiry is disabled.
        """
        if self.config.session_ttl_seconds is None:
            return 0
        return self.store.delete_expired(self.config.session_ttl_seconds)

    def list_sessions(self) -> list[SessionInfo]:
        """List current saved sessions without exposing snapshot bytes.

        Returns:
            Session metadata sorted by session ID.
        """
        self.delete_expired()
        return self.store.list_sessions()

    def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Read safe metadata for one current session.

        Args:
            session_id: Session key to inspect.

        Returns:
            Session metadata, or ``None`` when absent or expired.
        """
        stored = self.store.load(
            session_id,
            max_age_seconds=self.config.session_ttl_seconds,
        )
        return (
            SessionInfo.from_stored(session_id, stored) if stored is not None else None
        )

    def get_working_directory(self, session_id: str) -> str:
        """Read the saved virtual directory without executing startup code.

        Args:
            session_id: Session to inspect.

        Returns:
            Last saved directory, or the configured initial directory.
        """
        return (
            self._inspect_session(
                session_id,
                "import os as _arenero_inspect_os\n_arenero_inspect_os.getcwd()",
            )
            or self.default_working_directory
        )

    def variables(self, session_id: str) -> dict[str, str]:
        """Inspect saved public names and types without bootstrap or a state save.

        Args:
            session_id: Session to inspect.

        Returns:
            Name-to-type mapping, excluding private names and configured tools.
            An absent, expired, or stateless session returns an empty mapping.
        """
        return self._inspect_session(session_id, _variables_source(self)) or {}

    def _inspect_session(self, session_id: str, code: str) -> Any:
        """Read trusted metadata from a disposable copy of a saved session.

        Args:
            session_id: Session to read; no session is created when absent.
            code: Trusted metadata source, not caller-supplied Python.

        Returns:
            Raw metadata, or None when no snapshot exists. No state is saved.
        """
        _validate_session_id(session_id)
        if self._closed:
            raise RuntimeError("the kernel is closed")
        stored = self.store.load(
            session_id, max_age_seconds=self.config.session_ttl_seconds
        )
        if stored is None:
            return None
        if stored.monty_version != self.monty_version:
            raise ValueError(
                "the session uses another Monty version; reset it before inspection"
            )
        with self._pool.checkout(
            script_name=f"<arenero:inspect:{session_id}>",
            limits=self.config.resource_limits(),
            os_policy=self.config.os_policy,
        ) as session:
            session.load_session(stored.state)
            return session.feed_run(code, skip_type_check=True)

    def execute(
        self,
        session_id: str,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute one code snippet in a persistent session.

        Args:
            session_id: Session key with 1 to 255 characters.
            code: Monty-compatible Python code to execute.
            output_limit_chars: Optional output limit requested by the caller.
            output_file: Optional absolute virtual path used when output is trimmed.
            state_policy: Save state ``always``, only on ``success``, or ``never``.
            restart: Reset state before this execution when true.
            hosted_tools: Optional tools available for this operation only.
            inputs: Optional host values bound before this user feed, after bootstrap.

        Returns:
            A structured execution result.
        """
        return self._execute(
            session_id,
            code,
            output_limit_chars,
            file_path=None,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            inputs=inputs,
        )

    def execute_file(
        self,
        session_id: str,
        file_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> ExecutionResult:
        """Execute a virtual Python file and optional follow-up code.

        Args:
            session_id: Session key with 1 to 255 characters.
            file_path: Path to a readable Python file in the virtual filesystem.
            code: Optional code to run after the file succeeds. It can use names
                defined by the file.
            output_limit_chars: Optional output limit requested by the caller.
            output_file: Optional absolute virtual path used when output is trimmed.
            state_policy: Save state ``always``, only on ``success``, or ``never``.
            restart: Reset state before this execution when true.
            hosted_tools: Optional tools available for this operation only.

        Returns:
            A structured result for the file or the follow-up code when supplied.
        """
        return self._execute(
            session_id,
            code,
            output_limit_chars,
            file_path=file_path,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    def download_file(
        self,
        path: str,
        max_bytes: int | None = None,
    ) -> bytes:
        """Read one binary file through the configured sandbox filesystem.

        Args:
            path: Absolute virtual POSIX file path.
            max_bytes: Optional caller limit. The filesystem hard cap still applies.

        Returns:
            File content as bytes.

        Raises:
            ValueError: If the path is not absolute or the requested limit is invalid.
            MontyError: If the sandbox denies or cannot read the file.
        """
        if not path.startswith("/"):
            raise ValueError("path must be an absolute virtual POSIX path")
        limit = self._transfer_limit(max_bytes)
        source = (
            "from pathlib import Path\n"
            f"_arenero_transfer_path = Path({json.dumps(path)})\n"
            "_arenero_transfer_size = _arenero_transfer_path.stat().st_size\n"
            f"if _arenero_transfer_size > {limit}:\n"
            "    raise ValueError('file exceeds the transfer limit')\n"
            "_arenero_transfer_path.read_bytes()"
        )
        result = self._run_isolated(source)
        if not isinstance(result, bytes):
            raise TypeError("the virtual file did not return binary content")
        emit_event(
            self.event_sink,
            "filesystem",
            "downloaded",
            path=path,
            size_bytes=len(result),
        )
        return result

    def execute_module(
        self,
        session_id: str,
        entry_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        module_config: VirtualModuleConfig | None = None,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> ExecutionResult:
        """Resolve and execute a static virtual module graph.

        Args:
            session_id: Session key with 1 to 255 characters.
            entry_path: Absolute virtual entry ``.py`` file.
            code: Optional follow-up code in the same feed.
            output_limit_chars: Optional caller output limit.
            module_config: Optional module roots, aliases, and limits.
            state_policy: State save policy for this operation.
            restart: Reset saved Python state before execution when true.
            hosted_tools: Optional tools for this operation only.

        Returns:
            Structured result for the bundle or follow-up code.
        """

        def load(path: str) -> str | None:
            """Read one candidate virtual module as UTF-8 text.

            Args:
                path: Absolute candidate file path.

            Returns:
                Source text, or ``None`` when the file is not readable.
            """

            try:
                return self.download_file(path).decode()
            except MontyError:
                return None

        try:
            bundle = VirtualModuleResolver(load, module_config).bundle(entry_path, code)
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            return self._error_result(
                "VirtualModuleError",
                str(exc),
                output_limit_chars,
                state_policy=state_policy,
            )
        return self._execute(
            session_id,
            bundle.code,
            output_limit_chars,
            file_path=None,
            output_file=None,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            skip_type_check=True,
        )

    def upload_file(
        self,
        path: str,
        content: bytes,
        *,
        create_parents: bool = True,
    ) -> int:
        """Write one binary file through the configured sandbox filesystem.

        Args:
            path: Absolute virtual POSIX file path.
            content: Binary file content.
            create_parents: Create missing parent directories when true.

        Returns:
            Number of bytes written.

        Raises:
            ValueError: If the path is invalid or content exceeds the hard cap.
            MontyError: If the sandbox denies or cannot write the file.
        """
        if not path.startswith("/"):
            raise ValueError("path must be an absolute virtual POSIX path")
        if len(content) > self.filesystem_config.max_transfer_bytes:
            raise ValueError("content exceeds the filesystem transfer limit")
        parent_step = (
            "_arenero_transfer_path.parent.mkdir(parents=True, exist_ok=True)\n"
            if create_parents
            else ""
        )
        source = (
            "from pathlib import Path\n"
            f"_arenero_transfer_path = Path({json.dumps(path)})\n"
            f"{parent_step}"
            "_arenero_transfer_path.write_bytes(_arenero_transfer_content)"
        )
        result = self._run_isolated(
            source,
            inputs={"_arenero_transfer_content": bytes(content)},
        )
        if not isinstance(result, int):
            raise TypeError("the virtual file write did not return a byte count")
        emit_event(
            self.event_sink,
            "filesystem",
            "uploaded",
            path=path,
            size_bytes=result,
        )
        return result

    def call_filesystem(
        self,
        session_id: str,
        operation: FilesystemOperation,
        arguments: Mapping[str, FilesystemValue],
        output_limit_chars: int | None = None,
    ) -> ExecutionResult:
        """Call one sandbox filesystem helper.

        Args:
            session_id: Session key with 1 to 255 characters.
            operation: ``read``, ``write``, ``edit``, ``mkdir``, ``delete``,
                ``ls``, ``glob``, or ``grep``.
            arguments: Keyword arguments for the selected operation.
            output_limit_chars: Optional output limit requested by the caller.

        Returns:
            A structured execution result.
        """
        code = build_filesystem_call(operation, arguments)
        result = self.execute(session_id, code, output_limit_chars)
        emit_event(
            self.event_sink,
            "filesystem",
            "operation",
            session_id=session_id,
            operation=operation,
            ok=result["ok"],
        )
        return result

    def _execute(
        self,
        session_id: str,
        code: str | None,
        output_limit_chars: int | None,
        *,
        file_path: str | None,
        output_file: str | None,
        state_policy: StatePolicy,
        restart: bool,
        hosted_tools: HostedToolSet | None,
        skip_type_check: bool = False,
        inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        """Run one measured kernel operation.

        Args:
            session_id: Session key used to load, lock, and save state.
            code: Code to execute directly or after a virtual file.
            output_limit_chars: Optional output limit requested by the caller.
            file_path: Optional virtual Python file to execute first.
            output_file: Optional virtual path for bounded untrimmed output.
            state_policy: State save policy for this operation.
            restart: Whether to reset state before this operation.
            hosted_tools: Optional operation-only hosted tools.
            skip_type_check: Disable checking for generated support code.
            inputs: Optional values bound for the direct user feed.

        Returns:
            A structured execution result with stable statistics.
        """
        metrics = _ExecutionMetrics(uuid.uuid4().hex, time.monotonic())
        emit_event(
            self.event_sink,
            "execution",
            "started",
            execution_id=metrics.execution_id,
            session_id=session_id,
            state_policy=state_policy,
            source="file" if file_path is not None else "code",
        )
        result = self._execute_operation(
            session_id,
            code,
            output_limit_chars,
            file_path=file_path,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            skip_type_check=skip_type_check,
            metrics=metrics,
            inputs=inputs,
        )
        result["session_restarted"] = metrics.session_restarted
        result["host_calls"] = (
            metrics.hosted_scope.summary()
            if metrics.hosted_scope is not None
            else {"count": 0, "items": [], "truncated": False}
        )
        stats = result["stats"]
        stats["execution_id"] = metrics.execution_id
        stats["wall_time_seconds"] = time.monotonic() - metrics.started_at
        stats["checkout_wait_seconds"] = metrics.checkout_wait_seconds
        stats["state_before_bytes"] = metrics.state_before_bytes
        stats["state_after_bytes"] = metrics.state_after_bytes
        emit_event(
            self.event_sink,
            "execution",
            "finished",
            execution_id=metrics.execution_id,
            session_id=session_id,
            ok=result["ok"],
            error_code=result["error"]["code"] if result["error"] else None,
            wall_time_seconds=stats["wall_time_seconds"],
            checkout_wait_seconds=stats["checkout_wait_seconds"],
            state_saved=result["state_saved"],
        )
        return result

    def _execute_operation(
        self,
        session_id: str,
        code: str | None,
        output_limit_chars: int | None,
        *,
        file_path: str | None,
        output_file: str | None,
        state_policy: StatePolicy,
        restart: bool,
        hosted_tools: HostedToolSet | None,
        skip_type_check: bool,
        metrics: _ExecutionMetrics,
        inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        """Run feeds and save their final state once.

        Args:
            session_id: Session key used to load, lock, and save state.
            code: Code to execute directly or after a virtual file.
            output_limit_chars: Optional output limit requested by the caller.
            file_path: Optional virtual Python file to execute first.
            output_file: Optional virtual path for bounded untrimmed output.
            state_policy: State save policy for this operation.
            restart: Whether to delete saved state before loading.
            hosted_tools: Optional operation-only hosted tools.
            skip_type_check: Disable checking for generated support code.
            metrics: Mutable execution statistics.
            inputs: Optional values bound for the direct user feed.

        Returns:
            A structured execution result.
        """
        if self._closed:
            raise RuntimeError("the kernel is closed")
        if state_policy not in {"always", "success", "never"}:
            return self._error_result(
                "InvalidStatePolicy",
                "state_policy must be 'always', 'success', or 'never'",
                output_limit_chars,
                state_policy="never",
            )
        if not session_id or len(session_id) > 255:
            return self._error_result(
                "InvalidSessionId",
                "session_id must contain 1 to 255 characters",
                output_limit_chars,
                state_policy=state_policy,
            )

        limit, limit_was_capped = self._output_limit(output_limit_chars)
        with self._lock_for(session_id), self.filesystem_guard():
            if restart:
                existed = self.store.delete(session_id)
                metrics.session_restarted = True
                emit_event(
                    self.event_sink,
                    "session",
                    "reset",
                    session_id=session_id,
                    existed=existed,
                    execution_id=metrics.execution_id,
                )
            stored = self.store.load(
                session_id,
                max_age_seconds=self.config.session_ttl_seconds,
            )
            restored = stored is not None
            metrics.state_before_bytes = len(stored.state) if stored else 0

            if stored is not None and stored.monty_version != self.monty_version:
                return self._error_result(
                    "SessionVersionError",
                    (
                        f"The session uses Monty {stored.monty_version}, but this "
                        f"process uses {self.monty_version}. Reset the session."
                    ),
                    output_limit_chars,
                    state_restored=True,
                    state_policy=state_policy,
                )

            streams = CollectStreams(max_bytes=self.config.max_stream_bytes)
            result: Any = None
            execution_error: MontyError | None = None
            error_type_override: str | None = None
            terminal_resource_error = False
            bootstrap_failed = False
            state: bytes | None = None
            try:
                operation_tools = self._operation_hosted_tools(hosted_tools)
                external_lookup = self._operation_external_lookup(
                    metrics, operation_tools
                )
            except (TypeError, ValueError) as exc:
                return self._result(
                    result=None,
                    streams=streams,
                    error_type="HostedToolConfigurationError",
                    error_message=str(exc),
                    retryable=False,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=stored.revision if stored else None,
                    output_file=output_file,
                    state_policy=state_policy,
                )

            try:
                checkout_started = time.monotonic()
                with self._pool.checkout(
                    script_name=f"<arenero:{session_id}>",
                    limits=self.config.resource_limits(),
                    os_policy=self.config.os_policy,
                    type_check=(
                        False
                        if skip_type_check
                        else self._operation_type_check_enabled(operation_tools)
                    ),
                    type_check_stubs=self._operation_type_check_stubs(operation_tools),
                    type_check_format=self.config.type_check_format,
                ) as session:
                    metrics.checkout_wait_seconds = time.monotonic() - checkout_started
                    if stored is not None:
                        session.load_session(stored.state)

                    phase = "bootstrap"
                    try:
                        self._prepare_session(
                            session,
                            streams,
                            external_lookup,
                            new_session=stored is None,
                        )
                        phase = "execution"
                        result = self._run(
                            session, streams, code, file_path, external_lookup, inputs
                        )
                    except InfrastructureErrors:
                        raise
                    except MontyError as exc:
                        execution_error = exc
                        terminal_resource_error = _is_terminal_resource_error(exc)
                        if phase == "bootstrap":
                            bootstrap_failed = True
                            error_type_override = "SessionBootstrapError"

                    if (
                        self.store.stores_state
                        and not terminal_resource_error
                        and not bootstrap_failed
                        and state_policy != "never"
                        and (state_policy == "always" or execution_error is None)
                    ):
                        state = session.dump()
                        metrics.state_after_bytes = len(state)
            except InfrastructureErrors as exc:
                if isinstance(exc, MontyCrashedError):
                    self.store.delete(session_id)
                return self._result(
                    result=None,
                    streams=streams,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    retryable=True,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=stored.revision if stored else None,
                    output_file=output_file,
                    state_policy=state_policy,
                )
            except TimeoutError as exc:
                metrics.checkout_wait_seconds = time.monotonic() - checkout_started
                return self._result(
                    result=None,
                    streams=streams,
                    error_type="WorkerCheckoutTimeout",
                    error_message=str(exc),
                    retryable=True,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=stored.revision if stored else None,
                    output_file=output_file,
                    state_policy=state_policy,
                )
            except MontyError as exc:
                return self._result(
                    result=None,
                    streams=streams,
                    error_type="SessionRestoreError",
                    error_message=f"{exc}. Reset the session before retrying.",
                    retryable=False,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=stored.revision if stored else None,
                    output_file=output_file,
                    state_policy=state_policy,
                )

            if terminal_resource_error:
                self.store.delete(session_id)
                return self._result(
                    result=result,
                    streams=streams,
                    error_type=type(execution_error).__name__,
                    error_message=str(execution_error),
                    retryable=False,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=None,
                    output_file=output_file,
                    state_policy=state_policy,
                )

            if state is None:
                return self._result(
                    result=result,
                    streams=streams,
                    error_type=(
                        error_type_override
                        or (type(execution_error).__name__ if execution_error else None)
                    ),
                    error_message=str(execution_error) if execution_error else None,
                    retryable=False,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=stored.revision if stored else None,
                    output_file=output_file,
                    state_policy=state_policy,
                )

            if len(state) > self.config.max_state_bytes:
                return self._result(
                    result=None,
                    streams=streams,
                    error_type="SessionStateTooLarge",
                    error_message=(
                        f"The session state is {len(state)} bytes. The limit is "
                        f"{self.config.max_state_bytes} bytes. The previous state "
                        "was kept."
                    ),
                    retryable=False,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=stored.revision if stored else None,
                    output_file=output_file,
                    state_policy=state_policy,
                )

            try:
                revision = self.store.save(
                    session_id=session_id,
                    state=state,
                    monty_version=self.monty_version,
                    expected_revision=stored.revision if stored else None,
                )
            except ConcurrentSessionUpdate:
                return self._result(
                    result=None,
                    streams=streams,
                    error_type="ConcurrentSessionUpdate",
                    error_message=(
                        "The session changed during execution. Retry the tool call."
                    ),
                    retryable=True,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=restored,
                    revision=stored.revision if stored else None,
                    output_file=output_file,
                    state_policy=state_policy,
                )

            return self._result(
                result=result,
                streams=streams,
                error_type=type(execution_error).__name__ if execution_error else None,
                error_message=str(execution_error) if execution_error else None,
                retryable=False,
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_saved=True,
                state_restored=restored,
                revision=revision,
                output_file=output_file,
                state_policy=state_policy,
            )

    def _prepare_session(
        self,
        session: MontySession,
        streams: CollectStreams,
        external_lookup: dict[str, Any] | None,
        *,
        new_session: bool,
    ) -> None:
        """Install helpers and run configured startup sources for a new session.

        Args:
            session: Active Monty session.
            streams: Output collector shared by all feeds.
            external_lookup: Host functions for the operation.
            new_session: Whether no saved snapshot was loaded.
        """
        if (
            new_session
            or self._feed(
                session,
                streams,
                "locals().get('_arenero_fs_version')",
                external_lookup,
                skip_type_check=True,
            )
            != self._filesystem_version
        ):
            self._feed(
                session,
                streams,
                self._filesystem_prelude
                + f"\n_arenero_fs_version = {self._filesystem_version!r}",
                external_lookup,
                skip_type_check=True,
            )
        if not new_session:
            return
        if self.default_working_directory != "/":
            self._feed(
                session,
                streams,
                f"import os as _arenero_os\n_arenero_os.chdir({self.default_working_directory!r})",
                external_lookup,
                skip_type_check=True,
            )
        for path in self.bootstrap_files:
            source = self._feed(
                session,
                streams,
                f"open({json.dumps(path)}).read()",
                external_lookup,
            )
            self._feed(session, streams, source, external_lookup)
        if self.bootstrap_code:
            self._feed(session, streams, self.bootstrap_code, external_lookup)

    def _run(
        self,
        session: MontySession,
        streams: CollectStreams,
        code: str | None,
        file_path: str | None,
        external_lookup: dict[str, Any] | None,
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run direct code or a virtual file followed by direct code.

        Args:
            session: Active Monty session.
            streams: Output collector shared by all feeds in this operation.
            code: Direct or follow-up code to execute.
            file_path: Optional virtual file to read and execute first.
            external_lookup: Host functions for this complete operation.
            inputs: Optional values for a direct code feed.

        Returns:
            The value of the last feed that ran.
        """
        if file_path is None:
            return self._feed(
                session, streams, code or "", external_lookup, inputs=inputs
            )

        source = self._feed(
            session,
            streams,
            f"open({json.dumps(file_path)}).read()",
            external_lookup,
        )
        result = self._feed(session, streams, source, external_lookup)
        if code is not None:
            result = self._feed(session, streams, code, external_lookup)
        return result

    def _feed(
        self,
        session: MontySession,
        streams: CollectStreams,
        code: str,
        external_lookup: dict[str, Any] | None,
        *,
        skip_type_check: bool = False,
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run one feed with the kernel's host access settings.

        Args:
            session: Active Monty session.
            streams: Output collector for this kernel operation.
            code: Monty-compatible Python code to execute.
            external_lookup: Host functions for this complete operation.
            skip_type_check: Skip checking for trusted generated support code.
            inputs: Optional host values bound before execution.

        Returns:
            The feed result.
        """
        return session.feed_run(
            code,
            inputs=dict(inputs) if inputs is not None else None,
            external_lookup=external_lookup,
            print_callback=streams,
            mount=self.mounts,
            os=self.os_handler,
            skip_type_check=skip_type_check,
        )

    def _operation_hosted_tools(
        self, operation_tools: HostedToolSet | None
    ) -> HostedToolSet:
        """Combine configured, promoted MCP, and operation-only tools.

        Args:
            operation_tools: Optional tools supplied by one framework call.

        Returns:
            Complete hosted tool set for one execution.
        """

        promoted = self.mcp_hub.promoted_toolset() if self.mcp_hub is not None else None
        return self.hosted_tools.merged(promoted, operation_tools)

    def _operation_external_lookup(
        self,
        metrics: _ExecutionMetrics,
        hosted_tools: HostedToolSet,
    ) -> dict[str, Any] | None:
        """Build host functions with fresh MCP and hosted-tool budgets.

        Args:
            metrics: Current execution statistics.
            hosted_tools: Tools selected for this operation.

        Returns:
            Host functions for one complete kernel operation, or ``None``.
        """

        lookup = dict(self.external_lookup)
        if self.mcp_hub is not None:
            lookup.update(self.mcp_hub.host_functions())
        if hosted_tools.tools:
            scope = hosted_tools.sync_scope(metrics.execution_id, self.event_sink)
            hosted_lookup = scope.sync_lookup()
            collisions = lookup.keys() & hosted_lookup.keys()
            if collisions:
                names = ", ".join(sorted(collisions))
                raise ValueError(
                    f"hosted tool names conflict with host functions: {names}"
                )
            lookup.update(hosted_lookup)
            metrics.hosted_scope = scope
        return lookup or None

    def _operation_type_check_stubs(self, hosted_tools: HostedToolSet) -> str:
        """Build type stubs for one operation's exact host functions.

        Args:
            hosted_tools: Tools selected for this operation.

        Returns:
            Filesystem, MCP, application, and hosted-tool declarations.
        """

        hosted_stubs = hosted_tools.type_check_stubs(async_mode=False)
        host_stubs = "\n".join(
            value for value in (self._base_type_check_stubs, hosted_stubs) if value
        )
        return _type_check_stubs(host_stubs)

    def _operation_type_check_enabled(self, hosted_tools: HostedToolSet) -> bool:
        """Disable checking when an operation adds functions to a saved session.

        Args:
            hosted_tools: Complete hosted tool set for this operation.

        Returns:
            ``True`` when the session's stable stub set covers all hosted tools.
        """

        configured = {tool.alias for tool in self.hosted_tools.tools}
        active = {tool.alias for tool in hosted_tools.tools}
        return self.config.type_check and active <= configured

    def _run_isolated(
        self,
        code: str,
        *,
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run one filesystem transfer without changing persistent REPL state.

        Args:
            code: Monty-compatible transfer code.
            inputs: Optional host values bound before execution.

        Returns:
            Raw value returned across the Monty boundary.
        """
        if self._closed:
            raise RuntimeError("the kernel is closed")
        with (
            self.filesystem_guard(),
            self._pool.checkout(
                script_name="<arenero:transfer>",
                limits=self.config.resource_limits(),
                os_policy=self.config.os_policy,
            ) as session,
        ):
            return session.feed_run(
                code,
                inputs=dict(inputs) if inputs is not None else None,
                external_lookup=self.external_lookup or None,
                mount=self.mounts,
                os=self.os_handler,
            )

    @contextmanager
    def filesystem_guard(self) -> Iterator[None]:
        """Serialize shared filesystem work inside this kernel process.

        Yields:
            Control while this kernel owns its filesystem lock.
        """
        if not self._filesystem_enabled:
            yield
            return
        with self._filesystem_lock:
            yield

    def _lock_for(self, session_id: str) -> threading.Lock:
        """Get the process-local lock for one session.

        Args:
            session_id: Session key that owns the lock.

        Returns:
            The session lock.
        """
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _output_limit(self, requested: int | None) -> tuple[int, bool]:
        """Normalize a requested output limit.

        Args:
            requested: Caller limit, or ``None`` for the configured default.

        Returns:
            Effective limit and whether the hard maximum changed it.
        """
        if requested is None:
            return self.config.default_output_limit_chars, False
        normalized = max(0, requested)
        effective = min(normalized, self.config.max_output_limit_chars)
        return effective, effective != requested

    def _transfer_limit(self, requested: int | None) -> int:
        """Normalize a requested binary transfer limit.

        Args:
            requested: Caller byte limit, or ``None`` for the hard maximum.

        Returns:
            Effective byte limit.
        """
        if requested is None:
            return self.filesystem_config.max_transfer_bytes
        if requested < 0:
            raise ValueError("max_bytes must be at least 0")
        return min(requested, self.filesystem_config.max_transfer_bytes)

    def _error_result(
        self,
        error_type: str,
        message: str,
        requested_limit: int | None,
        state_restored: bool = False,
        state_policy: StatePolicy = "always",
    ) -> ExecutionResult:
        """Build a result for an error outside normal code execution.

        Args:
            error_type: Stable error category.
            message: Human-readable error details.
            requested_limit: Output limit requested by the caller.
            state_restored: Whether saved state was loaded before the error.
            state_policy: State save policy requested by the caller.

        Returns:
            A structured failed execution result.
        """
        limit, limit_was_capped = self._output_limit(requested_limit)
        streams = CollectStreams(max_bytes=self.config.max_stream_bytes)
        return self._result(
            result=None,
            streams=streams,
            error_type=error_type,
            error_message=message,
            retryable=False,
            limit=limit,
            limit_was_capped=limit_was_capped,
            state_saved=False,
            state_restored=state_restored,
            revision=None,
            state_policy=state_policy,
        )

    def _result(
        self,
        *,
        result: Any,
        streams: CollectStreams,
        error_type: str | None,
        error_message: str | None,
        retryable: bool,
        limit: int,
        limit_was_capped: bool,
        state_saved: bool,
        state_restored: bool,
        revision: int | None,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
    ) -> ExecutionResult:
        """Build and trim the public execution result.

        Args:
            result: Value returned by the final Monty feed.
            streams: Captured standard output and error streams.
            error_type: Optional stable error category.
            error_message: Optional human-readable error details.
            retryable: Whether the caller can safely retry.
            limit: Effective character limit for each output field.
            limit_was_capped: Whether the hard maximum changed the request.
            state_saved: Whether this operation saved Python state.
            state_restored: Whether this operation loaded Python state.
            revision: Saved session revision, when available.
            output_file: Optional virtual path for bounded untrimmed output.
            state_policy: State save policy for this operation.

        Returns:
            A structured execution result.
        """
        artifacts: list[str] = []
        if self.mcp_artifact_directory is not None and result is not None:
            result, artifacts = self.materialize_mcp_artifacts(
                result,
                self.mcp_artifact_directory,
            )
        public_result, payload = _prepare_execution_result(
            result=result,
            streams=streams,
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
            limit=limit,
            limit_was_capped=limit_was_capped,
            state_saved=state_saved,
            state_restored=state_restored,
            revision=revision,
            output_file=output_file,
            state_policy=state_policy,
        )
        public_result["mcp_artifacts"] = artifacts
        if payload is not None and output_file is not None:
            try:
                self.upload_file(output_file, payload)
            except (MontyError, ValueError, TypeError, RuntimeError) as exc:
                public_result["output_file_error"] = str(exc)
            else:
                public_result["output_file"] = output_file
        return public_result


def _prepare_execution_result(
    *,
    result: Any,
    streams: CollectStreams,
    error_type: str | None,
    error_message: str | None,
    retryable: bool,
    limit: int,
    limit_was_capped: bool,
    state_saved: bool,
    state_restored: bool,
    revision: int | None,
    output_file: str | None,
    state_policy: StatePolicy = "always",
) -> tuple[ExecutionResult, bytes | None]:
    """Build one public result and optional untrimmed output payload.

    Args:
        result: Value returned by the final Monty feed.
        streams: Captured standard output and error streams.
        error_type: Optional stable error category.
        error_message: Optional human-readable error details.
        retryable: Whether the caller can safely retry.
        limit: Effective character limit for each output field.
        limit_was_capped: Whether the hard maximum changed the request.
        state_saved: Whether this operation saved Python state.
        state_restored: Whether this operation loaded Python state.
        revision: Saved session revision, when available.
        output_file: Optional virtual path for bounded untrimmed output.
        state_policy: State save policy for this operation.

    Returns:
        Public result and an output payload when trimmed data must be saved.
    """
    full_stdout = "".join(text for stream, text in streams.output if stream == "stdout")
    full_stderr = "".join(text for stream, text in streams.output if stream == "stderr")
    stdout, stdout_trimmed = _trim(full_stdout, limit)
    stderr, stderr_trimmed = _trim(full_stderr, limit)
    safe_result, result_trimmed = _safe_result(result, limit)

    error: ExecutionError | None = None
    full_error: ExecutionError | None = None
    error_trimmed = False
    if error_type is not None and error_message is not None:
        error_code = _error_code(error_type, error_message)
        error_hint = _error_hint(error_type, error_message, retryable)
        recovery = _recovery_action(error_type, error_message, retryable)
        full_error = {
            "type": error_type,
            "code": error_code,
            "message": error_message,
            "hint": error_hint,
            "retryable": retryable,
            "recovery": recovery,
        }
        short_error, error_trimmed = _trim(error_message, limit)
        error = {
            "type": error_type,
            "code": error_code,
            "message": short_error,
            "hint": error_hint,
            "retryable": retryable,
            "recovery": recovery,
        }

    was_trimmed = result_trimmed or stdout_trimmed or stderr_trimmed or error_trimmed
    payload = None
    if output_file is not None and was_trimmed:
        payload = json.dumps(
            {
                "result": _json_value(result),
                "stdout": full_stdout,
                "stderr": full_stderr,
                "error": full_error,
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

    return (
        {
            "ok": error is None,
            "result": safe_result,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
            "truncated": {
                "result": result_trimmed,
                "stdout": stdout_trimmed,
                "stderr": stderr_trimmed,
                "error": error_trimmed,
            },
            "output_limit_chars": limit,
            "limit_was_capped": limit_was_capped,
            "state_saved": state_saved,
            "state_restored": state_restored,
            "session_revision": revision,
            "retryable": retryable,
            "output_file": None,
            "output_file_error": None,
            "state_policy": state_policy,
            "session_restarted": False,
            "stats": {
                "execution_id": "",
                "wall_time_seconds": 0.0,
                "checkout_wait_seconds": 0.0,
                "state_before_bytes": 0,
                "state_after_bytes": 0,
                "original_output_chars": {
                    "result": _result_length(result),
                    "stdout": len(full_stdout),
                    "stderr": len(full_stderr),
                    "error": len(error_message or ""),
                },
            },
            "mcp_artifacts": [],
            "host_calls": {"count": 0, "items": [], "truncated": False},
        },
        payload,
    )


def _variables_source(kernel: Any) -> str:
    """Build the common read-only name inspection used by both kernels.

    Args:
        kernel: Kernel whose configured helper and tool names must be hidden.

    Returns:
        Monty source that returns public names and their type names, not values.
    """
    excluded = {
        name for name, value in kernel.external_lookup.items() if callable(value)
    }
    excluded.update(tool.alias for tool in kernel._operation_hosted_tools(None).tools)
    excluded.update(f"fs_{name}" for name in get_args(FilesystemOperation))
    excluded.add("arenero_capture_shell_state")  # Alias in older snapshots.
    if kernel.mcp_hub is not None:
        excluded.update(kernel.mcp_hub.host_functions())
    return (
        "{name: type(value).__name__ for name, value in locals().items() "
        f"if not name.startswith('_') and name not in {sorted(excluded)!r}}}"
    )


def _mount_list(mounts: MountDir | list[MountDir] | None) -> list[MountDir]:
    """Normalize optional user mounts to a new list.

    Args:
        mounts: No mount, one mount, or a list of mounts.

    Returns:
        New mutable list of configured mounts.
    """
    if mounts is None:
        return []
    return list(mounts) if isinstance(mounts, list) else [mounts]


def _type_check_stubs(host_stubs: str | None) -> str:
    """Combine Arenero helper declarations with application declarations.

    Args:
        host_stubs: Optional stubs for approved host functions.

    Returns:
        Complete Monty type-check stub source.
    """
    if host_stubs:
        return f"{FILESYSTEM_TYPE_CHECK_STUBS}\n{host_stubs}"
    return FILESYSTEM_TYPE_CHECK_STUBS


def _safe_result(value: Any, limit: int) -> tuple[Any, bool]:
    """Convert and trim a result without failing on custom values.

    Args:
        value: Monty result value.
        limit: Maximum serialized character count.

    Returns:
        Safe result value and whether it was shortened.
    """
    if value is None:
        return None, False
    value = _json_value(value)
    serialized = json.dumps(value, ensure_ascii=False)

    if len(serialized) <= limit:
        return value, False
    return _trim(serialized, limit)


def _result_length(value: Any) -> int:
    """Measure the original serialized result length.

    Args:
        value: Raw Monty result value.

    Returns:
        JSON character length, or representation length for custom values.
    """
    if value is None:
        return 0
    return len(json.dumps(_json_value(value), ensure_ascii=False))


def _error_code(error_type: str, message: str) -> str:
    """Return a stable machine error code.

    Args:
        error_type: Public error class name.
        message: Human-readable error details.

    Returns:
        Lowercase snake-case error code.
    """
    if error_type == "VirtualModuleError":
        return "virtual_module_error"
    lowered = message.lower()
    if "suspension limit" in lowered:
        return "suspension_limit"
    if "sleep limit" in lowered:
        return "sleep_limit"
    if "hosted tool call limit" in lowered:
        return "hosted_tool_call_limit"
    if "hosted tool" in lowered and "time limit" in lowered:
        return "hosted_tool_time_limit"
    if "unresolved-reference" in lowered:
        return "unresolved_reference"
    if "import" in lowered or "no module" in lowered:
        return "unsupported_import"
    return re.sub(r"(?<!^)(?=[A-Z])", "_", error_type).lower()


def _error_hint(error_type: str, message: str, retryable: bool) -> str:
    """Return short recovery guidance for one execution error.

    Args:
        error_type: Public error class name.
        message: Human-readable error details.
        retryable: Whether a retry can succeed without changing code.

    Returns:
        Stable plain-language recovery hint.
    """
    if error_type == "HostedToolConfigurationError" and "async-only" in message:
        return "Use AsyncPersistentMontyKernel for this async-only hosted tool."
    hints = {
        "InvalidSessionId": "Use a session ID with 1 to 255 characters.",
        "InvalidStatePolicy": "Use state_policy='always', 'success', or 'never'.",
        "SessionVersionError": "Reset the session, then run the code again.",
        "WorkerCheckoutTimeout": "Retry later or increase the worker pool size.",
        "SessionRestoreError": "Reset the session before you run more code.",
        "SessionStateTooLarge": "Reduce saved values or reset the session.",
        "ConcurrentSessionUpdate": "Retry this operation once.",
        "SessionBootstrapError": "Check the trusted bootstrap code and files.",
        "HostedToolConfigurationError": (
            "Fix hosted tool names, schemas, or sync and async configuration."
        ),
        "VirtualModuleError": (
            "Use a readable .py file and only supported static virtual imports."
        ),
    }
    if error_type in hints:
        return hints[error_type]
    lowered = message.lower()
    if "suspension limit" in lowered:
        return "Use fewer host or OS calls, or split the work across executions."
    if "sleep limit" in lowered:
        return "Start fresh with less system sleep or a zero-sleep OS policy."
    if "hosted tool call limit" in lowered:
        return "Use fewer hosted calls or split the work across executions."
    if "hosted tool" in lowered and "time limit" in lowered:
        return "Use less hosted work or split the work across executions."
    if "unresolved-reference" in lowered:
        return (
            "Define the name. If it is a newly added hosted tool, use restart=True "
            "or configure it before this type-checked session starts. For exec-defined "
            "names, use execute_file or keep their use inside exec."
        )
    if "import" in lowered or "no module" in lowered:
        return (
            "Third-party packages are unavailable. Use execute_file for one virtual "
            "file or execute_module for a static graph."
        )
    if "nameerror" in lowered or "not defined" in lowered:
        return "Define the name or run its helper file before this code."
    if retryable:
        return "Retry the operation without changing the session ID."
    return "Change the code to fit the Monty subset and the active capability report."


def _recovery_action(error_type: str, message: str, retryable: bool) -> RecoveryAction:
    """Return the next useful action for an execution error.

    Args:
        error_type: Public error class name.
        message: Human-readable error details.
        retryable: Whether the same operation can be retried.

    Returns:
        Stable recovery action for agents and framework adapters.
    """

    if error_type in {"WorkerCheckoutTimeout", "MontyDisconnectError", "MontyShutdown"}:
        return "retry_later"
    if error_type == "ConcurrentSessionUpdate":
        return "retry_same_call"
    if error_type in {"SessionVersionError", "SessionRestoreError"}:
        return "restart_session"
    lowered = message.lower()
    if "hosted tool" in lowered and (
        "call limit" in lowered or "time limit" in lowered
    ):
        return "revise_code"
    if "time limit" in lowered or "sleep limit" in lowered or "memory" in lowered:
        return "restart_session"
    if error_type in {
        "InvalidSessionId",
        "InvalidStatePolicy",
        "SessionBootstrapError",
        "HostedToolConfigurationError",
    }:
        return "stop"
    if retryable:
        return "retry_same_call"
    return "revise_code"


def _validate_session_id(session_id: str) -> None:
    """Validate one public session ID.

    Args:
        session_id: Session key to validate.

    Raises:
        ValueError: If the ID does not contain 1 to 255 characters.
    """
    if not session_id or len(session_id) > 255:
        raise ValueError("session_id must contain 1 to 255 characters")


def _is_terminal_resource_error(error: MontyError) -> bool:
    """Return whether a Monty error makes the current session unsafe.

    Args:
        error: Error raised by one Monty feed.

    Returns:
        ``True`` for time and memory failures.
    """
    exception_method = getattr(error, "exception", None)
    if exception_method is None:
        return False
    try:
        exception = exception_method()
    except (AttributeError, TypeError):
        return False
    return isinstance(exception, (MemoryError, TimeoutError))


def _trim(text: str, limit: int) -> tuple[str, bool]:
    """Shorten text to a character limit.

    Args:
        text: Text to check.
        limit: Maximum returned length.

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
