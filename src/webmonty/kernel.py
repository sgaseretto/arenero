import importlib.metadata
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self, TypedDict

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
    ResourceLimits,
)

from .filesystem import (
    FilesystemConfig,
    FilesystemOperation,
    FilesystemValue,
    build_filesystem_call,
    build_filesystem_prelude,
)
from .store import ConcurrentSessionUpdate, InMemorySessionStore, SessionStore


class ExecutionError(TypedDict):
    """Describe an execution error returned to a caller."""

    type: str
    message: str
    retryable: bool


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


@dataclass(frozen=True)
class KernelConfig:
    """Set output, state, time, and memory limits for a kernel."""

    default_output_limit_chars: int = 8_000
    max_output_limit_chars: int = 50_000
    max_stream_bytes: int = 1_000_000
    max_state_bytes: int = 5_000_000
    max_duration_seconds: float = 5.0
    max_memory_bytes: int = 64 * 1024 * 1024
    max_recursion_depth: int = 100
    request_timeout_seconds: float = 10.0
    session_ttl_seconds: float | None = 24 * 60 * 60

    def __post_init__(self) -> None:
        """Validate related output and byte limit values."""

        if self.default_output_limit_chars < 0:
            raise ValueError("default_output_limit_chars must be at least 0")
        if self.max_output_limit_chars < self.default_output_limit_chars:
            raise ValueError("max_output_limit_chars must not be less than the default")
        if self.max_stream_bytes <= 0 or self.max_state_bytes <= 0:
            raise ValueError("byte limits must be greater than 0")


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
    ) -> None:
        """Create a persistent Monty kernel.

        Args:
            store: Backend that stores Python session dumps.
            config: Resource and output limits. Defaults to ``KernelConfig``.
            external_lookup: Approved host values and synchronous functions.
            os_handler: Optional virtual filesystem handler.
            mounts: Optional host directory mount or list of mounts.
            filesystem_config: Limits for the sandbox filesystem helpers.
        """

        self.store = store if store is not None else InMemorySessionStore()
        self.config = config or KernelConfig()
        self.external_lookup = dict(external_lookup or {})
        self.os_handler = os_handler
        self.mounts = mounts
        self.filesystem_config = filesystem_config or FilesystemConfig()
        self._filesystem_prelude = build_filesystem_prelude(self.filesystem_config)
        self.monty_version = importlib.metadata.version("pydantic-monty")
        self._pool_context = Monty(
            request_timeout=self.config.request_timeout_seconds,
        )
        self._pool = self._pool_context.__enter__()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
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
            self._pool_context.__exit__(None, None, None)
            self._closed = True

    def reset(self, session_id: str) -> bool:
        """Delete the saved Python state for one session.

        Args:
            session_id: Session key to reset.

        Returns:
            ``True`` when saved state existed.
        """
        with self._lock_for(session_id):
            return self.store.delete(session_id)

    def delete_expired(self) -> int:
        """Delete sessions older than the configured lifetime.

        Returns:
            Number of deleted sessions, or zero when expiry is disabled.
        """
        if self.config.session_ttl_seconds is None:
            return 0
        return self.store.delete_expired(self.config.session_ttl_seconds)

    def execute(
        self,
        session_id: str,
        code: str,
        output_limit_chars: int | None = None,
    ) -> ExecutionResult:
        """Execute one code snippet in a persistent session.

        Args:
            session_id: Session key with 1 to 255 characters.
            code: Monty-compatible Python code to execute.
            output_limit_chars: Optional output limit requested by the caller.

        Returns:
            A structured execution result.
        """
        return self._execute(session_id, code, output_limit_chars, file_path=None)

    def execute_file(
        self,
        session_id: str,
        file_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
    ) -> ExecutionResult:
        """Execute a virtual Python file and optional follow-up code.

        Args:
            session_id: Session key with 1 to 255 characters.
            file_path: Path to a readable Python file in the virtual filesystem.
            code: Optional code to run after the file succeeds. It can use names
                defined by the file.
            output_limit_chars: Optional output limit requested by the caller.

        Returns:
            A structured result for the file or the follow-up code when supplied.
        """
        return self._execute(session_id, code, output_limit_chars, file_path=file_path)

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
            f"_webmonty_transfer_path = Path({json.dumps(path)})\n"
            "_webmonty_transfer_size = _webmonty_transfer_path.stat().st_size\n"
            f"if _webmonty_transfer_size > {limit}:\n"
            "    raise ValueError('file exceeds the transfer limit')\n"
            "_webmonty_transfer_path.read_bytes()"
        )
        result = self._run_isolated(source)
        if not isinstance(result, bytes):
            raise TypeError("the virtual file did not return binary content")
        return result

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
            "_webmonty_transfer_path.parent.mkdir(parents=True, exist_ok=True)\n"
            if create_parents
            else ""
        )
        source = (
            "from pathlib import Path\n"
            f"_webmonty_transfer_path = Path({json.dumps(path)})\n"
            f"{parent_step}"
            "_webmonty_transfer_path.write_bytes(_webmonty_transfer_content)"
        )
        result = self._run_isolated(
            source,
            inputs={"_webmonty_transfer_content": bytes(content)},
        )
        if not isinstance(result, int):
            raise TypeError("the virtual file write did not return a byte count")
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
            operation: ``read``, ``write``, ``edit``, ``delete``, ``ls``,
                ``glob``, or ``grep``.
            arguments: Keyword arguments for the selected operation.
            output_limit_chars: Optional output limit requested by the caller.

        Returns:
            A structured execution result.
        """
        code = build_filesystem_call(operation, arguments)
        return self.execute(session_id, code, output_limit_chars)

    def _execute(
        self,
        session_id: str,
        code: str | None,
        output_limit_chars: int | None,
        *,
        file_path: str | None,
    ) -> ExecutionResult:
        """Run feeds and save their final state once.

        Args:
            session_id: Session key used to load, lock, and save state.
            code: Code to execute directly or after a virtual file.
            output_limit_chars: Optional output limit requested by the caller.
            file_path: Optional virtual Python file to execute first.

        Returns:
            A structured execution result.
        """
        if self._closed:
            raise RuntimeError("the kernel is closed")
        if not session_id or len(session_id) > 255:
            return self._error_result(
                "InvalidSessionId",
                "session_id must contain 1 to 255 characters",
                output_limit_chars,
            )

        limit, limit_was_capped = self._output_limit(output_limit_chars)
        with self._lock_for(session_id):
            stored = self.store.load(
                session_id,
                max_age_seconds=self.config.session_ttl_seconds,
            )
            restored = stored is not None

            if stored is not None and stored.monty_version != self.monty_version:
                return self._error_result(
                    "SessionVersionError",
                    (
                        f"The session uses Monty {stored.monty_version}, but this "
                        f"process uses {self.monty_version}. Reset the session."
                    ),
                    output_limit_chars,
                    state_restored=True,
                )

            streams = CollectStreams(max_bytes=self.config.max_stream_bytes)
            result: Any = None
            execution_error: MontyError | None = None
            terminal_resource_error = False
            state: bytes | None = None

            try:
                with self._pool.checkout(
                    script_name=f"<webmonty:{session_id}>",
                    limits=self._resource_limits(),
                ) as session:
                    if stored is not None:
                        session.load_session(stored.state)

                    try:
                        result = self._run(session, streams, code, file_path)
                    except InfrastructureErrors:
                        raise
                    except MontyError as exc:
                        execution_error = exc
                        terminal_resource_error = _is_terminal_resource_error(exc)

                    if self.store.stores_state and not terminal_resource_error:
                        state = session.dump()
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
                )

            if state is None:
                return self._result(
                    result=result,
                    streams=streams,
                    error_type=(
                        type(execution_error).__name__ if execution_error else None
                    ),
                    error_message=str(execution_error) if execution_error else None,
                    retryable=False,
                    limit=limit,
                    limit_was_capped=limit_was_capped,
                    state_saved=False,
                    state_restored=False,
                    revision=None,
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
            )

    def _run(
        self,
        session: MontySession,
        streams: CollectStreams,
        code: str | None,
        file_path: str | None,
    ) -> Any:
        """Run direct code or a virtual file followed by direct code.

        Args:
            session: Active Monty session.
            streams: Output collector shared by all feeds in this operation.
            code: Direct or follow-up code to execute.
            file_path: Optional virtual file to read and execute first.

        Returns:
            The value of the last feed that ran.
        """
        self._feed(session, streams, self._filesystem_prelude)
        if file_path is None:
            return self._feed(session, streams, code or "")

        source = self._feed(
            session,
            streams,
            f"open({json.dumps(file_path)}).read()",
        )
        result = self._feed(session, streams, source)
        if code is not None:
            result = self._feed(session, streams, code)
        return result

    def _feed(
        self,
        session: MontySession,
        streams: CollectStreams,
        code: str,
    ) -> Any:
        """Run one feed with the kernel's host access settings.

        Args:
            session: Active Monty session.
            streams: Output collector for this kernel operation.
            code: Monty-compatible Python code to execute.

        Returns:
            The feed result.
        """
        return session.feed_run(
            code,
            external_lookup=self.external_lookup or None,
            print_callback=streams,
            mount=self.mounts,
            os=self.os_handler,
        )

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
        with self._pool.checkout(
            script_name="<webmonty:transfer>",
            limits=self._resource_limits(),
        ) as session:
            return session.feed_run(
                code,
                inputs=inputs,
                external_lookup=self.external_lookup or None,
                mount=self.mounts,
                os=self.os_handler,
            )

    def _lock_for(self, session_id: str) -> threading.Lock:
        """Get the process-local lock for one session.

        Args:
            session_id: Session key that owns the lock.

        Returns:
            The session lock.
        """
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _resource_limits(self) -> ResourceLimits:
        """Build Monty resource limits from kernel configuration.

        Returns:
            Limits for time, memory, and recursion depth.
        """
        return {
            "max_duration_secs": self.config.max_duration_seconds,
            "max_memory": self.config.max_memory_bytes,
            "max_recursion_depth": self.config.max_recursion_depth,
        }

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
    ) -> ExecutionResult:
        """Build a result for an error outside normal code execution.

        Args:
            error_type: Stable error category.
            message: Human-readable error details.
            requested_limit: Output limit requested by the caller.
            state_restored: Whether saved state was loaded before the error.

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

        Returns:
            A structured execution result.
        """
        stdout = "".join(text for stream, text in streams.output if stream == "stdout")
        stderr = "".join(text for stream, text in streams.output if stream == "stderr")
        stdout, stdout_trimmed = _trim(stdout, limit)
        stderr, stderr_trimmed = _trim(stderr, limit)
        safe_result, result_trimmed = _safe_result(result, limit)

        error: ExecutionError | None = None
        error_trimmed = False
        if error_type is not None and error_message is not None:
            error_message, error_trimmed = _trim(error_message, limit)
            error = {
                "type": error_type,
                "message": error_message,
                "retryable": retryable,
            }

        return {
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
        }


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

    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return _trim(repr(value), limit)

    if len(serialized) <= limit:
        return value, False
    return _trim(serialized, limit)


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
