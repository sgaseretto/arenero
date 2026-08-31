"""Native asynchronous persistent Monty kernel."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Self

from pydantic_monty import (
    AsyncMonty,
    AsyncMontySession,
    CollectStreams,
    MontyCrashedError,
    MontyError,
    MountDir,
    OsHandler,
    ResourceLimits,
)

from .filesystem import FilesystemConfig, build_filesystem_prelude
from .kernel import (
    ExecutionResult,
    InfrastructureErrors,
    KernelConfig,
    _is_terminal_resource_error,
    _mount_list,
    _prepare_execution_result,
    _type_check_stubs,
)
from .store import ConcurrentSessionUpdate, InMemorySessionStore, SessionStore
from .workspace import WorkspaceConfig

if TYPE_CHECKING:
    from .mcp import McpHub


class AsyncPersistentMontyKernel:
    """Run persistent Python sessions on a native asynchronous Monty pool."""

    def __init__(
        self,
        store: SessionStore | None = None,
        config: KernelConfig | None = None,
        external_lookup: Mapping[str, Callable[..., Any]] | None = None,
        os_handler: OsHandler | None = None,
        mounts: MountDir | list[MountDir] | None = None,
        filesystem_config: FilesystemConfig | None = None,
        workspace: WorkspaceConfig | None = None,
        mcp_hub: McpHub | None = None,
    ) -> None:
        """Create a native asynchronous persistent kernel.

        Args:
            store: Backend that stores Python session dumps.
            config: Resource, pool, and output limits.
            external_lookup: Approved host values and sync or async functions.
            os_handler: Optional virtual filesystem handler.
            mounts: Optional host directory mount or list of mounts.
            filesystem_config: Limits for virtual ``fs_*`` helpers.
            workspace: Optional persistent directory mounted at a virtual path.
            mcp_hub: Optional configured MCP client hub. The caller owns it.
        """
        self.store = store if store is not None else InMemorySessionStore()
        self.config = config or KernelConfig()
        self.external_lookup = dict(external_lookup or {})
        self.os_handler = os_handler
        self.filesystem_config = filesystem_config or FilesystemConfig()
        self.workspace = workspace
        self.mcp_hub = mcp_hub
        if mcp_hub is not None:
            collisions = (
                self.external_lookup.keys() & mcp_hub.async_host_functions().keys()
            )
            if collisions:
                names = ", ".join(sorted(collisions))
                raise ValueError(f"external_lookup uses reserved MCP names: {names}")

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
        host_stubs = self.config.type_check_stubs
        if mcp_hub is not None:
            host_stubs = "\n".join(
                value for value in (host_stubs, mcp_hub.async_type_check_stubs) if value
            )
        self._type_check_stubs = _type_check_stubs(host_stubs)
        self.monty_version = importlib.metadata.version("pydantic-monty")
        self._pool_context = AsyncMonty(
            min_processes=self.config.min_worker_processes,
            max_processes=self.config.max_worker_processes,
            checkout_timeout=self.config.worker_checkout_timeout_seconds,
            request_timeout=self.config.request_timeout_seconds,
            max_checkouts_per_worker=self.config.max_checkouts_per_worker,
        )
        self._pool: AsyncMonty | None = None
        self._open_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._filesystem_lock = asyncio.Lock()
        self._filesystem_enabled = self.os_handler is not None or bool(self.mounts)
        self._closed = False

    async def __aenter__(self) -> Self:
        """Open and return this kernel.

        Returns:
            This open kernel.
        """
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the kernel when its context ends.

        Args:
            *exc_info: Optional exception details supplied by Python.
        """
        await self.close()

    async def open(self) -> None:
        """Open the asynchronous Monty worker pool once."""
        if self._closed:
            raise RuntimeError("the kernel is closed")
        if self._pool is not None:
            return
        async with self._open_lock:
            if self._pool is None:
                try:
                    self._pool = await self._pool_context.__aenter__()
                except BaseException:
                    self._close_owned_mount()
                    raise

    async def close(self) -> None:
        """Close the worker pool and owned mount once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._pool is not None:
                await self._pool_context.__aexit__(None, None, None)
        finally:
            self._pool = None
            self._close_owned_mount()

    async def reset(self, session_id: str) -> bool:
        """Delete saved Python state for one session.

        Args:
            session_id: Session key to reset.

        Returns:
            ``True`` when saved state existed.
        """
        async with self._lock_for(session_id), self.filesystem_guard():
            return await asyncio.to_thread(self.store.delete, session_id)

    async def execute(
        self,
        session_id: str,
        code: str,
        output_limit_chars: int | None = None,
    ) -> ExecutionResult:
        """Execute code in one persistent session.

        Args:
            session_id: Session key with 1 to 255 characters.
            code: Monty-compatible Python code.
            output_limit_chars: Optional caller output limit.

        Returns:
            A structured execution result.
        """
        return await self._execute(
            session_id,
            code,
            output_limit_chars,
            file_path=None,
        )

    async def execute_file(
        self,
        session_id: str,
        file_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
    ) -> ExecutionResult:
        """Execute a virtual Python file and optional follow-up code.

        Args:
            session_id: Session key with 1 to 255 characters.
            file_path: Path to a readable virtual Python file.
            code: Optional code to run after the file.
            output_limit_chars: Optional caller output limit.

        Returns:
            A structured execution result.
        """
        return await self._execute(
            session_id,
            code,
            output_limit_chars,
            file_path=file_path,
        )

    async def _execute(
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
            code: Direct or follow-up code.
            output_limit_chars: Optional caller output limit.
            file_path: Optional virtual Python file to execute first.

        Returns:
            A structured execution result.
        """
        await self.open()
        if not session_id or len(session_id) > 255:
            return self._error_result(
                "InvalidSessionId",
                "session_id must contain 1 to 255 characters",
                output_limit_chars,
            )
        limit, limit_was_capped = self._output_limit(output_limit_chars)
        async with self._lock_for(session_id), self.filesystem_guard():
            return await self._execute_locked(
                session_id,
                code,
                file_path,
                limit,
                limit_was_capped,
            )

    async def _execute_locked(
        self,
        session_id: str,
        code: str | None,
        file_path: str | None,
        limit: int,
        limit_was_capped: bool,
    ) -> ExecutionResult:
        """Execute while the session and optional file locks are held.

        Args:
            session_id: Locked session key.
            code: Direct or follow-up code.
            file_path: Optional virtual Python file.
            limit: Effective output limit.
            limit_was_capped: Whether the hard cap changed the request.

        Returns:
            A structured execution result.
        """
        stored = await asyncio.to_thread(
            self.store.load,
            session_id,
            self.config.session_ttl_seconds,
        )
        restored = stored is not None
        if stored is not None and stored.monty_version != self.monty_version:
            return self._result(
                error_type="SessionVersionError",
                error_message=(
                    f"The session uses Monty {stored.monty_version}, but this "
                    f"process uses {self.monty_version}. Reset the session."
                ),
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=True,
                revision=stored.revision,
            )

        streams = CollectStreams(max_bytes=self.config.max_stream_bytes)
        result: Any = None
        execution_error: MontyError | None = None
        terminal_resource_error = False
        state: bytes | None = None
        try:
            async with self._require_pool().checkout(
                script_name=f"<arenero:{session_id}>",
                limits=self._resource_limits(),
                type_check=self.config.type_check,
                type_check_stubs=self._type_check_stubs,
                type_check_format=self.config.type_check_format,
            ) as session:
                if stored is not None:
                    await session.load_session(stored.state)
                try:
                    result = await self._run(
                        session,
                        streams,
                        code,
                        file_path,
                        self._operation_external_lookup(),
                    )
                except InfrastructureErrors:
                    raise
                except MontyError as exc:
                    execution_error = exc
                    terminal_resource_error = _is_terminal_resource_error(exc)
                if self.store.stores_state and not terminal_resource_error:
                    state = await session.dump()
        except InfrastructureErrors as exc:
            if isinstance(exc, MontyCrashedError):
                await asyncio.to_thread(self.store.delete, session_id)
            return self._result(
                streams=streams,
                error_type=type(exc).__name__,
                error_message=str(exc),
                retryable=True,
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
            )
        except TimeoutError as exc:
            return self._result(
                streams=streams,
                error_type="WorkerCheckoutTimeout",
                error_message=str(exc),
                retryable=True,
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
            )
        except MontyError as exc:
            return self._result(
                streams=streams,
                error_type="SessionRestoreError",
                error_message=f"{exc}. Reset the session before retrying.",
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
            )

        if terminal_resource_error:
            await asyncio.to_thread(self.store.delete, session_id)
            return self._result(
                result=result,
                streams=streams,
                error_type=type(execution_error).__name__,
                error_message=str(execution_error),
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
            )
        if state is None:
            return self._result(
                result=result,
                streams=streams,
                error_type=type(execution_error).__name__ if execution_error else None,
                error_message=str(execution_error) if execution_error else None,
                limit=limit,
                limit_was_capped=limit_was_capped,
            )
        if len(state) > self.config.max_state_bytes:
            return self._result(
                streams=streams,
                error_type="SessionStateTooLarge",
                error_message=(
                    f"The session state is {len(state)} bytes. The limit is "
                    f"{self.config.max_state_bytes} bytes. The previous state was kept."
                ),
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
            )
        try:
            revision = await asyncio.to_thread(
                self.store.save,
                session_id,
                state,
                self.monty_version,
                stored.revision if stored else None,
            )
        except ConcurrentSessionUpdate:
            return self._result(
                streams=streams,
                error_type="ConcurrentSessionUpdate",
                error_message="The session changed during execution. Retry the tool call.",
                retryable=True,
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
            )
        return self._result(
            result=result,
            streams=streams,
            error_type=type(execution_error).__name__ if execution_error else None,
            error_message=str(execution_error) if execution_error else None,
            limit=limit,
            limit_was_capped=limit_was_capped,
            state_saved=True,
            state_restored=restored,
            revision=revision,
        )

    async def _run(
        self,
        session: AsyncMontySession,
        streams: CollectStreams,
        code: str | None,
        file_path: str | None,
        external_lookup: dict[str, Any] | None,
    ) -> Any:
        """Run direct code or a virtual file followed by direct code.

        Args:
            session: Active asynchronous Monty session.
            streams: Output collector shared by all feeds.
            code: Direct or follow-up code.
            file_path: Optional virtual file to read and run first.
            external_lookup: Host functions for this operation.

        Returns:
            Value from the last feed.
        """
        await self._feed(
            session,
            streams,
            self._filesystem_prelude,
            external_lookup,
            skip_type_check=True,
        )
        if file_path is None:
            return await self._feed(session, streams, code or "", external_lookup)
        source = await self._feed(
            session,
            streams,
            f"open({json.dumps(file_path)}).read()",
            external_lookup,
        )
        result = await self._feed(session, streams, source, external_lookup)
        if code is not None:
            result = await self._feed(session, streams, code, external_lookup)
        return result

    async def _feed(
        self,
        session: AsyncMontySession,
        streams: CollectStreams,
        code: str,
        external_lookup: dict[str, Any] | None,
        *,
        skip_type_check: bool = False,
    ) -> Any:
        """Run one asynchronous Monty feed.

        Args:
            session: Active asynchronous Monty session.
            streams: Output collector for this operation.
            code: Monty-compatible Python source.
            external_lookup: Host values and functions.
            skip_type_check: Skip checking trusted support code.

        Returns:
            Feed result.
        """
        return await session.feed_run(
            code,
            external_lookup=external_lookup,
            print_callback=streams,
            mount=self.mounts,
            os=self.os_handler,
            skip_type_check=skip_type_check,
        )

    def _operation_external_lookup(self) -> dict[str, Any] | None:
        """Build host functions with a fresh MCP call budget.

        Returns:
            Host functions for one operation, or ``None``.
        """
        lookup = dict(self.external_lookup)
        if self.mcp_hub is not None:
            lookup.update(self.mcp_hub.async_host_functions())
        return lookup or None

    @asynccontextmanager
    async def filesystem_guard(self) -> AsyncIterator[None]:
        """Serialize shared filesystem work inside this kernel.

        Yields:
            Control while the kernel owns its filesystem lock.
        """
        if not self._filesystem_enabled:
            yield
            return
        async with self._filesystem_lock:
            yield

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        """Get the event-loop-local lock for one session.

        Args:
            session_id: Session key that owns the lock.

        Returns:
            Session lock.
        """
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _resource_limits(self) -> ResourceLimits:
        """Build Monty resource limits from configuration.

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
            requested: Caller limit, or ``None`` for the default.

        Returns:
            Effective limit and whether the hard cap changed it.
        """
        if requested is None:
            return self.config.default_output_limit_chars, False
        effective = min(max(0, requested), self.config.max_output_limit_chars)
        return effective, effective != requested

    def _error_result(
        self,
        error_type: str,
        message: str,
        requested_limit: int | None,
    ) -> ExecutionResult:
        """Build an error result before execution starts.

        Args:
            error_type: Stable error category.
            message: Human-readable details.
            requested_limit: Caller output limit.

        Returns:
            A structured failed execution result.
        """
        limit, limit_was_capped = self._output_limit(requested_limit)
        return self._result(
            error_type=error_type,
            error_message=message,
            limit=limit,
            limit_was_capped=limit_was_capped,
        )

    def _result(
        self,
        *,
        result: Any = None,
        streams: CollectStreams | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        retryable: bool = False,
        limit: int,
        limit_was_capped: bool,
        state_saved: bool = False,
        state_restored: bool = False,
        revision: int | None = None,
    ) -> ExecutionResult:
        """Build and trim one native async execution result.

        Args:
            result: Value returned by the final Monty feed.
            streams: Captured standard streams.
            error_type: Optional stable error category.
            error_message: Optional error details.
            retryable: Whether the caller can safely retry.
            limit: Effective output limit.
            limit_was_capped: Whether the hard cap changed the request.
            state_saved: Whether state was saved.
            state_restored: Whether state was loaded.
            revision: Saved revision, when available.

        Returns:
            Public execution result.
        """
        public_result, _ = _prepare_execution_result(
            result=result,
            streams=streams or CollectStreams(max_bytes=self.config.max_stream_bytes),
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
            limit=limit,
            limit_was_capped=limit_was_capped,
            state_saved=state_saved,
            state_restored=state_restored,
            revision=revision,
            output_file=None,
        )
        return public_result

    def _require_pool(self) -> AsyncMonty:
        """Return the open worker pool.

        Returns:
            Open asynchronous Monty pool.
        """
        if self._pool is None:
            raise RuntimeError("the kernel is not open")
        return self._pool

    def _close_owned_mount(self) -> None:
        """Close the kernel-owned workspace mount once."""
        if self._owned_mount is not None:
            self._owned_mount.__exit__(None, None, None)
            self._owned_mount = None
