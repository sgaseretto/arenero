"""Native asynchronous persistent Monty kernel."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import time
import uuid
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

from .events import EventSink, emit_event
from .filesystem import FilesystemConfig, build_filesystem_prelude
from .hosted_tools import HostedToolSet
from .kernel import (
    ExecutionResult,
    InfrastructureErrors,
    KernelConfig,
    StatePolicy,
    _ExecutionMetrics,
    _is_terminal_resource_error,
    _mount_list,
    _prepare_execution_result,
    _type_check_stubs,
    _validate_session_id,
)
from .modules import VirtualModuleConfig, VirtualModuleResolver
from .store import (
    ConcurrentSessionUpdate,
    InMemorySessionStore,
    SessionInfo,
    SessionStore,
)
from .workspace import WorkspaceConfig

if TYPE_CHECKING:
    from .mcp import McpHub
    from .session import AsyncSessionHandle


@asynccontextmanager
async def _no_filesystem_guard() -> AsyncIterator[None]:
    """Yield without taking the kernel-wide filesystem lock.

    Yields:
        One no-op async context.
    """

    yield


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
        hosted_tools: HostedToolSet | None = None,
        bootstrap_code: str | None = None,
        bootstrap_files: tuple[str, ...] = (),
        event_sink: EventSink | None = None,
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
            hosted_tools: Optional schema-aware host functions for Monty code.
            bootstrap_code: Trusted code run once when a new session starts.
            bootstrap_files: Virtual Python files run once for each new session.
            event_sink: Optional observer for safe execution and session metadata.
        """
        self.store = store if store is not None else InMemorySessionStore()
        self.config = config or KernelConfig()
        self.external_lookup = dict(external_lookup or {})
        self.os_handler = os_handler
        self.filesystem_config = filesystem_config or FilesystemConfig()
        self.workspace = workspace
        self.mcp_hub = mcp_hub
        self.hosted_tools = hosted_tools or HostedToolSet()
        self.bootstrap_code = bootstrap_code
        self.bootstrap_files = tuple(bootstrap_files)
        self.event_sink = event_sink
        self.mcp_artifact_directory: str | None = None
        if any(not path.startswith("/") for path in self.bootstrap_files):
            raise ValueError("bootstrap file paths must be absolute virtual paths")
        if mcp_hub is not None:
            collisions = (
                self.external_lookup.keys() & mcp_hub.async_host_functions().keys()
            )
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
        host_stubs = self.config.type_check_stubs
        if mcp_hub is not None:
            host_stubs = "\n".join(
                value for value in (host_stubs, mcp_hub.async_type_check_stubs) if value
            )
        self._base_type_check_stubs = host_stubs
        self._type_check_stubs = self._operation_type_check_stubs(self.hosted_tools)
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

    async def reset(self, session_id: str, *, filesystem: bool = True) -> bool:
        """Delete saved Python state for one session.

        Args:
            session_id: Session key to reset.
            filesystem: Take the filesystem lock when true.

        Returns:
            ``True`` when saved state existed.
        """
        file_guard = self.filesystem_guard() if filesystem else _no_filesystem_guard()
        async with self._lock_for(session_id), file_guard:
            deleted = await asyncio.to_thread(self.store.delete, session_id)
        emit_event(
            self.event_sink,
            "session",
            "reset",
            session_id=session_id,
            existed=deleted,
        )
        return deleted

    def session(self, session_id: str) -> AsyncSessionHandle:
        """Bind this kernel to one session ID.

        Args:
            session_id: Session key with 1 to 255 characters.

        Returns:
            A bound ``AsyncSessionHandle``.
        """
        from .session import AsyncSessionHandle

        return AsyncSessionHandle(self, session_id)

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

    async def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Read safe metadata for one current session.

        Args:
            session_id: Session key to inspect.

        Returns:
            Session metadata, or ``None`` when absent or expired.
        """
        stored = await asyncio.to_thread(
            self.store.load,
            session_id,
            self.config.session_ttl_seconds,
        )
        return SessionInfo.from_stored(session_id, stored) if stored else None

    async def fork_session(self, source_id: str, target_id: str) -> SessionInfo:
        """Copy one saved Python snapshot to a new session.

        Args:
            source_id: Existing session to copy.
            target_id: New session that receives the snapshot.

        Returns:
            Metadata for the new session.
        """
        _validate_session_id(source_id)
        _validate_session_id(target_id)
        if source_id == target_id:
            raise ValueError("source and target session IDs must differ")
        if not self.store.stores_state:
            raise RuntimeError("the configured store does not save session state")
        first, second = sorted((source_id, target_id))
        async with self._lock_for(first), self._lock_for(second):
            source = await asyncio.to_thread(
                self.store.load,
                source_id,
                self.config.session_ttl_seconds,
            )
            if source is None:
                raise ValueError(f"source session does not exist: {source_id}")
            if source.monty_version != self.monty_version:
                raise ValueError("source session uses a different Monty version")
            target = await asyncio.to_thread(self.store.load, target_id)
            if target is not None:
                raise ValueError(f"target session already exists: {target_id}")
            await asyncio.to_thread(
                self.store.save,
                target_id,
                source.state,
                source.monty_version,
                None,
            )
        info = await self.get_session_info(target_id)
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

    async def execute(
        self,
        session_id: str,
        code: str,
        output_limit_chars: int | None = None,
        *,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
        inputs: Mapping[str, Any] | None = None,
        filesystem: bool = True,
    ) -> ExecutionResult:
        """Execute code in one persistent session.

        Args:
            session_id: Session key with 1 to 255 characters.
            code: Monty-compatible Python code.
            output_limit_chars: Optional caller output limit.
            state_policy: Save state ``always``, only on ``success``, or ``never``.
            restart: Reset state before this execution when true.
            hosted_tools: Optional tools available for this operation only.
            inputs: Optional plain values bound before this feed.
            filesystem: Allow configured mounts and virtual file callbacks.

        Returns:
            A structured execution result.
        """
        return await self._execute(
            session_id,
            code,
            output_limit_chars,
            file_path=None,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            inputs=inputs,
            filesystem=filesystem,
        )

    async def execute_file(
        self,
        session_id: str,
        file_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> ExecutionResult:
        """Execute a virtual Python file and optional follow-up code.

        Args:
            session_id: Session key with 1 to 255 characters.
            file_path: Path to a readable virtual Python file.
            code: Optional code to run after the file.
            output_limit_chars: Optional caller output limit.
            state_policy: Save state ``always``, only on ``success``, or ``never``.
            restart: Reset state before this execution when true.
            hosted_tools: Optional tools available for this operation only.

        Returns:
            A structured execution result.
        """
        return await self._execute(
            session_id,
            code,
            output_limit_chars,
            file_path=file_path,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    async def download_file(
        self,
        path: str,
        max_bytes: int | None = None,
    ) -> bytes:
        """Read one binary file without blocking the event loop.

        Args:
            path: Absolute virtual POSIX file path.
            max_bytes: Optional caller limit below the filesystem hard limit.

        Returns:
            File content as bytes.
        """

        if not path.startswith("/"):
            raise ValueError("path must be an absolute virtual POSIX path")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be at least 0")
        limit = self.filesystem_config.max_transfer_bytes
        if max_bytes is not None:
            limit = min(limit, max_bytes)
        source = (
            "from pathlib import Path\n"
            f"_arenero_transfer_path = Path({json.dumps(path)})\n"
            "_arenero_transfer_size = _arenero_transfer_path.stat().st_size\n"
            f"if _arenero_transfer_size > {limit}:\n"
            "    raise ValueError('file exceeds the transfer limit')\n"
            "_arenero_transfer_path.read_bytes()"
        )
        await self.open()
        async with (
            self.filesystem_guard(),
            self._require_pool().checkout(
                script_name="<arenero:download>",
                limits=self._resource_limits(),
                type_check=False,
            ) as session,
        ):
            result = await session.feed_run(
                source,
                mount=self.mounts,
                os=self.os_handler,
            )
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

    async def execute_module(
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

        loop = asyncio.get_running_loop()

        def load(path: str) -> str | None:
            """Read one candidate through the active asynchronous kernel.

            Args:
                path: Absolute candidate file path.

            Returns:
                Source text, or ``None`` when the file is not readable.
            """

            future = asyncio.run_coroutine_threadsafe(self.download_file(path), loop)
            try:
                return future.result().decode()
            except MontyError:
                return None

        def build() -> str:
            """Resolve source while the event loop services virtual file reads.

            Returns:
                Complete generated Monty source.
            """

            bundle = VirtualModuleResolver(load, module_config).bundle(entry_path, code)
            return bundle.code

        try:
            source = await asyncio.to_thread(build)
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            return self._error_result(
                "VirtualModuleError",
                str(exc),
                output_limit_chars,
                state_policy=state_policy,
            )
        return await self._execute(
            session_id,
            source,
            output_limit_chars,
            file_path=None,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
            skip_type_check=True,
        )

    async def _execute(
        self,
        session_id: str,
        code: str | None,
        output_limit_chars: int | None,
        *,
        file_path: str | None,
        state_policy: StatePolicy,
        restart: bool,
        hosted_tools: HostedToolSet | None,
        inputs: Mapping[str, Any] | None = None,
        filesystem: bool = True,
        skip_type_check: bool = False,
    ) -> ExecutionResult:
        """Run feeds and save their final state once.

        Args:
            session_id: Session key used to load, lock, and save state.
            code: Direct or follow-up code.
            output_limit_chars: Optional caller output limit.
            file_path: Optional virtual Python file to execute first.
            state_policy: State save policy for this operation.
            restart: Whether to reset state before this operation.
            hosted_tools: Optional operation-only hosted tools.
            inputs: Optional plain values bound before the direct code feed.
            filesystem: Allow configured mounts and virtual file callbacks.
            skip_type_check: Disable checking for generated support code.

        Returns:
            A structured execution result.
        """
        await self.open()
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
        if state_policy not in {"always", "success", "never"}:
            result = self._error_result(
                "InvalidStatePolicy",
                "state_policy must be 'always', 'success', or 'never'",
                output_limit_chars,
                state_policy="never",
            )
            return self._finish_result(result, session_id, metrics)
        if not session_id or len(session_id) > 255:
            result = self._error_result(
                "InvalidSessionId",
                "session_id must contain 1 to 255 characters",
                output_limit_chars,
                state_policy=state_policy,
            )
            return self._finish_result(result, session_id, metrics)
        limit, limit_was_capped = self._output_limit(output_limit_chars)
        file_guard = self.filesystem_guard() if filesystem else _no_filesystem_guard()
        async with self._lock_for(session_id), file_guard:
            result = await self._execute_locked(
                session_id,
                code,
                file_path,
                limit,
                limit_was_capped,
                state_policy,
                restart,
                hosted_tools,
                inputs,
                filesystem,
                skip_type_check,
                metrics,
            )
        result["session_restarted"] = metrics.session_restarted
        result["host_calls"] = (
            metrics.hosted_scope.summary()
            if metrics.hosted_scope is not None
            else {"count": 0, "items": [], "truncated": False}
        )
        return self._finish_result(result, session_id, metrics)

    async def _execute_locked(
        self,
        session_id: str,
        code: str | None,
        file_path: str | None,
        limit: int,
        limit_was_capped: bool,
        state_policy: StatePolicy,
        restart: bool,
        hosted_tools: HostedToolSet | None,
        inputs: Mapping[str, Any] | None,
        filesystem: bool,
        skip_type_check: bool,
        metrics: _ExecutionMetrics,
    ) -> ExecutionResult:
        """Execute while the session and optional file locks are held.

        Args:
            session_id: Locked session key.
            code: Direct or follow-up code.
            file_path: Optional virtual Python file.
            limit: Effective output limit.
            limit_was_capped: Whether the hard cap changed the request.
            state_policy: State save policy for this operation.
            restart: Whether to delete saved state before loading.
            hosted_tools: Optional operation-only hosted tools.
            inputs: Optional plain values bound before the direct code feed.
            filesystem: Allow configured mounts and virtual file callbacks.
            skip_type_check: Disable checking for generated support code.
            metrics: Mutable execution statistics.

        Returns:
            A structured execution result.
        """
        if restart:
            existed = await asyncio.to_thread(self.store.delete, session_id)
            metrics.session_restarted = True
            emit_event(
                self.event_sink,
                "session",
                "reset",
                session_id=session_id,
                existed=existed,
                execution_id=metrics.execution_id,
            )
        stored = await asyncio.to_thread(
            self.store.load,
            session_id,
            self.config.session_ttl_seconds,
        )
        restored = stored is not None
        metrics.state_before_bytes = len(stored.state) if stored else 0
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
            external_lookup = self._operation_external_lookup(metrics, operation_tools)
        except (TypeError, ValueError) as exc:
            return self._result(
                streams=streams,
                error_type="HostedToolConfigurationError",
                error_message=str(exc),
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
                state_policy=state_policy,
            )
        try:
            checkout_started = time.monotonic()
            async with self._require_pool().checkout(
                script_name=f"<arenero:{session_id}>",
                limits=self._resource_limits(),
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
                    await session.load_session(stored.state)
                phase = "bootstrap"
                try:
                    await self._prepare_session(
                        session,
                        streams,
                        external_lookup,
                        filesystem=filesystem,
                        new_session=stored is None,
                    )
                    phase = "execution"
                    result = await self._run(
                        session,
                        streams,
                        code,
                        file_path,
                        external_lookup,
                        inputs,
                        filesystem,
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
                    state = await session.dump()
                    metrics.state_after_bytes = len(state)
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
                state_policy=state_policy,
            )
        except TimeoutError as exc:
            metrics.checkout_wait_seconds = time.monotonic() - checkout_started
            return self._result(
                streams=streams,
                error_type="WorkerCheckoutTimeout",
                error_message=str(exc),
                retryable=True,
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
                state_policy=state_policy,
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
                state_policy=state_policy,
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
                state_policy=state_policy,
            )
        if state is None:
            return self._result(
                result=result,
                streams=streams,
                error_type=error_type_override
                or (type(execution_error).__name__ if execution_error else None),
                error_message=str(execution_error) if execution_error else None,
                limit=limit,
                limit_was_capped=limit_was_capped,
                state_restored=restored,
                revision=stored.revision if stored else None,
                state_policy=state_policy,
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
                state_policy=state_policy,
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
                state_policy=state_policy,
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
            state_policy=state_policy,
        )

    async def _prepare_session(
        self,
        session: AsyncMontySession,
        streams: CollectStreams,
        external_lookup: dict[str, Any] | None,
        *,
        filesystem: bool,
        new_session: bool,
    ) -> None:
        """Install helpers and run configured startup sources for a new session.

        Args:
            session: Active asynchronous Monty session.
            streams: Output collector shared by all feeds.
            external_lookup: Host functions for the operation.
            filesystem: Allow configured mounts and virtual file callbacks.
            new_session: Whether no saved snapshot was loaded.
        """
        await self._feed(
            session,
            streams,
            self._filesystem_prelude,
            external_lookup,
            filesystem=filesystem,
            skip_type_check=True,
        )
        if not new_session:
            return
        for path in self.bootstrap_files if filesystem else ():
            source = await self._feed(
                session,
                streams,
                f"open({json.dumps(path)}).read()",
                external_lookup,
                filesystem=filesystem,
            )
            await self._feed(
                session,
                streams,
                source,
                external_lookup,
                filesystem=filesystem,
            )
        if self.bootstrap_code:
            await self._feed(
                session,
                streams,
                self.bootstrap_code,
                external_lookup,
                filesystem=filesystem,
            )

    async def _run(
        self,
        session: AsyncMontySession,
        streams: CollectStreams,
        code: str | None,
        file_path: str | None,
        external_lookup: dict[str, Any] | None,
        inputs: Mapping[str, Any] | None,
        filesystem: bool,
    ) -> Any:
        """Run direct code or a virtual file followed by direct code.

        Args:
            session: Active asynchronous Monty session.
            streams: Output collector shared by all feeds.
            code: Direct or follow-up code.
            file_path: Optional virtual file to read and run first.
            external_lookup: Host functions for this operation.
            inputs: Optional values for a direct code feed.
            filesystem: Allow configured mounts and virtual file callbacks.

        Returns:
            Value from the last feed.
        """
        if file_path is None:
            return await self._feed(
                session,
                streams,
                code or "",
                external_lookup,
                inputs=inputs,
                filesystem=filesystem,
            )
        source = await self._feed(
            session,
            streams,
            f"open({json.dumps(file_path)}).read()",
            external_lookup,
            filesystem=filesystem,
        )
        result = await self._feed(
            session,
            streams,
            source,
            external_lookup,
            filesystem=filesystem,
        )
        if code is not None:
            result = await self._feed(
                session,
                streams,
                code,
                external_lookup,
                filesystem=filesystem,
            )
        return result

    async def _feed(
        self,
        session: AsyncMontySession,
        streams: CollectStreams,
        code: str,
        external_lookup: dict[str, Any] | None,
        *,
        inputs: Mapping[str, Any] | None = None,
        filesystem: bool = True,
        skip_type_check: bool = False,
    ) -> Any:
        """Run one asynchronous Monty feed.

        Args:
            session: Active asynchronous Monty session.
            streams: Output collector for this operation.
            code: Monty-compatible Python source.
            external_lookup: Host values and functions.
            inputs: Optional plain values bound for this feed.
            filesystem: Allow configured mounts and virtual file callbacks.
            skip_type_check: Skip checking trusted support code.

        Returns:
            Feed result.
        """
        return await session.feed_run(
            code,
            inputs=dict(inputs) if inputs is not None else None,
            external_lookup=external_lookup,
            print_callback=streams,
            mount=self.mounts if filesystem else None,
            os=self.os_handler if filesystem else None,
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
        """Build async host functions with fresh operation budgets.

        Args:
            metrics: Current execution statistics.
            hosted_tools: Tools selected for this operation.

        Returns:
            Host functions for one complete operation, or ``None``.
        """

        lookup = dict(self.external_lookup)
        if self.mcp_hub is not None:
            lookup.update(self.mcp_hub.async_host_functions())
        if hosted_tools.tools:
            scope = hosted_tools.async_scope(metrics.execution_id, self.event_sink)
            hosted_lookup = scope.async_lookup()
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
        """Build type stubs for one operation's exact async host functions.

        Args:
            hosted_tools: Tools selected for this operation.

        Returns:
            Filesystem, MCP, application, and hosted-tool declarations.
        """

        hosted_stubs = hosted_tools.type_check_stubs(async_mode=True)
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
        state_policy: StatePolicy = "always",
    ) -> ExecutionResult:
        """Build an error result before execution starts.

        Args:
            error_type: Stable error category.
            message: Human-readable details.
            requested_limit: Caller output limit.
            state_policy: State save policy requested by the caller.

        Returns:
            A structured failed execution result.
        """
        limit, limit_was_capped = self._output_limit(requested_limit)
        return self._result(
            error_type=error_type,
            error_message=message,
            limit=limit,
            limit_was_capped=limit_was_capped,
            state_policy=state_policy,
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
        state_policy: StatePolicy = "always",
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
            state_policy: State save policy for this operation.

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
            state_policy=state_policy,
        )
        return public_result

    def _finish_result(
        self,
        result: ExecutionResult,
        session_id: str,
        metrics: _ExecutionMetrics,
    ) -> ExecutionResult:
        """Finish statistics and emit the final async execution event.

        Args:
            result: Prepared public execution result.
            session_id: Session used by the operation.
            metrics: Mutable execution statistics.

        Returns:
            The same result with final statistics.
        """
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
