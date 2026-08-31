"""Bound session handles for concise and safer application code."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .hosted_tools import HostedToolSet
from .kernel import ExecutionResult, StatePolicy, _validate_session_id
from .modules import VirtualModuleConfig
from .store import SessionInfo

if TYPE_CHECKING:
    from .async_kernel import AsyncPersistentMontyKernel
    from .kernel import PersistentMontyKernel


class SessionHandle:
    """Bind a synchronous kernel to one session ID."""

    def __init__(self, kernel: PersistentMontyKernel, session_id: str) -> None:
        """Create one bound synchronous handle.

        Args:
            kernel: Kernel that owns the session.
            session_id: Session key with 1 to 255 characters.
        """
        _validate_session_id(session_id)
        self.kernel = kernel
        self.session_id = session_id

    def execute(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> ExecutionResult:
        """Execute code in this bound session.

        Args:
            code: Monty-compatible Python code.
            output_limit_chars: Optional caller output limit.
            output_file: Optional virtual path for trimmed output.
            state_policy: State save policy for this operation.
            restart: Reset this session before execution when true.
            hosted_tools: Optional tools for this operation only.

        Returns:
            Structured execution result.
        """
        return self.kernel.execute(
            self.session_id,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    def execute_file(
        self,
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
            file_path: Absolute virtual Python file path.
            code: Optional follow-up code.
            output_limit_chars: Optional caller output limit.
            output_file: Optional virtual path for trimmed output.
            state_policy: State save policy for this operation.
            restart: Reset this session before execution when true.
            hosted_tools: Optional tools for this operation only.

        Returns:
            Structured execution result.
        """
        return self.kernel.execute_file(
            self.session_id,
            file_path,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    def execute_cell(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        output_file: str | None = None,
        state_policy: StatePolicy = "always",
    ) -> dict[str, object]:
        """Execute a Python, shell, or small REPL command cell.

        Args:
            code: REPL cell source.
            output_limit_chars: Optional caller output limit.
            output_file: Optional virtual path for trimmed output.
            state_policy: State save policy for Python execution.

        Returns:
            Structured cell result.
        """
        from .cell import execute_repl_cell

        return execute_repl_cell(
            self.kernel,
            self.session_id,
            code,
            output_limit_chars,
            output_file=output_file,
            state_policy=state_policy,
        )

    def execute_module(
        self,
        entry_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        module_config: VirtualModuleConfig | None = None,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> ExecutionResult:
        """Resolve and run a static virtual module graph.

        Args:
            entry_path: Absolute virtual entry ``.py`` file.
            code: Optional follow-up code.
            output_limit_chars: Optional caller output limit.
            module_config: Optional module roots, aliases, and limits.
            state_policy: State save policy for this operation.
            restart: Reset this session before execution when true.
            hosted_tools: Optional tools for this operation only.

        Returns:
            Structured module execution result.
        """

        return self.kernel.execute_module(
            self.session_id,
            entry_path,
            code,
            output_limit_chars,
            module_config=module_config,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    def reset(self) -> bool:
        """Delete this session's saved Python state.

        Returns:
            ``True`` when saved state existed.
        """
        return self.kernel.reset(self.session_id)

    def info(self) -> SessionInfo | None:
        """Return safe metadata for this session.

        Returns:
            Session metadata, or ``None`` when no state is saved.
        """
        return self.kernel.get_session_info(self.session_id)

    def fork(self, target_id: str) -> SessionHandle:
        """Copy this session state and bind the new session.

        Args:
            target_id: New session key.

        Returns:
            Bound handle for the copied session.
        """
        self.kernel.fork_session(self.session_id, target_id)
        return SessionHandle(self.kernel, target_id)


class AsyncSessionHandle:
    """Bind a native asynchronous kernel to one session ID."""

    def __init__(self, kernel: AsyncPersistentMontyKernel, session_id: str) -> None:
        """Create one bound asynchronous handle.

        Args:
            kernel: Native asynchronous kernel.
            session_id: Session key with 1 to 255 characters.
        """
        _validate_session_id(session_id)
        self.kernel = kernel
        self.session_id = session_id

    async def execute(
        self,
        code: str,
        output_limit_chars: int | None = None,
        *,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> ExecutionResult:
        """Execute code in this bound session.

        Args:
            code: Monty-compatible Python code.
            output_limit_chars: Optional caller output limit.
            state_policy: State save policy for this operation.
            restart: Reset this session before execution when true.
            hosted_tools: Optional tools for this operation only.

        Returns:
            Structured execution result.
        """
        return await self.kernel.execute(
            self.session_id,
            code,
            output_limit_chars,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    async def execute_file(
        self,
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
            file_path: Absolute virtual Python file path.
            code: Optional follow-up code.
            output_limit_chars: Optional caller output limit.
            state_policy: State save policy for this operation.
            restart: Reset this session before execution when true.
            hosted_tools: Optional tools for this operation only.

        Returns:
            Structured execution result.
        """
        return await self.kernel.execute_file(
            self.session_id,
            file_path,
            code,
            output_limit_chars,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    async def reset(self) -> bool:
        """Delete this session's saved Python state.

        Returns:
            ``True`` when saved state existed.
        """
        return await self.kernel.reset(self.session_id)

    async def execute_module(
        self,
        entry_path: str,
        code: str | None = None,
        output_limit_chars: int | None = None,
        *,
        module_config: VirtualModuleConfig | None = None,
        state_policy: StatePolicy = "always",
        restart: bool = False,
        hosted_tools: HostedToolSet | None = None,
    ) -> ExecutionResult:
        """Resolve and run a static virtual module graph.

        Args:
            entry_path: Absolute virtual entry ``.py`` file.
            code: Optional follow-up code.
            output_limit_chars: Optional caller output limit.
            module_config: Optional module roots, aliases, and limits.
            state_policy: State save policy for this operation.
            restart: Reset this session before execution when true.
            hosted_tools: Optional tools for this operation only.

        Returns:
            Structured module execution result.
        """

        return await self.kernel.execute_module(
            self.session_id,
            entry_path,
            code,
            output_limit_chars,
            module_config=module_config,
            state_policy=state_policy,
            restart=restart,
            hosted_tools=hosted_tools,
        )

    async def info(self) -> SessionInfo | None:
        """Return safe metadata for this session.

        Returns:
            Session metadata, or ``None`` when no state is saved.
        """
        return await self.kernel.get_session_info(self.session_id)

    async def fork(self, target_id: str) -> AsyncSessionHandle:
        """Copy this session state and bind the new session.

        Args:
            target_id: New session key.

        Returns:
            Bound handle for the copied session.
        """
        await self.kernel.fork_session(self.session_id, target_id)
        return AsyncSessionHandle(self.kernel, target_id)
