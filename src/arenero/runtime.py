"""Small lifecycle owners for configured Arenero components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from .events import emit_event

if TYPE_CHECKING:
    from .async_kernel import AsyncPersistentMontyKernel
    from .kernel import PersistentMontyKernel
    from .session import AsyncSessionHandle, SessionHandle


class AreneroRuntime:
    """Own one synchronous kernel and its optional MCP hub."""

    def __init__(
        self, kernel: PersistentMontyKernel, *, close_mcp_hub: bool = True
    ) -> None:
        """Create a runtime around a configured kernel.

        Args:
            kernel: Configured synchronous kernel to own.
            close_mcp_hub: Close the attached MCP hub after the kernel.
        """
        self.kernel = kernel
        self.close_mcp_hub = close_mcp_hub
        self._closed = False

    def __enter__(self) -> Self:
        """Return this open runtime.

        Returns:
            This runtime.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Close owned components when the context ends."""
        self.close()

    def close(self) -> None:
        """Close the kernel, then its optional MCP hub, once."""
        if self._closed:
            return
        try:
            self.kernel.close()
        finally:
            if self.close_mcp_hub and self.kernel.mcp_hub is not None:
                self.kernel.mcp_hub.close()
            self._closed = True
            emit_event(self.kernel.event_sink, "runtime", "closed")

    def session(self, session_id: str) -> SessionHandle:
        """Bind the owned kernel to one session.

        Args:
            session_id: Session key.

        Returns:
            Bound synchronous session handle.
        """
        return self.kernel.session(session_id)

    def capabilities(self) -> dict[str, Any]:
        """Return the active capability report.

        Returns:
            JSON-compatible capability data.
        """
        return self.kernel.capabilities()

    def instructions(self) -> str:
        """Return generated model instructions.

        Returns:
            Plain model instructions for the active configuration.
        """
        return self.kernel.instructions()


class AsyncAreneroRuntime:
    """Own one native asynchronous kernel and its optional MCP hub."""

    def __init__(
        self,
        kernel: AsyncPersistentMontyKernel,
        *,
        close_mcp_hub: bool = True,
    ) -> None:
        """Create an async runtime around a configured kernel.

        Args:
            kernel: Configured native asynchronous kernel to own.
            close_mcp_hub: Close the attached MCP hub after the kernel.
        """
        self.kernel = kernel
        self.close_mcp_hub = close_mcp_hub
        self._closed = False

    async def __aenter__(self) -> Self:
        """Open and return this runtime.

        Returns:
            This runtime.
        """
        await self.kernel.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close owned components when the context ends."""
        await self.close()

    async def close(self) -> None:
        """Close the kernel, then its optional MCP hub, once."""
        if self._closed:
            return
        try:
            await self.kernel.close()
        finally:
            if self.close_mcp_hub and self.kernel.mcp_hub is not None:
                self.kernel.mcp_hub.close()
            self._closed = True
            emit_event(self.kernel.event_sink, "runtime", "closed")

    def session(self, session_id: str) -> AsyncSessionHandle:
        """Bind the owned kernel to one session.

        Args:
            session_id: Session key.

        Returns:
            Bound asynchronous session handle.
        """
        return self.kernel.session(session_id)

    def capabilities(self) -> dict[str, Any]:
        """Return the active capability report.

        Returns:
            JSON-compatible capability data.
        """
        return self.kernel.capabilities()

    def instructions(self) -> str:
        """Return generated model instructions.

        Returns:
            Plain model instructions for the active configuration.
        """
        return self.kernel.instructions()
