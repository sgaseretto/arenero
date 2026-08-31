"""Optional MCP clients that Monty code can call through host functions."""

from __future__ import annotations

import asyncio
import json
import keyword
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from .events import EventSink, emit_event
from .hosted_tools import (
    HostedTool,
    HostedToolConfig,
    HostedToolSet,
    python_identifier,
)

try:
    import httpx2
    from mcp import Client, StdioServerParameters
    from mcp.client.streamable_http import streamable_http_client
except ModuleNotFoundError as exc:  # pragma: no cover - covered in a clean environment
    raise ImportError(
        "Install Arenero with the MCP extra: uv add 'arenero[mcp]'"
    ) from exc


MCP_TYPE_CHECK_STUBS = """
from typing import Any

def mcp_servers() -> dict[str, Any]: ...
def mcp_search(
    query: str,
    server: str | None = None,
    limit: int = 20,
) -> dict[str, Any]: ...
def mcp_describe(name: str) -> dict[str, Any]: ...
def mcp_promote(name: str, python_name: str | None = None) -> dict[str, Any]: ...
def mcp_refresh(server: str | None = None) -> dict[str, Any]: ...
def mcp_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]: ...
"""

ASYNC_MCP_TYPE_CHECK_STUBS = """
from typing import Any

async def mcp_servers() -> dict[str, Any]: ...
async def mcp_search(
    query: str,
    server: str | None = None,
    limit: int = 20,
) -> dict[str, Any]: ...
async def mcp_describe(name: str) -> dict[str, Any]: ...
async def mcp_promote(
    name: str,
    python_name: str | None = None,
) -> dict[str, Any]: ...
async def mcp_refresh(server: str | None = None) -> dict[str, Any]: ...
async def mcp_call(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]: ...
"""

MCP_REPL_INSTRUCTIONS = """Configured MCP tools are available from Python.
Use mcp_servers() to inspect server status, mcp_search(query) to discover tools,
mcp_describe(name) to read one input schema, mcp_promote(name) to make a
discovered tool a direct Python function on the next REPL execution, and
mcp_call(name, arguments) to call it. Use the qualified server__tool name returned by search. You can call
tools in loops and pass one tool result to another without returning intermediate
data to the model. mcp_refresh() reconnects configured servers and reloads the
catalog. The application chooses servers, credentials, limits, and allowlists.
You cannot add servers or use MCP resources and prompts from this API. Results
are plain bounded dictionaries. Check ok, error, and truncated before use.
"""

ASYNC_MCP_REPL_INSTRUCTIONS = """Configured MCP tools are available as async
Python functions. Use await mcp_servers() to inspect server status, await
mcp_search(query) to discover tools, await mcp_describe(name) to read one input
schema, await mcp_promote(name) to make it a direct function on the next REPL
execution, and await mcp_call(name, arguments) to call it. Use qualified server__tool
names. Await mcp_refresh() to reconnect servers. Calls return plain bounded
dictionaries. Check ok, error, and truncated before use.
"""


_SERVER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_Transport = Literal["stdio", "streamable-http", "in-process"]


@dataclass(frozen=True)
class McpServerConfig:
    """Describe one user-configured MCP server.

    Attributes:
        name: Stable namespace used before each tool name.
        transport: Configured MCP transport.
        command: Executable used by a standard-input server.
        args: Arguments passed to a standard-input server.
        env: Extra environment values passed to a standard-input server.
        cwd: Working directory for a standard-input server.
        url: Streamable HTTP endpoint.
        headers: Trusted HTTP headers. They are never returned to Monty.
        target: In-process MCP server used by tests or embedded applications.
    """

    name: str
    transport: _Transport
    command: str | None = None
    args: tuple[str, ...] = field(default=(), repr=False)
    env: Mapping[str, str] | None = field(default=None, repr=False, compare=False)
    cwd: Path | str | None = None
    url: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] | None = field(default=None, repr=False, compare=False)
    target: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that fields match the selected transport."""
        if not _SERVER_NAME.fullmatch(self.name) or "__" in self.name:
            raise ValueError(
                "MCP server names must use 1 to 64 letters, digits, '-' or '_', "
                "must start with a letter or digit, and must not contain '__'"
            )
        if self.transport == "stdio" and not self.command:
            raise ValueError("a stdio MCP server requires command")
        if self.transport == "streamable-http" and not self.url:
            raise ValueError("a streamable HTTP MCP server requires url")
        if self.transport == "in-process" and self.target is None:
            raise ValueError("an in-process MCP server requires target")
        if self.transport != "stdio" and (
            self.command or self.args or self.env or self.cwd
        ):
            raise ValueError("command, args, env, and cwd are only valid for stdio")
        if self.transport != "streamable-http" and (self.url or self.headers):
            raise ValueError("url and headers are only valid for streamable HTTP")
        if self.env is not None:
            object.__setattr__(self, "env", dict(self.env))
        if self.headers is not None:
            object.__setattr__(self, "headers", dict(self.headers))

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        *,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> Self:
        """Create a configured standard-input MCP server.

        Args:
            name: Stable server namespace.
            command: Trusted executable selected by the application.
            args: Trusted executable arguments.
            env: Extra environment values for the server process.
            cwd: Optional server working directory.

        Returns:
            A standard-input server configuration.
        """
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=tuple(args),
            env=env,
            cwd=cwd,
        )

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Self:
        """Create a configured Streamable HTTP MCP server.

        Args:
            name: Stable server namespace.
            url: Trusted Streamable HTTP endpoint selected by the application.
            headers: Optional trusted authentication or routing headers.

        Returns:
            A Streamable HTTP server configuration.
        """
        return cls(
            name=name,
            transport="streamable-http",
            url=url,
            headers=headers,
        )

    @classmethod
    def in_process(cls, name: str, target: Any) -> Self:
        """Create an in-process MCP server configuration.

        Args:
            name: Stable server namespace.
            target: MCP ``Server`` or ``MCPServer`` instance.

        Returns:
            An in-process server configuration.
        """
        return cls(name=name, transport="in-process", target=target)


@dataclass(frozen=True)
class McpHubConfig:
    """Set MCP discovery, call, and transfer limits.

    Attributes:
        connect_timeout_seconds: Maximum time to open or refresh one server.
        call_timeout_seconds: Maximum time for one tool call.
        max_calls_per_operation: Tool calls allowed in one kernel operation.
        max_total_call_seconds_per_operation: Total MCP tool wait allowed in one
            kernel operation.
        max_concurrent_calls: Tool calls allowed across the hub at one time.
        max_argument_bytes: Maximum JSON argument size for one tool call.
        max_result_bytes: Maximum JSON result size returned to Monty.
        max_schema_bytes: Maximum schema data kept for one discovered tool.
        max_tools: Maximum tools kept across all configured servers.
        max_search_results: Hard maximum results from one discovery search.
        allowed_tools: Optional set of qualified ``server__tool`` names.
        retry_transport_errors: Whether one failed call reconnects and retries.
            This is off by default because an automatic retry can repeat a side
            effect when a server completed the first request but lost its reply.
    """

    connect_timeout_seconds: float = 15.0
    call_timeout_seconds: float = 30.0
    max_calls_per_operation: int = 20
    max_total_call_seconds_per_operation: float = 60.0
    max_concurrent_calls: int = 8
    max_argument_bytes: int = 100_000
    max_result_bytes: int = 1_000_000
    max_schema_bytes: int = 100_000
    max_tools: int = 1_000
    max_search_results: int = 50
    allowed_tools: frozenset[str] | None = None
    retry_transport_errors: bool = False

    def __post_init__(self) -> None:
        """Reject limits that cannot provide a usable MCP bridge."""
        numeric_values = (
            self.connect_timeout_seconds,
            self.call_timeout_seconds,
            self.max_calls_per_operation,
            self.max_total_call_seconds_per_operation,
            self.max_concurrent_calls,
            self.max_argument_bytes,
            self.max_result_bytes,
            self.max_schema_bytes,
            self.max_tools,
            self.max_search_results,
        )
        if any(value <= 0 for value in numeric_values):
            raise ValueError("all MCP limits must be greater than 0")
        if self.max_result_bytes < 512:
            raise ValueError("max_result_bytes must be at least 512")


@dataclass
class _ServerState:
    """Keep one live client and its safe connection status."""

    config: McpServerConfig
    client: Client | None = None
    stack: AsyncExitStack | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ToolEntry:
    """Keep one normalized MCP tool catalog entry."""

    server: str
    name: str
    qualified_name: str
    title: str | None
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any] | None
    schema_truncated: bool

    def summary(self) -> dict[str, Any]:
        """Return the small tool form used by search results.

        Returns:
            Safe tool identity and descriptive fields.
        """
        return {
            "name": self.qualified_name,
            "server": self.server,
            "tool": self.name,
            "title": self.title,
            "description": self.description,
        }

    def description_data(self) -> dict[str, Any]:
        """Return the full bounded tool description.

        Returns:
            Tool identity, schemas, annotations, and truncation state.
        """
        return {
            **self.summary(),
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "annotations": self.annotations,
            "schema_truncated": self.schema_truncated,
        }


class McpHub:
    """Manage configured MCP clients for sync and async Monty host calls."""

    instructions = MCP_REPL_INSTRUCTIONS
    async_instructions = ASYNC_MCP_REPL_INSTRUCTIONS
    type_check_stubs = MCP_TYPE_CHECK_STUBS
    async_type_check_stubs = ASYNC_MCP_TYPE_CHECK_STUBS

    def __init__(
        self,
        servers: Sequence[McpServerConfig],
        config: McpHubConfig | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        """Create an MCP hub without opening its connections.

        Args:
            servers: User-configured MCP servers. Names must be unique.
            config: Discovery, call, concurrency, and transfer limits.
            event_sink: Optional observer for safe MCP metadata.
        """
        server_list = list(servers)
        names = [server.name for server in server_list]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique")
        self.config = config or McpHubConfig()
        self.event_sink = event_sink
        self._states = {
            server.name: _ServerState(config=server) for server in server_list
        }
        self._tools: dict[str, _ToolEntry] = {}
        self._catalog_truncated = False
        self._promoted: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._start_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._call_semaphore: asyncio.Semaphore | None = None

    def __enter__(self) -> Self:
        """Open configured servers and return this hub.

        Returns:
            This open MCP hub.
        """
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close all MCP clients and the private event loop."""
        self.close()

    def open(self) -> None:
        """Start the private async loop and connect all configured servers."""
        with self._start_lock:
            if self._closed:
                raise RuntimeError("the MCP hub is closed")
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_event_loop,
                name="arenero-mcp",
                daemon=True,
            )
            self._thread.start()
            self._loop_ready.wait()
            self._submit(
                self._open_all(),
                self.config.connect_timeout_seconds * max(1, len(self._states)),
            )
            self._started = True

    def close(self) -> None:
        """Close all clients and stop the private event loop once."""
        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
            if loop is None or thread is None:
                return
            try:
                self._submit(
                    self._close_all(),
                    self.config.connect_timeout_seconds * max(1, len(self._states)),
                )
            finally:
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=self.config.connect_timeout_seconds)

    def host_functions(self) -> Mapping[str, Callable[..., Any]]:
        """Create MCP host functions with a fresh per-operation call budget.

        Returns:
            Functions for server listing, search, description, refresh, and calls.
        """
        calls = 0
        deadline = time.monotonic() + self.config.max_total_call_seconds_per_operation

        def mcp_servers() -> dict[str, Any]:
            """List configured MCP servers and safe connection status."""
            return self.servers()

        def mcp_search(
            query: str,
            server: str | None = None,
            limit: int = 20,
        ) -> dict[str, Any]:
            """Search the bounded MCP tool catalog."""
            return self.search(query, server=server, limit=limit)

        def mcp_describe(name: str) -> dict[str, Any]:
            """Get the bounded schema for one MCP tool."""
            return self.describe(name)

        def mcp_promote(name: str, python_name: str | None = None) -> dict[str, Any]:
            """Promote one discovered MCP tool for the next REPL execution."""

            return self.promote(name, python_name=python_name)

        def mcp_refresh(server: str | None = None) -> dict[str, Any]:
            """Reconnect and refresh one server or the full catalog."""
            return self.refresh(server)

        def mcp_call(
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Call one discovered MCP tool within this operation's limit."""
            nonlocal calls
            calls += 1
            if calls > self.config.max_calls_per_operation:
                return _error_result(
                    "McpCallLimitExceeded",
                    (
                        "This kernel operation exceeded "
                        f"{self.config.max_calls_per_operation} MCP tool calls."
                    ),
                    tool=name,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _error_result(
                    "McpOperationTimeout",
                    "This kernel operation exhausted its total MCP call time.",
                    tool=name,
                    retryable=True,
                )
            return self._call_bounded(
                name,
                arguments,
                min(remaining, self.config.call_timeout_seconds),
            )

        return {
            "mcp_servers": mcp_servers,
            "mcp_search": mcp_search,
            "mcp_describe": mcp_describe,
            "mcp_promote": mcp_promote,
            "mcp_refresh": mcp_refresh,
            "mcp_call": mcp_call,
        }

    def async_host_functions(self) -> Mapping[str, Callable[..., Any]]:
        """Create async MCP host functions with a fresh operation budget.

        Returns:
            Awaitable functions for discovery, refresh, and tool calls.
        """
        calls = 0
        deadline = time.monotonic() + self.config.max_total_call_seconds_per_operation

        async def mcp_servers() -> dict[str, Any]:
            """List configured MCP servers and safe connection status."""
            return await self.async_servers()

        async def mcp_search(
            query: str,
            server: str | None = None,
            limit: int = 20,
        ) -> dict[str, Any]:
            """Search the bounded MCP tool catalog."""
            return await self.async_search(query, server=server, limit=limit)

        async def mcp_describe(name: str) -> dict[str, Any]:
            """Get the bounded schema for one MCP tool."""
            return await self.async_describe(name)

        async def mcp_promote(
            name: str, python_name: str | None = None
        ) -> dict[str, Any]:
            """Promote one discovered MCP tool for the next REPL execution."""

            return await self.async_promote(name, python_name=python_name)

        async def mcp_refresh(server: str | None = None) -> dict[str, Any]:
            """Reconnect and refresh one server or the full catalog."""
            return await self.async_refresh(server)

        async def mcp_call(
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Call one discovered MCP tool within this operation's limit."""
            nonlocal calls
            calls += 1
            if calls > self.config.max_calls_per_operation:
                return _error_result(
                    "McpCallLimitExceeded",
                    (
                        "This kernel operation exceeded "
                        f"{self.config.max_calls_per_operation} MCP tool calls."
                    ),
                    tool=name,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _error_result(
                    "McpOperationTimeout",
                    "This kernel operation exhausted its total MCP call time.",
                    tool=name,
                    retryable=True,
                )
            return await self._async_call_bounded(
                name,
                arguments,
                min(remaining, self.config.call_timeout_seconds),
            )

        return {
            "mcp_servers": mcp_servers,
            "mcp_search": mcp_search,
            "mcp_describe": mcp_describe,
            "mcp_promote": mcp_promote,
            "mcp_refresh": mcp_refresh,
            "mcp_call": mcp_call,
        }

    async def async_servers(self) -> dict[str, Any]:
        """List configured server status without blocking the caller loop.

        Returns:
            A successful result with server status and tool counts.
        """
        await asyncio.to_thread(self.open)
        return await self._asubmit(
            self._server_data(), self.config.connect_timeout_seconds
        )

    async def async_search(
        self,
        query: str,
        *,
        server: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search tools without blocking the caller event loop.

        Args:
            query: Case-insensitive words or phrase.
            server: Optional configured server namespace.
            limit: Requested result count before the hard cap.

        Returns:
            A successful result with ranked tool summaries.
        """
        await asyncio.to_thread(self.open)
        if not isinstance(query, str):
            return _error_result("InvalidMcpSearch", "query must be a string")
        if server is not None and server not in self._states:
            return _error_result("UnknownMcpServer", f"Unknown MCP server: {server}")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return _error_result("InvalidMcpSearch", "limit must be a positive integer")
        hard_limit = min(limit, self.config.max_search_results)
        return await self._asubmit(
            self._search_data(query, server, hard_limit, limit > hard_limit),
            self.config.connect_timeout_seconds,
        )

    async def async_describe(self, name: str) -> dict[str, Any]:
        """Describe one tool without blocking the caller event loop.

        Args:
            name: Qualified or unambiguous short tool name.

        Returns:
            Bounded tool data or a structured lookup error.
        """
        await asyncio.to_thread(self.open)
        return await self._asubmit(
            self._describe_data(name), self.config.connect_timeout_seconds
        )

    async def async_promote(
        self, name: str, *, python_name: str | None = None
    ) -> dict[str, Any]:
        """Promote one MCP tool without blocking the caller event loop.

        Args:
            name: Qualified or unambiguous MCP tool name.
            python_name: Optional explicit Python alias.

        Returns:
            Promotion result with the next-call function declaration.
        """

        await asyncio.to_thread(self.open)
        return await self._asubmit(
            self._promote_data(name, python_name), self.config.connect_timeout_seconds
        )

    async def async_refresh(self, server: str | None = None) -> dict[str, Any]:
        """Refresh server connections without blocking the caller event loop.

        Args:
            server: Optional server namespace. ``None`` refreshes all servers.

        Returns:
            Updated server and catalog status.
        """
        await asyncio.to_thread(self.open)
        if server is not None and server not in self._states:
            return _error_result("UnknownMcpServer", f"Unknown MCP server: {server}")
        timeout = self.config.connect_timeout_seconds * (
            1 if server is not None else max(1, len(self._states))
        )
        return await self._asubmit(self._refresh_data(server), timeout)

    async def async_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call one MCP tool without blocking the caller event loop.

        Args:
            name: Qualified or unambiguous short tool name.
            arguments: JSON-compatible tool arguments.

        Returns:
            A bounded normalized MCP result.
        """
        return await self._async_call_bounded(
            name, arguments, self.config.call_timeout_seconds
        )

    async def _async_call_bounded(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Call one MCP tool within an async wall deadline.

        Args:
            name: Qualified or unique short MCP tool name.
            arguments: JSON-compatible tool arguments.
            timeout_seconds: Remaining wall time for the attempt.

        Returns:
            A normalized tool result or timeout error.
        """
        await asyncio.to_thread(self.open)
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error_result(
                "InvalidMcpArguments", "arguments must be a dictionary", tool=name
            )
        try:
            argument_size = _json_size(arguments)
        except (TypeError, ValueError) as exc:
            return _error_result("InvalidMcpArguments", str(exc), tool=name)
        if argument_size > self.config.max_argument_bytes:
            return _error_result(
                "McpArgumentsTooLarge",
                f"MCP arguments are {argument_size} bytes; the limit is {self.config.max_argument_bytes}.",
                tool=name,
            )
        try:
            return await self._asubmit(
                asyncio.wait_for(
                    self._call_data(name, arguments, timeout_seconds),
                    timeout=timeout_seconds,
                ),
                timeout_seconds + 1,
            )
        except TimeoutError:
            return _error_result(
                "McpCallTimeout",
                f"The MCP call exceeded {timeout_seconds:.3g} seconds.",
                tool=name,
                retryable=True,
            )

    def servers(self) -> dict[str, Any]:
        """List safe status for configured servers.

        Returns:
            A successful result with server names, transports, and tool counts.
        """
        self.open()
        return self._submit(self._server_data(), self.config.connect_timeout_seconds)

    def search(
        self,
        query: str,
        *,
        server: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search tool names and descriptions.

        Args:
            query: Case-insensitive words or phrase. An empty value lists tools.
            server: Optional configured server namespace.
            limit: Requested result count before the hard maximum is applied.

        Returns:
            A successful result with ranked tool summaries.
        """
        self.open()
        if not isinstance(query, str):
            return _error_result("InvalidMcpSearch", "query must be a string")
        if server is not None and server not in self._states:
            return _error_result("UnknownMcpServer", f"Unknown MCP server: {server}")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return _error_result("InvalidMcpSearch", "limit must be a positive integer")
        hard_limit = min(limit, self.config.max_search_results)
        return self._submit(
            self._search_data(query, server, hard_limit, limit > hard_limit),
            self.config.connect_timeout_seconds,
        )

    def describe(self, name: str) -> dict[str, Any]:
        """Describe one qualified or unambiguous short tool name.

        Args:
            name: ``server__tool`` name or unique short tool name.

        Returns:
            A bounded description or a structured lookup error.
        """
        self.open()
        return self._submit(
            self._describe_data(name), self.config.connect_timeout_seconds
        )

    def promote(self, name: str, *, python_name: str | None = None) -> dict[str, Any]:
        """Expose one discovered MCP tool as a direct hosted function.

        Args:
            name: Qualified or unambiguous MCP tool name.
            python_name: Optional explicit Python alias.

        Returns:
            Promotion result with the next-call function declaration.
        """

        self.open()
        return self._submit(
            self._promote_data(name, python_name), self.config.connect_timeout_seconds
        )

    def promoted_toolset(self) -> HostedToolSet:
        """Build hosted tools for all current MCP promotions.

        Returns:
            Promoted tools with synchronous and asynchronous invokers.
        """

        tools: list[HostedTool] = []
        for qualified_name, alias in list(self._promoted.items()):
            entry = self._tools.get(qualified_name)
            if entry is None:
                continue
            tools.append(self._hosted_tool(entry, alias))
        return HostedToolSet(tuple(tools), self._hosted_tool_config())

    def _hosted_tool(self, entry: _ToolEntry, alias: str) -> HostedTool:
        """Create one direct-call wrapper from a discovered MCP tool.

        Args:
            entry: Discovered MCP catalog entry.
            alias: Python name used inside Monty.

        Returns:
            Hosted tool with sync and async MCP invokers.
        """

        def invoke(arguments: dict[str, Any]) -> Any:
            """Call this MCP tool synchronously."""

            return self.call(entry.qualified_name, arguments)

        async def ainvoke(arguments: dict[str, Any]) -> Any:
            """Call this MCP tool asynchronously."""

            return await self.async_call(entry.qualified_name, arguments)

        return HostedTool(
            name=entry.qualified_name,
            description=entry.description,
            input_schema=entry.input_schema,
            output_schema=entry.output_schema,
            invoke=invoke,
            ainvoke=ainvoke,
            python_name=alias,
            metadata={"mcp": True, "server": entry.server},
        )

    def _hosted_tool_config(self) -> HostedToolConfig:
        """Return hosted-call limits derived from the MCP hub limits."""

        return HostedToolConfig(
            max_calls_per_operation=self.config.max_calls_per_operation,
            max_total_call_seconds=self.config.max_total_call_seconds_per_operation,
            max_parallel_calls=self.config.max_concurrent_calls,
        )

    def refresh(self, server: str | None = None) -> dict[str, Any]:
        """Refresh configured server connections and tool descriptions.

        Args:
            server: Optional server namespace. ``None`` refreshes all servers.

        Returns:
            Updated safe server and catalog status.
        """
        self.open()
        if server is not None and server not in self._states:
            return _error_result("UnknownMcpServer", f"Unknown MCP server: {server}")
        timeout = self.config.connect_timeout_seconds * (
            1 if server is not None else max(1, len(self._states))
        )
        return self._submit(self._refresh_data(server), timeout)

    def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call one MCP tool and normalize its result for Monty.

        Args:
            name: ``server__tool`` name or unique short tool name.
            arguments: JSON-compatible tool arguments.

        Returns:
            A bounded result envelope with content, value, blobs, and error data.
        """
        return self._call_bounded(name, arguments, self.config.call_timeout_seconds)

    def _call_bounded(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Call one tool within an operation-specific wall deadline.

        Args:
            name: Qualified or unique short MCP tool name.
            arguments: JSON-compatible tool arguments.
            timeout_seconds: Remaining wall time for the complete attempt.

        Returns:
            A normalized tool result or timeout error.
        """
        self.open()
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error_result(
                "InvalidMcpArguments", "arguments must be a dictionary", tool=name
            )
        try:
            argument_size = _json_size(arguments)
        except (TypeError, ValueError) as exc:
            return _error_result("InvalidMcpArguments", str(exc), tool=name)
        if argument_size > self.config.max_argument_bytes:
            return _error_result(
                "McpArgumentsTooLarge",
                f"MCP arguments are {argument_size} bytes; the limit is {self.config.max_argument_bytes}.",
                tool=name,
            )
        try:
            return self._submit(
                asyncio.wait_for(
                    self._call_data(name, arguments, timeout_seconds),
                    timeout=timeout_seconds,
                ),
                timeout_seconds + 1,
            )
        except TimeoutError:
            return _error_result(
                "McpCallTimeout",
                f"The MCP call exceeded {timeout_seconds:.3g} seconds.",
                tool=name,
                retryable=True,
            )

    def _run_event_loop(self) -> None:
        """Run the private event loop until ``close`` stops it."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        self._loop.close()

    def _submit(self, coroutine: Any, timeout: float) -> Any:
        """Run one coroutine on the hub loop and wait synchronously.

        Args:
            coroutine: Coroutine created for this hub event loop.
            timeout: Host wall-time deadline in seconds.

        Returns:
            The coroutine result.
        """
        if self._loop is None:
            raise RuntimeError("the MCP hub event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError("the MCP host operation timed out") from None

    async def _asubmit(self, coroutine: Any, timeout: float) -> Any:
        """Run one hub coroutine without blocking the caller event loop.

        Args:
            coroutine: Coroutine created for the hub event loop.
            timeout: Host wall-time deadline in seconds.

        Returns:
            The coroutine result.
        """
        if self._loop is None:
            raise RuntimeError("the MCP hub event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=timeout,
            )
        except TimeoutError:
            future.cancel()
            raise TimeoutError("the MCP host operation timed out") from None

    async def _open_all(self) -> None:
        """Initialize concurrency control and connect configured servers."""
        self._call_semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        for name in self._states:
            await self._connect(name)

    async def _close_all(self) -> None:
        """Close every live client in reverse connection order."""
        for name in reversed(self._states):
            await self._disconnect(name)

    async def _connect(self, name: str) -> None:
        """Open one client and load its tool catalog.

        Args:
            name: Configured server namespace.
        """
        state = self._states[name]
        await self._disconnect(name)
        stack = AsyncExitStack()
        try:
            target: Any
            server = state.config
            if server.transport == "stdio":
                target = StdioServerParameters(
                    command=server.command or "",
                    args=list(server.args),
                    env=dict(server.env) if server.env is not None else None,
                    cwd=server.cwd,
                )
            elif server.transport == "streamable-http":
                http_client = await stack.enter_async_context(
                    httpx2.AsyncClient(headers=dict(server.headers or {}))
                )
                target = streamable_http_client(
                    server.url or "", http_client=http_client
                )
            else:
                target = server.target
            client = await asyncio.wait_for(
                stack.enter_async_context(
                    Client(
                        target, read_timeout_seconds=self.config.call_timeout_seconds
                    )
                ),
                timeout=self.config.connect_timeout_seconds,
            )
            state.client = client
            state.stack = stack
            state.error = None
            await self._load_tools(name)
        except asyncio.CancelledError:
            await stack.aclose()
            state.client = None
            state.stack = None
            self._remove_server_tools(name)
            raise
        except Exception as exc:  # noqa: BLE001 - isolate one failed MCP server
            await stack.aclose()
            state.client = None
            state.stack = None
            state.error = _exception_text(exc)
            self._remove_server_tools(name)

    async def _disconnect(self, name: str) -> None:
        """Close one live client and remove its catalog entries.

        Args:
            name: Configured server namespace.
        """
        state = self._states[name]
        stack = state.stack
        state.client = None
        state.stack = None
        self._remove_server_tools(name)
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001 - client cleanup is best effort
                state.error = _exception_text(exc)

    async def _load_tools(self, name: str) -> None:
        """Replace one server's entries with a bounded paginated listing.

        Args:
            name: Connected server namespace.
        """
        state = self._states[name]
        if state.client is None:
            return
        self._remove_server_tools(name)
        cursor: str | None = None
        while True:
            page = await asyncio.wait_for(
                state.client.list_tools(cursor=cursor, cache_mode="refresh"),
                timeout=self.config.connect_timeout_seconds,
            )
            for tool in page.tools:
                if len(self._tools) >= self.config.max_tools:
                    self._catalog_truncated = True
                    return
                qualified_name = f"{name}__{tool.name}"
                if (
                    self.config.allowed_tools is not None
                    and qualified_name not in self.config.allowed_tools
                ):
                    continue
                tool_data = tool.model_dump(
                    mode="json", by_alias=False, exclude_none=True
                )
                input_schema, output_schema, annotations, truncated = _bounded_schema(
                    tool_data,
                    self.config.max_schema_bytes,
                )
                self._tools[qualified_name] = _ToolEntry(
                    server=name,
                    name=tool.name,
                    qualified_name=qualified_name,
                    title=tool.title,
                    description=(tool.description or "")[:4_000],
                    input_schema=input_schema,
                    output_schema=output_schema,
                    annotations=annotations,
                    schema_truncated=truncated,
                )
            cursor = page.next_cursor
            if cursor is None:
                return

    async def _server_data(self) -> dict[str, Any]:
        """Build safe server status on the event-loop thread.

        Returns:
            Safe server and catalog state.
        """
        servers = []
        for name, state in self._states.items():
            servers.append(
                {
                    "name": name,
                    "transport": state.config.transport,
                    "connected": state.client is not None,
                    "tool_count": sum(
                        1 for tool in self._tools.values() if tool.server == name
                    ),
                    "error": state.error,
                }
            )
        return {
            "ok": True,
            "servers": servers,
            "tool_count": len(self._tools),
            "catalog_truncated": self._catalog_truncated,
        }

    async def _search_data(
        self,
        query: str,
        server: str | None,
        limit: int,
        limit_was_capped: bool,
    ) -> dict[str, Any]:
        """Search catalog entries on the event-loop thread.

        Args:
            query: Normalized user search text.
            server: Optional server namespace.
            limit: Enforced result limit.
            limit_was_capped: Whether the request exceeded the hard maximum.

        Returns:
            Ranked search result data.
        """
        phrase = query.strip().lower()
        words = phrase.split()
        matches: list[tuple[int, str, _ToolEntry]] = []
        for tool in self._tools.values():
            if server is not None and tool.server != server:
                continue
            text = " ".join(
                part
                for part in (tool.qualified_name, tool.title, tool.description)
                if part
            ).lower()
            if words and not any(word in text for word in words):
                continue
            score = sum(text.count(word) for word in words)
            if phrase and phrase in text:
                score += 100
            matches.append((-score, tool.qualified_name, tool))
        matches.sort(key=lambda item: (item[0], item[1]))
        tools = [item[2].summary() for item in matches[:limit]]
        return {
            "ok": True,
            "tools": tools,
            "count": len(tools),
            "total_matches": len(matches),
            "limit": limit,
            "limit_was_capped": limit_was_capped,
            "catalog_truncated": self._catalog_truncated,
        }

    async def _describe_data(self, name: str) -> dict[str, Any]:
        """Resolve and describe one catalog entry.

        Args:
            name: Qualified or short tool name.

        Returns:
            A bounded description or lookup error.
        """
        tool, error = self._resolve_tool(name)
        if error is not None:
            return error
        assert tool is not None
        return {"ok": True, "tool": tool.description_data()}

    async def _promote_data(self, name: str, python_name: str | None) -> dict[str, Any]:
        """Record one promotion on the MCP event-loop thread.

        Args:
            name: Qualified or short MCP tool name.
            python_name: Optional explicit Python alias.

        Returns:
            Promotion result or structured lookup error.
        """

        tool, error = self._resolve_tool(name)
        if error is not None:
            return error
        assert tool is not None
        alias = python_name or python_identifier(tool.qualified_name)
        if not alias.isidentifier() or keyword.iskeyword(alias):
            return _error_result(
                "InvalidMcpPythonName",
                f"Invalid Python function name: {alias}",
                tool=tool.qualified_name,
            )
        other = next(
            (
                qualified
                for qualified, existing_alias in self._promoted.items()
                if existing_alias == alias and qualified != tool.qualified_name
            ),
            None,
        )
        if other is not None:
            return _error_result(
                "McpPythonNameCollision",
                f"Python name {alias!r} is already used by {other!r}.",
                tool=tool.qualified_name,
            )
        try:
            promoted_tool = self._hosted_tool(tool, alias)
        except (TypeError, ValueError) as exc:
            return _error_result(
                "McpToolNotPromotable",
                str(exc),
                tool=tool.qualified_name,
            )
        self._promoted[tool.qualified_name] = alias
        hosted = HostedToolSet((promoted_tool,), self._hosted_tool_config())
        return {
            "ok": True,
            "tool": tool.qualified_name,
            "python_name": alias,
            "available": "next_execution",
            "catalog": {
                "sync": hosted.catalog(async_mode=False),
                "async": hosted.catalog(async_mode=True),
            },
        }

    async def _refresh_data(self, server: str | None) -> dict[str, Any]:
        """Reconnect selected servers and return updated status.

        Args:
            server: Optional server namespace.

        Returns:
            Safe server and catalog state.
        """
        names = [server] if server is not None else list(self._states)
        self._catalog_truncated = False
        for name in names:
            assert name is not None
            await self._connect(name)
        return await self._server_data()

    async def _call_data(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Resolve, call, retry, and normalize one tool.

        Args:
            name: Qualified or short tool name.
            arguments: Validated JSON-compatible arguments.
            timeout_seconds: Remaining wall time for this complete attempt.

        Returns:
            A normalized bounded MCP result.
        """
        tool, error = self._resolve_tool(name)
        if error is not None:
            return error
        assert tool is not None
        emit_event(
            self.event_sink,
            "mcp",
            "call_started",
            server=tool.server,
            tool=tool.qualified_name,
            argument_count=len(arguments),
        )
        semaphore = self._call_semaphore
        assert semaphore is not None
        async with semaphore:
            try:
                result = await self._call_once(tool, arguments, timeout_seconds)
            except Exception as first_error:  # noqa: BLE001 - transport boundary
                await self._connect(tool.server)
                if not self.config.retry_transport_errors:
                    return _error_result(
                        "McpTransportError",
                        _exception_text(first_error),
                        server=tool.server,
                        tool=tool.qualified_name,
                        retryable=True,
                    )
                refreshed = self._tools.get(tool.qualified_name)
                if refreshed is None:
                    return _error_result(
                        "McpTransportError",
                        _exception_text(first_error),
                        server=tool.server,
                        tool=tool.qualified_name,
                        retryable=True,
                    )
                try:
                    result = await self._call_once(
                        refreshed,
                        arguments,
                        timeout_seconds,
                    )
                except Exception as second_error:  # noqa: BLE001 - transport boundary
                    return _error_result(
                        "McpTransportError",
                        _exception_text(second_error),
                        server=tool.server,
                        tool=tool.qualified_name,
                        retryable=True,
                    )
        normalized = _normalize_result(
            tool,
            result,
            self.config.max_result_bytes,
        )
        emit_event(
            self.event_sink,
            "mcp",
            "call_finished",
            server=tool.server,
            tool=tool.qualified_name,
            ok=normalized["ok"],
            truncated=normalized["truncated"],
        )
        return normalized

    async def _call_once(
        self,
        tool: _ToolEntry,
        arguments: dict[str, Any],
        timeout_seconds: float,
    ) -> Any:
        """Call one tool on its current live client.

        Args:
            tool: Resolved catalog entry.
            arguments: Validated JSON-compatible arguments.
            timeout_seconds: Maximum time for this individual request.

        Returns:
            The MCP SDK tool result.
        """
        client = self._states[tool.server].client
        if client is None:
            raise ConnectionError(f"MCP server {tool.server} is not connected")
        return await asyncio.wait_for(
            client.call_tool(
                tool.name,
                arguments,
                read_timeout_seconds=timeout_seconds,
            ),
            timeout=timeout_seconds,
        )

    def _resolve_tool(
        self,
        name: str,
    ) -> tuple[_ToolEntry | None, dict[str, Any] | None]:
        """Resolve a qualified name or one unique short name.

        Args:
            name: Tool name supplied by Monty code.

        Returns:
            A tool and no error, or no tool and a structured error.
        """
        if not isinstance(name, str) or not name:
            return None, _error_result(
                "InvalidMcpToolName", "tool name must be a non-empty string"
            )
        exact = self._tools.get(name)
        if exact is not None:
            return exact, None
        matches = [tool for tool in self._tools.values() if tool.name == name]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            names = ", ".join(sorted(tool.qualified_name for tool in matches))
            return None, _error_result(
                "AmbiguousMcpTool",
                f"Use a qualified MCP tool name. Matches: {names}",
                tool=name,
            )
        return None, _error_result(
            "UnknownMcpTool", f"Unknown MCP tool: {name}", tool=name
        )

    def _remove_server_tools(self, name: str) -> None:
        """Remove all catalog entries owned by one server.

        Args:
            name: Server namespace to remove.
        """
        for qualified_name in [
            key for key, tool in self._tools.items() if tool.server == name
        ]:
            del self._tools[qualified_name]


def _bounded_schema(
    tool_data: dict[str, Any],
    max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Keep tool schema fields only when they fit the configured limit.

    Args:
        tool_data: JSON tool model from the MCP SDK.
        max_bytes: Maximum combined schema size.

    Returns:
        Input schema, output schema, annotations, and truncation state.
    """
    input_schema = tool_data.get("input_schema") or {}
    output_schema = tool_data.get("output_schema")
    annotations = tool_data.get("annotations")
    data = {
        "input_schema": input_schema,
        "output_schema": output_schema,
        "annotations": annotations,
    }
    size = _json_size(data)
    if size <= max_bytes:
        return input_schema, output_schema, annotations, False
    return (
        {"omitted": True, "size_bytes": _json_size(input_schema)},
        None,
        None,
        True,
    )


def _normalize_result(
    tool: _ToolEntry,
    result: Any,
    max_bytes: int,
) -> dict[str, Any]:
    """Convert one MCP result into a bounded plain-value envelope.

    Args:
        tool: Called catalog entry.
        result: MCP SDK ``CallToolResult``.
        max_bytes: Maximum JSON bytes returned to Monty.

    Returns:
        A plain result envelope safe for the Monty boundary.
    """
    data = result.model_dump(mode="json", by_alias=False, exclude_none=True)
    content = data.get("content", [])
    blobs = _blob_metadata(content)
    is_error = bool(data.get("is_error", False))
    envelope = {
        "ok": not is_error,
        "server": tool.server,
        "tool": tool.qualified_name,
        "value": data.get("structured_content"),
        "content": content,
        "blobs": blobs,
        "error": (
            {
                "type": "McpToolError",
                "message": _content_text(content) or "The MCP tool returned an error.",
                "retryable": False,
            }
            if is_error
            else None
        ),
        "truncated": False,
    }
    if _json_size(envelope) <= max_bytes:
        return envelope
    return _truncate_result(envelope, max_bytes)


def _truncate_result(envelope: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Remove large payloads and keep a bounded result preview.

    Args:
        envelope: Complete normalized MCP result.
        max_bytes: Maximum result size.

    Returns:
        A result that fits the configured maximum.
    """
    value = envelope.get("value")
    value_size = _json_size(value)
    content: list[dict[str, Any]] = []
    text_budget = max(0, min(16_000, max_bytes // 3))
    for block in envelope.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", ""))
            content.append(
                {
                    "type": "text",
                    "text": text[:text_budget],
                    "original_chars": len(text),
                    "truncated": len(text) > text_budget,
                }
            )
            text_budget = 0
        elif block_type in {"image", "audio"}:
            content.append(
                {
                    "type": block_type,
                    "mime_type": block.get("mime_type"),
                    "data_omitted": True,
                }
            )
        else:
            content.append({"type": block_type or "unknown", "data_omitted": True})
    trimmed = {
        **envelope,
        "value": {"omitted": True, "size_bytes": value_size}
        if value is not None
        else None,
        "content": content,
        "truncated": True,
    }
    if _json_size(trimmed) <= max_bytes:
        return trimmed
    error = envelope.get("error")
    if error is not None:
        error = {
            **error,
            "message": str(error.get("message", ""))[: max(0, max_bytes // 4)],
        }
    minimal = {
        "ok": envelope["ok"],
        "server": envelope["server"],
        "tool": envelope["tool"],
        "value": None,
        "content": [],
        "blobs": envelope.get("blobs", [])[:1],
        "error": error,
        "truncated": True,
    }
    if _json_size(minimal) <= max_bytes:
        return minimal
    return {
        "ok": envelope["ok"],
        "server": str(envelope["server"])[:64],
        "tool": str(envelope["tool"])[:128],
        "value": None,
        "content": [],
        "blobs": [],
        "error": None,
        "truncated": True,
    }


def _blob_metadata(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe binary MCP content without duplicating its Base64 data.

    Args:
        content: Normalized MCP content blocks.

    Returns:
        Binary block indexes, types, media types, and approximate byte sizes.
    """
    blobs = []
    for index, block in enumerate(content):
        if block.get("type") not in {"image", "audio"}:
            continue
        data = block.get("data", "")
        blobs.append(
            {
                "content_index": index,
                "type": block.get("type"),
                "mime_type": block.get("mime_type"),
                "size_bytes": len(data) * 3 // 4,
                "encoding": "base64",
            }
        )
    return blobs


def _content_text(content: list[dict[str, Any]]) -> str:
    """Join MCP text blocks for an error message.

    Args:
        content: Normalized MCP content blocks.

    Returns:
        Joined non-empty text content.
    """
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if block.get("type") == "text" and block.get("text")
    )


def _error_result(
    error_type: str,
    message: str,
    *,
    server: str | None = None,
    tool: str | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    """Build one plain MCP error envelope.

    Args:
        error_type: Stable error category.
        message: Human-readable error text.
        server: Optional server namespace.
        tool: Optional qualified or supplied tool name.
        retryable: Whether an immediate retry can help.

    Returns:
        A normalized failed MCP result.
    """
    return {
        "ok": False,
        "server": server,
        "tool": tool,
        "value": None,
        "content": [],
        "blobs": [],
        "error": {
            "type": error_type,
            "message": message,
            "retryable": retryable,
        },
        "truncated": False,
    }


def _json_size(value: Any) -> int:
    """Measure compact UTF-8 JSON bytes.

    Args:
        value: JSON-compatible value.

    Returns:
        Encoded byte length.
    """
    return len(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _exception_text(exc: Exception) -> str:
    """Return an exception category without endpoint or credential data.

    Args:
        exc: Host-side MCP or transport exception.

    Returns:
        A safe generic message with the exception class.
    """
    return f"{type(exc).__name__}: The MCP client operation failed."
