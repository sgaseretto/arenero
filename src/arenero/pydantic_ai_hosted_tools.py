"""Optional Pydantic AI tools callable from an Arenero REPL."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai import AbstractToolset, RunContext
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDenied

from .hosted_tools import HostedTool, HostedToolConfig, HostedToolSet


async def hosted_tools_from_pydantic_ai(
    toolset: AbstractToolset[Any],
    ctx: RunContext[Any],
    *,
    limits: HostedToolConfig | None = None,
    names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_code_execution: bool = False,
) -> HostedToolSet:
    """Adapt one Pydantic AI toolset for an Arenero operation.

    Args:
        toolset: Entered Pydantic AI toolset used by the current run.
        ctx: Current Pydantic AI run context.
        limits: Optional hosted-call and record limits.
        names: Optional accepted tool names.
        metadata: Optional required tool metadata values.
        include_code_execution: Include nested code tools when true.

    Returns:
        Hosted tools that use Pydantic AI validation, hooks, and execution.
    """

    tools = await toolset.get_tools(ctx)
    parent_manager = ctx.tool_manager
    if parent_manager is None:
        raise RuntimeError("Pydantic AI RunContext has no tool manager")
    manager = ToolManager(
        toolset=toolset,
        root_capability=parent_manager.root_capability,
        ctx=ctx,
        tools=tools,
    )
    hosted = HostedToolSet(
        tuple(
            _adapt_tool(name, tool.tool_def, manager)
            for name, tool in tools.items()
            if ctx.is_tool_available(tool.tool_def)
        ),
        limits or HostedToolConfig(),
    )
    return hosted.select(
        names=names,
        metadata=metadata,
        include_code_execution=include_code_execution,
    )


def _adapt_tool(name: str, definition: Any, manager: ToolManager[Any]) -> HostedTool:
    """Convert one Pydantic AI tool definition and manager.

    Args:
        name: Tool name used by the manager.
        definition: Pydantic AI tool definition.
        manager: Current run's standard tool manager.

    Returns:
        Framework-neutral asynchronous hosted tool.
    """

    async def ainvoke(arguments: dict[str, Any]) -> Any:
        """Execute through Pydantic AI's normal validation and hook path."""

        call = ToolCallPart(
            tool_name=name,
            args=arguments,
            tool_call_id=f"arenero_{uuid.uuid4().hex}",
        )
        result = await manager.handle_call(call, wrap_validation_errors=False)
        if isinstance(result, ToolDenied):
            raise PermissionError(f"Tool {name!r} was denied: {result.message}")
        if isinstance(result, ToolReturn):
            return result.return_value
        return result

    tool_metadata = dict(definition.metadata or {})
    kind = (
        "control"
        if definition.tool_kind is not None
        else "code-execution"
        if "code_arg_name" in tool_metadata
        else "function"
    )
    return HostedTool(
        name=name,
        description=definition.description or "",
        input_schema=definition.parameters_json_schema,
        output_schema=definition.return_schema,
        ainvoke=ainvoke,
        python_name=tool_metadata.get("python_name"),
        sequential=bool(definition.sequential),
        metadata=tool_metadata,
        kind=kind,
    )
