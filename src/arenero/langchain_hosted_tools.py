"""Optional LangChain tools callable from an Arenero REPL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, get_type_hints
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .hosted_tools import HostedTool, HostedToolConfig, HostedToolSet

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


def hosted_tools_from_langchain(
    tools: Sequence[BaseTool],
    *,
    config: RunnableConfig | None = None,
    runtime: ToolRuntime | None = None,
    limits: HostedToolConfig | None = None,
    names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_code_execution: bool = False,
) -> HostedToolSet:
    """Adapt LangChain tools to schema-aware Arenero host functions.

    Args:
        tools: LangChain tools to consider.
        config: Current LangChain or LangGraph runtime configuration.
        runtime: Current tool runtime. Use this for injected state, store,
            context, and tool-call data during programmatic calls.
        limits: Optional hosted-call and record limits.
        names: Optional accepted tool names.
        metadata: Optional required tool metadata values.
        include_code_execution: Include nested code tools when true.

    Returns:
        Hosted tool set that invokes ``BaseTool.invoke`` and ``ainvoke``.
    """

    hosted = HostedToolSet(
        tuple(_adapt_tool(tool, config, runtime) for tool in tools),
        limits or HostedToolConfig(),
    )
    return hosted.select(
        names=names,
        metadata=metadata,
        include_code_execution=include_code_execution,
    )


def _adapt_tool(
    tool: BaseTool,
    config: RunnableConfig | None,
    runtime: ToolRuntime | None,
) -> HostedTool:
    """Convert one LangChain tool without bypassing its normal call path.

    Args:
        tool: LangChain tool.
        config: Current runtime configuration.
        runtime: Current tool runtime, when called through a ToolNode.

    Returns:
        Framework-neutral hosted tool.
    """

    tool_metadata = dict(tool.metadata or {})
    input_model = tool.tool_call_schema
    if isinstance(input_model, Mapping):
        schema = dict(input_model)
    else:
        schema_method = getattr(input_model, "model_json_schema", None)
        schema = schema_method() if schema_method is not None else input_model.schema()

    def invoke(arguments: dict[str, Any]) -> Any:
        """Call the LangChain tool through its synchronous public API.

        Args:
            arguments: Model-visible keyword arguments.

        Returns:
            Boundary-ready tool value.
        """

        enriched, tool_call_id = _inject_runtime(tool, arguments, runtime)
        if tool_call_id is not None and _uses_injected_tool_call_id(tool):
            result = tool.run(enriched, config=config, tool_call_id=tool_call_id)
        else:
            result = tool.invoke(enriched, config=config)
        return _programmatic_result(tool, result)

    async def ainvoke(arguments: dict[str, Any]) -> Any:
        """Call the LangChain tool through its asynchronous public API.

        Args:
            arguments: Model-visible keyword arguments.

        Returns:
            Boundary-ready awaited tool value.
        """

        enriched, tool_call_id = _inject_runtime(tool, arguments, runtime)
        if tool_call_id is not None and _uses_injected_tool_call_id(tool):
            result = await tool.arun(enriched, config=config, tool_call_id=tool_call_id)
        else:
            result = await tool.ainvoke(enriched, config=config)
        return _programmatic_result(tool, result)

    configured_kind = tool_metadata.get("arenero_kind")
    if configured_kind == "control":
        kind = "control"
    elif "code_arg_name" in tool_metadata or configured_kind == "code-execution":
        kind = "code-execution"
    else:
        kind = "function"
    output_schema = tool_metadata.get("output_schema")
    return HostedTool(
        name=tool.name,
        description=tool.description or "",
        input_schema=schema,
        output_schema=output_schema if isinstance(output_schema, Mapping) else None,
        invoke=invoke,
        ainvoke=ainvoke,
        python_name=tool_metadata.get("python_name"),
        sequential=bool(tool_metadata.get("sequential", False)),
        metadata=tool_metadata,
        kind=kind,
    )


def _inject_runtime(
    tool: BaseTool,
    arguments: dict[str, Any],
    runtime: ToolRuntime | None,
) -> tuple[dict[str, Any], str | None]:
    """Inject LangGraph runtime values for a programmatic tool call.

    Args:
        tool: Tool whose hidden arguments must be detected.
        arguments: Model-visible keyword arguments.
        runtime: Outer REPL tool runtime, or ``None`` outside LangGraph.

    Returns:
        Enriched arguments and a synthetic child tool-call ID when available.
    """

    if runtime is None:
        return dict(arguments), None
    try:
        from langgraph.prebuilt.tool_node import _get_all_injected_args
    except ImportError:
        return dict(arguments), None

    injected = _get_all_injected_args(tool)
    child_id = f"arenero_{tool.name}_{uuid4().hex[:8]}"
    derived = type(runtime)(
        state=runtime.state,
        context=runtime.context,
        config=runtime.config,
        stream_writer=runtime.stream_writer,
        tool_call_id=child_id,
        store=runtime.store,
        tools=runtime.tools,
        execution_info=getattr(runtime, "execution_info", None),
        server_info=getattr(runtime, "server_info", None),
    )
    enriched = dict(arguments)
    if injected.runtime:
        enriched[injected.runtime] = derived
    for name, field in injected.state.items():
        if field is None:
            enriched[name] = runtime.state
        elif isinstance(runtime.state, Mapping):
            enriched[name] = runtime.state.get(field)
        else:
            enriched[name] = getattr(runtime.state, field, None)
    if injected.store and runtime.store is not None:
        enriched[injected.store] = runtime.store
    return enriched, child_id


def _uses_injected_tool_call_id(tool: BaseTool) -> bool:
    """Check whether a tool requests ``InjectedToolCallId``.

    Args:
        tool: LangChain tool to inspect.

    Returns:
        ``True`` when the tool needs a separate call ID argument.
    """

    try:
        from langchain_core.tools.base import (
            InjectedToolCallId,
            _is_injected_arg_type,
            get_all_basemodel_annotations,
        )
    except ImportError:
        return False
    schema_annotations = get_all_basemodel_annotations(tool.get_input_schema())
    function = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    try:
        function_annotations = (
            get_type_hints(function, include_extras=True) if function else {}
        )
    except (NameError, TypeError):
        function_annotations = {}
    annotations = {**function_annotations, **schema_annotations}
    return any(
        _is_injected_arg_type(value, injected_type=InjectedToolCallId)
        for value in annotations.values()
    )


def _programmatic_result(tool: BaseTool, result: Any) -> Any:
    """Reduce a Deep Agents task command to its returned message.

    Args:
        tool: Tool that produced the value.
        result: Raw LangChain tool result.

    Returns:
        Task response text, or the original value for other tools.
    """

    if tool.name != "task":
        return result
    update = getattr(result, "update", None)
    if not isinstance(update, Mapping):
        return result
    messages = update.get("messages")
    if not isinstance(messages, Sequence) or not messages:
        return result
    return getattr(messages[-1], "content", messages[-1])
