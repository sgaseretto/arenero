"""Small Deep Agents task bridge for the Python compatibility adapter."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import replace
from typing import Any

from langchain.agents.structured_output import AutoStrategy
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp

from .hosted_tools import HostedTool
from .langchain_hosted_tools import _inject_runtime, _programmatic_result

try:
    from deepagents.middleware.subagents import (
        SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY as _FORMAT_KEY,
    )
except ImportError:
    _FORMAT_KEY = "__deepagents_subagent_response_format"


def task_bridge(
    tool: BaseTool,
    runtime: ToolRuntime,
    *,
    concurrency: int,
    max_calls: int,
) -> HostedTool:
    """Make one task function with its own concurrency and dispatch budget.

    Args:
        tool: Current Deep Agents task tool.
        runtime: Parent eval runtime, including its replay-stable call ID.
        concurrency: Maximum active subagents in this eval.
        max_calls: Maximum task dispatches in this eval.

    Returns:
        Hosted task with optional response schema and stream labels.
    """
    gate = asyncio.Semaphore(concurrency)
    ordinal = 0

    async def invoke(arguments: dict[str, Any]) -> Any:
        """Dispatch a task with a stable identity and custom lifecycle events.

        Args:
            arguments: Description, subagent_type, response_schema, and label.

        Returns:
            Structured response when requested, otherwise task response text.
        """
        nonlocal ordinal
        index = ordinal
        ordinal += 1
        if ordinal > max_calls:
            raise ValueError("subagent call limit exceeded")
        schema = arguments.get("response_schema")
        if schema is not None:
            _validate_schema(schema)
            schema = dict(schema)
            if not isinstance(schema.get("title"), str) or not schema["title"].strip():
                schema["title"] = "subagent_response"
        payload = json.dumps(
            [runtime.tool_call_id, index, arguments],
            sort_keys=True,
            separators=(",", ":"),
        )
        dispatch_id = (
            "arenero_task_" + hashlib.sha256(payload.encode()).hexdigest()[:32]
        )
        config: RunnableConfig = {**(runtime.config or {})}
        if schema is not None:
            config["configurable"] = {
                **config.get("configurable", {}),
                _FORMAT_KEY: AutoStrategy(schema),
            }
        child = replace(runtime, config=config)
        args, _ = _inject_runtime(
            tool,
            {key: arguments[key] for key in ("description", "subagent_type")},
            child,
            dispatch_id,
        )
        event = {"type": "subagent", "id": dispatch_id, "eval_id": runtime.tool_call_id}
        async with gate:
            started = time.monotonic()
            _emit(
                runtime,
                {
                    **event,
                    "phase": "start",
                    "subagent_type": arguments["subagent_type"],
                    "description": arguments["description"][:200],
                    "label": (arguments.get("label") or arguments["description"])[:120],
                },
            )
            try:
                result = await tool.arun(args, config=config, tool_call_id=dispatch_id)
                result = _programmatic_result(tool, result)
                if schema is not None and isinstance(result, str):
                    result = json.loads(result)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                _emit(
                    runtime,
                    {
                        **event,
                        "phase": "error",
                        "error": str(exc)[:200],
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    },
                )
                raise
            _emit(
                runtime,
                {
                    **event,
                    "phase": "complete",
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
        return result

    return HostedTool(
        name="task",
        description="Run a subagent. Optional bounded JSON response_schema and label.",
        input_schema={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "subagent_type": {"type": "string"},
                "response_schema": {"type": ["object", "null"], "default": None},
                "label": {"type": ["string", "null"], "default": None},
            },
            "required": ["description", "subagent_type"],
            "additionalProperties": False,
        },
        ainvoke=invoke,
    )


def _validate_schema(schema: dict[str, Any]) -> None:
    """Check a task schema before it reaches a model adapter.

    Args:
        schema: JSON schema, bounded to 4096 UTF-8 bytes, depth 5, 32 properties.
    """
    if len(json.dumps(schema).encode()) > 4096:
        raise ValueError("response_schema exceeds 4096 bytes")
    stack = [(schema, 0)]
    properties = 0
    while stack:
        node, depth = stack.pop()
        if depth > 5:
            raise ValueError("response_schema exceeds depth 5")
        if isinstance(node, dict):
            properties += len(node.get("properties", {}))
            stack.extend(
                (value, depth + 1)
                for value in node.values()
                if isinstance(value, (dict, list))
            )
        elif isinstance(node, list):
            stack.extend((value, depth) for value in node)
        if properties > 32:
            raise ValueError("response_schema exceeds 32 properties")


def _emit(runtime: ToolRuntime, event: dict[str, Any]) -> None:
    """Send an optional custom event without changing dispatch behavior.

    Args:
        runtime: Parent runtime with its stream writer.
        event: Bounded subagent event record.
    """
    try:
        runtime.stream_writer(event)
    except Exception:
        logging.getLogger(__name__).debug("Subagent event sink failed", exc_info=True)
