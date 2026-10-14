"""Framework-neutral tools that Monty code can call through the host."""

from __future__ import annotations

import asyncio
import inspect
import json
import keyword
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, get_args, get_origin

from .events import EventSink, emit_event

_INVALID_NAME = re.compile(r"[^A-Za-z0-9_]")
_PLAIN_TYPES = (str, int, float, bool, bytes, type(None))


@dataclass(frozen=True)
class HostedToolConfig:
    """Set operation limits for hosted tools.

    Attributes:
        max_calls_per_operation: Maximum hosted calls in one kernel operation.
        max_total_call_seconds: Maximum wall time available to hosted calls.
        max_parallel_calls: Maximum hosted calls that can run at the same time.
        max_recorded_calls: Maximum call records returned to the caller.
        preview_chars: Maximum argument, result, or error preview length.
        include_value_previews: Whether records include bounded value previews.
    """

    max_calls_per_operation: int = 100
    max_total_call_seconds: float = 30.0
    max_parallel_calls: int = 10
    max_recorded_calls: int = 20
    preview_chars: int = 200
    include_value_previews: bool = False

    def __post_init__(self) -> None:
        """Validate all hosted-tool limits."""

        if self.max_calls_per_operation <= 0:
            raise ValueError("max_calls_per_operation must be greater than 0")
        if self.max_total_call_seconds <= 0:
            raise ValueError("max_total_call_seconds must be greater than 0")
        if self.max_parallel_calls <= 0:
            raise ValueError("max_parallel_calls must be greater than 0")
        if self.max_recorded_calls < 0 or self.preview_chars < 0:
            raise ValueError("record and preview limits must not be negative")


SyncToolInvoker = Callable[[dict[str, Any]], Any]
AsyncToolInvoker = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class HostedTool:
    """Describe one host tool that can become a Monty function.

    Attributes:
        name: Stable external tool name.
        description: Short model-facing purpose.
        input_schema: JSON Schema for keyword arguments.
        output_schema: Optional JSON Schema for the return value.
        invoke: Optional synchronous framework invocation function.
        ainvoke: Optional asynchronous framework invocation function.
        python_name: Optional explicit Python alias.
        sequential: Whether this call must not overlap other hosted calls.
        metadata: Framework-neutral selection metadata.
        kind: Tool kind. Code-execution tools are excluded by default.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None = None
    invoke: SyncToolInvoker | None = None
    ainvoke: AsyncToolInvoker | None = None
    python_name: str | None = None
    sequential: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kind: Literal["function", "code-execution", "control"] = "function"

    def __post_init__(self) -> None:
        """Validate the hosted tool identity and invocation methods."""

        if not self.name:
            raise ValueError("hosted tool name must not be empty")
        if self.invoke is None and self.ainvoke is None:
            raise ValueError(f"hosted tool {self.name!r} has no invoker")
        alias = self.python_name or python_identifier(self.name)
        if not alias.isidentifier() or keyword.iskeyword(alias):
            raise ValueError(f"invalid hosted tool Python name: {alias!r}")
        if self.input_schema.get("type", "object") != "object":
            raise ValueError("hosted tool input_schema must describe an object")
        properties = self.input_schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise TypeError("hosted tool input_schema properties must be an object")
        if any(not isinstance(schema, Mapping) for schema in properties.values()):
            raise TypeError("each hosted tool input property must have a schema object")
        invalid = [
            name
            for name in properties
            if not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
        ]
        if invalid:
            raise ValueError(
                f"hosted tool {self.name!r} has invalid Python argument names: {invalid}"
            )

    @property
    def alias(self) -> str:
        """Return the Python-safe name used inside Monty.

        Returns:
            Explicit alias or a deterministic sanitized tool name.
        """

        return self.python_name or python_identifier(self.name)

    @classmethod
    def from_callable(
        cls,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        python_name: str | None = None,
        sequential: bool = False,
        metadata: Mapping[str, Any] | None = None,
        kind: Literal["function", "code-execution", "control"] = "function",
    ) -> HostedTool:
        """Create a hosted tool from one annotated Python function.

        Args:
            function: Synchronous or asynchronous callable.
            name: Optional public tool name.
            description: Optional model-facing description.
            python_name: Optional Python alias.
            sequential: Whether calls must run without overlap.
            metadata: Optional tool selection metadata.
            kind: Function, code-execution, or control classification.

        Returns:
            Hosted tool with schemas derived from the function signature.
        """

        tool_name = name or getattr(function, "__name__", type(function).__name__)
        input_schema, output_schema = _callable_schemas(function)
        summary = (
            description if description is not None else inspect.getdoc(function) or ""
        )
        if inspect.iscoroutinefunction(function):

            async def async_invoke(arguments: dict[str, Any]) -> Any:
                """Invoke the original asynchronous callable with keyword arguments."""

                return await function(**arguments)

            return cls(
                name=tool_name,
                description=summary,
                input_schema=input_schema,
                output_schema=output_schema,
                ainvoke=async_invoke,
                python_name=python_name,
                sequential=sequential,
                metadata=metadata or {},
                kind=kind,
            )

        def sync_invoke(arguments: dict[str, Any]) -> Any:
            """Invoke the original synchronous callable with keyword arguments."""

            return function(**arguments)

        return cls(
            name=tool_name,
            description=summary,
            input_schema=input_schema,
            output_schema=output_schema,
            invoke=sync_invoke,
            python_name=python_name,
            sequential=sequential,
            metadata=metadata or {},
            kind=kind,
        )


@dataclass(frozen=True)
class HostedToolSet:
    """Hold selected hosted tools and their shared operation limits.

    Attributes:
        tools: Hosted tools exposed to Monty.
        config: Per-operation call and record limits.
    """

    tools: tuple[HostedTool, ...] = ()
    config: HostedToolConfig = field(default_factory=HostedToolConfig)

    def __post_init__(self) -> None:
        """Reject duplicate external names and Python aliases."""

        names: set[str] = set()
        aliases: set[str] = set()
        for tool in self.tools:
            if tool.name in names:
                raise ValueError(f"duplicate hosted tool name: {tool.name}")
            if tool.alias in aliases:
                raise ValueError(f"duplicate hosted tool Python name: {tool.alias}")
            names.add(tool.name)
            aliases.add(tool.alias)

    @classmethod
    def from_callables(
        cls,
        functions: Sequence[Callable[..., Any]],
        *,
        config: HostedToolConfig | None = None,
    ) -> HostedToolSet:
        """Create a hosted tool set from annotated Python functions.

        Args:
            functions: Functions to expose.
            config: Optional operation limits.

        Returns:
            New hosted tool set.
        """

        return cls(
            tuple(HostedTool.from_callable(function) for function in functions),
            config or HostedToolConfig(),
        )

    def select(
        self,
        *,
        names: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_code_execution: bool = False,
    ) -> HostedToolSet:
        """Return tools that match simple name and metadata selectors.

        Args:
            names: Optional accepted external names.
            metadata: Optional required metadata values.
            include_code_execution: Include nested code-execution tools when true.

        Returns:
            Selected tool set with the same limits.
        """

        accepted = set(names) if names is not None else None
        selected = tuple(
            tool
            for tool in self.tools
            if (accepted is None or tool.name in accepted)
            and tool.kind != "control"
            and (include_code_execution or tool.kind != "code-execution")
            and _metadata_contains(tool.metadata, metadata or {})
        )
        return HostedToolSet(selected, self.config)

    def merged(self, *others: HostedToolSet | None) -> HostedToolSet:
        """Combine tool sets and preserve this set's limits.

        Args:
            *others: Optional additional tool sets.

        Returns:
            Combined tool set.
        """

        tools = list(self.tools)
        config = self.config if self.tools else None
        for other in others:
            if other is not None:
                tools.extend(other.tools)
                if other.tools:
                    config = (
                        other.config
                        if config is None
                        else _strictest_config(config, other.config)
                    )
        return HostedToolSet(tuple(tools), config or self.config)

    def catalog(self, *, async_mode: bool) -> str:
        """Render model-facing Python declarations.

        Args:
            async_mode: Render functions as awaitable when true.

        Returns:
            Python code block with types and exact top-level signatures.
        """

        declarations = self.type_check_stubs(async_mode=async_mode)
        if not declarations:
            return ""
        mode = (
            "Await these functions." if async_mode else "Call these functions directly."
        )
        notes = self.schema_warnings()
        warning_text = f"\nSchema notes: {'; '.join(notes)}" if notes else ""
        return (
            f"Hosted functions available inside the REPL. {mode}"
            f"{warning_text}\n```python\n{declarations}\n```"
        )

    def schema_warnings(self) -> list[str]:
        """Return concise warnings for incomplete hosted-tool schemas.

        Returns:
            Model-facing warnings that explain where generated types use ``Any``.
        """

        warnings: list[str] = []
        for tool in self.tools:
            if tool.output_schema is None:
                warnings.append(f"{tool.alias} has no output schema")
            properties = tool.input_schema.get("properties", {})
            unknown = [name for name, schema in properties.items() if not schema]
            if unknown:
                warnings.append(
                    f"{tool.alias} has unknown input types for {', '.join(unknown)}"
                )
        return warnings

    def type_check_stubs(self, *, async_mode: bool) -> str:
        """Render Monty type-check declarations for hosted functions.

        Args:
            async_mode: Render asynchronous declarations when true.

        Returns:
            Type definitions and function stubs.
        """

        if not self.tools:
            return ""
        definitions: list[str] = [
            "from typing import Any, Literal, NotRequired, TypedDict"
        ]
        functions: list[str] = []
        for tool in self.tools:
            type_blocks, signature = _render_tool(tool, async_mode=async_mode)
            definitions.extend(type_blocks)
            functions.append(signature)
        return "\n\n".join([*definitions, *functions])

    def sync_scope(
        self,
        execution_id: str,
        event_sink: EventSink | None = None,
    ) -> HostedToolScope:
        """Create fresh synchronous wrappers and call state.

        Args:
            execution_id: Parent kernel execution ID.
            event_sink: Optional safe event observer.

        Returns:
            Fresh hosted-tool operation scope.
        """

        return HostedToolScope(self, execution_id, event_sink)

    def async_scope(
        self,
        execution_id: str,
        event_sink: EventSink | None = None,
    ) -> HostedToolScope:
        """Create fresh asynchronous wrappers and call state.

        Args:
            execution_id: Parent kernel execution ID.
            event_sink: Optional safe event observer.

        Returns:
            Fresh hosted-tool operation scope.
        """

        return HostedToolScope(self, execution_id, event_sink)


class HostedToolScope:
    """Own hosted-tool limits and call records for one kernel operation."""

    def __init__(
        self,
        toolset: HostedToolSet,
        execution_id: str,
        event_sink: EventSink | None,
    ) -> None:
        """Create one operation scope.

        Args:
            toolset: Selected tools and operation limits.
            execution_id: Parent kernel execution ID.
            event_sink: Optional safe event observer.
        """

        self.toolset = toolset
        self.execution_id = execution_id
        self.event_sink = event_sink
        self._deadline: float | None = None
        self._guard = threading.Lock()
        self._calls = 0
        self._records: list[dict[str, Any]] = []
        self._records_truncated = False
        self._sync_gate = threading.Condition()
        self._sync_parallel = 0
        self._sync_sequential = False
        self._sync_slots = threading.BoundedSemaphore(toolset.config.max_parallel_calls)
        self._async_gate = asyncio.Condition()
        self._async_parallel = 0
        self._async_sequential = False
        self._async_slots = asyncio.Semaphore(toolset.config.max_parallel_calls)

    def sync_lookup(self) -> dict[str, Callable[..., Any]]:
        """Build synchronous Monty host functions.

        Returns:
            Python alias to synchronous wrapper mapping.

        Raises:
            ValueError: If a selected tool has no synchronous invoker.
        """

        lookup: dict[str, Callable[..., Any]] = {}
        for tool in self.toolset.tools:
            if tool.invoke is None:
                raise ValueError(
                    f"hosted tool {tool.name!r} is async-only; use AsyncPersistentMontyKernel"
                )
            lookup[tool.alias] = self._sync_wrapper(tool)
        return lookup

    def async_lookup(self) -> dict[str, Callable[..., Any]]:
        """Build asynchronous Monty host functions.

        Returns:
            Python alias to asynchronous wrapper mapping.
        """

        return {tool.alias: self._async_wrapper(tool) for tool in self.toolset.tools}

    def summary(self) -> dict[str, Any]:
        """Return bounded nested-call metadata.

        Returns:
            Exact call count, bounded records, and truncation state.
        """

        with self._guard:
            return {
                "count": self._calls,
                "items": [dict(record) for record in self._records],
                "truncated": self._records_truncated,
            }

    def _reserve(
        self, tool: HostedTool, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Reserve call budget before host work starts.

        Args:
            tool: Tool about to run.
            arguments: Validated or raw keyword arguments.

        Returns:
            Stable call ID and mutable call record.

        Raises:
            RuntimeError: If the operation call or time limit is exhausted.
        """

        with self._guard:
            now = time.monotonic()
            if self._deadline is None:
                self._deadline = now + self.toolset.config.max_total_call_seconds
            if self._calls >= self.toolset.config.max_calls_per_operation:
                raise RuntimeError(
                    "hosted tool call limit exceeded; split the work across executions"
                )
            if now >= self._deadline:
                raise RuntimeError(
                    "hosted tool time limit exceeded; split the work across executions"
                )
            self._calls += 1
            call_id = f"{self.execution_id}__host_{self._calls}"
            record: dict[str, Any] = {
                "call_id": call_id,
                "tool": tool.name,
                "python_name": tool.alias,
                "status": "running",
                "duration_seconds": 0.0,
                "may_have_side_effects": False,
            }
            if self.toolset.config.include_value_previews:
                record["arguments"] = _preview(
                    arguments, self.toolset.config.preview_chars
                )
            if len(self._records) < self.toolset.config.max_recorded_calls:
                self._records.append(record)
            else:
                self._records_truncated = True
            return call_id, record

    def _finish(
        self,
        record: dict[str, Any],
        started_at: float,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """Finish one record and emit safe metadata.

        Args:
            record: Mutable record created at reservation.
            started_at: Monotonic host-call start time.
            result: Optional successful result.
            error: Optional raised error.
        """

        duration = time.monotonic() - started_at
        with self._guard:
            record["duration_seconds"] = duration
            if error is None:
                record["status"] = "success"
                if self.toolset.config.include_value_previews:
                    record["result"] = _preview(
                        result, self.toolset.config.preview_chars
                    )
            else:
                record["status"] = "error"
                record["may_have_side_effects"] = True
                record["error"] = _preview(
                    f"{type(error).__name__}: {error}",
                    self.toolset.config.preview_chars,
                )
        emit_event(
            self.event_sink,
            "hosted_tool",
            "finished",
            execution_id=self.execution_id,
            call_id=record["call_id"],
            tool=record["tool"],
            status=record["status"],
            duration_seconds=duration,
        )

    def _sync_wrapper(self, tool: HostedTool) -> Callable[..., Any]:
        """Build one synchronous host wrapper.

        Args:
            tool: Tool to invoke.

        Returns:
            Keyword-only host wrapper.
        """

        def call(*args: Any, **kwargs: Any) -> Any:
            """Reserve, invoke, and record one synchronous hosted call."""

            if args:
                raise TypeError(f"{tool.alias}() accepts keyword arguments only")
            _, record = self._reserve(tool, kwargs)
            started_at = time.monotonic()
            emit_event(
                self.event_sink,
                "hosted_tool",
                "started",
                execution_id=self.execution_id,
                call_id=record["call_id"],
                tool=tool.name,
            )
            self._enter_sync(tool.sequential)
            try:
                with self._sync_slots:
                    assert tool.invoke is not None
                    result = tool.invoke(kwargs)
            except BaseException as exc:
                self._finish(record, started_at, error=exc)
                raise
            finally:
                self._leave_sync(tool.sequential)
            self._finish(record, started_at, result=result)
            return _boundary_value(result)

        return call

    def _async_wrapper(self, tool: HostedTool) -> Callable[..., Awaitable[Any]]:
        """Build one asynchronous host wrapper.

        Args:
            tool: Tool to invoke.

        Returns:
            Keyword-only asynchronous host wrapper.
        """

        async def call(*args: Any, **kwargs: Any) -> Any:
            """Reserve, invoke, and record one asynchronous hosted call."""

            if args:
                raise TypeError(f"{tool.alias}() accepts keyword arguments only")
            _, record = self._reserve(tool, kwargs)
            started_at = time.monotonic()
            emit_event(
                self.event_sink,
                "hosted_tool",
                "started",
                execution_id=self.execution_id,
                call_id=record["call_id"],
                tool=tool.name,
            )
            await self._enter_async(tool.sequential)
            try:
                async with self._async_slots:
                    assert self._deadline is not None
                    remaining = self._deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("hosted tool operation time limit exceeded")
                    result = await asyncio.wait_for(
                        _invoke_tool(tool, kwargs), timeout=remaining
                    )
            except BaseException as exc:
                self._finish(record, started_at, error=exc)
                raise
            finally:
                await self._leave_async(tool.sequential)
            self._finish(record, started_at, result=result)
            return _boundary_value(result)

        return call

    def _enter_sync(self, sequential: bool) -> None:
        """Enter the synchronous parallel or sequential gate.

        Args:
            sequential: Whether this call needs exclusive access.
        """

        with self._sync_gate:
            if sequential:
                self._sync_gate.wait_for(
                    lambda: not self._sync_sequential and self._sync_parallel == 0
                )
                self._sync_sequential = True
            else:
                self._sync_gate.wait_for(lambda: not self._sync_sequential)
                self._sync_parallel += 1

    def _leave_sync(self, sequential: bool) -> None:
        """Leave the synchronous execution gate.

        Args:
            sequential: Whether this call held exclusive access.
        """

        with self._sync_gate:
            if sequential:
                self._sync_sequential = False
            else:
                self._sync_parallel -= 1
            self._sync_gate.notify_all()

    async def _enter_async(self, sequential: bool) -> None:
        """Enter the asynchronous parallel or sequential gate.

        Args:
            sequential: Whether this call needs exclusive access.
        """

        async with self._async_gate:
            if sequential:
                await self._async_gate.wait_for(
                    lambda: not self._async_sequential and self._async_parallel == 0
                )
                self._async_sequential = True
            else:
                await self._async_gate.wait_for(lambda: not self._async_sequential)
                self._async_parallel += 1

    async def _leave_async(self, sequential: bool) -> None:
        """Leave the asynchronous execution gate.

        Args:
            sequential: Whether this call held exclusive access.
        """

        async with self._async_gate:
            if sequential:
                self._async_sequential = False
            else:
                self._async_parallel -= 1
            self._async_gate.notify_all()


def python_identifier(name: str) -> str:
    """Convert an external tool name to a valid Python identifier.

    Args:
        name: External tool name.

    Returns:
        Deterministic Python-safe name.
    """

    value = _INVALID_NAME.sub("_", name)
    if not value:
        value = "_"
    if value[0].isdigit():
        value = f"_{value}"
    if keyword.iskeyword(value):
        value = f"{value}_"
    return value


def _callable_schemas(
    function: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Derive small JSON schemas from a callable signature.

    Args:
        function: Annotated Python callable.

    Returns:
        Input object schema and optional return schema.
    """

    signature = inspect.signature(function)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in {
            parameter.POSITIONAL_ONLY,
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        }:
            raise ValueError(
                "hosted callables must accept named parameters without *args or **kwargs"
            )
        schema = _annotation_schema(parameter.annotation)
        if parameter.default is parameter.empty:
            required.append(parameter.name)
        else:
            schema["default"] = parameter.default
        properties[parameter.name] = schema
    input_schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        input_schema["required"] = required
    output_schema = (
        None
        if signature.return_annotation in {signature.empty, Any}
        else _annotation_schema(signature.return_annotation)
    )
    return input_schema, output_schema


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    """Convert a common Python annotation to small JSON Schema.

    Args:
        annotation: Python type annotation.

    Returns:
        JSON-compatible schema.
    """

    if annotation in {inspect.Signature.empty, Any}:
        return {}
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        return {"enum": list(arguments)}
    if origin in {list, tuple, Sequence}:
        return {
            "type": "array",
            "items": _annotation_schema(arguments[0]) if arguments else {},
        }
    if origin in {dict, Mapping}:
        return {"type": "object"}
    if origin is not None and arguments:
        return {"anyOf": [_annotation_schema(value) for value in arguments]}
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        bytes: "string",
        type(None): "null",
    }
    return {"type": mapping[annotation]} if annotation in mapping else {}


def _render_tool(tool: HostedTool, *, async_mode: bool) -> tuple[list[str], str]:
    """Render type definitions and one function stub.

    Args:
        tool: Hosted tool definition.
        async_mode: Render an asynchronous function when true.

    Returns:
        Type definition blocks and the function signature.
    """

    input_schema = dict(tool.input_schema)
    definitions, aliases = _render_definitions(tool.alias, input_schema)
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    parameters: list[str] = []
    for name, schema in properties.items():
        type_name = _schema_type(schema, aliases)
        if name in required:
            parameters.append(f"{name}: {type_name}")
        else:
            default = schema.get("default")
            if "default" not in schema and "None" not in type_name:
                type_name = f"{type_name} | None"
            parameters.append(f"{name}: {type_name} = {_python_literal(default)}")
    args = "*, " + ", ".join(parameters) if parameters else ""
    return_type = _schema_type(tool.output_schema or {}, aliases)
    prefix = "async " if async_mode else ""
    signature = f"{prefix}def {tool.alias}({args}) -> {return_type}: ..."
    if tool.description:
        signature = f"# {tool.description.splitlines()[0]}\n{signature}"
    return definitions, signature


def _render_definitions(
    prefix: str, schema: Mapping[str, Any]
) -> tuple[list[str], dict[str, str]]:
    """Render JSON Schema definitions as unique TypedDict classes.

    Args:
        prefix: Tool alias used to prevent type-name collisions.
        schema: Tool input schema.

    Returns:
        Rendered blocks and JSON reference aliases.
    """

    raw_definitions = schema.get("$defs", {})
    aliases = {name: _class_name(f"{prefix}_{name}") for name in raw_definitions}
    blocks: list[str] = []
    for name, definition in raw_definitions.items():
        if definition.get("type") != "object":
            continue
        required = set(definition.get("required", []))
        fields: list[str] = []
        for field_name, field_schema in definition.get("properties", {}).items():
            field_type = _schema_type(field_schema, aliases)
            if field_name not in required:
                field_type = f"NotRequired[{field_type}]"
            fields.append(f"    {field_name}: {field_type}")
        body = "\n".join(fields) if fields else "    pass"
        blocks.append(f"class {aliases[name]}(TypedDict):\n{body}")
    return blocks, aliases


def _schema_type(schema: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    """Render a JSON Schema value as a Monty-compatible type expression.

    Args:
        schema: JSON Schema value.
        aliases: JSON definition names mapped to generated class names.

    Returns:
        Python type expression.
    """

    reference = schema.get("$ref")
    if isinstance(reference, str):
        return aliases.get(reference.rsplit("/", 1)[-1], "Any")
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return f"Literal[{', '.join(_python_literal(value) for value in enum)}]"
    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list) and variants:
        return " | ".join(_schema_type(value, aliases) for value in variants)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(
            _schema_type({"type": value}, aliases) for value in schema_type
        )
    if schema_type == "array":
        return f"list[{_schema_type(schema.get('items', {}), aliases)}]"
    if schema_type == "object":
        return "dict[str, Any]"
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "null": "None",
    }.get(schema_type, "Any")


def _class_name(value: str) -> str:
    """Create a simple PascalCase class name.

    Args:
        value: Source identifier.

    Returns:
        Valid class name.
    """

    parts = [part for part in _INVALID_NAME.sub("_", value).split("_") if part]
    name = "".join(part[:1].upper() + part[1:] for part in parts) or "Value"
    return f"_{name}" if name[0].isdigit() else name


def _python_literal(value: Any) -> str:
    """Render a JSON-like value as a Python literal.

    Args:
        value: Default or enum value.

    Returns:
        Python literal text.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value)
    return repr(value)


def _metadata_contains(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Test whether tool metadata contains selected values.

    Args:
        actual: Tool metadata.
        expected: Required metadata values.

    Returns:
        ``True`` when every expected value matches.
    """

    return all(actual.get(key) == value for key, value in expected.items())


def _strictest_config(
    left: HostedToolConfig, right: HostedToolConfig
) -> HostedToolConfig:
    """Combine hosted-tool configurations using the safer limits.

    Args:
        left: First operation configuration.
        right: Additional operation configuration.

    Returns:
        Configuration with the lower numeric limits.
    """

    return HostedToolConfig(
        max_calls_per_operation=min(
            left.max_calls_per_operation, right.max_calls_per_operation
        ),
        max_total_call_seconds=min(
            left.max_total_call_seconds, right.max_total_call_seconds
        ),
        max_parallel_calls=min(left.max_parallel_calls, right.max_parallel_calls),
        max_recorded_calls=min(left.max_recorded_calls, right.max_recorded_calls),
        preview_chars=min(left.preview_chars, right.preview_chars),
        include_value_previews=(
            left.include_value_previews and right.include_value_previews
        ),
    )


def _preview(value: Any, limit: int) -> str:
    """Return a bounded representation without expanding arbitrary objects.

    Args:
        value: Value to preview.
        limit: Maximum character count.

    Returns:
        Bounded safe representation.
    """

    if isinstance(value, str):
        text = value
    elif isinstance(value, _PLAIN_TYPES):
        text = repr(value)
    else:
        try:
            text = json.dumps(_boundary_value(value), ensure_ascii=False)
        except (TypeError, ValueError):
            text = f"<{type(value).__name__}>"
    if len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text)} chars)"


async def _invoke_tool(tool: HostedTool, arguments: dict[str, Any]) -> Any:
    """Invoke one hosted tool without blocking the caller event loop.

    Args:
        tool: Tool with an async invoker, a sync invoker, or both.
        arguments: Concrete keyword arguments from Monty or speculation.

    Returns:
        Raw framework result. The async invoker takes priority; sync-only tools
        run in a host thread. Exceptions and cancellation reach the caller.
    """
    if tool.ainvoke is not None:
        return await tool.ainvoke(arguments)
    assert tool.invoke is not None
    return await asyncio.to_thread(tool.invoke, arguments)


def _boundary_value(value: Any) -> Any:
    """Convert common framework values to Monty boundary values.

    Args:
        value: Raw hosted-tool return value.

    Returns:
        Plain value suitable for the Monty host boundary when possible.
    """

    if isinstance(value, _PLAIN_TYPES):
        return value
    if isinstance(value, Mapping):
        return {str(key): _boundary_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_boundary_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _boundary_value(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _boundary_value(model_dump(mode="python"))
    return value
