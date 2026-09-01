from collections.abc import Callable, Mapping
from typing import Any

from .async_kernel import AsyncPersistentMontyKernel
from .hosted_tools import HostedToolConfig, HostedToolSet
from .kernel import PersistentMontyKernel
from .packages import PackageConfig
from .rlm import RLMModelCall, RLMModelStreamCall


def create_python_repl_tools(
    kernel: PersistentMontyKernel,
    session_id_resolver: Callable[[Any], str | None] | None = None,
    hosted_tools: tuple[Any, ...] = (),
    hosted_tool_config: HostedToolConfig | None = None,
) -> list[Any]:
    """Create LangChain tools when the optional integration is installed.

    Args:
        kernel: Kernel used by every generated tool.
        session_id_resolver: Optional function that maps runtime data to a
            kernel session ID.
        hosted_tools: LangChain tools callable from the REPL.
        hosted_tool_config: Optional nested-call and record limits.

    Returns:
        LangChain tools for execution, files, shell commands, and reset.

    Raises:
        ImportError: If the ``langchain`` package extra is not installed.
    """
    try:
        from .langchain_tools import create_python_repl_tools as create_tools
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "pydantic" or exc.name.startswith("langchain")):
            raise ImportError(
                "Install Arenero with the langchain extra: uv add 'arenero[langchain]'"
            ) from exc
        raise
    return create_tools(
        kernel,
        session_id_resolver,
        hosted_tools,
        hosted_tool_config,
    )


def create_async_python_repl_tools(
    kernel: AsyncPersistentMontyKernel,
    session_id_resolver: Callable[[Any], str | None] | None = None,
    hosted_tools: tuple[Any, ...] = (),
    hosted_tool_config: HostedToolConfig | None = None,
) -> list[Any]:
    """Create native async LangChain tools when the extra is installed.

    Args:
        kernel: Native async kernel used by every generated tool.
        session_id_resolver: Optional runtime-to-session mapping function.
        hosted_tools: LangChain tools callable from the REPL.
        hosted_tool_config: Optional nested-call and record limits.

    Returns:
        Async execute-code, execute-file, and reset tools.

    Raises:
        ImportError: If the ``langchain`` package extra is not installed.
    """
    try:
        from .langchain_tools import (
            create_async_python_repl_tools as create_tools,
        )
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "pydantic" or exc.name.startswith("langchain")):
            raise ImportError(
                "Install Arenero with the langchain extra: uv add 'arenero[langchain]'"
            ) from exc
        raise
    return create_tools(
        kernel,
        session_id_resolver,
        hosted_tools,
        hosted_tool_config,
    )


def create_package_tools(
    kernel: PersistentMontyKernel,
    config: PackageConfig | None = None,
) -> list[Any]:
    """Create LangChain package tools when the optional integration is installed.

    Args:
        kernel: Synchronous kernel with a writable virtual filesystem.
        config: Optional package roots and hard limits.

    Returns:
        Package initialization, check, test, and export tools.

    Raises:
        ImportError: If the ``langchain`` package extra is not installed.
    """

    try:
        from .langchain_package_tools import create_package_tools as create_tools
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "pydantic" or exc.name.startswith("langchain")):
            raise ImportError(
                "Install Arenero with the langchain extra: uv add 'arenero[langchain]'"
            ) from exc
        raise
    return create_tools(kernel, config)


def adapt_langchain_tools(
    tools: tuple[Any, ...],
    *,
    config: Any = None,
    runtime: Any = None,
    limits: HostedToolConfig | None = None,
) -> HostedToolSet:
    """Adapt LangChain tools when the optional integration is installed.

    Args:
        tools: LangChain tools to expose.
        config: Optional LangChain runtime configuration.
        runtime: Optional active ToolRuntime for injected arguments.
        limits: Optional nested-call and record limits.

    Returns:
        Framework-neutral hosted tool set.

    Raises:
        ImportError: If the ``langchain`` package extra is not installed.
    """

    try:
        from .langchain_hosted_tools import hosted_tools_from_langchain
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Install Arenero with the langchain extra: uv add 'arenero[langchain]'"
        ) from exc
    return hosted_tools_from_langchain(
        tools,
        config=config,
        runtime=runtime,
        limits=limits,
    )


async def adapt_pydantic_ai_tools(
    toolset: Any,
    ctx: Any,
    *,
    limits: HostedToolConfig | None = None,
) -> HostedToolSet:
    """Adapt the current Pydantic AI toolset and run context.

    Args:
        toolset: Entered Pydantic AI toolset.
        ctx: Current Pydantic AI run context.
        limits: Optional nested-call and record limits.

    Returns:
        Framework-neutral hosted tool set for one operation.

    Raises:
        ImportError: If the ``pydantic-ai`` extra is not installed.
    """

    try:
        from .pydantic_ai_hosted_tools import hosted_tools_from_pydantic_ai
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Install Arenero with the pydantic-ai extra: uv add 'arenero[pydantic-ai]'"
        ) from exc
    return await hosted_tools_from_pydantic_ai(
        toolset,
        ctx,
        limits=limits,
    )


def adapt_langchain_model(
    model: Any,
    *,
    models: Mapping[str, Any] | None = None,
) -> RLMModelCall:
    """Adapt LangChain chat models for recursive model calls.

    Args:
        model: Default LangChain chat model.
        models: Optional mapping of selectable model names to chat models.

    Returns:
        Async framework-neutral RLM model callback.

    Raises:
        ImportError: If the ``langchain`` extra is not installed.
    """

    try:
        from .langchain_rlm import model_call_from_langchain
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("langchain"):
            raise ImportError(
                "Install Arenero with the langchain extra: uv add 'arenero[langchain]'"
            ) from exc
        raise
    return model_call_from_langchain(model, models=models)


def adapt_langchain_model_stream(
    model: Any,
    *,
    models: Mapping[str, Any] | None = None,
) -> RLMModelStreamCall:
    """Adapt LangChain streaming for speculative RLM planning.

    Args:
        model: Default LangChain chat model.
        models: Optional mapping of selectable model names to chat models.

    Returns:
        Async framework-neutral RLM stream callback.

    Raises:
        ImportError: If the ``langchain`` extra is not installed.
    """

    try:
        from .langchain_rlm import model_stream_from_langchain
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("langchain"):
            raise ImportError(
                "Install Arenero with the langchain extra: uv add 'arenero[langchain]'"
            ) from exc
        raise
    return model_stream_from_langchain(model, models=models)


def create_deepagents_middleware(**kwargs: Any) -> Any:
    """Create optional Deep Agents interpreter middleware.

    Args:
        **kwargs: Arguments for ``AreneroInterpreterMiddleware``.

    Returns:
        Configured asynchronous interpreter middleware.

    Raises:
        ImportError: If the ``deepagents`` extra is not installed.
    """

    try:
        from .deepagents import AreneroInterpreterMiddleware
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name == "deepagents"
            or exc.name.startswith("langchain")
            or exc.name.startswith("langgraph")
        ):
            raise ImportError(
                "Install Arenero with the deepagents extra: "
                "uv add 'arenero[deepagents]'"
            ) from exc
        raise
    return AreneroInterpreterMiddleware(**kwargs)
