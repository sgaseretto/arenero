from collections.abc import Callable
from typing import Any

from .async_kernel import AsyncPersistentMontyKernel
from .kernel import PersistentMontyKernel


def create_python_repl_tools(
    kernel: PersistentMontyKernel,
    session_id_resolver: Callable[[Any], str | None] | None = None,
) -> list[Any]:
    """Create LangChain tools when the optional integration is installed.

    Args:
        kernel: Kernel used by every generated tool.
        session_id_resolver: Optional function that maps runtime data to a
            kernel session ID.

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
    return create_tools(kernel, session_id_resolver)


def create_async_python_repl_tools(
    kernel: AsyncPersistentMontyKernel,
    session_id_resolver: Callable[[Any], str | None] | None = None,
) -> list[Any]:
    """Create native async LangChain tools when the extra is installed.

    Args:
        kernel: Native async kernel used by every generated tool.
        session_id_resolver: Optional runtime-to-session mapping function.

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
    return create_tools(kernel, session_id_resolver)
