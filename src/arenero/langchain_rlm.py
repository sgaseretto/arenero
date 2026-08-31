"""Optional LangChain model adapter for Arenero RLM calls."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .rlm import RLMModelCall, RLMModelResponse, RLMModelStreamCall


def model_call_from_langchain(
    model: BaseChatModel,
    *,
    models: Mapping[str, BaseChatModel] | None = None,
) -> RLMModelCall:
    """Adapt LangChain chat models to the framework-neutral RLM callback.

    Args:
        model: Default chat model.
        models: Optional names that Monty code can select with ``model=``.

    Returns:
        Async callback accepted by :class:`arenero.RLM`.
    """

    named = dict(models or {})

    async def call(prompt: str, requested_model: str | None) -> RLMModelResponse:
        """Invoke one selected LangChain model and normalize its usage.

        Args:
            prompt: Complete RLM model input.
            requested_model: Optional configured model name.

        Returns:
            Text and available LangChain usage metadata.
        """

        selected = _select_model(model, named, requested_model)
        message = await selected.ainvoke(prompt)
        usage = getattr(message, "usage_metadata", None) or {}
        metadata = getattr(message, "response_metadata", None) or {}
        return RLMModelResponse(
            content=_message_text(message),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cost=float(metadata.get("cost", 0.0) or 0.0),
        )

    return call


def model_stream_from_langchain(
    model: BaseChatModel,
    *,
    models: Mapping[str, BaseChatModel] | None = None,
) -> RLMModelStreamCall:
    """Adapt LangChain chat-model streaming to RLM text deltas.

    Args:
        model: Default chat model.
        models: Optional names that Monty code can select with ``model=``.

    Returns:
        Async delta iterator factory accepted by :class:`arenero.RLM`.
    """

    named = dict(models or {})

    async def stream(
        prompt: str, requested_model: str | None
    ) -> AsyncIterator[RLMModelResponse]:
        """Yield text and per-chunk usage from one selected model.

        Args:
            prompt: Complete RLM model input.
            requested_model: Optional configured model name.

        Yields:
            Text deltas with any usage reported on that chunk.
        """

        selected = _select_model(model, named, requested_model)
        async for message in selected.astream(prompt):
            usage = getattr(message, "usage_metadata", None) or {}
            metadata = getattr(message, "response_metadata", None) or {}
            yield RLMModelResponse(
                content=_message_text(message),
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cost=float(metadata.get("cost", 0.0) or 0.0),
            )

    return stream


def _select_model(
    default: BaseChatModel,
    named: Mapping[str, BaseChatModel],
    requested: str | None,
) -> BaseChatModel:
    """Resolve one optional public model name.

    Args:
        default: Default chat model.
        named: Allowed named models.
        requested: Optional requested name.

    Returns:
        Selected chat model.

    Raises:
        ValueError: If the requested model name is not configured.
    """

    if requested is None:
        return default
    selected = named.get(requested)
    if selected is None:
        raise ValueError(f"unknown RLM model name: {requested}")
    return selected


def _message_text(message: Any) -> str:
    """Read text from a LangChain message without losing text blocks.

    Args:
        message: LangChain model response message.

    Returns:
        Combined response text.
    """

    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content if isinstance(content, list) else ():
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)
