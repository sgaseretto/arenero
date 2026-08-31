"""Optional LangChain model adapter for Arenero RLM calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .rlm import RLMModelCall, RLMModelResponse


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

        if requested_model is None:
            selected = model
        else:
            selected = named.get(requested_model)
            if selected is None:
                raise ValueError(f"unknown RLM model name: {requested_model}")
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
