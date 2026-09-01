"""End-to-end tests for a model-driven LangChain agent using Arenero."""

import json
import unittest
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from arenero import AsyncPersistentMontyKernel, create_async_python_repl_tools


class ToolCallingFakeChatModel(GenericFakeChatModel):
    """Provide deterministic tool calls without a real model provider."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Accept agent tool binding and keep the scripted fake model.

        Args:
            tools: Tools exposed to the agent.
            tool_choice: Optional tool-selection constraint.
            **kwargs: Additional model binding options.

        Returns:
            This deterministic fake model.
        """

        del tools, tool_choice, kwargs
        return self


def named_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> AIMessage:
    """Build one scripted agent tool call.

    Args:
        call_id: Stable synthetic model call ID.
        name: Tool name selected by the fake model.
        arguments: Model-visible tool arguments.

    Returns:
        AI message with one tool call.
    """

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def python_tool_call(
    call_id: str,
    code: str,
    **arguments: Any,
) -> AIMessage:
    """Build one scripted Python REPL tool call.

    Args:
        call_id: Stable synthetic model call ID.
        code: Monty source sent to the REPL tool.
        **arguments: Optional additional REPL arguments.

    Returns:
        AI message with one `python_repl` call.
    """

    return named_tool_call(call_id, "python_repl", {"code": code, **arguments})


def tool_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode every JSON tool result in an agent state.

    Args:
        state: Completed LangGraph agent state.

    Returns:
        Parsed tool results in message order.
    """

    return [
        json.loads(message.content)
        for message in state["messages"]
        if isinstance(message, ToolMessage)
    ]


class LangChainAgentPersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Check REPL persistence through a complete two-turn agent graph."""

    async def test_repl_session_persists_across_two_human_turns(self) -> None:
        """Restore Monty state when the same agent thread calls the REPL again."""

        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("store-value", "value = 41"),
                    AIMessage(content="I stored the value."),
                    python_tool_call("increment-value", "value += 1\nvalue"),
                    AIMessage(content="The result is 42."),
                ]
            )
        )
        config: RunnableConfig = {"configurable": {"thread_id": "two-turn-agent"}}

        async with AsyncPersistentMontyKernel() as kernel:
            agent = create_agent(
                model,
                create_async_python_repl_tools(kernel),
                checkpointer=InMemorySaver(),
            )

            first = await agent.ainvoke(
                {"messages": [HumanMessage(content="Store the number 41 in Python.")]},
                config,
            )
            second = await agent.ainvoke(
                {"messages": [HumanMessage(content="Add one to the stored number.")]},
                config,
            )

            tool_messages = [
                message
                for message in second["messages"]
                if isinstance(message, ToolMessage)
            ]
            first_tool = json.loads(tool_messages[0].content)
            second_tool = json.loads(tool_messages[1].content)
            persisted = await kernel.execute(
                "two-turn-agent", "value", state_policy="never"
            )
            sessions = kernel.store.list_sessions()

        self.assertEqual(first["messages"][-1].content, "I stored the value.")
        self.assertEqual(second["messages"][-1].content, "The result is 42.")
        self.assertEqual(
            [
                message.content
                for message in second["messages"]
                if isinstance(message, HumanMessage)
            ],
            [
                "Store the number 41 in Python.",
                "Add one to the stored number.",
            ],
        )
        self.assertEqual(len(tool_messages), 2)
        self.assertFalse(first_tool["state_restored"])
        self.assertEqual(first_tool["session_revision"], 1)
        self.assertTrue(second_tool["state_restored"])
        self.assertEqual(second_tool["session_revision"], 2)
        self.assertEqual(second_tool["result"], 42)
        self.assertEqual(persisted["result"], 42)
        self.assertEqual(
            [session.session_id for session in sessions], ["two-turn-agent"]
        )


if __name__ == "__main__":
    unittest.main()
