"""Deterministic LangChain and LangGraph tests for speculative RLM execution."""

import operator
import unittest
from typing import Annotated

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from arenero import (
    RLM,
    AsyncPersistentMontyKernel,
    KernelConfig,
    RLMConfig,
    SpeculationConfig,
    adapt_langchain_model,
    adapt_langchain_model_stream,
)


class WorkflowState(TypedDict, total=False):
    """Define input and accumulated output for the test graph."""

    context: str
    answers: Annotated[list[str], operator.add]


def _kernel() -> AsyncPersistentMontyKernel:
    """Create an async kernel for framework integration tests.

    Returns:
        Kernel with enough workers for root and shadow sessions.
    """

    return AsyncPersistentMontyKernel(
        config=KernelConfig(
            max_worker_processes=6,
            worker_checkout_timeout_seconds=2,
            request_timeout_seconds=5,
        )
    )


def _fake_model(*responses: str) -> GenericFakeChatModel:
    """Create a deterministic model with an ordered response iterator.

    Args:
        responses: Responses consumed by root and nested model calls.

    Returns:
        Generic LangChain fake chat model.
    """

    return GenericFakeChatModel(
        messages=iter(AIMessage(content=response) for response in responses)
    )


def _runner(model: GenericFakeChatModel) -> RLM:
    """Adapt one LangChain fake model for streaming speculative RLM use.

    Args:
        model: Deterministic LangChain fake chat model.

    Returns:
        Configured RLM runner.
    """

    return RLM(
        adapt_langchain_model(model),
        model_stream=adapt_langchain_model_stream(model),
        config=RLMConfig(timeout_seconds=5),
        speculation_config=SpeculationConfig(mode="stream"),
    )


def _controller(prompt: str) -> str:
    """Build one controller response that calls the model from Monty.

    Args:
        prompt: Exact nested model prompt.

    Returns:
        Complete Python code fence for the RLM controller.
    """

    return (
        "```python\n"
        f"value = await llm_query(prompt={prompt!r})\n"
        "answer['content'] = value\n"
        "answer['ready'] = True\n"
        "```"
    )


class LangChainSpeculationTests(unittest.IsolatedAsyncioTestCase):
    """Run speculative RLM behavior with LangChain's generic fake model."""

    async def test_fake_model_runs_stream_and_nested_call_end_to_end(self) -> None:
        """Use one ordered fake for a streamed root and claimed nested call."""

        model = _fake_model(_controller("nested"), "deterministic result")
        async with _kernel() as kernel:
            result = await _runner(model).complete(kernel, "context")

        self.assertEqual(result.response, "deterministic result")
        self.assertEqual(result.usage.model_calls, 2)
        self.assertEqual(result.usage.speculation.dispatched, 1)
        self.assertEqual(result.usage.speculation.hits, 1)
        self.assertEqual(result.usage.speculation.misses, 0)


class LangGraphSpeculationTests(unittest.IsolatedAsyncioTestCase):
    """Run RLM nodes with a fresh graph and in-memory checkpointer."""

    async def test_checkpointer_preserves_prior_answers_across_turns(self) -> None:
        """Keep graph state across two turns while RLM sessions stay temporary."""

        model = _fake_model(
            _controller("first"),
            "FIRST",
            _controller("second"),
            "SECOND",
        )
        runner = _runner(model)

        async with _kernel() as kernel:

            async def complete_node(state: WorkflowState) -> dict[str, list[str]]:
                """Run one isolated RLM operation for the current graph input.

                Args:
                    state: Current graph state and accumulated answers.

                Returns:
                    One answer for the graph reducer.
                """

                result = await runner.complete(kernel, state["context"])
                return {"answers": [result.response]}

            # LangGraph accepts typing_extensions.TypedDict at runtime. Ty does not
            # yet recognize it as LangGraph's private TypedDictLike bound.
            builder = StateGraph(WorkflowState)  # ty: ignore[invalid-argument-type]
            builder.add_node("complete", complete_node)
            builder.add_edge(START, "complete")
            builder.add_edge("complete", END)
            graph = builder.compile(checkpointer=InMemorySaver())
            config: RunnableConfig = {"configurable": {"thread_id": "test-thread"}}

            first = await graph.ainvoke(
                {"context": "first context", "answers": []}, config
            )
            second = await graph.ainvoke({"context": "second context"}, config)
            sessions = kernel.store.list_sessions()

        self.assertEqual(first["answers"], ["FIRST"])
        self.assertEqual(second["answers"], ["FIRST", "SECOND"])
        self.assertEqual(sessions, [])

    async def test_node_can_be_called_directly_without_graph_execution(self) -> None:
        """Test an RLM node as a plain function with no graph runtime."""

        model = _fake_model(_controller("direct"), "DIRECT")
        runner = _runner(model)
        async with _kernel() as kernel:

            async def complete_node(state: WorkflowState) -> dict[str, list[str]]:
                """Run the node logic without graph compilation.

                Args:
                    state: Direct node input.

                Returns:
                    One answer compatible with the graph reducer.
                """

                result = await runner.complete(kernel, state["context"])
                return {"answers": [result.response]}

            update = await complete_node({"context": "direct context", "answers": []})

        self.assertEqual(update, {"answers": ["DIRECT"]})


if __name__ == "__main__":
    unittest.main()
