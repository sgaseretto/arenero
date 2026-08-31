"""Tests for Recursive Language Model support."""

import asyncio
import unittest

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic_monty import OSAccess

from arenero import (
    RLM,
    AsyncPersistentMontyKernel,
    KernelConfig,
    RLMConfig,
    RLMModelResponse,
    adapt_langchain_model,
    adapt_langchain_model_stream,
)


def _kernel(*, filesystem: bool = False) -> AsyncPersistentMontyKernel:
    """Create a kernel with enough workers for nested RLM sessions.

    Args:
        filesystem: Add an in-memory filesystem when true.

    Returns:
        Configured asynchronous test kernel.
    """

    return AsyncPersistentMontyKernel(
        config=KernelConfig(
            max_worker_processes=4,
            worker_checkout_timeout_seconds=2,
            request_timeout_seconds=5,
        ),
        os_handler=OSAccess() if filesystem else None,
    )


class RLMTests(unittest.IsolatedAsyncioTestCase):
    """Check plain, batched, recursive, and bounded model calls."""

    async def test_plain_batch_calls_report_usage(self) -> None:
        """Run ordered concurrent calls and collect adapter usage."""

        async def model(prompt: str, model_name: str | None) -> RLMModelResponse:
            """Return controller code or one uppercase plain response.

            Args:
                prompt: Root prompt or plain subquery.
                model_name: Unused optional model selector.

            Returns:
                Text with deterministic usage.
            """

            if prompt.startswith("You are an RLM controller"):
                content = """```python
values = await llm_query_batched(prompts=['alpha', 'beta'])
answer['content'] = values[0] + ':' + values[1]
answer['ready'] = True
```"""
            else:
                await asyncio.sleep(0)
                content = prompt.upper()
            return RLMModelResponse(content, input_tokens=1, output_tokens=2, cost=0.1)

        rlm = RLM(model, config=RLMConfig(timeout_seconds=5))
        async with _kernel() as kernel:
            result = await rlm.complete(
                kernel,
                "large external context",
                "Combine two model calls.",
            )

        self.assertEqual(result.response, "ALPHA:BETA")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.usage.model_calls, 3)
        self.assertEqual(result.usage.total_tokens, 9)
        self.assertAlmostEqual(result.usage.cost, 0.3)
        self.assertEqual(rlm.capabilities()["limits"]["max_depth"], 1)

    async def test_recursive_child_gets_an_isolated_repl(self) -> None:
        """Run a child RLM with a separate temporary Monty session."""

        async def model(prompt: str, model_name: str | None) -> str:
            """Return root or child controller code.

            Args:
                prompt: Current root prompt.
                model_name: Unused optional model selector.

            Returns:
                Deterministic RLM code.
            """

            if "RLM depth: 0" in prompt:
                return """```python
child = await rlm_query(prompt='child request')
answer['content'] = child
answer['ready'] = True
```"""
            return """```python
child_only = 42
answer['content'] = context.upper()
answer['ready'] = True
```"""

        kernel = _kernel(filesystem=True)
        async with kernel:
            result = await RLM(
                model,
                config=RLMConfig(max_depth=1, timeout_seconds=5),
            ).complete(kernel, "root request")
            sessions = kernel.store.list_sessions()

        self.assertEqual(result.response, "CHILD REQUEST")
        self.assertEqual(result.usage.recursive_calls, 1)
        self.assertEqual(result.usage.deepest_level, 1)
        self.assertEqual(sessions, [])

    async def test_depth_limit_falls_back_to_plain_model_call(self) -> None:
        """Use a plain model call when a child asks for deeper recursion."""

        async def model(prompt: str, model_name: str | None) -> str:
            """Drive a root, child, and depth-limited leaf call.

            Args:
                prompt: Root prompt or exact leaf prompt.
                model_name: Unused optional model selector.

            Returns:
                Controller code or leaf text.
            """

            if "RLM depth: 0" in prompt:
                return """```python
answer['content'] = await rlm_query(prompt='child')
answer['ready'] = True
```"""
            if "RLM depth: 1" in prompt:
                return """```python
answer['content'] = await rlm_query(prompt='leaf')
answer['ready'] = True
```"""
            return f"plain:{prompt}"

        async with _kernel() as kernel:
            result = await RLM(
                model,
                config=RLMConfig(max_depth=1, timeout_seconds=5),
            ).complete(kernel, "root")

        self.assertEqual(result.response, "plain:leaf")
        self.assertEqual(result.usage.model_calls, 3)
        self.assertEqual(result.usage.recursive_calls, 1)

    async def test_model_call_budget_stops_generated_code(self) -> None:
        """Stop a hosted subquery after the shared call budget is spent."""

        async def model(prompt: str, model_name: str | None) -> str:
            """Always ask for one model call from generated code.

            Args:
                prompt: Current model prompt.
                model_name: Unused optional model selector.

            Returns:
                One controller code block.
            """

            return """```python
answer['content'] = await llm_query(prompt='over budget')
answer['ready'] = True
```"""

        async with _kernel() as kernel:
            with self.assertRaisesRegex(RuntimeError, "RLMCallLimitExceeded"):
                await RLM(
                    model,
                    config=RLMConfig(
                        max_model_calls=1,
                        max_errors=1,
                        timeout_seconds=5,
                    ),
                ).complete(kernel, "root")

    async def test_langchain_adapter_preserves_text_and_usage(self) -> None:
        """Convert a LangChain chat response into an RLM response."""

        model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="adapter result",
                    usage_metadata={
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                    },
                )
            ]
        )
        call = adapt_langchain_model(model)

        response = await call("prompt", None)

        self.assertIsInstance(response, RLMModelResponse)
        assert isinstance(response, RLMModelResponse)
        self.assertEqual(response.content, "adapter result")
        self.assertEqual(response.input_tokens, 3)
        self.assertEqual(response.output_tokens, 4)

    async def test_langchain_stream_adapter_yields_text_deltas(self) -> None:
        """Convert LangChain streamed chunks into RLM response deltas."""

        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="streamed result")]
        )
        stream = adapt_langchain_model_stream(model)

        chunks = [chunk async for chunk in stream("prompt", None)]

        text = "".join(
            chunk if isinstance(chunk, str) else chunk.content for chunk in chunks
        )
        self.assertEqual(text, "streamed result")
