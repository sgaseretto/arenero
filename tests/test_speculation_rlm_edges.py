"""Integration tests for speculative RLM budgets and lifecycle cleanup."""

import asyncio
import unittest
from collections.abc import AsyncIterator

from arenero import (
    RLM,
    AsyncPersistentMontyKernel,
    KernelConfig,
    RLMConfig,
    RLMModelResponse,
    SpeculationConfig,
)


def _kernel() -> AsyncPersistentMontyKernel:
    """Create an async kernel for recursive and speculative test sessions.

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


class SpeculativeRLMBudgetTests(unittest.IsolatedAsyncioTestCase):
    """Check shared usage, concurrency, and failure limits with speculation."""

    async def test_stream_and_speculative_call_usage_is_counted_once(self) -> None:
        """Count root stream and claimed subquery usage one time each."""

        async def model(prompt: str, model_name: str | None) -> RLMModelResponse:
            """Return one deterministic subquery with reported usage.

            Args:
                prompt: Plain subquery prompt.
                model_name: Unused model selector.

            Returns:
                Uppercase prompt and synthetic usage.
            """

            del model_name
            return RLMModelResponse(
                prompt.upper(), input_tokens=5, output_tokens=6, cost=0.3
            )

        async def stream(
            prompt: str, model_name: str | None
        ) -> AsyncIterator[RLMModelResponse]:
            """Stream one controller program with usage on the last delta.

            Args:
                prompt: Root controller prompt.
                model_name: Unused model selector.

            Yields:
                Controller source and one usage report.
            """

            del prompt, model_name
            yield RLMModelResponse("```python\n")
            yield RLMModelResponse("value = await llm_query(prompt='one')\n")
            yield RLMModelResponse("answer['content'] = value\n")
            yield RLMModelResponse(
                "answer['ready'] = True\n```",
                input_tokens=3,
                output_tokens=4,
                cost=0.2,
            )

        runner = RLM(
            model,
            model_stream=stream,
            config=RLMConfig(timeout_seconds=5),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with _kernel() as kernel:
            result = await runner.complete(kernel, "context")

        self.assertEqual(result.response, "ONE")
        self.assertEqual(result.usage.model_calls, 2)
        self.assertEqual(result.usage.input_tokens, 8)
        self.assertEqual(result.usage.output_tokens, 10)
        self.assertAlmostEqual(result.usage.cost, 0.5)
        self.assertEqual(result.usage.speculation.hits, 1)

    async def test_parallel_limit_applies_to_speculative_model_calls(self) -> None:
        """Keep speculative subqueries inside the shared parallel-call limit."""

        active = 0
        maximum_active = 0

        async def model(prompt: str, model_name: str | None) -> str:
            """Measure active subqueries and return their prompt.

            Args:
                prompt: Plain subquery prompt.
                model_name: Unused model selector.

            Returns:
                Original subquery prompt.
            """

            nonlocal active, maximum_active
            del model_name
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return prompt

        async def stream(prompt: str, model_name: str | None) -> AsyncIterator[str]:
            """Stream two independent calls before final controller code.

            Args:
                prompt: Root controller prompt.
                model_name: Unused model selector.

            Yields:
                Controller source deltas.
            """

            del prompt, model_name
            yield "```python\n"
            yield "one = await llm_query(prompt='one')\n"
            yield "two = await llm_query(prompt='two')\n"
            yield "answer['content'] = one + ':' + two\n"
            yield "answer['ready'] = True\n```"

        runner = RLM(
            model,
            model_stream=stream,
            config=RLMConfig(max_parallel_calls=1, timeout_seconds=5),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with _kernel() as kernel:
            result = await runner.complete(kernel, "context")

        self.assertEqual(result.response, "one:two")
        self.assertEqual(maximum_active, 1)
        self.assertEqual(result.usage.speculation.hits, 2)

    async def test_shared_call_budget_rejects_speculative_subquery(self) -> None:
        """Reject a speculative call after the root uses the call budget."""

        subquery_calls = 0

        async def model(prompt: str, model_name: str | None) -> str:
            """Record a subquery that must not reach the adapter.

            Args:
                prompt: Plain subquery prompt.
                model_name: Unused model selector.

            Returns:
                Unused response text.
            """

            nonlocal subquery_calls
            del prompt, model_name
            subquery_calls += 1
            return "unexpected"

        async def stream(prompt: str, model_name: str | None) -> AsyncIterator[str]:
            """Stream code that requests one call beyond the shared limit.

            Args:
                prompt: Root controller prompt.
                model_name: Unused model selector.

            Yields:
                Controller source deltas.
            """

            del prompt, model_name
            yield "```python\n"
            yield "answer['content'] = await llm_query(prompt='blocked')\n"
            yield "answer['ready'] = True\n```"

        runner = RLM(
            model,
            model_stream=stream,
            config=RLMConfig(
                max_model_calls=1,
                max_errors=1,
                timeout_seconds=5,
            ),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with _kernel() as kernel:
            with self.assertRaisesRegex(RuntimeError, "RLMCallLimitExceeded"):
                await runner.complete(kernel, "context")
            sessions = kernel.store.list_sessions()

        self.assertEqual(subquery_calls, 0)
        self.assertEqual(sessions, [])


class SpeculativeRLMLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Check that cancellation and stream errors remove temporary sessions."""

    async def test_cancellation_removes_root_and_shadow_sessions(self) -> None:
        """Cancel a blocked completion and remove all temporary state."""

        streamed_call = asyncio.Event()
        keep_stream_open = asyncio.Event()

        async def model(prompt: str, model_name: str | None) -> str:
            """Return a subquery result if the pending call gets a slot.

            Args:
                prompt: Plain subquery prompt.
                model_name: Unused model selector.

            Returns:
                Original prompt.
            """

            del model_name
            return prompt

        async def stream(prompt: str, model_name: str | None) -> AsyncIterator[str]:
            """Open a speculative call and then keep the root stream active.

            Args:
                prompt: Root controller prompt.
                model_name: Unused model selector.

            Yields:
                Partial controller source before blocking.
            """

            del prompt, model_name
            yield "```python\n"
            yield "value = await llm_query(prompt='pending')\n"
            streamed_call.set()
            await keep_stream_open.wait()
            yield "answer['content'] = value\nanswer['ready'] = True\n```"

        runner = RLM(
            model,
            model_stream=stream,
            config=RLMConfig(max_parallel_calls=1, timeout_seconds=5),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with _kernel() as kernel:
            task = asyncio.create_task(runner.complete(kernel, "context"))
            await asyncio.wait_for(streamed_call.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            sessions = kernel.store.list_sessions()

        self.assertEqual(sessions, [])

    async def test_stream_failure_removes_root_and_shadow_sessions(self) -> None:
        """Remove all temporary state when the root model stream fails."""

        async def model(prompt: str, model_name: str | None) -> str:
            """Return a plain subquery response.

            Args:
                prompt: Plain subquery prompt.
                model_name: Unused model selector.

            Returns:
                Original prompt.
            """

            del model_name
            return prompt

        async def stream(prompt: str, model_name: str | None) -> AsyncIterator[str]:
            """Start one speculative call and then raise a stream error.

            Args:
                prompt: Root controller prompt.
                model_name: Unused model selector.

            Yields:
                Partial controller source.

            Raises:
                RuntimeError: Always, after the partial source.
            """

            del prompt, model_name
            yield "```python\n"
            yield "value = await llm_query(prompt='pending')\n"
            raise RuntimeError("synthetic stream failure")

        runner = RLM(
            model,
            model_stream=stream,
            config=RLMConfig(timeout_seconds=5),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with _kernel() as kernel:
            with self.assertRaisesRegex(RuntimeError, "synthetic stream failure"):
                await runner.complete(kernel, "context")
            sessions = kernel.store.list_sessions()

        self.assertEqual(sessions, [])


if __name__ == "__main__":
    unittest.main()
