"""Tests for Monty-native speculative programmatic tool calls."""

import asyncio
import unittest
from collections.abc import AsyncIterator

from arenero import (
    RLM,
    AsyncPersistentMontyKernel,
    HostedTool,
    HostedToolSet,
    KernelConfig,
    RLMConfig,
    SpeculationConfig,
    SpeculativeToolPolicy,
    SpeculativeTurn,
)


def _kernel() -> AsyncPersistentMontyKernel:
    """Create a test kernel with room for a real and shadow session.

    Returns:
        Openable async kernel.
    """

    return AsyncPersistentMontyKernel(
        config=KernelConfig(
            min_worker_processes=2,
            max_worker_processes=4,
            worker_checkout_timeout_seconds=2,
            request_timeout_seconds=5,
        )
    )


class SpeculationTests(unittest.IsolatedAsyncioTestCase):
    """Check planning, claims, shadows, failure, and cleanup."""

    async def test_streamed_rlm_calls_start_early_and_run_once(self) -> None:
        """Claim two model calls that started while root code streamed."""

        calls: list[str] = []

        async def model(prompt: str, model_name: str | None) -> str:
            """Return one delayed subquery result.

            Args:
                prompt: Plain subquery prompt.
                model_name: Unused model selector.

            Returns:
                Uppercase prompt.
            """

            calls.append(prompt)
            await asyncio.sleep(0.02)
            return prompt.upper()

        async def stream(prompt: str, model_name: str | None) -> AsyncIterator[str]:
            """Stream one controller program by complete source lines.

            Args:
                prompt: Root controller prompt.
                model_name: Unused model selector.

            Yields:
                Controller source deltas.
            """

            del prompt, model_name
            yield "```python\n"
            yield "first = await llm_query(prompt='one')\n"
            yield "second = await llm_query(prompt='two')\n"
            await asyncio.sleep(0.04)
            self.assertCountEqual(calls, ["one", "two"])
            yield "answer['content'] = first + ':' + second\n"
            yield "answer['ready'] = True\n```"

        runner = RLM(
            model,
            model_stream=stream,
            config=RLMConfig(timeout_seconds=5),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with _kernel() as kernel:
            result = await runner.complete(kernel, "context")

        self.assertEqual(result.response, "ONE:TWO")
        self.assertCountEqual(calls, ["one", "two"])
        self.assertEqual(result.usage.model_calls, 3)
        self.assertEqual(result.usage.speculation.dispatched, 2)
        self.assertEqual(result.usage.speculation.hits, 2)
        self.assertEqual(result.usage.speculation.misses, 0)
        self.assertEqual(result.usage.speculation.already_ready, 2)

    async def test_full_response_fallback_still_speculates(self) -> None:
        """Plan a full response when no streaming adapter is supplied."""

        calls: list[str] = []

        async def model(prompt: str, model_name: str | None) -> str:
            """Return controller code or one plain response.

            Args:
                prompt: Root prompt or plain subquery.
                model_name: Unused model selector.

            Returns:
                Controller code or uppercase subquery.
            """

            del model_name
            if prompt.startswith("You are an RLM controller"):
                return """```python
value = await llm_query(prompt='lazy')
answer['content'] = value
answer['ready'] = True
```"""
            calls.append(prompt)
            return prompt.upper()

        runner = RLM(
            model,
            config=RLMConfig(timeout_seconds=5),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with _kernel() as kernel:
            result = await runner.complete(kernel, "context")

        self.assertEqual(result.response, "LAZY")
        self.assertEqual(calls, ["lazy"])
        self.assertEqual(result.usage.speculation.hits, 1)

    async def test_shadow_resolves_state_without_changing_real_state(self) -> None:
        """Use a forked Monty state for an argument the planner cannot resolve."""

        calls: list[str] = []

        async def ask(prompt: str) -> str:
            """Record and return one shadow or real prompt.

            Args:
                prompt: Resolved prompt.

            Returns:
                Same prompt.
            """

            calls.append(prompt)
            return prompt

        tools = HostedToolSet.from_callables([ask])
        source = """counter += 1
def make_prompt(value):
    return 'Q:' + value
value = await ask(prompt=make_prompt(context[:2]))
value
"""
        async with _kernel() as kernel:
            await kernel.execute("real", "counter = 0", inputs={"context": "abcdef"})
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                SpeculationConfig(mode="stream", tools=(SpeculativeToolPolicy("ask"),)),
            )
            await turn.start()
            await turn.feed(f"```python\n{source}```")
            await turn.finish_generation()
            result = await kernel.execute(
                "real",
                source + "\ncounter",
                hosted_tools=turn.real_tools,
            )
            usage = await turn.close()

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"], 1)
        self.assertEqual(calls, ["Q:ab"])
        self.assertEqual(usage.hits, 1)
        self.assertGreaterEqual(usage.shadow_executions, 3)

    async def test_failed_bet_falls_back_to_the_real_call(self) -> None:
        """Ignore a failed early call and preserve normal execution behavior."""

        calls = 0

        async def flaky(prompt: str) -> str:
            """Fail once and then return the prompt.

            Args:
                prompt: Input prompt.

            Returns:
                Prompt after the first call.

            Raises:
                RuntimeError: On the first invocation.
            """

            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("early failure")
            return prompt

        tools = HostedToolSet.from_callables([flaky])
        source = "await flaky(prompt='safe')"
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                SpeculationConfig(
                    mode="lazy",
                    shadow=False,
                    tools=(SpeculativeToolPolicy("flaky"),),
                ),
            )
            await turn.start()
            await turn.feed(f"```python\n{source}\n```")
            await turn.finish_generation()
            await asyncio.sleep(0)
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], "safe")
        self.assertEqual(calls, 2)
        self.assertEqual(usage.failed, 1)
        self.assertEqual(usage.misses, 1)

    async def test_deterministic_policy_reuses_one_result(self) -> None:
        """Reuse one deterministic bet for repeated identical real calls."""

        calls = 0

        async def lookup(prompt: str) -> str:
            """Return a counted deterministic value.

            Args:
                prompt: Lookup key.

            Returns:
                Counted result.
            """

            nonlocal calls
            calls += 1
            return f"{prompt}:{calls}"

        tools = HostedToolSet.from_callables([lookup])
        source = """one = await lookup(prompt='same')
two = await lookup(prompt='same')
one + ':' + two
"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                SpeculationConfig(
                    mode="lazy",
                    shadow=False,
                    tools=(SpeculativeToolPolicy("lookup", deterministic=True),),
                ),
            )
            await turn.start()
            await turn.feed(f"```python\n{source}```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], "same:1:same:1")
        self.assertEqual(calls, 1)
        self.assertEqual(usage.dispatched, 1)
        self.assertEqual(usage.hits, 2)

    async def test_nondeterministic_policy_keeps_duplicate_call_order(self) -> None:
        """Keep separate FIFO bets for identical nondeterministic calls."""

        calls = 0

        async def sample(prompt: str) -> str:
            """Return a new value for every call.

            Args:
                prompt: Sample label.

            Returns:
                Label with a call sequence.
            """

            nonlocal calls
            calls += 1
            return f"{prompt}:{calls}"

        tools = HostedToolSet.from_callables([sample])
        source = """one = await sample(prompt='same')
two = await sample(prompt='same')
one + ':' + two
"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                SpeculationConfig(
                    mode="lazy",
                    shadow=False,
                    tools=(SpeculativeToolPolicy("sample"),),
                ),
            )
            await turn.start()
            await turn.feed(f"```python\n{source}```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], "same:1:same:2")
        self.assertEqual(calls, 2)
        self.assertEqual(usage.dispatched, 2)
        self.assertEqual(usage.hits, 2)

    async def test_unused_pending_bet_is_cancelled(self) -> None:
        """Cancel an unclaimed pending call when a turn ends."""

        started = asyncio.Event()

        async def slow(prompt: str) -> str:
            """Wait until cancellation.

            Args:
                prompt: Unused prompt.

            Returns:
                Prompt if the task is not cancelled.
            """

            started.set()
            await asyncio.sleep(10)
            return prompt

        tools = HostedToolSet.from_callables([slow])
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                SpeculationConfig(
                    mode="stream",
                    shadow=False,
                    tools=(SpeculativeToolPolicy("slow"),),
                ),
            )
            await turn.start()
            await turn.feed("```python\nunused = await slow(prompt='unused')\n")
            await asyncio.wait_for(started.wait(), timeout=1)
            usage = await turn.close()

        self.assertEqual(usage.dispatched, 1)
        self.assertEqual(usage.cancelled, 1)

    async def test_static_planner_unrolls_a_small_loop(self) -> None:
        """Plan bounded loop calls without executing code in the host."""

        calls: list[str] = []

        async def lookup(prompt: str) -> str:
            """Return one recorded loop prompt.

            Args:
                prompt: Loop value.

            Returns:
                Uppercase value.
            """

            calls.append(prompt)
            return prompt.upper()

        tools = HostedToolSet.from_callables([lookup])
        source = """values = []
for prompt in ['a', 'b']:
    value = await lookup(prompt=prompt)
    values.append(value)
values
"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                SpeculationConfig(
                    mode="stream",
                    shadow=False,
                    tools=(SpeculativeToolPolicy("lookup"),),
                ),
            )
            await turn.start()
            await turn.feed(f"```python\n{source}```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], ["A", "B"])
        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(usage.hits, 2)

    async def test_sequential_tool_is_not_speculated(self) -> None:
        """Keep a sequential hosted tool on the normal execution path."""

        calls = 0

        async def change(value: int) -> int:
            """Return one counted value.

            Args:
                value: Value to return.

            Returns:
                Supplied value.
            """

            nonlocal calls
            calls += 1
            return value

        tool = HostedTool.from_callable(change, sequential=True)
        tools = HostedToolSet((tool,))
        source = "await change(value=42)"
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                SpeculationConfig(
                    mode="stream",
                    tools=(SpeculativeToolPolicy("change"),),
                ),
            )
            await turn.start()
            await turn.feed(f"```python\n{source}\n```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], 42)
        self.assertEqual(calls, 1)
        self.assertEqual(usage.turns, 0)
        self.assertEqual(usage.dispatched, 0)


if __name__ == "__main__":
    unittest.main()
