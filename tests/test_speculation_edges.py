"""Integration tests for speculative execution boundaries and races."""

import asyncio
import json
import unittest

from pydantic_monty import OSAccess, OsHandler

from arenero import (
    AreneroEvent,
    AsyncPersistentMontyKernel,
    EventSink,
    HostedToolSet,
    KernelConfig,
    SessionStore,
    SpeculationConfig,
    SpeculativeToolPolicy,
    SpeculativeTurn,
    StatelessSessionStore,
)


def _kernel(
    *,
    store: SessionStore | None = None,
    os_handler: OsHandler | None = None,
    event_sink: EventSink | None = None,
) -> AsyncPersistentMontyKernel:
    """Create a bounded async kernel for integration tests.

    Args:
        store: Optional session store.
        os_handler: Optional virtual filesystem handler.
        event_sink: Optional safe event observer.

    Returns:
        Async kernel with enough workers for concurrent shadows.
    """

    return AsyncPersistentMontyKernel(
        store=store,
        config=KernelConfig(
            min_worker_processes=2,
            max_worker_processes=6,
            worker_checkout_timeout_seconds=2,
            request_timeout_seconds=5,
        ),
        os_handler=os_handler,
        event_sink=event_sink,
    )


def _config(
    tool: str,
    *,
    shadow: bool = False,
    max_inflight: int = 8,
    max_calls_per_turn: int = 64,
    max_argument_chars: int = 100_000,
    max_code_chars: int = 200_000,
    max_loop_unroll: int = 64,
    shadow_drain_seconds: float = 30.0,
) -> SpeculationConfig:
    """Create one explicit speculative tool policy.

    Args:
        tool: Hosted-tool name or alias.
        shadow: Enable the forked Monty shadow.
        max_inflight: Maximum early calls active at once.
        max_calls_per_turn: Maximum early dispatches.
        max_argument_chars: Maximum canonical argument size.
        max_code_chars: Maximum generated text retained.
        max_loop_unroll: Maximum statically inspected iterations.
        shadow_drain_seconds: Maximum shadow drain time.

    Returns:
        Stream-mode configuration for the selected tool.
    """

    return SpeculationConfig(
        mode="stream",
        shadow=shadow,
        tools=(SpeculativeToolPolicy(tool),),
        max_inflight=max_inflight,
        max_calls_per_turn=max_calls_per_turn,
        max_argument_chars=max_argument_chars,
        max_code_chars=max_code_chars,
        max_loop_unroll=max_loop_unroll,
        shadow_drain_seconds=shadow_drain_seconds,
    )


class SpeculationConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    """Check isolation and cancellation across concurrent turns."""

    async def test_concurrent_sessions_do_not_share_claims(self) -> None:
        """Keep identical call identities inside their owning turn."""

        calls: list[str] = []

        def tools_for(owner: str) -> HostedToolSet:
            """Create one same-named tool with an owner-specific result.

            Args:
                owner: Session owner returned by the tool.

            Returns:
                Hosted tool set with a `lookup` alias.
            """

            async def lookup(prompt: str) -> str:
                """Return the tool owner after one scheduling point.

                Args:
                    prompt: Shared lookup prompt.

                Returns:
                    Owner and prompt.
                """

                await asyncio.sleep(0)
                calls.append(owner)
                return f"{owner}:{prompt}"

            return HostedToolSet.from_callables([lookup])

        source = "await lookup(prompt='same')"
        async with _kernel() as kernel:
            await asyncio.gather(
                kernel.execute("agent-a", "owner = 'a'"),
                kernel.execute("agent-b", "owner = 'b'"),
            )
            turn_a = SpeculativeTurn(
                kernel, "agent-a", tools_for("a"), _config("lookup")
            )
            turn_b = SpeculativeTurn(
                kernel, "agent-b", tools_for("b"), _config("lookup")
            )
            await asyncio.gather(turn_a.start(), turn_b.start())
            await asyncio.gather(
                turn_a.feed(f"```python\n{source}\n```"),
                turn_b.feed(f"```python\n{source}\n```"),
            )
            await asyncio.gather(turn_a.finish_generation(), turn_b.finish_generation())
            result_a, result_b = await asyncio.gather(
                kernel.execute("agent-a", source, hosted_tools=turn_a.real_tools),
                kernel.execute("agent-b", source, hosted_tools=turn_b.real_tools),
            )
            usage_a, usage_b = await asyncio.gather(turn_a.close(), turn_b.close())

        self.assertEqual(result_a["result"], "a:same")
        self.assertEqual(result_b["result"], "b:same")
        self.assertCountEqual(calls, ["a", "b"])
        self.assertEqual((usage_a.hits, usage_b.hits), (1, 1))

    async def test_closing_one_turn_does_not_cancel_another(self) -> None:
        """Cancel one owner while another identical call stays active."""

        started_a = asyncio.Event()
        started_b = asyncio.Event()
        cancelled_a = asyncio.Event()
        release_b = asyncio.Event()

        async def lookup_a(prompt: str) -> str:
            """Wait until the owning turn cancels this call.

            Args:
                prompt: Shared prompt.

            Returns:
                Unreachable owner result.
            """

            started_a.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled_a.set()
                raise
            return f"a:{prompt}"

        async def lookup_b(prompt: str) -> str:
            """Wait for release and return the second owner result.

            Args:
                prompt: Shared prompt.

            Returns:
                Second owner result.
            """

            started_b.set()
            await release_b.wait()
            return f"b:{prompt}"

        tools_a = HostedToolSet.from_callables([lookup_a])
        tools_b = HostedToolSet.from_callables([lookup_b])
        async with _kernel() as kernel:
            await asyncio.gather(
                kernel.execute("agent-a", "seed = 1"),
                kernel.execute("agent-b", "seed = 2"),
            )
            turn_a = SpeculativeTurn(kernel, "agent-a", tools_a, _config("lookup_a"))
            turn_b = SpeculativeTurn(kernel, "agent-b", tools_b, _config("lookup_b"))
            await asyncio.gather(turn_a.start(), turn_b.start())
            await asyncio.gather(
                turn_a.feed("```python\nawait lookup_a(prompt='same')\n```"),
                turn_b.feed("```python\nawait lookup_b(prompt='same')\n```"),
            )
            await asyncio.wait_for(
                asyncio.gather(started_a.wait(), started_b.wait()), timeout=1
            )
            usage_a = await turn_a.close()
            await asyncio.wait_for(cancelled_a.wait(), timeout=1)
            release_b.set()
            result_b = await kernel.execute(
                "agent-b",
                "await lookup_b(prompt='same')",
                hosted_tools=turn_b.real_tools,
            )
            usage_b = await turn_b.close()

        self.assertEqual(result_b["result"], "b:same")
        self.assertEqual(usage_a.cancelled, 1)
        self.assertEqual(usage_b.hits, 1)

    async def test_max_inflight_limits_early_overlap(self) -> None:
        """Run no more than the configured number of early calls."""

        active = 0
        maximum = 0
        two_started = asyncio.Event()
        release = asyncio.Event()

        async def lookup(prompt: str) -> str:
            """Record active calls and wait for test release.

            Args:
                prompt: Unique prompt.

            Returns:
                Same prompt.
            """

            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                two_started.set()
            try:
                await release.wait()
                return prompt
            finally:
                active -= 1

        tools = HostedToolSet.from_callables([lookup])
        response = """```python
one = await lookup(prompt='one')
two = await lookup(prompt='two')
three = await lookup(prompt='three')
```"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                _config("lookup", max_inflight=2),
            )
            await turn.start()
            await turn.feed(response)
            await asyncio.wait_for(two_started.wait(), timeout=1)
            self.assertEqual(maximum, 2)
            release.set()
            await asyncio.sleep(0)
            usage = await turn.close()

        self.assertEqual(maximum, 2)
        self.assertEqual(usage.dispatched, 3)


class SpeculationStreamAndLimitTests(unittest.IsolatedAsyncioTestCase):
    """Check fragmented generation and every turn-level limit."""

    async def test_character_stream_and_multiple_fences_plan_once(self) -> None:
        """Accept one-character deltas without duplicate dispatches."""

        calls: list[str] = []

        async def lookup(prompt: str) -> str:
            """Record one planned prompt.

            Args:
                prompt: Planned prompt.

            Returns:
                Same prompt.
            """

            calls.append(prompt)
            return prompt

        tools = HostedToolSet.from_callables([lookup])
        response = """Some text before code.
```python
one = await lookup(prompt='one')
```
More text.
```repl
two = await lookup(prompt='two')
```"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(kernel, "real", tools, _config("lookup"))
            await turn.start()
            for character in response:
                await turn.feed(character)
            await turn.finish_generation()
            one = await kernel.execute(
                "real",
                "await lookup(prompt='one')",
                hosted_tools=turn.real_tools,
            )
            two = await kernel.execute(
                "real",
                "await lookup(prompt='two')",
                hosted_tools=turn.real_tools,
            )
            usage = await turn.close()

        self.assertEqual([one["result"], two["result"]], ["one", "two"])
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual((usage.dispatched, usage.hits), (2, 2))

    async def test_incomplete_source_does_not_dispatch(self) -> None:
        """Wait for a parseable complete call before starting host work."""

        started = asyncio.Event()

        async def lookup(prompt: str) -> str:
            """Mark a valid dispatch.

            Args:
                prompt: Planned prompt.

            Returns:
                Same prompt.
            """

            started.set()
            return prompt

        tools = HostedToolSet.from_callables([lookup])
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(kernel, "real", tools, _config("lookup"))
            await turn.start()
            await turn.feed("```python\nvalue = await lookup(prompt=")
            await asyncio.sleep(0)
            self.assertFalse(started.is_set())
            await turn.feed("'ready')\n```")
            await asyncio.wait_for(started.wait(), timeout=1)
            usage = await turn.close()

        self.assertEqual(usage.dispatched, 1)

    async def test_dispatch_argument_and_code_limits(self) -> None:
        """Apply call-count, argument-size, and retained-code limits."""

        async def lookup(prompt: str) -> str:
            """Return one limit-test prompt.

            Args:
                prompt: Planned prompt.

            Returns:
                Same prompt.
            """

            return prompt

        tools = HostedToolSet.from_callables([lookup])
        cases = [
            (
                "calls",
                _config("lookup", max_calls_per_turn=2),
                """```python
await lookup(prompt='one')
await lookup(prompt='two')
await lookup(prompt='three')
```""",
                2,
            ),
            (
                "arguments",
                _config("lookup", max_argument_chars=20),
                "```python\nawait lookup(prompt='" + ("x" * 100) + "')\n```",
                0,
            ),
            (
                "code",
                _config("lookup", max_code_chars=24),
                "natural language before ```python\nawait lookup(prompt='late')\n```",
                0,
            ),
        ]
        async with _kernel() as kernel:
            for name, config, response, expected in cases:
                with self.subTest(limit=name):
                    session_id = f"limit-{name}"
                    await kernel.execute(session_id, "seed = 1")
                    turn = SpeculativeTurn(kernel, session_id, tools, config)
                    await turn.start()
                    await turn.feed(response)
                    await turn.finish_generation()
                    usage = await turn.close()
                    self.assertEqual(usage.dispatched, expected)

    async def test_loop_unroll_limit_does_not_materialize_large_range(self) -> None:
        """Inspect only a bounded prefix of a very large range."""

        calls: list[int] = []

        async def lookup(value: int) -> int:
            """Record one bounded loop value.

            Args:
                value: Planned loop index.

            Returns:
                Same index.
            """

            calls.append(value)
            return value

        tools = HostedToolSet.from_callables([lookup])
        response = """```python
for index in range(1000000000):
    await lookup(value=index)
```"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                _config("lookup", max_loop_unroll=3),
            )
            await turn.start()
            await turn.feed(response)
            await turn.finish_generation()
            await asyncio.sleep(0)
            usage = await turn.close()

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(usage.dispatched, 3)

    async def test_shadow_timeout_cleans_snapshot_and_call(self) -> None:
        """Cancel a blocked shadow and delete all temporary state."""

        started = asyncio.Event()

        async def slow(prompt: str) -> str:
            """Block until the shadow drain timeout cancels the call.

            Args:
                prompt: State-derived prompt.

            Returns:
                Unreachable prompt.
            """

            started.set()
            await asyncio.Future()
            return prompt

        tools = HostedToolSet.from_callables([slow])
        response = """```python
def make_prompt():
    return prefix + '-slow'
value = await slow(prompt=make_prompt())
```"""
        async with _kernel() as kernel:
            await kernel.execute("real", "prefix = 'state'")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                _config(
                    "slow",
                    shadow=True,
                    shadow_drain_seconds=0.02,
                ),
            )
            await turn.start()
            await turn.feed(response)
            finish = asyncio.create_task(turn.finish_generation())
            await asyncio.wait_for(started.wait(), timeout=1)
            await finish
            usage = await turn.close()
            sessions = [info.session_id for info in kernel.store.list_sessions()]

        self.assertEqual(sessions, ["real"])
        self.assertEqual(usage.shadow_errors, 1)
        self.assertEqual(usage.cancelled, 1)


class SpeculationIdentityAndBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Check canonical matching, host gates, shadows, and safe events."""

    async def test_nested_arguments_and_bytes_have_canonical_identity(self) -> None:
        """Match reordered nested dictionaries and bytes deterministically."""

        calls = 0

        async def identify(payload: dict[str, int], blob: bytes) -> str:
            """Return a stable result for canonical identity data.

            Args:
                payload: Nested mapping with reordered keys.
                blob: Binary identity value.

            Returns:
                Stable marker.
            """

            nonlocal calls
            calls += 1
            return f"{sum(payload.values())}:{blob.decode()}"

        tools = HostedToolSet.from_callables([identify])
        policy = SpeculativeToolPolicy("identify", deterministic=True)
        config = SpeculationConfig(mode="stream", shadow=False, tools=(policy,))
        source = """one = await identify(payload={'a': 1, 'b': 2}, blob=b'x')
two = await identify(blob=b'x', payload={'b': 2, 'a': 1})
one + ':' + two
"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(kernel, "real", tools, config)
            await turn.start()
            await turn.feed(f"```python\n{source}```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], "3:x:3:x")
        self.assertEqual(calls, 1)
        self.assertEqual((usage.dispatched, usage.hits), (1, 2))

    async def test_custom_key_and_gate_control_eligibility(self) -> None:
        """Reuse an approved key and run rejected arguments normally."""

        calls: list[int] = []

        async def parity(value: int) -> str:
            """Return parity while recording actual calls.

            Args:
                value: Integer to classify.

            Returns:
                Even or odd marker.
            """

            calls.append(value)
            return "even" if value % 2 == 0 else "odd"

        tools = HostedToolSet.from_callables([parity])
        policy = SpeculativeToolPolicy(
            "parity",
            deterministic=True,
            key=lambda args: args["value"] % 2,
            gate=lambda args: args["value"] >= 0,
        )
        config = SpeculationConfig(mode="stream", shadow=False, tools=(policy,))
        source = """one = await parity(value=2)
two = await parity(value=4)
three = await parity(value=-1)
[one, two, three]
"""
        async with _kernel() as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(kernel, "real", tools, config)
            await turn.start()
            await turn.feed(f"```python\n{source}```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], ["even", "even", "odd"])
        self.assertEqual(calls, [2, -1])
        self.assertEqual((usage.dispatched, usage.hits, usage.misses), (1, 2, 1))

    async def test_stateless_store_uses_static_planning_without_shadow(self) -> None:
        """Keep static speculation when snapshots are unavailable."""

        calls: list[str] = []

        async def lookup(prompt: str) -> str:
            """Record one stateless prompt.

            Args:
                prompt: Planned prompt.

            Returns:
                Same prompt.
            """

            calls.append(prompt)
            return prompt

        tools = HostedToolSet.from_callables([lookup])
        source = "await lookup(prompt='static')"
        async with _kernel(store=StatelessSessionStore()) as kernel:
            turn = SpeculativeTurn(
                kernel, "real", tools, _config("lookup", shadow=True)
            )
            await turn.start()
            await turn.feed(f"```python\n{source}\n```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], "static")
        self.assertEqual(calls, ["static"])
        self.assertEqual((usage.hits, usage.shadow_executions), (1, 0))

    async def test_shadow_cannot_read_the_real_virtual_filesystem(self) -> None:
        """Leave a file-dependent call for normal filesystem execution."""

        calls: list[str] = []

        async def lookup(prompt: str) -> str:
            """Record content read by the real session.

            Args:
                prompt: Virtual file content.

            Returns:
                Same content.
            """

            calls.append(prompt)
            return prompt

        tools = HostedToolSet.from_callables([lookup])
        source = """content = fs_read('/secret.txt')
await lookup(prompt=content)
"""
        async with _kernel(os_handler=OSAccess()) as kernel:
            await kernel.execute("real", "fs_write('/secret.txt', 'file-secret')")
            turn = SpeculativeTurn(
                kernel, "real", tools, _config("lookup", shadow=True)
            )
            await turn.start()
            await turn.feed(f"```python\n{source}```")
            await turn.finish_generation()
            self.assertEqual(calls, [])
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        self.assertEqual(result["result"], "file-secret")
        self.assertEqual(calls, ["file-secret"])
        self.assertEqual((usage.hits, usage.misses), (0, 1))
        self.assertGreaterEqual(usage.shadow_errors, 1)

    async def test_events_are_ordered_and_do_not_include_values(self) -> None:
        """Report safe lifecycle data without arguments or results."""

        events: list[AreneroEvent] = []
        secret = "do-not-record-this"

        async def lookup(prompt: str) -> str:
            """Return a secret value through the hosted boundary.

            Args:
                prompt: Secret test prompt.

            Returns:
                Same secret prompt.
            """

            return prompt

        tools = HostedToolSet.from_callables([lookup])
        source = f"await lookup(prompt={secret!r})"
        async with _kernel(event_sink=events.append) as kernel:
            await kernel.execute("real", "seed = 1")
            turn = SpeculativeTurn(
                kernel,
                "real",
                tools,
                _config("lookup"),
                event_sink=events.append,
            )
            await turn.start()
            await turn.feed(f"```python\n{source}\n```")
            await turn.finish_generation()
            result = await kernel.execute("real", source, hosted_tools=turn.real_tools)
            usage = await turn.close()

        actions = [event.action for event in events if event.category == "speculation"]
        serialized = json.dumps(
            [event.data for event in events], default=str, ensure_ascii=False
        )
        self.assertEqual(result["result"], secret)
        self.assertEqual(usage.hits, 1)
        self.assertEqual(
            actions,
            ["turn_started", "dispatched", "claim_hit", "turn_finished"],
        )
        self.assertNotIn(secret, serialized)

    async def test_failed_and_waited_calls_have_exact_statistics(self) -> None:
        """Record failure fallback and real wait time without double counting."""

        events: list[AreneroEvent] = []
        attempts = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def lookup(prompt: str) -> str:
            """Fail one key and block another until the real claim waits.

            Args:
                prompt: Failure or wait selector.

            Returns:
                Wait selector after release.

            Raises:
                RuntimeError: For the failure selector.
            """

            nonlocal attempts
            attempts += 1
            if prompt == "fail" and attempts == 1:
                raise RuntimeError("planned failure")
            if prompt == "wait":
                started.set()
                await release.wait()
            return prompt

        tools = HostedToolSet.from_callables([lookup])
        async with _kernel(event_sink=events.append) as kernel:
            await kernel.execute("real", "seed = 1")
            failed = SpeculativeTurn(
                kernel,
                "real",
                tools,
                _config("lookup"),
                event_sink=events.append,
            )
            await failed.start()
            await failed.feed("```python\nawait lookup(prompt='fail')\n```")
            await failed.finish_generation()
            await asyncio.sleep(0)
            failed_result = await kernel.execute(
                "real",
                "await lookup(prompt='fail')",
                hosted_tools=failed.real_tools,
            )
            failed_usage = await failed.close()

            waiting = SpeculativeTurn(
                kernel,
                "real",
                tools,
                _config("lookup"),
                event_sink=events.append,
            )
            await waiting.start()
            await waiting.feed("```python\nawait lookup(prompt='wait')\n```")
            await asyncio.wait_for(started.wait(), timeout=1)
            execution = asyncio.create_task(
                kernel.execute(
                    "real",
                    "await lookup(prompt='wait')",
                    hosted_tools=waiting.real_tools,
                )
            )
            await asyncio.sleep(0.02)
            release.set()
            waited_result = await execution
            waited_usage = await waiting.close()

        speculation_actions = [
            event.action for event in events if event.category == "speculation"
        ]
        self.assertEqual(failed_result["result"], "fail")
        self.assertEqual(waited_result["result"], "wait")
        self.assertEqual((failed_usage.failed, failed_usage.misses), (1, 1))
        self.assertEqual(waited_usage.hits, 1)
        self.assertGreaterEqual(waited_usage.waited_seconds, 0.015)
        self.assertIn("call_failed", speculation_actions)


if __name__ == "__main__":
    unittest.main()
