"""Regression tests for filtered PTC, results, errors, and RLM loop ownership."""

import asyncio
import json
import unittest
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic_monty import OSAccess

from arenero import (
    RLM,
    AsyncPersistentMontyKernel,
    HostedToolConfig,
    KernelConfig,
    PersistentMontyKernel,
    RLMConfig,
    RLMModelResponse,
    SpeculationConfig,
)
from arenero.deepagents import (
    AreneroInterpreterMiddleware,
    CodeInterpreterMiddleware,
    _tool_value,
)
from tests.test_deepagents_compat import _eval, _model
from tests.test_langchain_agent import named_tool_call


class FilterTools(AgentMiddleware):
    """Remove named tools before Arenero sees the active model request.

    Args:
        hidden: Tools hidden on each request, or on turns containing 'hide'.
        on_hide_turn: Filter only when the latest human turn says 'hide'.
    """

    def __init__(self, *hidden: str, on_hide_turn: bool = False) -> None:
        """Store the test filter names and optional human-turn switch."""
        self.hidden = set(hidden)
        self.on_hide_turn = on_hide_turn

    def _filter(self, request: Any) -> Any:
        """Filter the actual request sent to the next middleware.

        Args:
            request: Current model request.

        Returns:
            Request with only currently allowed tools.
        """
        turns = [m for m in request.state["messages"] if isinstance(m, HumanMessage)]
        active = not self.on_hide_turn or "hide" in turns[-1].content
        return request.override(
            tools=[t for t in request.tools if not active or t.name not in self.hidden]
        )

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Apply the same filter to a synchronous model request.

        Args:
            request: Current model request.
            handler: Next middleware or fake model.

        Returns:
            Fake model response.
        """
        return handler(self._filter(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Apply the same filter to an asynchronous model request.

        Args:
            request: Current model request.
            handler: Next middleware or fake model.

        Returns:
            Fake model response.
        """
        return await handler(self._filter(request))


def _results(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Read either middleware's unchanged result format.

    Args:
        state: Completed fake agent state.

    Returns:
        Rich results from interpreter messages in order.
    """
    return [
        m.artifact if m.artifact is not None else json.loads(m.content)
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    ]


def _packet_tool(packet: Command) -> BaseTool:
    """Make a fixed nested-command tool without mutable loop captures.

    Args:
        packet: Command returned to the interpreter, not the parent graph.

    Returns:
        Standalone fake LangChain tool.
    """

    @tool
    async def wrapped() -> Any:
        """Return one fixed envelope.

        Returns:
            Nested command with synthetic content or update data.
        """
        return packet

    return wrapped


class InterpreterFixTests(unittest.IsolatedAsyncioTestCase):
    """Use complete fake-model graphs and real local Monty workers."""

    async def test_request_filter_removes_a_previously_used_tool(self) -> None:
        """Refresh instructions and dispatch together across two human turns."""
        calls = []
        for middleware_type in (
            CodeInterpreterMiddleware,
            AreneroInterpreterMiddleware,
        ):
            calls.clear()

            @tool
            async def hidden() -> int:
                """Record one allowed call.

                Returns:
                    Integer 42.
                """
                calls.append(True)
                return 42

            async with middleware_type(ptc=("hidden",)) as middleware:
                name = middleware.tools[0].name
                agent = create_agent(
                    _model(
                        named_tool_call(
                            "first",
                            name,
                            {
                                "code": "async def saved():\n    return await hidden()\nawait saved()"
                            },
                        ),
                        AIMessage(content="stored"),
                        named_tool_call("second", name, {"code": "await saved()"}),
                        AIMessage(content="removed"),
                    ),
                    tools=[hidden],
                    middleware=[FilterTools("hidden", on_hide_turn=True), middleware],
                    checkpointer=InMemorySaver(),
                )
                config = {"configurable": {"thread_id": "filtered"}}
                first = await agent.ainvoke(
                    {"messages": [HumanMessage(content="use tool")]}, config
                )
                second = await agent.ainvoke(
                    {"messages": [HumanMessage(content="hide tool")]}, config
                )
                self.assertEqual(_results(first)[-1]["result"], 42)
                self.assertFalse(_results(second)[-1]["ok"])
                self.assertEqual(calls, [True])
                self.assertNotIn(
                    "hidden", agent.get_state(config).values["_arenero_active_tools"]
                )
                self.assertEqual(middleware._active_tools, {})

    async def test_filter_survives_interrupt_and_adapter_recreation(self) -> None:
        """Restore allowed names from private graph state, not a host cache."""
        calls = []

        @tool
        async def gate() -> str:
            """Pause the outer graph.

            Returns:
                Host approval reply.
            """
            return interrupt("approve")

        @tool
        async def hidden() -> int:
            """Record an unexpected filtered call.

            Returns:
                Integer 42.
            """
            calls.append(True)
            return 42

        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "filter-resume"}}
        async with CodeInterpreterMiddleware(ptc=("gate", "hidden")) as first:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "paused", "eval", {"code": "await gate()\nawait hidden()"}
                    )
                ),
                tools=[gate, hidden],
                middleware=[FilterTools("hidden"), first],
                checkpointer=saver,
            )
            paused = await agent.ainvoke(
                {"messages": [HumanMessage(content="test")]}, config
            )
            self.assertTrue(paused.get("__interrupt__"))
            self.assertNotIn(
                "hidden", agent.get_state(config).values["_arenero_active_tools"]
            )
        async with CodeInterpreterMiddleware(ptc=("gate", "hidden")) as second:
            agent = create_agent(
                _model(AIMessage(content="done")),
                tools=[gate, hidden],
                middleware=[FilterTools("hidden"), second],
                checkpointer=saver,
            )
            result = await agent.ainvoke(Command(resume="yes"), config)
            self.assertFalse(_results(result)[-1]["ok"])
            self.assertEqual(calls, [])

    async def test_explicit_standalone_tool_remains_available(self) -> None:
        """Keep the documented explicit-BaseTool capability without registration."""

        @tool
        async def standalone() -> int:
            """Return a standalone value.

            Returns:
                Integer 42.
            """
            return 42

        async with CodeInterpreterMiddleware(ptc=(standalone,)) as middleware:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "one", "eval", {"code": "await tools.standalone()"}
                    ),
                    AIMessage(content="42"),
                ),
                middleware=[middleware],
            )
            result = await agent.ainvoke({"messages": [HumanMessage(content="test")]})
            self.assertEqual(_results(result)[0]["result"], 42)

    async def test_restored_names_override_a_newer_host_cache(self) -> None:
        """Keep historical checkpoint permissions when a live cache is newer."""
        from tests.test_deepagents_compat import _runtime

        @tool
        async def hidden() -> int:
            """Return a tool that the old checkpoint did not expose.

            Returns:
                Integer 42.
            """
            return 42

        async with CodeInterpreterMiddleware(ptc=("hidden",)) as middleware:
            state = await middleware.abefore_agent({}, None)
            state["_arenero_active_tools"] = []
            middleware._active_tools[state["_arenero_slot_id"]] = (hidden,)
            self.assertEqual(middleware._runtime_tools(_runtime(state, [hidden])), ())

    async def test_concurrent_graphs_do_not_share_filtered_inventories(self) -> None:
        """Use separate private slots with one shared middleware and pool."""

        @tool
        async def red() -> str:
            """Return a red marker.

            Returns:
                Red text.
            """
            await asyncio.sleep(0)
            return "red"

        @tool
        async def blue() -> str:
            """Return a blue marker.

            Returns:
                Blue text.
            """
            await asyncio.sleep(0)
            return "blue"

        async with CodeInterpreterMiddleware(ptc=("red", "blue")) as middleware:
            agents = [
                create_agent(
                    _model(
                        named_tool_call(
                            "call-" + allowed,
                            "eval",
                            {"code": f"await tools.{allowed}()"},
                        ),
                        AIMessage(content="done"),
                    ),
                    tools=[red, blue],
                    middleware=[FilterTools(hidden), middleware],
                )
                for allowed, hidden in (("red", "blue"), ("blue", "red"))
            ]
            results = await asyncio.gather(
                *(
                    agent.ainvoke({"messages": [HumanMessage(content="test")]})
                    for agent in agents
                )
            )
            self.assertEqual(
                [_results(result)[0]["result"] for result in results], ["red", "blue"]
            )
            self.assertEqual(middleware._active_tools, {})

    async def test_command_envelopes_return_content_or_typed_update(self) -> None:
        """Extract content without applying nested graph updates."""
        cases = [
            (Command(update={"messages": [AIMessage(content="hello")]}), "hello"),
            (
                Command(
                    update={"messages": [{"role": "assistant", "content": "dict"}]}
                ),
                "dict",
            ),
            (Command(update={"messages": AIMessage(content="single")}), "single"),
            (
                Command(update={"messages": None, "value": 42}),
                {"messages": None, "value": 42},
            ),
            (Command(update={"value": 42}), {"value": 42}),
            (
                Command(update={"messages": [AIMessage(content="first"), object()]}),
                "first",
            ),
        ]
        for packet, expected in cases:
            wrapped = _packet_tool(packet)

            async with CodeInterpreterMiddleware(ptc=("wrapped",)) as middleware:
                agent = create_agent(
                    _model(
                        named_tool_call(
                            "one", "eval", {"code": "await tools.wrapped()"}
                        ),
                        AIMessage(content="done"),
                    ),
                    tools=[wrapped],
                    middleware=[middleware],
                )
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content="test")]}
                )
                self.assertEqual(_results(result)[0]["result"], expected)
                self.assertNotIn("value", result)
        self.assertEqual(
            _tool_value([ToolMessage(content="old", tool_call_id="x"), cases[0][0]]),
            "hello",
        )
        for plain in ({"content": "plain"}, [1, 2], True, 42):
            self.assertIs(_tool_value(plain), plain)

    async def test_agent_gets_native_traceback_and_can_fix_code(self) -> None:
        """Keep error.message unchanged and expose bounded frames separately."""
        async with CodeInterpreterMiddleware() as middleware:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "bad",
                        "eval",
                        {"code": "def divide(value):\n    return 10/value\ndivide(0)"},
                    ),
                    named_tool_call("fixed", "eval", {"code": "divide(2)"}),
                    AIMessage(content="5"),
                ),
                middleware=[middleware],
            )
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="calculate")]}
            )
            first, second = _results(result)
            self.assertEqual(
                first["error"]["message"], "ZeroDivisionError: division by zero"
            )
            self.assertIn("line 2, in divide", first["error"]["traceback"])
            self.assertIn("return 10/value", first["error"]["traceback"])
            self.assertEqual(second["result"], 5)
            message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
            self.assertIn("<traceback>", message.content)

    async def test_traceback_is_bounded_and_full_overflow_is_available(self) -> None:
        """Apply one error cap to message plus stack in sync and async kernels."""
        source = "def fail():\n    return 1/0\nfail()"
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            result = kernel.execute("sync", source, 50, output_file="/error.json")
            error = result["error"]
            self.assertLessEqual(len(error["message"]) + len(error["traceback"]), 50)
            self.assertTrue(result["truncated"]["error"])
            full = json.loads(kernel.download_file("/error.json"))
            self.assertIn("line 2, in fail", full["error"]["traceback"])
        async with AsyncPersistentMontyKernel() as kernel:
            result = await kernel.execute("async", source, 0)
            self.assertEqual(result["error"]["message"], "")
            self.assertEqual(result["error"]["traceback"], "")
            self.assertTrue(result["truncated"]["error"])

    async def test_model_text_includes_overflow_path_error_and_original_counts(
        self,
    ) -> None:
        """Report output-file success and failure without reading the artifact."""
        async with CodeInterpreterMiddleware(
            os_handler=OSAccess(), max_result_chars=30
        ) as middleware:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "ok",
                        "eval",
                        {
                            "code": "print('x'*100)\n'large'*100",
                            "output_file": "/saved.json",
                        },
                    ),
                    named_tool_call(
                        "bad",
                        "eval",
                        {"code": "'large'*100", "output_file": "/saved.json/fail.json"},
                    ),
                    AIMessage(content="done"),
                ),
                middleware=[middleware],
            )
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="calculate")]}
            )
            first, second = [
                m for m in result["messages"] if isinstance(m, ToolMessage)
            ]
            self.assertIn("<output_file>/saved.json</output_file>", first.content)
            self.assertIn("&quot;stdout&quot;: 101", first.content)
            self.assertIn("<output_file_error>", second.content)
            self.assertIn("saved.json", second.content)
            self.assertNotIn("<output_file>", second.content)

    async def test_rlm_plain_batch_and_recursive_calls_use_caller_loop(self) -> None:
        """Keep model clients on their owner loop and child kernels on theirs."""
        owner = asyncio.get_running_loop()
        seen = []

        async def model(prompt: str, name: str | None) -> RLMModelResponse:
            """Use a loop-bound fake client with synthetic usage.

            Args:
                prompt: Plain query or child controller prompt.
                name: Optional model selector.

            Returns:
                Uppercase text or deterministic child controller code.
            """
            self.assertIs(asyncio.get_running_loop(), owner)
            seen.append(prompt)
            future = owner.create_future()
            text = (
                "```python\nanswer['content']='child'\nanswer['ready']=True\n```"
                if prompt.startswith("You are an RLM controller")
                else prompt.upper()
            )
            owner.call_soon(future.set_result, RLMModelResponse(text, input_tokens=1))
            return await future

        runner = RLM(model, config=RLMConfig(timeout_seconds=5))
        async with CodeInterpreterMiddleware(
            rlm=runner, kernel_config=KernelConfig(max_worker_processes=4)
        ) as middleware:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "one",
                        "eval",
                        {
                            "code": "plain=await llm_query(prompt='one')\nbatch=await llm_query_batched(prompts=['two','three'])\nchild=await rlm_query(prompt='child')\n[plain,batch,child]"
                        },
                    ),
                    AIMessage(content="done"),
                ),
                middleware=[middleware],
            )
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="run RLM")]}
            )
            self.assertEqual(
                _results(result)[0]["result"], ["ONE", ["TWO", "THREE"], "child"]
            )
            self.assertEqual(len(seen), 4)
            self.assertIs(runner.model_call, model)

    async def test_rlm_stream_and_speculation_stay_on_the_correct_loops(self) -> None:
        """Stream on the application loop, while planning stays on Monty's loop."""
        owner = asyncio.get_running_loop()
        seen, closed = [], []

        async def model(prompt: str, name: str | None) -> RLMModelResponse:
            """Return one speculative plain response.

            Args:
                prompt: Plain query.
                name: Optional model selector.

            Returns:
                Uppercase query and synthetic usage.
            """
            self.assertIs(asyncio.get_running_loop(), owner)
            seen.append(prompt)
            return RLMModelResponse(prompt.upper(), input_tokens=1)

        async def stream(prompt: str, name: str | None) -> AsyncIterator[str]:
            """Read fake streaming model deltas on the application loop.

            Args:
                prompt: Child controller prompt.
                name: Optional model selector.

            Yields:
                Bounded deterministic controller source.
            """
            try:
                for chunk in (
                    "```python\n",
                    "value=await llm_query(prompt='one')\n",
                    "answer['content']=value\nanswer['ready']=True\n```",
                ):
                    self.assertIs(asyncio.get_running_loop(), owner)
                    future = owner.create_future()
                    owner.call_soon(future.set_result, chunk)
                    yield await future
            finally:
                self.assertIs(asyncio.get_running_loop(), owner)
                closed.append(True)

        runner = RLM(
            model,
            model_stream=stream,
            config=RLMConfig(timeout_seconds=5),
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with CodeInterpreterMiddleware(
            rlm=runner, kernel_config=KernelConfig(max_worker_processes=5)
        ) as middleware:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "one", "eval", {"code": "await rlm_query(prompt='stream')"}
                    ),
                    AIMessage(content="done"),
                ),
                middleware=[middleware],
            )
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="run RLM")]}
            )
            self.assertEqual(_results(result)[0]["result"], "ONE")
            self.assertEqual(seen, ["one"])
            self.assertEqual(closed, [True])
            self.assertIs(runner.model_stream, stream)

    async def test_rlm_stream_cancellation_drains_client_cleanup(self) -> None:
        """Cancel an application-loop stream without abandoning its cleanup."""
        owner = asyncio.get_running_loop()
        started, closed = asyncio.Event(), asyncio.Event()

        async def model(prompt: str, name: str | None) -> str:
            """Return unused fake text.

            Args:
                prompt: Unused prompt.
                name: Optional model selector.

            Returns:
                Fixed text.
            """
            return "unused"

        async def stream(prompt: str, name: str | None) -> AsyncIterator[str]:
            """Wait as a loop-bound streaming client would.

            Args:
                prompt: Unused prompt.
                name: Optional model selector.

            Yields:
                A controller header before cancellation.
            """
            try:
                yield "```python\n"
                started.set()
                await asyncio.Future()
            finally:
                self.assertIs(asyncio.get_running_loop(), owner)
                await asyncio.sleep(0.01)
                closed.set()

        runner = RLM(
            model,
            model_stream=stream,
            speculation_config=SpeculationConfig(mode="stream"),
        )
        async with CodeInterpreterMiddleware(rlm=runner) as middleware:
            state = await middleware.abefore_agent({}, None)
            await _eval(middleware, state, "value=42")
            work = asyncio.create_task(
                _eval(middleware, state, "await rlm_query(prompt='wait')")
            )
            await asyncio.wait_for(started.wait(), 2)
            work.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await work
            await asyncio.wait_for(closed.wait(), 2)
            self.assertEqual(
                (await _eval(middleware, state, "value")).artifact["result"], 42
            )

    async def test_rlm_callback_failures_and_deadlines_return_normal_errors(
        self,
    ) -> None:
        """Keep model errors and RLM wall budgets through the loop bridge."""
        owner = asyncio.get_running_loop()

        async def model(prompt: str, name: str | None) -> str:
            """Fail or wait without contacting a provider.

            Args:
                prompt: 'fail' raises; any other prompt waits for the deadline.
                name: Optional model selector.

            Returns:
                No normal response in these tests.
            """
            self.assertIs(asyncio.get_running_loop(), owner)
            if prompt == "fail":
                raise ValueError("fake model failure")
            await asyncio.Future()
            return "unreachable"

        runner = RLM(
            model,
            config=RLMConfig(timeout_seconds=0.1),
            hosted_tool_config=HostedToolConfig(max_total_call_seconds=1),
        )
        async with CodeInterpreterMiddleware(rlm=runner) as middleware:
            state = await middleware.abefore_agent({}, None)
            failed = await _eval(middleware, state, "await llm_query(prompt='fail')")
            self.assertFalse(failed.artifact["ok"])
            self.assertIn("fake model failure", failed.content)
            timed = await _eval(middleware, state, "await llm_query(prompt='wait')")
            self.assertFalse(timed.artifact["ok"])
            self.assertIn("RLMTimeout", timed.content)

    async def test_rlm_plain_cancellation_reaches_the_caller_loop(self) -> None:
        """Cancel the plain model callback and retain the last graph snapshot."""
        started, closed = asyncio.Event(), asyncio.Event()

        async def model(prompt: str, name: str | None) -> str:
            """Wait for cancellation like an async model client.

            Args:
                prompt: Unused prompt.
                name: Optional model selector.

            Returns:
                No normal result.
            """
            started.set()
            try:
                await asyncio.Future()
            finally:
                closed.set()
            return "unreachable"

        async with CodeInterpreterMiddleware(rlm=RLM(model)) as middleware:
            state = await middleware.abefore_agent({}, None)
            await _eval(middleware, state, "value=42")
            work = asyncio.create_task(
                _eval(middleware, state, "await llm_query(prompt='wait')")
            )
            await asyncio.wait_for(started.wait(), 2)
            work.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await work
            await asyncio.wait_for(closed.wait(), 2)
            self.assertEqual(
                (await _eval(middleware, state, "value")).artifact["result"], 42
            )

    async def test_shared_rlm_runner_can_serve_two_application_loops(self) -> None:
        """Bind each operation separately instead of changing the shared runner."""
        owners = {}

        async def model(prompt: str, name: str | None) -> str:
            """Use the fake client loop assigned to this graph request.

            Args:
                prompt: Request identity.
                name: Optional model selector.

            Returns:
                Request identity, after one loop-bound future.
            """
            owner = owners[prompt]
            self.assertIs(asyncio.get_running_loop(), owner)
            future = owner.create_future()
            owner.call_soon(future.set_result, prompt)
            return await future

        runner = RLM(model)
        async with CodeInterpreterMiddleware(rlm=runner) as middleware:

            async def invoke(prompt: str) -> dict[str, Any]:
                """Run a complete fake-model graph on one application loop.

                Args:
                    prompt: Independent request identity.

                Returns:
                    Completed graph result.
                """
                owners[prompt] = asyncio.get_running_loop()
                agent = create_agent(
                    _model(
                        named_tool_call(
                            prompt,
                            "eval",
                            {"code": f"await llm_query(prompt={prompt!r})"},
                        ),
                        AIMessage(content="done"),
                    ),
                    middleware=[middleware],
                )
                return await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

            def run(prompt: str) -> dict[str, Any]:
                """Give one fake graph a fresh application loop.

                Args:
                    prompt: Request identity.

                Returns:
                    Completed graph state.
                """
                return asyncio.run(invoke(prompt))

            results = await asyncio.gather(
                *(asyncio.to_thread(run, prompt) for prompt in ("one", "two"))
            )
            self.assertEqual(
                [_results(result)[0]["result"] for result in results], ["one", "two"]
            )
            self.assertIsNot(owners["one"], owners["two"])
            self.assertIs(runner.model_call, model)


class SyncInterpreterFilterTests(unittest.TestCase):
    """Keep filtering correct through the synchronous graph bridge too."""

    def test_sync_graph_cannot_call_a_filtered_tool(self) -> None:
        """Reject a filtered named PTC tool before its body runs."""
        calls = []

        @tool
        def hidden() -> int:
            """Record an unexpected call.

            Returns:
                Integer 42.
            """
            calls.append(True)
            return 42

        with CodeInterpreterMiddleware(ptc=("hidden",)) as middleware:
            agent = create_agent(
                _model(
                    named_tool_call("one", "eval", {"code": "await tools.hidden()"}),
                    AIMessage(content="done"),
                ),
                tools=[hidden],
                middleware=[FilterTools("hidden"), middleware],
            )
            result = agent.invoke({"messages": [HumanMessage(content="test")]})
            self.assertFalse(_results(result)[0]["ok"])
            self.assertEqual(calls, [])
            self.assertEqual(middleware._active_tools, {})

    def test_sync_graph_rlm_callback_uses_the_owned_kernel_loop(self) -> None:
        """Keep synchronous graphs working without an application event loop."""
        seen = []

        async def model(prompt: str, name: str | None) -> str:
            """Use the kernel loop for a synchronous graph.

            Args:
                prompt: Fake model query.
                name: Optional model selector.

            Returns:
                Uppercase query.
            """
            seen.append(asyncio.get_running_loop())
            return prompt.upper()

        with CodeInterpreterMiddleware(rlm=RLM(model)) as middleware:
            agent = create_agent(
                _model(
                    named_tool_call(
                        "one", "eval", {"code": "await llm_query(prompt='sync')"}
                    ),
                    AIMessage(content="done"),
                ),
                middleware=[middleware],
            )
            result = agent.invoke({"messages": [HumanMessage(content="test")]})
            self.assertEqual(_results(result)[0]["result"], "SYNC")
            self.assertEqual(seen, [middleware._bridge.loop])
