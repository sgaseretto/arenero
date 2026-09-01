"""Model-driven agent tests for session lifecycle and worker behavior."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from arenero import (
    AsyncPersistentMontyKernel,
    KernelConfig,
    SQLiteSessionStore,
    create_async_python_repl_tools,
)
from tests.test_langchain_agent import (
    ToolCallingFakeChatModel,
    python_tool_call,
    tool_results,
)


def _config(thread_id: str) -> RunnableConfig:
    """Build one LangGraph configuration with a persistent thread ID.

    Args:
        thread_id: Agent and REPL session identity.

    Returns:
        Runnable configuration for an agent invocation.
    """

    return {"configurable": {"thread_id": thread_id}}


class AgentIsolationAndDurabilityTests(unittest.IsolatedAsyncioTestCase):
    """Check complete agent isolation, restarts, and state ownership."""

    async def test_concurrent_agent_threads_keep_separate_repl_state(self) -> None:
        """Run two complete agents concurrently without crossing session state."""

        model_a = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("a-store", "value = 10"),
                    AIMessage(content="A stored."),
                    python_tool_call("a-read", "value += 1\nvalue"),
                    AIMessage(content="A finished."),
                ]
            )
        )
        model_b = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("b-store", "value = 20"),
                    AIMessage(content="B stored."),
                    python_tool_call("b-read", "value += 2\nvalue"),
                    AIMessage(content="B finished."),
                ]
            )
        )

        async with AsyncPersistentMontyKernel() as kernel:
            tools = create_async_python_repl_tools(kernel)
            agent_a = create_agent(model_a, tools, checkpointer=InMemorySaver())
            agent_b = create_agent(model_b, tools, checkpointer=InMemorySaver())
            await asyncio.gather(
                agent_a.ainvoke(
                    {"messages": [HumanMessage(content="Store A.")]},
                    _config("agent-a"),
                ),
                agent_b.ainvoke(
                    {"messages": [HumanMessage(content="Store B.")]},
                    _config("agent-b"),
                ),
            )
            second_a, second_b = await asyncio.gather(
                agent_a.ainvoke(
                    {"messages": [HumanMessage(content="Update A.")]},
                    _config("agent-a"),
                ),
                agent_b.ainvoke(
                    {"messages": [HumanMessage(content="Update B.")]},
                    _config("agent-b"),
                ),
            )
            sessions = kernel.store.list_sessions()

        result_a = tool_results(second_a)[-1]
        result_b = tool_results(second_b)[-1]
        self.assertEqual(result_a["result"], 11)
        self.assertEqual(result_b["result"], 22)
        self.assertTrue(result_a["state_restored"])
        self.assertTrue(result_b["state_restored"])
        self.assertEqual(
            sorted(session.session_id for session in sessions),
            ["agent-a", "agent-b"],
        )

    async def test_agent_repl_state_survives_kernel_and_graph_restart(self) -> None:
        """Restore SQLite state through new kernel, model, and graph instances."""

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "agent-sessions.db"
            first_model = ToolCallingFakeChatModel(
                messages=iter(
                    [
                        python_tool_call("persist", "value = 40"),
                        AIMessage(content="Stored."),
                    ]
                )
            )
            async with AsyncPersistentMontyKernel(
                SQLiteSessionStore(database)
            ) as first_kernel:
                first_agent = create_agent(
                    first_model,
                    create_async_python_repl_tools(first_kernel),
                    checkpointer=InMemorySaver(),
                )
                first = await first_agent.ainvoke(
                    {"messages": [HumanMessage(content="Store forty.")]},
                    _config("durable-agent"),
                )

            second_model = ToolCallingFakeChatModel(
                messages=iter(
                    [
                        python_tool_call("restore", "value + 2"),
                        AIMessage(content="Restored."),
                    ]
                )
            )
            async with AsyncPersistentMontyKernel(
                SQLiteSessionStore(database)
            ) as second_kernel:
                second_agent = create_agent(
                    second_model,
                    create_async_python_repl_tools(second_kernel),
                    checkpointer=InMemorySaver(),
                )
                second = await second_agent.ainvoke(
                    {"messages": [HumanMessage(content="Use the stored value.")]},
                    _config("durable-agent"),
                )

        self.assertTrue(tool_results(first)[0]["state_saved"])
        restored = tool_results(second)[0]
        self.assertTrue(restored["state_restored"])
        self.assertEqual(restored["result"], 42)

    async def test_graph_and_repl_state_have_independent_lifetimes(self) -> None:
        """Reset either graph or REPL state without silently resetting the other."""

        reset_model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("store-reset", "value = 41"),
                    AIMessage(content="Stored."),
                    python_tool_call("read-reset", "value"),
                    AIMessage(content="Read attempted."),
                ]
            )
        )
        kept_model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("store-kept", "kept = 42"),
                    AIMessage(content="Stored."),
                ]
            )
        )
        new_graph_model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("read-kept", "kept"),
                    AIMessage(content="Read."),
                ]
            )
        )

        async with AsyncPersistentMontyKernel() as kernel:
            tools = create_async_python_repl_tools(kernel)
            reset_agent = create_agent(reset_model, tools, checkpointer=InMemorySaver())
            await reset_agent.ainvoke(
                {"messages": [HumanMessage(content="Store a value.")]},
                _config("reset-repl"),
            )
            await kernel.reset("reset-repl")
            graph_kept = await reset_agent.ainvoke(
                {"messages": [HumanMessage(content="Read it again.")]},
                _config("reset-repl"),
            )

            old_graph = create_agent(kept_model, tools, checkpointer=InMemorySaver())
            await old_graph.ainvoke(
                {"messages": [HumanMessage(content="Store another value.")]},
                _config("kept-repl"),
            )
            new_graph = create_agent(
                new_graph_model, tools, checkpointer=InMemorySaver()
            )
            repl_kept = await new_graph.ainvoke(
                {"messages": [HumanMessage(content="Read existing REPL state.")]},
                _config("kept-repl"),
            )

        missing = tool_results(graph_kept)[-1]
        restored = tool_results(repl_kept)[0]
        self.assertFalse(missing["ok"])
        self.assertIn("NameError", missing["error"]["message"])
        self.assertEqual(
            len(
                [
                    message
                    for message in graph_kept["messages"]
                    if isinstance(message, HumanMessage)
                ]
            ),
            2,
        )
        self.assertTrue(restored["state_restored"])
        self.assertEqual(restored["result"], 42)
        self.assertEqual(
            len(
                [
                    message
                    for message in repl_kept["messages"]
                    if isinstance(message, HumanMessage)
                ]
            ),
            1,
        )


class AgentStatePolicyTests(unittest.IsolatedAsyncioTestCase):
    """Check state-save policies through complete model and tool loops."""

    async def test_error_state_policy_controls_the_next_agent_turn(self) -> None:
        """Keep or discard changes before a normal Monty execution error."""

        expected = {"always": 42, "success": 41, "never": 41}
        async with AsyncPersistentMontyKernel() as kernel:
            for policy, expected_value in expected.items():
                with self.subTest(policy=policy):
                    model = ToolCallingFakeChatModel(
                        messages=iter(
                            [
                                python_tool_call("store", "value = 41"),
                                AIMessage(content="Stored."),
                                python_tool_call(
                                    "fail",
                                    "value = 42\n1 / 0",
                                    state_policy=policy,
                                ),
                                AIMessage(content="Failure handled."),
                                python_tool_call("read", "value"),
                                AIMessage(content="Read."),
                            ]
                        )
                    )
                    agent = create_agent(
                        model,
                        create_async_python_repl_tools(kernel),
                        checkpointer=InMemorySaver(),
                    )
                    config = _config(f"policy-{policy}")
                    await agent.ainvoke(
                        {"messages": [HumanMessage(content="Store.")]}, config
                    )
                    failed = await agent.ainvoke(
                        {"messages": [HumanMessage(content="Change then fail.")]},
                        config,
                    )
                    read = await agent.ainvoke(
                        {"messages": [HumanMessage(content="Read.")]}, config
                    )

                    failure = tool_results(failed)[-1]
                    current = tool_results(read)[-1]
                    self.assertFalse(failure["ok"])
                    self.assertEqual(failure["state_saved"], policy == "always")
                    self.assertEqual(current["result"], expected_value)


class AgentInterruptionAndReplayTests(unittest.IsolatedAsyncioTestCase):
    """Check graph interruption and the defined duplicate-delivery behavior."""

    async def test_interrupt_before_tool_resumes_without_duplicate_execution(
        self,
    ) -> None:
        """Pause before the REPL call and execute it once after graph resume."""

        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("approved", "value = 42"),
                    AIMessage(content="Done."),
                ]
            )
        )
        async with AsyncPersistentMontyKernel() as kernel:
            agent = create_agent(
                model,
                create_async_python_repl_tools(kernel),
                checkpointer=InMemorySaver(),
                interrupt_before=["tools"],
            )
            config = _config("interrupted-agent")
            paused = await agent.ainvoke(
                {"messages": [HumanMessage(content="Set a value.")]}, config
            )
            sessions_before = kernel.store.list_sessions()
            resumed = await agent.ainvoke(None, config)
            sessions_after = kernel.store.list_sessions()

        self.assertEqual(tool_results(paused), [])
        self.assertEqual(sessions_before, [])
        result = tool_results(resumed)[0]
        self.assertEqual(result["session_revision"], 1)
        self.assertEqual(len(sessions_after), 1)
        self.assertEqual(sessions_after[0].revision, 1)

    async def test_duplicate_delivery_has_at_least_once_semantics(self) -> None:
        """Execute a repeated tool-call ID twice and document no deduplication."""

        initializer = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("initialize", "counter = 0"),
                    AIMessage(content="Initialized."),
                ]
            )
        )
        first_replay = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("same-delivery", "counter += 1\ncounter"),
                    AIMessage(content="Applied."),
                ]
            )
        )
        second_replay = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("same-delivery", "counter += 1\ncounter"),
                    AIMessage(content="Applied again."),
                ]
            )
        )

        async with AsyncPersistentMontyKernel() as kernel:
            tools = create_async_python_repl_tools(kernel)
            await create_agent(initializer, tools).ainvoke(
                {"messages": [HumanMessage(content="Initialize.")]},
                _config("replayed-agent"),
            )
            first = await create_agent(first_replay, tools).ainvoke(
                {"messages": [HumanMessage(content="Apply delivery.")]},
                _config("replayed-agent"),
            )
            second = await create_agent(second_replay, tools).ainvoke(
                {"messages": [HumanMessage(content="Replay delivery.")]},
                _config("replayed-agent"),
            )
            persisted = await kernel.execute(
                "replayed-agent", "counter", state_policy="never"
            )

        self.assertEqual(tool_results(first)[0]["result"], 1)
        self.assertEqual(tool_results(second)[0]["result"], 2)
        self.assertEqual(persisted["result"], 2)


class AgentCancellationAndCapacityTests(unittest.IsolatedAsyncioTestCase):
    """Check cancellation cleanup and worker-pool retry behavior."""

    async def test_cancelled_agent_releases_worker_and_session_lock(self) -> None:
        """Cancel a hosted wait and run another agent on the same thread."""

        started = asyncio.Event()
        release = asyncio.Event()

        @tool
        async def hold() -> int:
            """Wait until the test releases the hosted call."""

            started.set()
            await release.wait()
            return 1

        blocked_model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("blocked", "await hold()"),
                    AIMessage(content="Not reached."),
                ]
            )
        )
        recovery_model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("recover", "1 + 1"),
                    AIMessage(content="Recovered."),
                ]
            )
        )
        config = KernelConfig(
            max_worker_processes=1,
            worker_checkout_timeout_seconds=0.2,
        )

        async with AsyncPersistentMontyKernel(config=config) as kernel:
            hosted_tools = (hold,)
            blocked = create_agent(
                blocked_model,
                create_async_python_repl_tools(kernel, hosted_tools=hosted_tools),
            )
            task = asyncio.create_task(
                blocked.ainvoke(
                    {"messages": [HumanMessage(content="Wait.")]},
                    _config("cancelled-agent"),
                )
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()

            recovery = create_agent(
                recovery_model,
                create_async_python_repl_tools(kernel, hosted_tools=hosted_tools),
            )
            recovered = await recovery.ainvoke(
                {"messages": [HumanMessage(content="Try again.")]},
                _config("cancelled-agent"),
            )

        self.assertEqual(tool_results(recovered)[0]["result"], 2)

    async def test_worker_saturation_returns_retryable_then_recovers(self) -> None:
        """Return busy guidance to one agent and succeed after capacity returns."""

        started = asyncio.Event()
        release = asyncio.Event()

        @tool
        async def hold() -> int:
            """Keep the only worker busy until released by the test."""

            started.set()
            await release.wait()
            return 1

        owner_model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("owner", "await hold()"),
                    AIMessage(content="Released."),
                ]
            )
        )
        waiting_model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("busy", "2 + 2"),
                    AIMessage(content="Busy."),
                    python_tool_call("retry", "2 + 2"),
                    AIMessage(content="Succeeded."),
                ]
            )
        )
        config = KernelConfig(
            max_worker_processes=1,
            worker_checkout_timeout_seconds=0.02,
        )

        async with AsyncPersistentMontyKernel(config=config) as kernel:
            tools = create_async_python_repl_tools(kernel, hosted_tools=(hold,))
            owner = create_agent(owner_model, tools)
            waiter = create_agent(waiting_model, tools, checkpointer=InMemorySaver())
            owner_task = asyncio.create_task(
                owner.ainvoke(
                    {"messages": [HumanMessage(content="Own worker.")]},
                    _config("worker-owner"),
                )
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            busy = await waiter.ainvoke(
                {"messages": [HumanMessage(content="Try while busy.")]},
                _config("waiting-agent"),
            )
            release.set()
            await owner_task
            retried = await waiter.ainvoke(
                {"messages": [HumanMessage(content="Retry now.")]},
                _config("waiting-agent"),
            )

        busy_result = tool_results(busy)[0]
        retry_result = tool_results(retried)[-1]
        self.assertEqual(busy_result["error"]["type"], "WorkerCheckoutTimeout")
        self.assertTrue(busy_result["retryable"])
        self.assertEqual(busy_result["error"]["recovery"], "retry_later")
        self.assertEqual(retry_result["result"], 4)


if __name__ == "__main__":
    unittest.main()
