import asyncio
import unittest

from langchain_core.tools import tool
from pydantic_ai import FunctionToolset, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.usage import RunUsage

from arenero import (
    AsyncPersistentMontyKernel,
    HostedTool,
    HostedToolConfig,
    HostedToolSet,
    KernelConfig,
    OpenTelemetryEventSink,
    PersistentMontyKernel,
    adapt_langchain_tools,
    adapt_pydantic_ai_tools,
    create_python_repl_tools,
)


class HostedToolTests(unittest.TestCase):
    """Test framework-neutral host tools in the synchronous kernel."""

    def test_callable_catalog_invocation_and_records(self) -> None:
        """Generate an exact alias and record a successful hosted call."""

        def add(a: int, b: int = 2) -> int:
            """Add two integers."""

            return a + b

        toolset = HostedToolSet((HostedTool.from_callable(add, name="math.add"),))
        with PersistentMontyKernel(
            config=KernelConfig(type_check=True),
            hosted_tools=toolset,
        ) as kernel:
            result = kernel.execute("thread", "answer = math_add(a=40)\nanswer")
            persisted = kernel.execute("thread", "answer + 1")
            report = kernel.capabilities()

        self.assertEqual(result["result"], 42)
        self.assertEqual(persisted["result"], 43)
        self.assertEqual(result["host_calls"]["count"], 1)
        self.assertEqual(result["host_calls"]["items"][0]["tool"], "math.add")
        self.assertIn(
            "def math_add(*, a: int, b: int = 2)", report["hosted_tools"]["catalog"]
        )

    def test_call_limit_and_value_previews_are_bounded(self) -> None:
        """Reserve call budget before work and keep only bounded previews."""

        def echo(value: str) -> str:
            """Return one value."""

            return value

        toolset = HostedToolSet.from_callables(
            [echo],
            config=HostedToolConfig(
                max_calls_per_operation=1,
                include_value_previews=True,
                preview_chars=4,
            ),
        )
        with PersistentMontyKernel(hosted_tools=toolset) as kernel:
            result = kernel.execute(
                "thread",
                'first = echo(value="abcdef")\necho(value="again")',
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "hosted_tool_call_limit")
        self.assertEqual(result["error"]["recovery"], "revise_code")
        self.assertEqual(result["host_calls"]["count"], 1)
        record = result["host_calls"]["items"][0]
        self.assertIn("...", record["arguments"])
        self.assertIn("...", record["result"])

    def test_atomic_restart_removes_old_state_before_code(self) -> None:
        """Reset and run one operation without a separate reset call."""

        with PersistentMontyKernel() as kernel:
            kernel.execute("thread", "old_value = 10")
            restarted = kernel.execute("thread", "new_value = 20", restart=True)
            old = kernel.execute("thread", "old_value")
            new = kernel.execute("thread", "new_value")

        self.assertTrue(restarted["session_restarted"])
        self.assertFalse(restarted["state_restored"])
        self.assertFalse(old["ok"])
        self.assertEqual(new["result"], 20)

    def test_operation_tools_do_not_change_kernel_configuration(self) -> None:
        """Expose selected tools for one call without adding them globally."""

        def square(value: int) -> int:
            """Square one integer."""

            return value * value

        operation_tools = HostedToolSet.from_callables([square])
        with PersistentMontyKernel() as kernel:
            first = kernel.execute(
                "thread", "square(value=6)", hosted_tools=operation_tools
            )
            second = kernel.execute("thread", "square(value=7)")

        self.assertEqual(first["result"], 36)
        self.assertFalse(second["ok"])
        self.assertIn("NameError", second["error"]["message"])

    def test_invalid_schema_argument_name_is_rejected(self) -> None:
        """Reject schemas that cannot become Python keyword arguments."""

        with self.assertRaisesRegex(ValueError, "invalid Python argument"):
            HostedTool(
                name="bad",
                description="Bad schema",
                input_schema={
                    "type": "object",
                    "properties": {"not-valid": {"type": "string"}},
                },
                invoke=lambda arguments: arguments,
            )

    def test_selection_excludes_nested_execution_and_control(self) -> None:
        """Select normal tools and require explicit nested code execution."""

        schema = {"type": "object", "properties": {}}
        normal = HostedTool("normal", "Normal", schema, invoke=lambda _: None)
        code = HostedTool(
            "code",
            "Code",
            schema,
            invoke=lambda _: None,
            kind="code-execution",
        )
        control = HostedTool(
            "control",
            "Control",
            schema,
            invoke=lambda _: None,
            kind="control",
        )
        toolset = HostedToolSet((normal, code, control))

        self.assertEqual(
            [item.name for item in toolset.select().tools],
            ["normal"],
        )
        self.assertEqual(
            [item.name for item in toolset.select(include_code_execution=True).tools],
            ["normal", "code"],
        )

    def test_async_only_tool_has_structured_sync_guidance(self) -> None:
        """Return a configuration error when a sync kernel gets an async tool."""

        async def async_value() -> int:
            """Return one value asynchronously."""

            return 1

        hosted = HostedToolSet.from_callables([async_value])
        with PersistentMontyKernel(hosted_tools=hosted) as kernel:
            result = kernel.execute("thread", "async_value()")

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"]["code"],
            "hosted_tool_configuration_error",
        )
        self.assertEqual(result["error"]["recovery"], "stop")

    def test_langchain_tools_keep_the_public_invoke_path(self) -> None:
        """Call a LangChain tool through its standard validation path."""

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""

            return a * b

        hosted = adapt_langchain_tools((multiply,))
        with PersistentMontyKernel(hosted_tools=hosted) as kernel:
            result = kernel.execute("thread", "multiply(a=6, b=7)")
            repl_tools = create_python_repl_tools(
                kernel,
                hosted_tools=(multiply,),
            )

        self.assertEqual(result["result"], 42)
        self.assertEqual(result["host_calls"]["items"][0]["tool"], "multiply")
        self.assertIn("def multiply", repl_tools[0].description)

    def test_opentelemetry_sink_works_without_an_sdk_provider(self) -> None:
        """Send events to the optional no-op OpenTelemetry provider."""

        with PersistentMontyKernel(event_sink=OpenTelemetryEventSink()) as kernel:
            result = kernel.execute("thread", "21 * 2")

        self.assertEqual(result["result"], 42)


class AsyncHostedToolTests(unittest.IsolatedAsyncioTestCase):
    """Test hosted-tool adapters in the native asynchronous kernel."""

    async def test_async_callable_is_awaitable_inside_monty(self) -> None:
        """Await an asynchronous host function and record its result."""

        async def add(a: int, b: int) -> int:
            """Add two integers after an event-loop yield."""

            await asyncio.sleep(0)
            return a + b

        hosted = HostedToolSet.from_callables([add])
        async with AsyncPersistentMontyKernel(hosted_tools=hosted) as kernel:
            result = await kernel.execute("thread", "await add(a=19, b=23)")

        self.assertEqual(result["result"], 42)
        self.assertEqual(result["host_calls"]["count"], 1)

    async def test_parallel_limit_and_sequential_tool_are_enforced(self) -> None:
        """Limit normal overlap and give a sequential tool exclusive access."""

        active = 0
        maximum = 0
        exclusive_overlap = False

        async def observe(arguments: dict[str, int]) -> int:
            """Record overlap and return the supplied value."""

            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01)
                return arguments["value"]
            finally:
                active -= 1

        async def exclusive(arguments: dict[str, int]) -> int:
            """Record whether an exclusive call overlaps another call."""

            nonlocal active, exclusive_overlap
            active += 1
            exclusive_overlap = active != 1
            try:
                await asyncio.sleep(0.01)
                return arguments["value"]
            finally:
                active -= 1

        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        toolset = HostedToolSet(
            (
                HostedTool("observe", "Observe", schema, ainvoke=observe),
                HostedTool(
                    "exclusive",
                    "Exclusive",
                    schema,
                    ainvoke=exclusive,
                    sequential=True,
                ),
            ),
            HostedToolConfig(max_parallel_calls=2),
        )
        scope = toolset.async_scope("execution")
        functions = scope.async_lookup()
        results = await asyncio.gather(
            functions["observe"](value=1),
            functions["observe"](value=2),
            functions["exclusive"](value=3),
        )

        self.assertEqual(results, [1, 2, 3])
        self.assertEqual(maximum, 2)
        self.assertFalse(exclusive_overlap)
        self.assertEqual(scope.summary()["count"], 3)

    async def test_pydantic_ai_toolset_uses_its_tool_manager(self) -> None:
        """Keep Pydantic AI validation and execution in the reverse bridge."""

        def add(a: int, b: int) -> int:
            """Add two integers."""

            return a + b

        toolset = FunctionToolset([add], include_return_schema=True)
        context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        context.tool_manager = ToolManager(toolset, ctx=context)
        hosted = await adapt_pydantic_ai_tools(toolset, context)

        async with AsyncPersistentMontyKernel(hosted_tools=hosted) as kernel:
            result = await kernel.execute("thread", "await add(a=17, b=25)")

        self.assertEqual(result["result"], 42)
        self.assertEqual(result["host_calls"]["items"][0]["status"], "success")
