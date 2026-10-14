"""Validate the six runtime changes with workers, files, and a mocked agent."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, date, datetime, time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic_monty import MemoryFile, OSAccess

from arenero import (
    AsyncPersistentMontyKernel,
    FilesystemConfig,
    HostedToolSet,
    InMemorySessionStore,
    KernelConfig,
    PersistentMontyKernel,
    SandboxConfig,
    StatelessSessionStore,
    create_python_repl_tools,
    execute_repl_cell,
    open_sandbox,
)
from arenero.easy import SandboxResult
from tests.test_langchain_agent import (
    ToolCallingFakeChatModel,
    named_tool_call,
    python_tool_call,
    tool_results,
)


class RuntimeSimplificationTests(unittest.TestCase):
    """Exercise helper reuse, inputs, metadata, directories, and values."""

    def test_helpers_are_not_compiled_again_after_restore(self) -> None:
        """Keep repeated trivial calls below a small snapshot-size limit."""
        config = KernelConfig(max_state_bytes=100_000)
        with PersistentMontyKernel(config=config) as kernel:
            first = kernel.execute("agent", "value = 42")
            for _ in range(30):
                later = kernel.execute("agent", "value")
                self.assertTrue(later["ok"], later["error"])
            self.assertLess(
                later["stats"]["state_after_bytes"]
                - first["stats"]["state_after_bytes"],
                10_000,
            )

    def test_helpers_refresh_for_new_limits_and_missing_marker(self) -> None:
        """Rebuild helpers once when configuration or the saved marker changes."""
        files = OSAccess(files=[MemoryFile("/text.txt", "abcdef")])
        store = InMemorySessionStore()
        with PersistentMontyKernel(
            store,
            os_handler=files,
            filesystem_config=FilesystemConfig(default_read_limit_chars=1),
        ) as kernel:
            self.assertEqual(
                kernel.execute("agent", "fs_read('/text.txt')")["result"], "a"
            )
        with PersistentMontyKernel(
            store,
            os_handler=files,
            filesystem_config=FilesystemConfig(default_read_limit_chars=2),
        ) as kernel:
            refreshed = kernel.execute("agent", "fs_read('/text.txt')")
            self.assertEqual(refreshed["result"], "ab")
            size = refreshed["stats"]["state_after_bytes"]
            later = kernel.execute("agent", "fs_read('/text.txt')")
            self.assertLess(later["stats"]["state_after_bytes"] - size, 2_000)
            kernel.execute("agent", "_arenero_fs_version = None")
            restored = kernel.execute("agent", "fs_read('/text.txt')")
            self.assertEqual(restored["result"], "ab")
            self.assertGreater(
                restored["stats"]["state_after_bytes"]
                - later["stats"]["state_after_bytes"],
                10_000,
            )

    def test_inputs_persist_and_obey_state_policies(self) -> None:
        """Bind values after bootstrap and save them only under the chosen policy."""
        with PersistentMontyKernel(
            config=KernelConfig(
                type_check=True, type_check_stubs="values: list[int]\n"
            ),
            bootstrap_code="base = 1",
        ) as kernel:
            session = kernel.session("agent")
            first = session.execute(
                "sum(values) + base", inputs={"values": [20, 21], "base": 1}
            )
            self.assertEqual(first["result"], 42)
            self.assertEqual(session.execute("sum(values)")["result"], 41)
            session.execute(
                "sum(values)", inputs={"values": [99]}, state_policy="never"
            )
            session.execute("1 / 0", inputs={"values": [88]}, state_policy="success")
            self.assertEqual(session.execute("sum(values)")["result"], 41)
            self.assertEqual(
                session.execute("values", inputs={"values": [42]}, restart=True)[
                    "result"
                ],
                [42],
            )

    def test_checked_inputs_need_stable_declarations(self) -> None:
        """Keep native input binding separate from Monty's static declarations."""
        with PersistentMontyKernel(config=KernelConfig(type_check=True)) as kernel:
            result = kernel.execute("agent", "sum(values)", inputs={"values": [42]})
            self.assertEqual(result["error"]["type"], "MontyTypingError")

    def test_inspection_does_not_create_state_or_run_startup(self) -> None:
        """Inspect absent and saved state without changing revisions or callbacks."""
        calls = []

        def startup() -> int:
            """Record startup and return a constant.

            Returns:
                The integer 42.
            """
            calls.append("started")
            return 42

        with PersistentMontyKernel(
            external_lookup={"startup": startup}, bootstrap_code="answer = startup()"
        ) as kernel:
            self.assertEqual(kernel.variables("new"), {})
            self.assertEqual(execute_repl_cell(kernel, "new", "%who")["result"], [])
            self.assertEqual(calls, [])
            self.assertIsNone(kernel.store.load("new"))
            kernel.execute("new", "_private = 'hidden'\nvalues = [42]")
            before = kernel.store.load("new")
            self.assertEqual(
                kernel.session("new").variables(), {"answer": "int", "values": "list"}
            )
            self.assertEqual(
                execute_repl_cell(kernel, "new", "%who")["result"], ["answer", "values"]
            )
            self.assertEqual(
                execute_repl_cell(kernel, "new", "%whos")["result"],
                {"answer": "int", "values": "list"},
            )
            after = kernel.store.load("new")
            self.assertEqual(
                (before.state, before.revision), (after.state, after.revision)
            )
            self.assertEqual(calls, ["started"])

    def test_inspection_hides_hosted_helpers_and_trims_model_output(self) -> None:
        """Keep names useful and preserve internal data when output is small."""

        def add(a: int, b: int) -> int:
            """Add the supplied integers.

            Args:
                a: First integer.
                b: Second integer.

            Returns:
                Their sum.
            """
            return a + b

        with PersistentMontyKernel(
            hosted_tools=HostedToolSet.from_callables([add]), os_handler=OSAccess()
        ) as kernel:
            kernel.execute("agent", "a_long_public_name = add(a=20, b=22)")
            execute_repl_cell(kernel, "agent", "!pwd")
            self.assertEqual(kernel.variables("agent"), {"a_long_public_name": "int"})
            result = execute_repl_cell(kernel, "agent", "%whos", 1)
            self.assertTrue(result["truncated"]["result"])
            self.assertEqual(result["output_limit_chars"], 1)
            self.assertFalse(result["state_saved"])

    def test_inspection_rejects_other_versions_and_stateless_has_no_names(self) -> None:
        """Do not load incompatible state or invent state in stateless mode."""
        store = InMemorySessionStore()
        store.save("old", b"invalid", "0.0.21", expected_revision=None)
        with PersistentMontyKernel(store) as kernel:
            result = execute_repl_cell(kernel, "old", "%whos")
            self.assertEqual(result["error"]["code"], "session_inspection_failed")
            self.assertEqual(store.load("old").state, b"invalid")
        with PersistentMontyKernel(StatelessSessionStore()) as kernel:
            kernel.execute("agent", "answer = 42")
            self.assertEqual(kernel.variables("agent"), {})

    def test_one_directory_is_used_by_python_shell_and_files(self) -> None:
        """Observe native directory changes through every managed surface."""
        with open_sandbox("agent") as sandbox:
            self.assertEqual(sandbox.cwd, "/workspace")
            self.assertEqual(
                sandbox.run_python("import os\nos.getcwd()").value, "/workspace"
            )
            sandbox.mkdir("sub")
            self.assertTrue(sandbox.chdir("sub").ok)
            self.assertEqual(sandbox.run_python("os.getcwd()").value, "/workspace/sub")
            self.assertTrue(sandbox.run_python("open('value.txt', 'w').write('42')").ok)
            self.assertEqual(sandbox.read("value.txt").value, "42")
            self.assertTrue(sandbox.run_shell("cd /workspace").ok)
            self.assertEqual(sandbox.cwd, "/workspace")
            self.assertTrue(sandbox.run_python("os.chdir('/workspace/sub')").ok)
            self.assertEqual(sandbox.run_shell("pwd").value, "/workspace/sub\n")
            self.assertFalse(sandbox.chdir("missing").ok)
            self.assertEqual(sandbox.cwd, "/workspace/sub")

    def test_managed_shell_does_not_spend_a_command_on_cd(self) -> None:
        """Run a single command when the configured shell budget is one."""
        config = SandboxConfig.conversation(kernel=KernelConfig(max_shell_commands=1))
        with open_sandbox("agent", config=config) as sandbox:
            result = sandbox.run_shell("pwd")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.value, "/workspace\n")

    def test_initial_directory_precedes_bootstrap_and_stateless_calls_reset_it(
        self,
    ) -> None:
        """Use the configured root before startup without retaining stateless cd."""
        config = SandboxConfig.ephemeral(
            workspace_root="/project",
            bootstrap_code="import os\nstarted_at = os.getcwd()",
        )
        with open_sandbox("agent", config=config) as sandbox:
            self.assertEqual(sandbox.run_python("started_at").value, "/project")
            sandbox.mkdir("sub")
            self.assertTrue(sandbox.chdir("sub").ok)
            self.assertEqual(sandbox.cwd, "/project")
            self.assertEqual(sandbox.run_shell("cd sub && pwd").value, "/project/sub\n")
            self.assertEqual(sandbox.run_shell("pwd").value, "/project\n")
            self.assertEqual(sandbox.run_cell("%who").value, [])

    def test_saved_directory_obeys_state_policy_fork_reset_and_checkpoint(self) -> None:
        """Use the saved directory rather than a second handle-level value."""
        with open_sandbox("agent") as sandbox:
            sandbox.mkdir("sub")
            sandbox.chdir("sub")
            sandbox.run_python(
                "import os\nos.chdir('/workspace')", state_policy="never"
            )
            self.assertEqual(sandbox.cwd, "/workspace/sub")
            sandbox.run_python("os.chdir('/workspace')\n1 / 0", state_policy="success")
            self.assertEqual(sandbox.cwd, "/workspace/sub")
            child = sandbox.fork("child").value
            self.assertEqual(child.cwd, "/workspace/sub")
            child.chdir("/workspace")
            self.assertEqual(sandbox.cwd, "/workspace/sub")
            checkpoint = sandbox.checkpoint().value
            sandbox.chdir("/workspace")
            self.assertTrue(sandbox.restore(checkpoint).ok)
            self.assertEqual(sandbox.cwd, "/workspace/sub")
            sandbox.reset()
            self.assertEqual(sandbox.cwd, "/workspace")

    def test_directory_survives_durable_reopen_and_is_not_output_trimmed(self) -> None:
        """Read the full native directory after a process-local handle restart."""
        with tempfile.TemporaryDirectory() as directory:
            config = SandboxConfig.durable(
                directory,
                kernel=KernelConfig(
                    default_output_limit_chars=1, max_output_limit_chars=1
                ),
            )
            with open_sandbox("agent", config=config) as sandbox:
                Path(directory, "workspace", "long-directory").mkdir()
                self.assertTrue(sandbox.chdir("/workspace/long-directory").ok)
                self.assertEqual(sandbox.cwd, "/workspace/long-directory")
            with open_sandbox("agent", config=config) as sandbox:
                self.assertEqual(sandbox.cwd, "/workspace/long-directory")
                self.assertTrue(sandbox.run_python("42").ok)

    def test_results_keep_nested_bytes_complex_dates_and_instance_attributes(
        self,
    ) -> None:
        """Use one explicit converter for kernel and managed result values."""
        code = (
            "from datetime import date\nclass Record:\n    def __init__(self):\n        self.value = 1+2j\n"
            "{'blob': b'OK', 'day': date(2026, 1, 2), 'record': Record()}"
        )
        expected = {
            "blob": {"encoding": "base64", "content": "T0s="},
            "day": "2026-01-02",
            "record": {
                "type": "monty_instance",
                "class": "Record",
                "attributes": {"value": {"type": "complex", "real": 1.0, "imag": 2.0}},
                "is_dataclass": False,
            },
        }
        with PersistentMontyKernel() as kernel:
            result = kernel.execute("agent", code)
            self.assertEqual(result["result"], expected)
            self.assertEqual(kernel.execute("agent", "(1+2j).imag")["result"], 2)
            json.dumps(result)
        with open_sandbox("agent") as sandbox:
            result = sandbox.run_python(code)
            self.assertEqual(result.value, expected)
            self.assertEqual(result.to_dict()["value"], expected)
        self.assertEqual(
            SandboxResult(True, {"day": date(2026, 1, 2), "blob": b"OK"}).to_dict()[
                "value"
            ],
            {"day": "2026-01-02", "blob": expected["blob"]},
        )

    def test_structured_result_trimming_and_offload_use_the_same_encoding(self) -> None:
        """Keep the full encoded value in overflow output and report its length."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            result = kernel.execute(
                "agent",
                "{'blob': b'OK', 'number': 1+2j}",
                1,
                output_file="/result.json",
            )
            value = json.loads(kernel.download_file("/result.json"))["result"]
        self.assertTrue(result["truncated"]["result"])
        self.assertEqual(value["blob"], {"encoding": "base64", "content": "T0s="})
        self.assertEqual(value["number"], {"type": "complex", "real": 1.0, "imag": 2.0})
        self.assertEqual(
            result["stats"]["original_output_chars"]["result"],
            len(json.dumps(value, ensure_ascii=False)),
        )

    def test_dataclass_and_other_result_shapes_keep_the_shared_contract(self) -> None:
        """Convert field copies and host dates without changing binary transfers."""
        with PersistentMontyKernel() as kernel:
            result = kernel.execute(
                "agent",
                "from dataclasses import dataclass\n@dataclass\nclass Row:\n    value: int\nRow(42)",
            )
            self.assertEqual(
                result["result"],
                {
                    "type": "monty_instance",
                    "class": "Row",
                    "attributes": {"value": 42},
                    "is_dataclass": True,
                },
            )
            cyclic = kernel.execute("agent", "items = []\nitems.append(items)\nitems")
            self.assertEqual(cyclic["result"], ["[...]"])
        result = SandboxResult(
            True,
            {1: (datetime(2026, 1, 2, 3, 4, tzinfo=UTC), time(3, 4), Path("/data"))},
        ).to_dict()
        self.assertEqual(
            result["value"], {"1": ["2026-01-02T03:04:00+00:00", "03:04:00", "/data"]}
        )
        with open_sandbox("agent") as sandbox:
            sandbox.write("data.bin", b"OK")
            binary = sandbox.read("data.bin", encoding=None)
            self.assertEqual(binary.value, b"OK")
            self.assertEqual(
                binary.to_dict()["value"], {"encoding": "base64", "content": "T0s="}
            )

    def test_managed_inputs_and_inspection_results_are_json_safe(self) -> None:
        """Expose host inputs and percent inspection through the managed API."""
        with open_sandbox("agent") as sandbox:
            result = sandbox.run_python("sum(values)", inputs={"values": [20, 22]})
            self.assertEqual(result.value, 42)
            self.assertEqual(
                sandbox.run_cell("%whos").to_dict()["value"], {"values": "list"}
            )

    def test_nested_bundled_imports_run_but_virtual_and_wildcard_imports_fail(
        self,
    ) -> None:
        """Permit standard imports in supported blocks without relaxing file imports."""
        source = (
            "def score(values):\n    import math as m\n    from functools import reduce\n"
            "    if values:\n        import json\n    return reduce(lambda a,b: a+b, values) + m.sqrt(4)\n"
        )
        files = OSAccess(files=[MemoryFile("/helper.py", source)])
        with PersistentMontyKernel(os_handler=files) as kernel:
            result = kernel.execute_module("agent", "/helper.py", "score([20,20])")
            self.assertEqual(result["result"], 42)
            self.assertEqual(kernel.execute("agent", "score([1,1])")["result"], 4)
            for invalid in [
                "def f():\n    import other",
                "if True:\n    from .other import x",
                "def f():\n    import math, other",
                "def f():\n    from math import *",
            ]:
                kernel.upload_file("/bad.py", invalid.encode())
                bad = kernel.execute_module("agent", "/bad.py")
                self.assertEqual(bad["error"]["type"], "VirtualModuleError")


class AsyncRuntimeSimplificationTests(unittest.IsolatedAsyncioTestCase):
    """Check helper, input, inspection, and output parity in AsyncMonty."""

    async def test_async_helpers_inputs_and_readonly_inspection(self) -> None:
        """Reuse helper code and inspect the same stored values as the sync API."""
        async with AsyncPersistentMontyKernel(
            config=KernelConfig(
                max_state_bytes=100_000,
                type_check=True,
                type_check_stubs="values: list[int]\n",
            )
        ) as kernel:
            session = kernel.session("agent")
            first = await session.execute("sum(values)", inputs={"values": [20, 22]})
            for _ in range(30):
                later = await session.execute("sum(values)")
                self.assertTrue(later["ok"], later["error"])
            self.assertLess(
                later["stats"]["state_after_bytes"]
                - first["stats"]["state_after_bytes"],
                10_000,
            )
            before = kernel.store.load("agent")
            self.assertEqual(await session.variables(), {"values": "list"})
            after = kernel.store.load("agent")
            self.assertEqual(
                (before.state, before.revision), (after.state, after.revision)
            )
            self.assertEqual(await kernel.variables("missing"), {})
            self.assertIsNone(kernel.store.load("missing"))

    async def test_async_nested_imports_and_structured_results(self) -> None:
        """Keep new module rules and the result encoding consistent with sync."""
        files = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "def value():\n    import math\n    return complex(math.sqrt(4), 1)",
                )
            ]
        )
        async with AsyncPersistentMontyKernel(os_handler=files) as kernel:
            result = await kernel.execute_module(
                "agent", "/helper.py", "{'number': value(), 'blob': b'OK'}"
            )
            self.assertEqual(
                result["result"],
                {
                    "number": {"type": "complex", "real": 2.0, "imag": 1.0},
                    "blob": {"encoding": "base64", "content": "T0s="},
                },
            )


class SimplifiedAgentGraphTests(unittest.TestCase):
    """Use the stable REPL tool in a complete scripted two-human-turn agent."""

    def test_agent_inspects_state_then_reuses_directory_and_nested_helper(self) -> None:
        """Combine host inputs, inspection, modules, shell, and structured results."""
        files = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "def score(values):\n    import math\n    return sum(values) + math.sqrt(4)",
                )
            ]
        )
        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call("inspect", "%whos"),
                    python_tool_call("directory", "%%shell\nmkdir -p /work\ncd /work"),
                    named_tool_call(
                        "calculate",
                        "python_repl_execute_module",
                        {
                            "entry_path": "/helper.py",
                            "code": "answer = score(values)\n{'answer': answer, 'number': complex(answer, 1)}",
                        },
                    ),
                    AIMessage(content="The answer is 42."),
                    python_tool_call("names", "%who"),
                    python_tool_call(
                        "reuse",
                        "import os\n{'answer': answer, 'cwd': os.getcwd(), 'blob': b'OK'}",
                    ),
                    AIMessage(content="I reused the saved answer and directory."),
                ]
            )
        )
        with PersistentMontyKernel(os_handler=files) as kernel:
            kernel.execute("agent", "values", inputs={"values": [20, 20]})
            agent = create_agent(
                model, create_python_repl_tools(kernel), checkpointer=InMemorySaver()
            )
            config = {"configurable": {"thread_id": "agent"}}
            first = agent.invoke(
                {
                    "messages": [
                        HumanMessage(content="Calculate from the supplied values.")
                    ]
                },
                config,
            )
            second = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Reuse the result and tell me the directory."
                        )
                    ]
                },
                config,
            )
            results = tool_results(second)
            self.assertTrue(all(result["ok"] for result in results), results)
            self.assertEqual(results[0]["result"], {"values": "list"})
            self.assertFalse(results[0]["state_saved"])
            self.assertEqual(
                results[2]["result"]["number"],
                {"type": "complex", "real": 42.0, "imag": 1.0},
            )
            self.assertIn("answer", results[3]["result"])
            self.assertFalse(any(name.startswith("_") for name in results[3]["result"]))
            self.assertEqual(
                results[4]["result"],
                {
                    "answer": 42.0,
                    "cwd": "/work",
                    "blob": {"encoding": "base64", "content": "T0s="},
                },
            )
            self.assertLess(len(kernel.store.load("agent").state), 100_000)
        self.assertEqual(first["messages"][-1].content, "The answer is 42.")
        self.assertEqual(
            second["messages"][-1].content, "I reused the saved answer and directory."
        )
