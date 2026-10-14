"""Check execution API parity, raw package metadata, imports, and binary assets."""

from __future__ import annotations

import asyncio
import io
import json
import unittest
import zipfile
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic_monty import MemoryFile, MontyError, OSAccess

from arenero import (
    AreneroSandbox,
    AsyncAreneroSandbox,
    AsyncPersistentMontyKernel,
    FilesystemConfig,
    KernelConfig,
    MontyPackageManager,
    PackageConfig,
    PersistentMontyKernel,
    SandboxConfig,
    VirtualModuleConfig,
    create_async_python_repl_tools,
    execute_repl_cell,
    open_async_sandbox,
    open_sandbox,
)
from arenero.mcp import McpHub
from tests.test_langchain_agent import (
    ToolCallingFakeChatModel,
    named_tool_call,
    python_tool_call,
    tool_results,
)


class ExecutionApiTests(unittest.TestCase):
    """Check host input forwarding and module output across sync surfaces."""

    def test_file_inputs_bind_after_bootstrap_and_only_once(self) -> None:
        """Keep file mutations visible to follow-up code and later typed calls."""
        files = OSAccess(files=[MemoryFile("/helper.py", "values.append(22)")])
        config = KernelConfig(type_check=True, type_check_stubs="values: list[int]\n")
        with PersistentMontyKernel(
            os_handler=files, config=config, bootstrap_code="values = [100]"
        ) as kernel:
            session = kernel.session("agent")
            supplied = [20]
            result = session.execute_file(
                "/helper.py", "sum(values)", inputs={"values": supplied}
            )
            self.assertEqual(result["result"], 42)
            self.assertEqual(supplied, [20])
            self.assertEqual(session.execute("values")["result"], [20, 22])
            session.execute_file(
                "/helper.py", inputs={"values": [99]}, state_policy="never"
            )
            kernel.upload_file("/bad.py", b"values.append(99)\n1 / 0")
            session.execute_file(
                "/bad.py", inputs={"values": [88]}, state_policy="success"
            )
            self.assertEqual(session.execute("values")["result"], [20, 22])

    def test_cells_and_facades_forward_inputs_without_shell_mutations(self) -> None:
        """Use inputs on Python and %run, and reject them on shell/control cells."""
        files = OSAccess(
            files=[MemoryFile("/helper.py", "values.append(22)\nsum(values)")]
        )
        with PersistentMontyKernel(os_handler=files) as kernel:
            facade = AreneroSandbox(kernel, session_id="agent")
            self.assertEqual(
                facade.execute_python("sum(values)", inputs={"values": [42]}).output,
                "42",
            )
            self.assertEqual(
                facade.execute_file("/helper.py", inputs={"values": [20]}).output, "42"
            )
            self.assertEqual(
                facade.execute_cell("%run /helper.py", inputs={"values": [20]})[
                    "result"
                ],
                42,
            )
            before = kernel.store.load("agent")
            for cell in ("%reset", "%limits", "!touch /bad", "%%shell\ntouch /bad"):
                result = execute_repl_cell(
                    kernel, "agent", cell, inputs={"value": 1}, restart=True
                )
                self.assertEqual(result["error"]["code"], "invalid_cell_inputs")
            self.assertEqual(kernel.store.load("agent").state, before.state)
            self.assertEqual(kernel.store.load("agent").revision, before.revision)
            with self.assertRaises(MontyError):
                kernel.download_file("/bad")

    def test_module_inputs_and_overflow_work_through_bound_and_facade_apis(
        self,
    ) -> None:
        """Keep the same module value and JSON file through both sync wrappers."""
        files = OSAccess(
            files=[MemoryFile("/main.py", "def double(x):\n    return x * 2")]
        )
        with PersistentMontyKernel(os_handler=files) as kernel:
            session = kernel.session("agent")
            result = session.execute_module(
                "/main.py",
                "{'answer': double(value), 'blob': b'OK'}",
                0,
                inputs={"value": 21},
                output_file="/output.json",
            )
            self.assertTrue(result["ok"], result["error"])
            self.assertEqual(result["output_file"], "/output.json")
            self.assertEqual(
                json.loads(kernel.download_file("/output.json"))["result"]["answer"], 42
            )
            facade = AreneroSandbox(kernel, session_id="agent")
            self.assertEqual(
                facade.execute_module(
                    "/main.py", "double(value)", inputs={"value": 21}
                ).output,
                "42",
            )

    def test_managed_file_module_and_run_cells_forward_all_options(self) -> None:
        """Resolve managed paths and keep input and overflow behavior consistent."""
        with open_sandbox("agent") as sandbox:
            sandbox.write("helper.py", "values.append(22)\nsum(values)")
            result = sandbox.run_cell(
                "%run helper.py", 0, inputs={"values": [20]}, output_file="result.json"
            )
            self.assertTrue(result.ok, result.error)
            self.assertIn("/workspace/result.json", result.artifacts)
            self.assertEqual(
                json.loads(sandbox.read("result.json").value)["result"], 42
            )
            self.assertEqual(
                sandbox.run_file(
                    "helper.py", "sum(values)", inputs={"values": [20]}
                ).value,
                42,
            )
            sandbox.write("main.py", "def double(x):\n    return x * 2")
            result = sandbox.run_module(
                "main.py",
                "double(value)",
                0,
                inputs={"value": 21},
                output_file="module.json",
            )
            self.assertTrue(result.ok, result.error)
            self.assertEqual(
                json.loads(sandbox.read("module.json").value)["result"], 42
            )
            self.assertFalse(sandbox.run_cell("!touch bad", inputs={"x": 1}).ok)
            self.assertFalse(sandbox.exists("bad").value)


class PackageCompletionTests(unittest.TestCase):
    """Check raw bookkeeping and opt-in assets through the complete package flow."""

    def test_zero_output_limit_does_not_change_package_decisions(self) -> None:
        """Create, check, verify, and export while every model field is trimmed."""
        config = SandboxConfig.conversation(
            kernel=KernelConfig(default_output_limit_chars=0, max_output_limit_chars=0)
        )
        with open_sandbox("agent", config=config) as sandbox:
            self.assertTrue(sandbox.package_init("smalllib").ok)
            self.assertTrue(sandbox.package_check("smalllib").ok)
            self.assertTrue(sandbox.package_test("smalllib").ok)
            self.assertTrue(sandbox.package_export("smalllib").ok)
            for value in ("False", "1", "'True'"):
                sandbox.write("smalllib/verify.py", f"passed = {value}")
                failed = sandbox.package_test("smalllib")
                self.assertEqual(failed.error.code, "invalid_verification_result")

    def test_inventory_does_not_run_bootstrap_or_save_a_session(self) -> None:
        """Use raw filesystem scans rather than temporary persistent executions."""
        files = OSAccess()
        with PersistentMontyKernel(os_handler=files) as kernel:
            self.assertTrue(MontyPackageManager(kernel).init("smalllib")["ok"])
        calls = []
        with PersistentMontyKernel(
            os_handler=files,
            external_lookup={"startup": lambda: calls.append(True)},
            bootstrap_code="startup()",
        ) as kernel:
            result = MontyPackageManager(kernel).check("/workspace/smalllib")
            self.assertTrue(result["ok"], result.get("error"))
            self.assertEqual(calls, [])
            self.assertEqual(kernel.list_sessions(), [])

    def test_binary_assets_keep_revision_limits_and_deterministic_archives(
        self,
    ) -> None:
        """Hash, verify, and export raw assets without adding a package loader."""
        config = SandboxConfig.conversation(allow_binary_assets=True)
        with open_sandbox("agent", config=config) as sandbox:
            self.assertTrue(sandbox.package_init("smalllib").ok)
            content = b"\x89PNG\r\n\x00\xff"
            sandbox.write("smalllib/assets/image.png", content)
            sandbox.write(
                "smalllib/verify.py",
                "from pathlib import Path\n"
                "assert Path(__file__).parent.joinpath('assets/image.png').read_bytes() == b'\\x89PNG\\r\\n\\x00\\xff'\n"
                "passed = True",
            )
            checked = sandbox.package_check("smalllib")
            self.assertTrue(checked.ok, checked.error)
            self.assertTrue(sandbox.package_test("smalllib").ok)
            first = sandbox.package_export("smalllib")
            self.assertTrue(first.ok, first.error)
            first_bytes = sandbox.read(first.value["archive_path"], encoding=None).value
            second = sandbox.package_export("smalllib")
            self.assertEqual(
                first_bytes,
                sandbox.read(second.value["archive_path"], encoding=None).value,
            )
            with zipfile.ZipFile(io.BytesIO(first_bytes)) as archive:
                self.assertEqual(
                    archive.read("smalllib-0.1.0/assets/image.png"), content
                )
            sandbox.write("smalllib/assets/image.png", content + b"new")
            changed = sandbox.package_check("smalllib")
            self.assertNotEqual(checked.value["revision"], changed.value["revision"])
            self.assertEqual(
                sandbox.package_export(
                    "smalllib", checked.value["revision"]
                ).error.code,
                "package_revision_conflict",
            )

    def test_asset_opt_in_does_not_relax_python_or_root_rules(self) -> None:
        """Keep assets disabled by default and keep all .py source UTF-8."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            manager = MontyPackageManager(kernel)
            self.assertTrue(manager.init("smalllib")["ok"])
            kernel.upload_file("/workspace/smalllib/assets/image.png", b"\x89PNG")
            self.assertEqual(
                manager.check("/workspace/smalllib")["error"]["code"],
                "unsupported_package_file",
            )
            manager = MontyPackageManager(
                kernel, PackageConfig(allow_binary_assets=True)
            )
            self.assertTrue(manager.check("/workspace/smalllib")["ok"])
            kernel.upload_file("/workspace/smalllib/assets/bad.py", b"\xff")
            self.assertEqual(
                manager.check("/workspace/smalllib")["error"]["code"],
                "non_utf8_package_file",
            )
        with open_sandbox(
            "agent", config=SandboxConfig.conversation(allow_binary_assets=True)
        ) as sandbox:
            sandbox.package_init("smalllib")
            sandbox.write("smalllib/image.png", b"\x89PNG")
            self.assertEqual(
                sandbox.package_check("smalllib").error.code, "unsupported_package_file"
            )

    def test_assets_count_toward_existing_file_and_byte_limits(self) -> None:
        """Apply package limits to binary artifacts without a second quota system."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            manager = MontyPackageManager(
                kernel, PackageConfig(allow_binary_assets=True)
            )
            manager.init("smalllib")
            kernel.upload_file("/workspace/smalllib/assets/data.bin", b"x" * 1000)
            for config, code in (
                (
                    PackageConfig(allow_binary_assets=True, max_files=5),
                    "too_many_package_files",
                ),
                (
                    PackageConfig(allow_binary_assets=True, max_file_bytes=999),
                    "package_file_too_large",
                ),
                (
                    PackageConfig(allow_binary_assets=True, max_total_bytes=1000),
                    "package_source_too_large",
                ),
            ):
                with self.subTest(code=code):
                    result = MontyPackageManager(kernel, config).check(
                        "/workspace/smalllib"
                    )
                    self.assertEqual(result["error"]["code"], code)

            exported = MontyPackageManager(
                kernel, PackageConfig(allow_binary_assets=True, max_archive_bytes=100)
            ).export("/workspace/smalllib")
            self.assertEqual(exported["error"]["code"], "package_archive_too_large")


class SiblingImportTests(unittest.TestCase):
    """Check explicit relative module bindings through the existing static graph."""

    def test_siblings_aliases_packages_and_parent_levels(self) -> None:
        """Resolve each sibling as a module object with the correct local binding."""
        files = OSAccess(
            files=[
                MemoryFile("/app/helper.py", "def double(x):\n    return x * 2"),
                MemoryFile("/app/math.py", "VALUE = 21"),
                MemoryFile(
                    "/app/pkg/__init__.py",
                    "from .. import helper as h, math as local_math\nANSWER = h.double(local_math.VALUE)",
                ),
                MemoryFile("/app/main.py", "from . import pkg as p\nANSWER = p.ANSWER"),
            ]
        )
        with PersistentMontyKernel(os_handler=files) as kernel:
            result = kernel.execute_module("agent", "/app/main.py", "ANSWER")
            self.assertEqual(result["result"], 42)
            self.assertEqual(kernel.execute("agent", "p.ANSWER")["result"], 42)
            for config in (
                VirtualModuleConfig(roots=("/app/pkg",)),
                VirtualModuleConfig(max_modules=1),
            ):
                limited = kernel.execute_module(
                    "limited", "/app/pkg/__init__.py", module_config=config
                )
                self.assertFalse(limited["ok"])

    def test_nested_missing_wildcard_and_circular_sibling_imports_still_fail(
        self,
    ) -> None:
        """Keep the source loader static and bounded after the new import form."""
        files = OSAccess(files=[MemoryFile("/helper.py", "from . import main")])
        with PersistentMontyKernel(os_handler=files) as kernel:
            for source in (
                "def f():\n    from . import helper",
                "from . import missing",
                "from . import *",
                "from . import helper",
            ):
                with self.subTest(source=source):
                    kernel.upload_file("/main.py", source.encode())
                    result = kernel.execute_module("agent", "/main.py")
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["error"]["type"], "VirtualModuleError")


class AsyncCompletionTests(unittest.IsolatedAsyncioTestCase):
    """Check native async utilities, overflow files, cells, and the threaded facade."""

    async def test_upload_download_controls_and_listing_do_not_create_state(
        self,
    ) -> None:
        """Transfer bytes and inspect state without bootstrap or output trimming."""
        async with AsyncPersistentMontyKernel(os_handler=OSAccess()) as kernel:
            self.assertEqual(
                await kernel.upload_file("/files/data.bin", b"\x00\xff"), 2
            )
            self.assertEqual(await kernel.download_file("/files/data.bin"), b"\x00\xff")
            self.assertEqual(await kernel.list_sessions(), [])
            self.assertEqual((await kernel.execute_cell("new", "%who"))["result"], [])
            self.assertIsNone(kernel.store.load("new"))
            session = kernel.session("agent")
            await session.execute("value = 42")
            before = kernel.store.load("agent")
            self.assertEqual(
                (await session.execute_cell("%whos"))["result"], {"value": "int"}
            )
            self.assertEqual(
                (await session.execute_cell("%sessions"))["result"]["current_session"][
                    "session_id"
                ],
                "agent",
            )
            self.assertEqual(
                (await session.execute_cell("%limits"))["result"]["runtime"]["mode"],
                "native-async",
            )
            self.assertEqual(
                [info.session_id for info in await kernel.list_sessions()], ["agent"]
            )
            self.assertEqual(kernel.store.load("agent").state, before.state)
            self.assertEqual(kernel.store.load("agent").revision, before.revision)
            self.assertTrue((await session.execute_cell("%reset"))["result"]["reset"])
            self.assertEqual(await kernel.list_sessions(), [])

    async def test_file_inputs_bind_once_and_native_output_uses_one_worker(
        self,
    ) -> None:
        """Finish the execution checkout before the raw overflow writer starts."""
        config = KernelConfig(
            max_worker_processes=1,
            type_check=True,
            type_check_stubs="values: list[int]\n",
        )
        async with AsyncPersistentMontyKernel(
            os_handler=OSAccess(), config=config, bootstrap_code="values = [100]"
        ) as kernel:
            await kernel.upload_file("/helper.py", b"values.append(22)")
            result = await asyncio.wait_for(
                kernel.session("agent").execute_file(
                    "/helper.py",
                    "{'values': values, 'blob': b'OK', 'number': 1+2j}",
                    0,
                    inputs={"values": [20]},
                    output_file="/full.json",
                ),
                timeout=2,
            )
            self.assertTrue(result["ok"], result["error"])
            payload = json.loads(await kernel.download_file("/full.json"))
            self.assertEqual(payload["result"]["values"], [20, 22])
            self.assertEqual(
                payload["result"]["blob"], {"encoding": "base64", "content": "T0s="}
            )
            self.assertEqual(
                payload["result"]["number"],
                {"type": "complex", "real": 1.0, "imag": 2.0},
            )
            self.assertEqual(
                result["stats"]["original_output_chars"]["result"],
                len(json.dumps(payload["result"], ensure_ascii=False)),
            )
            self.assertEqual(
                (await kernel.execute("agent", "sum(values)"))["result"], 42
            )

    async def test_overflow_failure_does_not_undo_python_state(self) -> None:
        """Report bad paths, disabled files, and transfer caps separately."""
        async with AsyncPersistentMontyKernel(os_handler=OSAccess()) as kernel:
            for path, filesystem in (
                ("relative.json", True),
                ("/disabled.json", False),
            ):
                result = await kernel.execute(
                    "agent",
                    "value = 42\nvalue",
                    0,
                    output_file=path,
                    filesystem=filesystem,
                )
                self.assertTrue(result["ok"])
                self.assertIsNone(result["output_file"])
                self.assertIsNotNone(result["output_file_error"])
                self.assertEqual((await kernel.execute("agent", "value"))["result"], 42)
            complete = await kernel.execute(
                "agent", "42", output_file="/not-needed.json"
            )
            self.assertIsNone(complete["output_file"])
            with self.assertRaises(MontyError):
                await kernel.download_file("/not-needed.json")
        async with AsyncPersistentMontyKernel(
            os_handler=OSAccess(),
            filesystem_config=FilesystemConfig(max_transfer_bytes=10),
        ) as kernel:
            result = await kernel.execute(
                "agent", "'x' * 100", 0, output_file="/large.json"
            )
            self.assertIn("transfer limit", result["output_file_error"])
            with self.assertRaises(ValueError):
                await kernel.upload_file("/large.bin", b"x" * 11)
            with self.assertRaises(ValueError):
                await kernel.upload_file("relative.bin", b"x")
            with self.assertRaises(MontyError):
                await kernel.upload_file(
                    "/missing/data.bin", b"x", create_parents=False
                )

    async def test_run_cells_module_imports_and_control_errors_are_consistent(
        self,
    ) -> None:
        """Run source and sibling modules while keeping native async shell unsupported."""
        async with AsyncPersistentMontyKernel(os_handler=OSAccess()) as kernel:
            await kernel.upload_file("/helper.py", b"values.append(22)\nsum(values)")
            run = await kernel.execute_cell(
                "agent", "%run /helper.py", inputs={"values": [20]}
            )
            self.assertEqual(run["result"], 42)
            await kernel.upload_file(
                "/main.py", b"from . import helper as h\nANSWER = 42"
            )
            result = await kernel.session("module").execute_module(
                "/main.py",
                "ANSWER + extra",
                0,
                inputs={"values": [20], "extra": 0},
                output_file="/module.json",
            )
            self.assertTrue(result["ok"], result["error"])
            self.assertEqual(
                json.loads(await kernel.download_file("/module.json"))["result"], 42
            )
            for cell in ("%run", "%who extra", "%unknown", "%who\n%whos"):
                self.assertEqual(
                    (await kernel.execute_cell("agent", cell))["error"]["code"],
                    "invalid_repl_command",
                )
            before = kernel.store.load("agent")
            self.assertEqual(
                (await kernel.execute_cell("agent", "%reset", inputs={"x": 1}))[
                    "error"
                ]["code"],
                "invalid_cell_inputs",
            )
            self.assertEqual(
                (await kernel.execute_cell("agent", "!pwd"))["error"]["code"],
                "unsupported_shell_cell",
            )
            self.assertEqual(kernel.store.load("agent").state, before.state)
            self.assertEqual(
                (await kernel.execute_cell("agent", "%mcp"))["error"]["code"],
                "mcp_unavailable",
            )
        with McpHub() as hub:
            async with AsyncPersistentMontyKernel(mcp_hub=hub) as kernel:
                self.assertTrue((await kernel.execute_cell("agent", "%mcp"))["ok"])

    async def test_session_lock_covers_overflow_until_completion(self) -> None:
        """Keep a following same-session feed behind the pending output writer."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_upload(path: str, content: bytes) -> int:
            """Pause a mocked output writer without using a Monty worker.

            Args:
                path: Requested overflow path.
                content: Full JSON output bytes.

            Returns:
                Byte count after the test releases the writer.
            """
            started.set()
            await release.wait()
            return len(content)

        async with AsyncPersistentMontyKernel(os_handler=OSAccess()) as kernel:
            with patch.object(kernel, "upload_file", side_effect=delayed_upload):
                first = asyncio.create_task(
                    kernel.execute(
                        "agent", "value = 1\nvalue", 0, output_file="/full.json"
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=2)
                second = asyncio.create_task(
                    kernel.execute("agent", "value += 1\nvalue")
                )
                other = await kernel.execute("other", "42", filesystem=False)
                self.assertTrue(other["ok"])
                self.assertFalse(second.done())
                release.set()
                result, following = await asyncio.wait_for(
                    asyncio.gather(first, second), timeout=2
                )
                self.assertEqual(result["output_file"], "/full.json")
                self.assertEqual(following["result"], 2)

    async def test_threaded_facade_and_managed_async_forward_inputs(self) -> None:
        """Keep the complete threaded facade distinct from native AsyncMonty."""
        with PersistentMontyKernel(
            os_handler=OSAccess(files=[MemoryFile("/helper.py", "sum(values)")])
        ) as kernel:
            facade = AsyncAreneroSandbox(kernel, session_id="agent")
            self.assertEqual(
                (
                    await facade.execute_python("sum(values)", inputs={"values": [42]})
                ).output,
                "42",
            )

            self.assertEqual(
                (
                    await facade.execute_file("/helper.py", inputs={"values": [20, 22]})
                ).output,
                "42",
            )
            self.assertEqual(
                (await facade.execute_cell("%run /helper.py", inputs={"values": [42]}))[
                    "result"
                ],
                42,
            )
            self.assertEqual(
                (
                    await facade.execute_module(
                        "/helper.py", "sum(values)", inputs={"values": [20, 22]}
                    )
                ).output,
                "42",
            )

    async def test_managed_async_file_and_module_inputs(self) -> None:
        """Forward the same mappings through the managed thread wrappers."""
        async with open_async_sandbox("agent") as sandbox:
            await sandbox.call("write", "helper.py", "sum(values)")
            result = await sandbox.call(
                "run_file", "helper.py", inputs={"values": [42]}
            )
            self.assertEqual(result.value, 42)
            result = await sandbox.call(
                "run_module", "helper.py", "sum(values)", inputs={"values": [20, 22]}
            )
            self.assertEqual(result.value, 42)
            self.assertEqual(
                (
                    await sandbox.run_cell("%run helper.py", inputs={"values": [42]})
                ).value,
                42,
            )

    async def test_mocked_native_async_agent_uses_controls_and_module_overflow_across_turns(
        self,
    ) -> None:
        """Keep the agent tool names stable while new controls and output options work."""
        model = ToolCallingFakeChatModel(
            messages=iter(
                [
                    python_tool_call(
                        "run",
                        "%run /helper.py",
                        output_limit_chars=0,
                        output_file="/first.json",
                    ),
                    AIMessage(content="The answer is 42."),
                    python_tool_call("who", "%whos"),
                    named_tool_call(
                        "module",
                        "python_repl_execute_module",
                        {
                            "entry_path": "/main.py",
                            "code": "sum(values) + extra",
                            "output_limit_chars": 0,
                            "output_file": "/second.json",
                        },
                    ),
                    AIMessage(content="I reused the saved values."),
                ]
            )
        )
        async with AsyncPersistentMontyKernel(os_handler=OSAccess()) as kernel:
            await kernel.upload_file("/helper.py", b"values.append(22)\nsum(values)")
            await kernel.upload_file("/main.py", b"extra = 0")
            await kernel.execute("agent", "values", inputs={"values": [20]})
            agent = create_agent(
                model,
                create_async_python_repl_tools(kernel),
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "agent"}}
            await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="Calculate from the saved values.")
                    ]
                },
                config,
            )
            second = await agent.ainvoke(
                {"messages": [HumanMessage(content="Inspect and reuse the values.")]},
                config,
            )
            results = tool_results(second)
            self.assertTrue(all(result["ok"] for result in results), results)
            self.assertEqual(results[1]["result"], {"values": "list"})
            self.assertFalse(results[1]["state_saved"])
            self.assertEqual(
                json.loads(await kernel.download_file("/first.json"))["result"], 42
            )
            self.assertEqual(
                json.loads(await kernel.download_file("/second.json"))["result"], 42
            )
