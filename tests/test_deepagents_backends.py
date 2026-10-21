"""Backend profiles, capture, and permission tests without real model calls."""

import base64
import unittest
from typing import Any
from unittest.mock import patch

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol
from deepagents.backends.sandbox import BaseSandbox
from deepagents.middleware.filesystem import GREP_TOOL_DESCRIPTION, FilesystemMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic_monty import OSAccess

from arenero import (
    AreneroSandbox,
    FilesystemConfig,
    KernelConfig,
    PersistentMontyKernel,
)
from arenero.deepagents import CodeInterpreterMiddleware
from arenero.deepagents_backend import (
    AreneroBackend,
    AreneroBackendMiddleware,
    AreneroFilesBackend,
    _backend_request,
)
from tests.test_deepagents_compat import _model
from tests.test_interpreter_fixes import FilterTools
from tests.test_langchain_agent import named_tool_call


class ObserveRequests(AgentMiddleware):
    """Record model requests after the backend and interpreter middleware."""

    def __init__(self) -> None:
        """Start with no recorded model requests."""
        self.requests = []

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Record a synchronous request.

        Args:
            request: Filtered model request.
            handler: Scripted model or next middleware.

        Returns:
            Scripted model response.
        """
        self.requests.append(request)
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Record an asynchronous request.

        Args:
            request: Filtered model request.
            handler: Scripted model or next middleware.

        Returns:
            Scripted model response.
        """
        self.requests.append(request)
        return await handler(request)


def _messages(state: dict) -> list[ToolMessage]:
    """Select model-visible tool messages from graph output.

    Args:
        state: Completed or interrupted graph output.

    Returns:
        Tool messages in execution order.
    """
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


class BackendProfileTests(unittest.IsolatedAsyncioTestCase):
    """Test complete model-driven graphs against real local Monty workers."""

    async def test_file_only_profile_has_all_files_but_no_execution(self) -> None:
        """Reuse native records and async forwarding without the sandbox marker."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            self.assertIsInstance(backend, BackendProtocol)
            self.assertNotIsInstance(backend, SandboxBackendProtocol)
            self.assertFalse(hasattr(backend, "execute"))
            self.assertIsNone((await backend.awrite("/a.txt", "one\ntwo")).error)
            self.assertEqual(
                (await backend.aread("/a.txt", 1, 1)).file_data["content"], "two"
            )
            self.assertIsNone((await backend.aedit("/a.txt", "two", "TWO")).error)
            self.assertEqual(len((await backend.als("/")).entries), 1)
            self.assertEqual(len((await backend.aglob("*.txt")).matches), 1)
            self.assertEqual((await backend.agrep("TWO")).matches[0]["line"], 2)
            binary = b"\x00\xff\x80"
            self.assertIsNone(
                (await backend.aupload_files([("/data.bin", binary)]))[0].error
            )
            self.assertEqual(
                (await backend.adownload_files(["/data.bin"]))[0].content, binary
            )
            self.assertIsNone((await backend.adelete("/a.txt")).error)

    async def test_backend_only_descriptions_and_virtual_execution(self) -> None:
        """Give the fake model accurate stock descriptions and run supported code."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            observer = ObserveRequests()
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "run", "execute", {"command": "python -c 'print(6*7)'"}
                    ),
                    AIMessage(content="42"),
                ),
                backend=backend,
                middleware=[AreneroBackendMiddleware(backend), observer],
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Calculate"}]}
            )
            self.assertIn("42", _messages(result)[0].content)
            tools = {t.name: t for t in observer.requests[0].tools}
            self.assertIn("not Bash or CPython", tools["execute"].description)
            self.assertIn("grep -E", tools["grep"].description)
            self.assertNotIn("rg '<regex>'", tools["grep"].description)
            self.assertIn(
                "Arenero virtual shell",
                str(observer.requests[0].system_message.content),
            )

    async def test_combined_profile_rewrites_descriptions_without_extra_middleware(
        self,
    ) -> None:
        """Reuse the same guidance automatically with the shared interpreter."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            observer = ObserveRequests()
            async with CodeInterpreterMiddleware(backend=backend) as interpreter:
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call("run", "eval", {"code": "42"}),
                        AIMessage(content="done"),
                    ),
                    backend=backend,
                    middleware=[interpreter, observer],
                )
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "Calculate"}]}
                )
                self.assertEqual(_messages(result)[0].artifact["result"], 42)
                self.assertIn(
                    "not Bash or CPython",
                    next(
                        t.description
                        for t in observer.requests[0].tools
                        if t.name == "execute"
                    ),
                )

    async def test_hidden_execute_is_not_restored_or_suggested_by_grep(self) -> None:
        """Honor earlier request filters instead of widening tool authority."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            observer = ObserveRequests()
            agent = create_deep_agent(
                model=_model(AIMessage(content="done")),
                backend=backend,
                middleware=[
                    FilterTools("execute"),
                    AreneroBackendMiddleware(backend),
                    observer,
                ],
            )
            await agent.ainvoke({"messages": [{"role": "user", "content": "work"}]})
            tools = {t.name: t for t in observer.requests[0].tools}
            self.assertNotIn("execute", tools)
            self.assertNotIn("execute", tools["grep"].description)

    async def test_custom_descriptions_and_tool_schemas_are_unchanged(self) -> None:
        """Copy only standard descriptions, including dictionary-form tools."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            tools = FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions={
                    "execute": "Executes a shell command in an isolated sandbox: custom contract"
                },
            ).tools
            request = ModelRequest(
                model=_model(AIMessage(content="done")), messages=[], tools=tools
            )
            changed = _backend_request(request, backend)
            before, after = (
                {t.name: t for t in group} for group in (request.tools, changed.tools)
            )
            self.assertIs(before["execute"], after["execute"])
            for name in before:
                self.assertIs(before[name].args_schema, after[name].args_schema)
                self.assertIs(before[name].func, after[name].func)
                self.assertIs(before[name].coroutine, after[name].coroutine)
            raw = {
                "name": "grep",
                "description": GREP_TOOL_DESCRIPTION,
                "parameters": {"type": "object"},
            }
            dict_request = request.override(tools=[raw])
            rewritten = _backend_request(dict_request, backend)
            self.assertEqual(raw["description"], dict_request.tools[0]["description"])
            self.assertNotIn("execute", rewritten.tools[0]["description"])
            self.assertIs(raw["parameters"], rewritten.tools[0]["parameters"])

    async def test_file_only_deny_rules_filter_all_read_tools(self) -> None:
        """Protect data in exact reads and bulk listing, glob, and grep results."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            backend.write("/secret/a.txt", "secret marker")
            backend.write("/public/a.txt", "public marker")
            observer = ObserveRequests()
            calls = [
                ("ls", {"path": "/"}),
                ("glob", {"pattern": "**/*.txt", "path": "/"}),
                ("grep", {"pattern": "marker", "path": "/"}),
                ("read_file", {"file_path": "/secret/a.txt"}),
            ]
            agent = create_deep_agent(
                model=_model(
                    *(
                        named_tool_call(str(i), name, args)
                        for i, (name, args) in enumerate(calls)
                    ),
                    AIMessage(content="done"),
                ),
                backend=backend,
                permissions=[
                    FilesystemPermission(
                        operations=["read"],
                        paths=["/secret", "/secret/**"],
                        mode="deny",
                    )
                ],
                middleware=[AreneroBackendMiddleware(backend), observer],
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "search"}]}
            )
            self.assertNotIn("execute", [t.name for t in observer.requests[0].tools])
            self.assertNotIn(
                "execute",
                next(
                    t.description
                    for t in observer.requests[0].tools
                    if t.name == "grep"
                ),
            )
            for message in _messages(result)[:3]:
                self.assertNotIn("secret", str(message.content))
            self.assertEqual(_messages(result)[-1].status, "error")
            self.assertIn("denied", _messages(result)[-1].content.lower())

    async def test_permission_order_and_directory_delete_are_enforced(self) -> None:
        """Use upstream first-match rules and avoid a partly deleted directory."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            backend.write("/tree/keep.txt", "keep")
            backend.write("/tree/other.txt", "other")
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "allow",
                        "write_file",
                        {"file_path": "/allowed.txt", "content": "yes"},
                    ),
                    named_tool_call(
                        "deny",
                        "write_file",
                        {"file_path": "/denied.txt", "content": "no"},
                    ),
                    named_tool_call(
                        "edit",
                        "edit_file",
                        {
                            "file_path": "/tree/keep.txt",
                            "old_string": "keep",
                            "new_string": "lost",
                        },
                    ),
                    named_tool_call("delete", "delete", {"file_path": "/tree"}),
                    AIMessage(content="done"),
                ),
                backend=backend,
                permissions=[
                    FilesystemPermission(
                        operations=["write"], paths=["/allowed.txt"], mode="allow"
                    ),
                    FilesystemPermission(
                        operations=["write"], paths=["/tree/keep.txt"], mode="deny"
                    ),
                    FilesystemPermission(
                        operations=["write"], paths=["/tree", "/tree/**"], mode="allow"
                    ),
                    FilesystemPermission(
                        operations=["write"], paths=["/**"], mode="deny"
                    ),
                ],
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "work"}]}
            )
            self.assertEqual(
                [m.status for m in _messages(result)],
                ["success", "error", "error", "error"],
            )
            self.assertEqual(
                backend.download_files(["/allowed.txt"])[0].content, b"yes"
            )
            self.assertTrue(
                all(
                    r.error is None
                    for r in backend.download_files(
                        ["/tree/keep.txt", "/tree/other.txt"]
                    )
                )
            )

    async def test_file_permission_interrupt_approve_and_reject(self) -> None:
        """Pause before file changes and resume with the reviewer decision."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            for decision in ("approve", "reject"):
                path = f"/{decision}.txt"
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            decision,
                            "write_file",
                            {"file_path": path, "content": "yes"},
                        ),
                        AIMessage(content="done"),
                    ),
                    backend=backend,
                    permissions=[
                        FilesystemPermission(
                            operations=["write"], paths=["/**"], mode="interrupt"
                        )
                    ],
                    checkpointer=InMemorySaver(),
                )
                config = {"configurable": {"thread_id": decision}}
                paused = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "write"}]}, config
                )
                self.assertTrue(paused.get("__interrupt__"))
                self.assertEqual(
                    backend.download_files([path])[0].error, "file_not_found"
                )
                resumed = await agent.ainvoke(
                    Command(resume={"decisions": [{"type": decision}]}), config
                )
                self.assertFalse(resumed.get("__interrupt__"))
                self.assertEqual(
                    backend.download_files([path])[0].content,
                    b"yes" if decision == "approve" else None,
                )

    async def test_interpreter_file_tools_keep_deny_rules_across_two_turns(
        self,
    ) -> None:
        """Keep Python state while checking permission on every nested file call."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            backend.write("/public.txt", "41")
            async with CodeInterpreterMiddleware(
                ptc=("read_file", "write_file")
            ) as interpreter:
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            "first",
                            "eval",
                            {
                                "code": "value=41\nawait tools.read_file(file_path='/public.txt')"
                            },
                        ),
                        AIMessage(content="stored"),
                        named_tool_call(
                            "second",
                            "eval",
                            {
                                "code": "value += 1\nblocked=await tools.write_file(file_path='/blocked.txt', content=str(value))\n[value, blocked]"
                            },
                        ),
                        AIMessage(content="done"),
                    ),
                    backend=backend,
                    middleware=[interpreter],
                    permissions=[
                        FilesystemPermission(
                            operations=["write"], paths=["/**"], mode="deny"
                        )
                    ],
                    checkpointer=InMemorySaver(),
                )
                config = {"configurable": {"thread_id": "permissions-two-turns"}}
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "remember"}]}, config
                )
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "continue"}]}, config
                )
                value, blocked = _messages(result)[-1].artifact["result"]
                self.assertEqual(value, 42)
                self.assertIn("denied", str(blocked).lower())
                self.assertEqual(
                    backend.download_files(["/blocked.txt"])[0].error, "file_not_found"
                )

    async def test_ptc_approval_boundary_and_outer_eval_approval(self) -> None:
        """Record that nested calls bypass parent HITL, but outer eval can pause."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            for gate in ("write_file", "eval"):
                async with CodeInterpreterMiddleware(
                    ptc=("write_file",)
                ) as interpreter:
                    path = f"/{gate}.txt"
                    agent = create_deep_agent(
                        model=_model(
                            named_tool_call(
                                gate,
                                "eval",
                                {
                                    "code": f"await tools.write_file(file_path={path!r}, content='yes')"
                                },
                            ),
                            AIMessage(content="done"),
                        ),
                        backend=backend,
                        middleware=[interpreter],
                        interrupt_on={gate: True},
                        checkpointer=InMemorySaver(),
                    )
                    config = {"configurable": {"thread_id": gate}}
                    result = await agent.ainvoke(
                        {"messages": [{"role": "user", "content": "write"}]}, config
                    )
                    self.assertEqual(bool(result.get("__interrupt__")), gate == "eval")
                    if gate == "eval":
                        self.assertIsNone(backend.download_files([path])[0].content)
                        result = await agent.ainvoke(
                            Command(resume={"decisions": [{"type": "approve"}]}), config
                        )
                    self.assertEqual(backend.download_files([path])[0].content, b"yes")

    async def test_direct_file_bridge_is_a_separate_host_grant(self) -> None:
        """Do not claim middleware permissions govern directly granted fs_* calls."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            async with CodeInterpreterMiddleware(backend=backend) as interpreter:
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            "direct",
                            "eval",
                            {
                                "code": "await fs_write(file_path='/direct.txt', content='yes')"
                            },
                        ),
                        AIMessage(content="done"),
                    ),
                    backend=backend,
                    middleware=[interpreter],
                    permissions=[
                        FilesystemPermission(
                            operations=["write"], paths=["/**"], mode="deny"
                        )
                    ],
                )
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "write"}]}
                )
                self.assertEqual(
                    backend.download_files(["/direct.txt"])[0].content, b"yes"
                )

    async def test_real_subagent_inherits_or_replaces_file_permissions(self) -> None:
        """Run actual child graphs with scripted models and shared file storage."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            for override in (False, True):
                path = f"/child-{override}.txt"
                spec = {
                    "name": "writer",
                    "description": "Write files",
                    "system_prompt": "Write",
                    "model": _model(
                        named_tool_call(
                            "write", "write_file", {"file_path": path, "content": "yes"}
                        ),
                        AIMessage(content="done"),
                    ),
                }
                if override:
                    spec["permissions"] = [
                        FilesystemPermission(
                            operations=["write"], paths=["/**"], mode="allow"
                        )
                    ]
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            "task",
                            "task",
                            {"description": "Write", "subagent_type": "writer"},
                        ),
                        AIMessage(content="done"),
                    ),
                    backend=backend,
                    subagents=[spec],
                    permissions=[
                        FilesystemPermission(
                            operations=["write"], paths=["/**"], mode="deny"
                        )
                    ],
                )
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "delegate"}]}
                )
                self.assertEqual(
                    backend.download_files([path])[0].content,
                    b"yes" if override else None,
                )

    async def test_sandbox_permissions_and_composite_routes_keep_upstream_rules(
        self,
    ) -> None:
        """Reject sandbox path rules; allow rules scoped to a separate route."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            rule = FilesystemPermission(
                operations=["write"], paths=["/**"], mode="deny"
            )
            with self.assertRaises(NotImplementedError):
                create_deep_agent(
                    model=_model(AIMessage(content="done")),
                    backend=backend,
                    permissions=[rule],
                )
            composite = CompositeBackend(
                default=backend, routes={"/route/": StateBackend()}
            )
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "deny",
                        "write_file",
                        {"file_path": "/route/secret.txt", "content": "no"},
                    ),
                    named_tool_call(
                        "allow",
                        "write_file",
                        {"file_path": "/route/public.txt", "content": "yes"},
                    ),
                    named_tool_call(
                        "read", "read_file", {"file_path": "/route/public.txt"}
                    ),
                    AIMessage(content="done"),
                ),
                backend=composite,
                permissions=[
                    FilesystemPermission(
                        operations=["write"], paths=["/route/secret.txt"], mode="deny"
                    )
                ],
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "work"}]}
            )
            self.assertEqual(_messages(result)[0].status, "error")
            self.assertIn("yes", _messages(result)[-1].content)
            self.assertEqual(
                backend.download_files(["/route/public.txt"])[0].error, "file_not_found"
            )

    async def test_native_image_read_returns_a_model_content_block(self) -> None:
        """Keep images on the native multimodal tool path without an LLM call."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroFilesBackend(AreneroSandbox(kernel))
            image = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aH1kAAAAASUVORK5CYII="
            )
            backend.upload_files([("/image.png", image)])
            agent = create_deep_agent(
                model=_model(
                    named_tool_call("image", "read_file", {"file_path": "/image.png"}),
                    AIMessage(content="done"),
                ),
                backend=backend,
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "read"}]}
            )
            blocks = _messages(result)[0].content
            self.assertIsInstance(blocks, list)
            self.assertTrue(
                any(b.get("type") in ("image", "image_url") for b in blocks)
            )


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    """Check native capture without OS wrappers or command retries."""

    async def test_capture_is_opt_in_and_keeps_native_async_file_paths(self) -> None:
        """Leave old execute defaults intact and never inherit shell file scripts."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            self.assertIsInstance(backend, BaseSandbox)
            self.assertFalse(backend.enable_capture_offload)
            plain = backend.execute("echo yes")
            captured = await backend.aexecute_with_offload(
                "echo yes", "/unused", max_inline_bytes=0
            )
            self.assertFalse(captured.offloaded)
            self.assertEqual(captured.response, plain)
            self.assertEqual(
                backend.download_files(["/unused"])[0].error, "file_not_found"
            )
            with patch.object(
                backend, "execute", side_effect=AssertionError("file wrapper used")
            ):
                self.assertIsNone((await backend.awrite("/a", "one")).error)
                self.assertEqual(
                    (await backend.aread("/a")).file_data["content"], "one"
                )
                self.assertEqual(len((await backend.als("/")).entries), 1)
                self.assertEqual(len((await backend.aglob("*", "/")).matches), 1)
                self.assertEqual(len((await backend.agrep("one")).matches), 1)
                self.assertIsNone((await backend.aedit("/a", "one", "two")).error)
                self.assertIsNone((await backend.adelete("/a")).error)

    async def test_model_graph_reads_capture_and_preserves_exit_and_single_run(
        self,
    ) -> None:
        """Use Deep Agents' native file pointer and read it in the next tool call."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            config=KernelConfig(
                default_output_limit_chars=100, max_output_limit_chars=20_000
            ),
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            backend.enable_capture_offload = True
            text = "start\n" + "content\n" * 1000 + "end\n"
            backend.write("/source.txt", text)
            path = "/large_tool_results/capture"
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "capture",
                        "execute",
                        {"command": "cat /source.txt; echo once >> /runs.txt; false"},
                    ),
                    named_tool_call(
                        "read",
                        "read_file",
                        {"file_path": path, "offset": 1001, "limit": 1},
                    ),
                    AIMessage(content="done"),
                ),
                backend=backend,
                middleware=[AreneroBackendMiddleware(backend)],
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "run"}]}
            )
            messages = _messages(result)
            self.assertIn(path, messages[0].content)
            self.assertIn("exit code 1", messages[0].content)
            self.assertNotIn("saved file is incomplete", messages[0].content)
            self.assertIn("end", messages[1].content)
            self.assertEqual(backend.download_files([path])[0].content, text.encode())
            self.assertEqual(
                backend.download_files(["/runs.txt"])[0].content, b"once\n"
            )

    async def test_capture_caps_and_utf8_previews_are_honest(self) -> None:
        """Bound bytes, preserve UTF-8, and report hard truncation, not previewing."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            config=KernelConfig(
                default_output_limit_chars=100, max_output_limit_chars=200
            ),
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            backend.enable_capture_offload = True
            for inline, cap in ((0, 20), (5, 20), (20, 7), (1, 0)):
                result = await backend.aexecute_with_offload(
                    "echo '😀😀😀😀😀😀'",
                    f"/capture-{inline}-{cap}",
                    max_inline_bytes=inline,
                    max_capture_bytes=cap,
                )
                self.assertLessEqual(len(result.response.output.encode()), inline)
                self.assertTrue(result.response.truncated)
                if result.offloaded:
                    data = backend.download_files([f"/capture-{inline}-{cap}"])[
                        0
                    ].content
                    self.assertLessEqual(len(data), cap)
                    data.decode("utf-8")
            hard = backend.execute_with_offload(
                "echo '" + "x" * 500 + "'", "/hard", max_inline_bytes=50
            )
            self.assertTrue(hard.response.truncated)
            self.assertLessEqual(len(backend.download_files(["/hard"])[0].content), 200)

    async def test_graph_marks_a_hard_capped_capture_incomplete(self) -> None:
        """Tell the fake model when the saved file cannot contain all output."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            config=KernelConfig(
                default_output_limit_chars=20, max_output_limit_chars=200
            ),
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            backend.enable_capture_offload = True
            backend.write("/source", "x" * 500)
            agent = create_deep_agent(
                model=_model(
                    named_tool_call(
                        "incomplete", "execute", {"command": "cat /source"}
                    ),
                    AIMessage(content="done"),
                ),
                backend=backend,
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "run"}]}
            )
            self.assertIn("saved file is incomplete", _messages(result)[0].content)
            self.assertLessEqual(
                len(
                    backend.download_files(["/large_tool_results/incomplete"])[
                        0
                    ].content
                ),
                200,
            )

    async def test_capture_validates_before_execution_and_does_not_retry_failed_writes(
        self,
    ) -> None:
        """Reject bad paths/budgets early and keep completed effects on write failure."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel), root="/safe")
            backend.enable_capture_offload = True
            for path, inline, cap in (
                ("/outside", 1, None),
                ("/safe/out", -1, None),
                ("/safe/out", 1, -1),
            ):
                with self.assertRaises(ValueError):
                    backend.execute_with_offload(
                        "echo once >> /safe/runs",
                        path,
                        max_inline_bytes=inline,
                        max_capture_bytes=cap,
                    )
            self.assertIsNone(backend.download_files(["/safe/runs"])[0].content)
            with patch.object(
                backend, "_write_output", side_effect=ValueError("read-only capture")
            ):
                result = backend.execute_with_offload(
                    "echo once >> /safe/runs; echo '" + "x" * 1000 + "'",
                    "/safe/out",
                    max_inline_bytes=500,
                )
            self.assertFalse(result.offloaded)
            self.assertTrue(result.response.truncated)
            self.assertIn("capture failed", result.response.output)
            self.assertEqual(
                backend.download_files(["/safe/runs"])[0].content, b"once\n"
            )

    async def test_transfer_limit_bounds_capture_without_changing_kernel_config(
        self,
    ) -> None:
        """Reuse transfer quotas and cooperative timeouts rather than global mutation."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            filesystem_config=FilesystemConfig(max_transfer_bytes=64),
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            backend.enable_capture_offload = True
            config = kernel.config
            result = backend.execute_with_offload(
                "echo '" + "x" * 100 + "'", "/cap", max_inline_bytes=10
            )
            self.assertTrue(result.offloaded)
            self.assertTrue(result.response.truncated)
            self.assertEqual(len(backend.download_files(["/cap"])[0].content), 64)
            timeout = await backend.aexecute_with_offload(
                "echo should-not-run > /never",
                "/timeout",
                max_inline_bytes=500,
                timeout=-1,
            )
            self.assertEqual(timeout.response.exit_code, 2)
            self.assertIsNone(backend.download_files(["/never"])[0].content)
            self.assertIs(config, kernel.config)

    async def test_composite_capture_uses_the_configured_artifact_root(self) -> None:
        """Keep capture in the executing root and skip paths routed elsewhere."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            config=KernelConfig(
                default_output_limit_chars=40, max_output_limit_chars=2000
            ),
        ) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel), root="/workspace")
            backend.enable_capture_offload = True
            backend.write("/workspace/source", "x" * 1000)
            for reroute in (False, True):
                composite = CompositeBackend(
                    default=backend,
                    routes={"/workspace/large_tool_results/": StateBackend()}
                    if reroute
                    else {},
                    artifacts_root="/workspace",
                )
                call_id = f"route-{reroute}"
                path = f"/workspace/large_tool_results/{call_id}"
                agent = create_deep_agent(
                    model=_model(
                        named_tool_call(
                            call_id, "execute", {"command": "cat /workspace/source"}
                        ),
                        AIMessage(content="done"),
                    ),
                    backend=composite,
                )
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "run"}]}
                )
                self.assertEqual(path in _messages(result)[0].content, not reroute)
                self.assertEqual(
                    backend.download_files([path])[0].content,
                    None if reroute else b"x" * 1000,
                )

    async def test_cooperative_capture_timeout_does_not_run_later_commands(
        self,
    ) -> None:
        """Keep completed effects and the normal timeout exit through capture."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            backend.enable_capture_offload = True
            from arenero.shell import VirtualShell

            original = VirtualShell._echo

            def expired(shell: Any, args: Any, stdin: Any, cwd: Any) -> Any:
                """Write the first command, then trigger a cooperative timeout.

                Args:
                    shell: Active virtual shell.
                    args: Normal echo arguments.
                    stdin: Pipeline input.
                    cwd: Virtual current directory.

                Returns:
                    Echo result; the patched deadline check then raises.
                """
                result = original(shell, args, stdin, cwd)
                checks.side_effect = TimeoutError("shell deadline exceeded")
                return result

            with (
                patch("arenero.shell.remaining") as checks,
                patch.object(VirtualShell, "_echo", expired),
            ):
                result = backend.execute_with_offload(
                    "touch /before; echo first; echo later > /later",
                    "/capture",
                    max_inline_bytes=100,
                    timeout=1,
                )
            self.assertEqual(result.response.exit_code, 124)
            self.assertIn("deadline", result.response.output)
            self.assertIsNone(backend.download_files(["/before"])[0].error)
            self.assertEqual(
                backend.download_files(["/later"])[0].error, "file_not_found"
            )


class BackendSyncTests(unittest.TestCase):
    """Keep synchronous graph entry points equivalent to asynchronous ones."""

    def test_sync_graph_has_correct_descriptions_and_native_capture(self) -> None:
        """Use the same backend middleware, shell capture, and file tools in invoke."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            backend = AreneroBackend(AreneroSandbox(kernel))
            backend.enable_capture_offload = True
            observer = ObserveRequests()
            agent = create_deep_agent(
                model=_model(
                    named_tool_call("sync", "execute", {"command": "echo yes"}),
                    AIMessage(content="done"),
                ),
                backend=backend,
                middleware=[AreneroBackendMiddleware(backend), observer],
            )
            result = agent.invoke({"messages": [{"role": "user", "content": "work"}]})
            self.assertIn("yes", _messages(result)[0].content)
            self.assertIn(
                "not Bash or CPython",
                next(
                    t.description
                    for t in observer.requests[0].tools
                    if t.name == "execute"
                ),
            )
