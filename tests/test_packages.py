"""Tests for dependency-free Monty source package workflows."""

import io
import json
import unittest
import zipfile
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic_monty import OSAccess

from arenero import (
    MontyPackageManager,
    PackageConfig,
    PersistentMontyKernel,
    create_package_tools,
    create_python_repl_tools,
)


class PackageToolCallingModel(GenericFakeChatModel):
    """Return a fixed series of package and filesystem tool calls."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Accept agent tool binding without changing the fixed model.

        Args:
            tools: Tools exposed to the agent.
            tool_choice: Optional tool-selection constraint.
            **kwargs: Additional model binding options.

        Returns:
            This fixed fake model.
        """

        del tools, tool_choice, kwargs
        return self


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> AIMessage:
    """Create one fixed model tool call.

    Args:
        call_id: Unique test call identifier.
        name: Tool name.
        args: Tool arguments.

    Returns:
        AI message with one tool call.
    """

    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args, "type": "tool_call"}],
    )


class MontyPackageManagerTests(unittest.TestCase):
    """Check the complete source package contract and its limits."""

    def setUp(self) -> None:
        """Create one in-memory filesystem and synchronous kernel."""

        self.kernel = PersistentMontyKernel(os_handler=OSAccess())
        self.addCleanup(self.kernel.close)
        self.manager = MontyPackageManager(self.kernel)

    def test_scaffold_check_test_and_export_form_one_valid_workflow(self) -> None:
        """Create, validate, test, and export the standard package layout."""

        initialized = self.manager.init("mathkit")
        checked = self.manager.check("/workspace/mathkit", initialized.get("revision"))
        tested = self.manager.test("/workspace/mathkit", checked.get("revision"))
        exported = self.manager.export("/workspace/mathkit", tested.get("revision"))

        self.assertTrue(initialized["ok"])
        self.assertTrue(checked["ok"])
        self.assertTrue(tested["ok"])
        self.assertTrue(exported["ok"])
        self.assertEqual(self.kernel.store.list_sessions(), [])
        archive = self.kernel.download_file(exported["archive_path"])
        with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
            names = source_zip.namelist()
            manifest = json.loads(source_zip.read("mathkit-0.1.0/arenero-package.json"))
        self.assertEqual(
            names,
            [
                "mathkit-0.1.0/README.md",
                "mathkit-0.1.0/arenero-package.json",
                "mathkit-0.1.0/mathkit/__init__.py",
                "mathkit-0.1.0/mathkit/core.py",
                "mathkit-0.1.0/verify.py",
            ],
        )
        self.assertEqual(manifest["dependencies"], [])

    def test_fresh_test_session_observes_source_edits(self) -> None:
        """Reject an edited package even when an old REPL loaded valid code."""

        initialized = self.manager.init("mathkit")
        self.assertTrue(initialized["ok"])
        loaded = self.kernel.execute_module(
            "old-repl",
            "/workspace/mathkit/mathkit/__init__.py",
            code="identity(42)",
        )
        self.assertEqual(loaded["result"], 42)
        self.kernel.upload_file(
            "/workspace/mathkit/mathkit/core.py",
            b"def identity(value):\n    return value + 1\n",
        )

        tested = self.manager.test("/workspace/mathkit")

        self.assertFalse(tested["ok"])
        self.assertEqual(tested["error"]["code"], "package_verification_failed")
        self.assertEqual(
            [item.session_id for item in self.kernel.store.list_sessions()],
            ["old-repl"],
        )

    def test_revision_prevents_test_or_export_after_an_edit(self) -> None:
        """Detect a source change between check, test, and export operations."""

        initialized = self.manager.init("mathkit")
        revision = initialized["revision"]
        self.kernel.upload_file(
            "/workspace/mathkit/README.md", b"# Changed after package_check\n"
        )

        tested = self.manager.test("/workspace/mathkit", revision)
        exported = self.manager.export("/workspace/mathkit", revision)

        self.assertEqual(tested["error"]["code"], "package_revision_conflict")
        self.assertEqual(exported["error"]["code"], "package_revision_conflict")

    def test_manifest_rejects_dependencies_and_unsupported_imports(self) -> None:
        """Reject third-party dependencies and unresolved source imports."""

        self.assertTrue(self.manager.init("mathkit")["ok"])
        manifest_path = "/workspace/mathkit/arenero-package.json"
        manifest = json.loads(self.kernel.download_file(manifest_path))
        manifest["dependencies"] = ["requests"]
        self.kernel.upload_file(manifest_path, json.dumps(manifest).encode())
        dependencies = self.manager.check("/workspace/mathkit")
        manifest["dependencies"] = []
        self.kernel.upload_file(manifest_path, json.dumps(manifest).encode())
        self.kernel.upload_file(
            "/workspace/mathkit/mathkit/core.py",
            b"from missing_helper import value\n",
        )
        module = self.manager.check("/workspace/mathkit")

        self.assertEqual(
            dependencies["error"]["code"], "package_dependencies_not_supported"
        )
        self.assertEqual(module["error"]["code"], "unsupported_package_module")

    def test_existing_package_and_invalid_verification_are_rejected(self) -> None:
        """Do not overwrite a package or accept a false verification flag."""

        initialized = self.manager.init("mathkit")
        duplicate = self.manager.init("mathkit")
        self.kernel.upload_file(
            "/workspace/mathkit/verify.py",
            b"from mathkit import identity\npassed = False\n",
        )
        tested = self.manager.test("/workspace/mathkit")

        self.assertTrue(initialized["ok"])
        self.assertEqual(duplicate["error"]["code"], "package_exists")
        self.assertEqual(tested["error"]["code"], "invalid_verification_result")

    def test_missing_manifest_fields_return_structured_errors(self) -> None:
        """Report a missing entry field without raising an application error."""

        self.assertTrue(self.manager.init("mathkit")["ok"])
        manifest_path = "/workspace/mathkit/arenero-package.json"
        manifest = json.loads(self.kernel.download_file(manifest_path))
        del manifest["entry"]
        self.kernel.upload_file(manifest_path, json.dumps(manifest).encode())

        checked = self.manager.check("/workspace/mathkit")

        self.assertFalse(checked["ok"])
        self.assertEqual(checked["error"]["code"], "invalid_entry")

    def test_package_limits_and_root_are_enforced(self) -> None:
        """Keep package reads and writes inside explicit host limits."""

        outside = self.manager.init("mathkit", root="/tmp/mathkit")
        limited = MontyPackageManager(
            self.kernel,
            PackageConfig(max_files=5, max_archive_bytes=100),
        )
        initialized = limited.init("smallkit")
        self.kernel.upload_file("/workspace/smallkit/extra.txt", b"extra")
        too_many = limited.check("/workspace/smallkit")
        self.kernel.call_filesystem(
            "cleanup", "delete", {"path": "/workspace/smallkit/extra.txt"}
        )
        archive = limited.export("/workspace/smallkit")

        self.assertEqual(outside["error"]["code"], "invalid_package_root")
        self.assertTrue(initialized["ok"])
        self.assertEqual(too_many["error"]["code"], "too_many_package_files")
        self.assertEqual(archive["error"]["code"], "package_archive_too_large")

    def test_export_is_deterministic_and_excludes_dist(self) -> None:
        """Create the same source ZIP for an unchanged package revision."""

        self.assertTrue(self.manager.init("mathkit")["ok"])
        first = self.manager.export("/workspace/mathkit")
        first_bytes = self.kernel.download_file(first["archive_path"])
        second = self.manager.export("/workspace/mathkit", first["revision"])
        second_bytes = self.kernel.download_file(second["archive_path"])

        self.assertEqual(first["revision"], second["revision"])
        self.assertEqual(first_bytes, second_bytes)
        with zipfile.ZipFile(io.BytesIO(second_bytes)) as source_zip:
            self.assertFalse(any("/dist/" in name for name in source_zip.namelist()))


class MontyPackageAgentTests(unittest.TestCase):
    """Check that a model-driven agent can follow the package workflow."""

    def test_mock_agent_builds_and_exports_a_valid_package(self) -> None:
        """Drive all package steps through fixed model tool calls."""

        model = PackageToolCallingModel(
            messages=iter(
                [
                    _tool_call("init", "package_init", {"name": "mathkit"}),
                    _tool_call(
                        "source",
                        "write_file",
                        {
                            "path": "/workspace/mathkit/mathkit/core.py",
                            "content": (
                                "def identity(value):\n"
                                '    """Return twice the supplied value."""\n'
                                "    return value * 2\n"
                            ),
                        },
                    ),
                    _tool_call(
                        "verify",
                        "write_file",
                        {
                            "path": "/workspace/mathkit/verify.py",
                            "content": (
                                "from mathkit import identity\n\n"
                                "assert identity(21) == 42\n"
                                "passed = True\n"
                            ),
                        },
                    ),
                    _tool_call(
                        "check", "package_check", {"root": "/workspace/mathkit"}
                    ),
                    _tool_call("test", "package_test", {"root": "/workspace/mathkit"}),
                    _tool_call(
                        "export",
                        "package_export",
                        {"root": "/workspace/mathkit"},
                    ),
                    AIMessage(content="The Monty source package is ready."),
                ]
            )
        )
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            package_tools = create_package_tools(kernel)
            agent = create_agent(
                model,
                [*create_python_repl_tools(kernel), *package_tools],
            )
            state = agent.invoke(
                {
                    "messages": [
                        HumanMessage(content="Build a dependency-free Monty package.")
                    ]
                },
                {"configurable": {"thread_id": "package-agent"}},
            )
            results = [
                json.loads(message.content)
                for message in state["messages"]
                if isinstance(message, ToolMessage)
            ]
            archive_path = results[-1]["archive_path"]
            archive = kernel.download_file(archive_path)

        self.assertEqual(
            state["messages"][-1].content, "The Monty source package is ready."
        )
        self.assertEqual([result["ok"] for result in results], [True] * 6)
        self.assertGreater(len(archive), 0)
        descriptions = "\n".join(tool.description for tool in package_tools)
        self.assertIn("dependencies empty", descriptions)
        self.assertIn("fresh temporary Monty session", descriptions)
        self.assertIn("source ZIP", descriptions)
        self.assertIn("not a wheel", descriptions)


if __name__ == "__main__":
    unittest.main()
