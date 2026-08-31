"""Tests for limited static virtual modules."""

import unittest

from pydantic_monty import MemoryFile, OSAccess

from arenero import (
    AsyncPersistentMontyKernel,
    KernelConfig,
    PersistentMontyKernel,
    VirtualModuleConfig,
    VirtualModuleError,
    VirtualModuleResolver,
)


class VirtualModuleResolverTests(unittest.TestCase):
    """Check resolution rules without starting Monty."""

    def test_bundle_resolves_imports_packages_and_aliases(self) -> None:
        """Resolve absolute, relative, package, and configured alias imports."""

        sources = {
            "/lib/math_helpers.py": "def double(value):\n    return value * 2\n",
            "/pkg/__init__.py": "from .values import VALUE\n",
            "/pkg/values.py": "VALUE = 20\n",
            "/main.py": (
                "from libalias.math_helpers import double\n"
                "import pkg as package\n"
                "ANSWER = double(package.VALUE + 1)\n"
            ),
        }
        resolver = VirtualModuleResolver(
            sources.get,
            VirtualModuleConfig(aliases={"libalias": "/lib"}),
        )

        bundle = resolver.bundle("/main.py", "ANSWER")

        self.assertEqual(bundle.entry_name, "main")
        self.assertEqual(bundle.files[-1], "/main.py")
        self.assertIn("ANSWER", bundle.exports)
        self.assertIn("_AreneroVirtualModule", bundle.code)

    def test_bundle_rejects_cycles_nested_imports_and_limits(self) -> None:
        """Reject graphs that the minimal resolver cannot represent safely."""

        cycle = {"/a.py": "from b import value\n", "/b.py": "from a import value\n"}
        nested = {"/a.py": "def load():\n    import json\n"}
        count = {"/a.py": "from b import value\n", "/b.py": "value = 1\n"}

        with self.assertRaisesRegex(VirtualModuleError, "circular"):
            VirtualModuleResolver(cycle.get).bundle("/a.py")
        with self.assertRaisesRegex(VirtualModuleError, "inside functions"):
            VirtualModuleResolver(nested.get).bundle("/a.py")
        with self.assertRaisesRegex(VirtualModuleError, "count limit"):
            VirtualModuleResolver(count.get, VirtualModuleConfig(max_modules=1)).bundle(
                "/a.py"
            )


class KernelVirtualModuleTests(unittest.TestCase):
    """Run generated module code with the synchronous kernel."""

    def test_execute_module_keeps_exports_in_the_session(self) -> None:
        """Use a dependency, follow-up code, and a later persistent call."""

        filesystem = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "FACTOR = 2\ndef double(value):\n    return value * FACTOR\n",
                ),
                MemoryFile(
                    "/main.py",
                    "from helper import double\nANSWER = double(21)\n",
                ),
            ]
        )
        kernel = PersistentMontyKernel(os_handler=filesystem)
        self.addCleanup(kernel.close)

        first = kernel.execute_module("module", "/main.py", "ANSWER")
        later = kernel.execute("module", "double(5)")

        self.assertTrue(first["ok"])
        self.assertEqual(first["result"], 42)
        self.assertEqual(later["result"], 10)

    def test_execute_module_returns_a_structured_resolution_error(self) -> None:
        """Return a normal result object for an unresolved virtual import."""

        kernel = PersistentMontyKernel(
            os_handler=OSAccess(files=[MemoryFile("/main.py", "from gone import x\n")])
        )
        self.addCleanup(kernel.close)

        result = kernel.execute_module("module", "/main.py")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "VirtualModuleError")

    def test_generated_module_feed_skips_type_check(self) -> None:
        """Allow generated module attributes when normal cells are type checked."""

        filesystem = OSAccess(
            files=[
                MemoryFile("/helper.py", "def inc(value):\n    return value + 1\n"),
                MemoryFile("/main.py", "from helper import inc\n"),
            ]
        )
        kernel = PersistentMontyKernel(
            config=KernelConfig(type_check=True),
            os_handler=filesystem,
        )
        self.addCleanup(kernel.close)

        result = kernel.execute_module("module", "/main.py", "inc(1)")

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], 2)


class AsyncKernelVirtualModuleTests(unittest.IsolatedAsyncioTestCase):
    """Run static virtual modules on the asynchronous worker pool."""

    async def test_async_execute_module(self) -> None:
        """Resolve files asynchronously and retain the resulting exports."""

        filesystem = OSAccess(
            files=[
                MemoryFile("/helper.py", "def inc(value):\n    return value + 1\n"),
                MemoryFile("/main.py", "from helper import inc\nVALUE = inc(4)\n"),
            ]
        )
        async with AsyncPersistentMontyKernel(os_handler=filesystem) as kernel:
            first = await kernel.execute_module("module", "/main.py", "VALUE")
            later = await kernel.execute("module", "inc(9)")

        self.assertTrue(first["ok"])
        self.assertEqual(first["result"], 5)
        self.assertEqual(later["result"], 10)
