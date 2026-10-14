"""Check the Monty 1.1 runtime contract through real Arenero workers."""

import json
import tempfile
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pydantic_monty import MemoryFile, Monty, MontyError, OSAccess

from arenero import (
    AreneroSandbox,
    AsyncPersistentMontyKernel,
    InMemorySessionStore,
    KernelConfig,
    PersistentMontyKernel,
    SandboxConfig,
    WorkspaceConfig,
    open_sandbox,
)


class Monty110Tests(unittest.TestCase):
    """Check native features, persistence, limits, and rejected features."""

    def test_new_limit_configuration_is_validated(self) -> None:
        """Reject invalid budgets and the removed cumulative-duration field."""
        for values in (
            {"max_feed_duration_seconds": 0},
            {"max_turn_duration_seconds": 0},
            {"max_suspensions": 0},
            {"max_total_sleep_seconds": -1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                KernelConfig(**values)
        with self.assertRaises(TypeError):
            KernelConfig(max_duration_seconds=5)

    def test_native_exec_bypasses_checks_but_not_resource_limits(self) -> None:
        """Bound dynamic code even when its source bypasses static checking."""
        config = KernelConfig(type_check=True, max_feed_duration_seconds=0.01)
        with PersistentMontyKernel(config=config) as kernel:
            unchecked = kernel.execute("agent", 'exec("answer: int = 3.5")')
            self.assertTrue(unchecked["ok"], unchecked["error"])
            follow_up = kernel.execute("agent", "answer")
            self.assertEqual(follow_up["error"]["type"], "MontyTypingError")
            failed = kernel.execute("agent", "exec('while True:\\n    pass')")
            self.assertFalse(failed["ok"])
            self.assertIsNone(kernel.store.load("agent"))

    def test_system_sleep_budget_resets_on_restore(self) -> None:
        """Charge sleep within a checkout, not across saved agent turns."""
        with PersistentMontyKernel(
            config=KernelConfig(max_total_sleep_seconds=0.015)
        ) as kernel:
            for _ in range(3):
                result = kernel.execute("agent", "import time\ntime.sleep(0.01)")
                self.assertTrue(result["ok"], result["error"])
            exceeded = kernel.execute("agent", "time.sleep(0.01)\ntime.sleep(0.01)")
            self.assertEqual(exceeded["error"]["code"], "sleep_limit")
            self.assertIsNone(kernel.store.load("agent"))

    def test_native_features_and_remaining_limits(self) -> None:
        """Exercise useful native features and keep unsupported code rejected."""
        accepted = {
            "eval('20 + 22')": 42,
            "exec('answer = 42')\nlocals()['answer']": 42,
            "format(42, '04d')": "0042",
            "'%s %d' % ('answer', 42)": "answer 42",
            "import base64\nbase64.b64decode(base64.b64encode(b'hi')).decode()": "hi",
            "import binascii\nbinascii.hexlify(b'hi').decode()": "6869",
            "import copy\ncopy.deepcopy({'values': [42]})": {"values": [42]},
            "from functools import partial, reduce\npartial(reduce, lambda a, b: a + b)([20, 22])": 42,
            "from itertools import batched\n[list(batch) for batch in batched([1, 2, 3], 2)]": [
                [1, 2],
                [3],
            ],
            "(1 + 2j) * (1 - 2j)": {"type": "complex", "real": 5.0, "imag": 0.0},
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Point:\n    x: int\nPoint(1) == Point(1)": True,
            "[type(value).__name__ for value in [map(str, [1]), (x for x in [1])]]": [
                "list",
                "list",
            ],
        }
        rejected = [
            "import numpy",
            "import helper",
            "import hashlib",
            "import os.path",
            "compile('42', '<test>', 'eval')",
            "globals()",
            "__import__('math')",
            "from pathlib import Path\nPath('/').glob('*')",
            "from dataclasses import field",
            "def values():\n    yield 1",
        ]
        with PersistentMontyKernel() as kernel:
            for index, (source, expected) in enumerate(accepted.items()):
                with self.subTest(source=source):
                    result = kernel.execute(f"accepted-{index}", source)
                    self.assertTrue(result["ok"], result["error"])
                    self.assertEqual(result["result"], expected)
            for index, source in enumerate(rejected):
                with self.subTest(source=source):
                    self.assertFalse(kernel.execute(f"rejected-{index}", source)["ok"])

    def test_exec_reads_helper_and_calls_host_across_turns(self) -> None:
        """Run virtual helper source in Monty, including a normal host call."""
        files = OSAccess(
            files=[MemoryFile("/helper.py", "def double(x):\n    return add(x, x)")]
        )
        with PersistentMontyKernel(
            os_handler=files, external_lookup={"add": lambda a, b: a + b}
        ) as kernel:
            first = kernel.execute(
                "agent", "exec(open('/helper.py').read())\ndouble(21)"
            )
            later = kernel.execute("agent", "double(5)")
            isolated = kernel.execute(
                "isolated", "ns = {}\nexec('value = add(1, 2)', ns)"
            )
        self.assertEqual(first["result"], 42)
        self.assertEqual(later["result"], 10)
        self.assertFalse(isolated["ok"])
        self.assertIn("NameError", isolated["error"]["message"])

    def test_shell_and_python_share_native_directory_after_restore(self) -> None:
        """Use one saved working directory for Python, shell, and file helpers."""
        files = OSAccess(
            files=[MemoryFile("/data/a.txt", "42"), MemoryFile("/other/b.txt", "7")]
        )
        store = InMemorySessionStore()
        with PersistentMontyKernel(store, os_handler=files) as kernel:
            shell = AreneroSandbox(kernel, session_id="agent")
            self.assertEqual(shell.execute("cd /data").exit_code, 0)
            self.assertEqual(
                kernel.execute("agent", "open('a.txt').read()")["result"], "42"
            )
            changed = kernel.execute(
                "agent", "import os\nos.chdir('/other')\nfs_read('b.txt')"
            )
            self.assertEqual(changed["result"], "7")
            self.assertEqual(shell.execute("pwd").output, "/other\n")
            self.assertEqual(
                kernel.execute("other-agent", "import os\nos.getcwd()")["result"], "/"
            )
        with PersistentMontyKernel(store, os_handler=files) as kernel:
            self.assertEqual(
                AreneroSandbox(kernel, session_id="agent").execute("pwd").output,
                "/other\n",
            )

    def test_shell_directory_is_not_trimmed_as_internal_output(self) -> None:
        """Capture the directory as host data even with a small result limit."""
        config = KernelConfig(default_output_limit_chars=1, max_output_limit_chars=1)
        files = OSAccess(files=[MemoryFile("/long-directory/a.txt", "42")])
        with PersistentMontyKernel(config=config, os_handler=files) as kernel:
            self.assertTrue(
                kernel.execute("agent", "import os\nos.chdir('/long-directory')")["ok"]
            )
            shell = AreneroSandbox(kernel, session_id="agent").execute("pwd")
        self.assertEqual(shell.exit_code, 0)
        self.assertTrue(shell.truncated)

    def test_workspace_sets_native_directory_before_bootstrap(self) -> None:
        """Let relative startup file reads use the configured workspace."""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "answer.txt").write_text("42")
            with PersistentMontyKernel(
                workspace=WorkspaceConfig(directory),
                bootstrap_code="answer = int(open('answer.txt').read())",
            ) as kernel:
                result = kernel.execute("agent", "answer")
        self.assertEqual(result["result"], 42)

    def test_seeded_random_state_and_exec_function_survive_fork(self) -> None:
        """Copy native RNG state and dynamically defined functions in a fork."""
        config = KernelConfig(os_policy={"random_start": {"seed": 42}, "sleep": "zero"})
        with PersistentMontyKernel(config=config) as kernel:
            self.assertTrue(
                kernel.execute(
                    "parent",
                    "import random\nexec('def sample():\\n    return random.random()')\nsample()",
                )["ok"]
            )
            kernel.fork_session("parent", "child")
            parent = kernel.execute("parent", "sample()")
            child = kernel.execute("child", "sample()")
        self.assertEqual(parent["result"], child["result"])

    def test_fixed_os_policy_and_stderr_are_reported(self) -> None:
        """Report fixed clocks safely and collect the native stderr stream."""
        instant = datetime(2026, 1, 2, tzinfo=UTC)
        config = KernelConfig(
            os_policy={
                "datetime": instant,
                "sleep": "zero",
                "random_start": {"seed": b"42"},
            }
        )
        with open_sandbox(
            "agent", config=SandboxConfig.conversation(kernel=config)
        ) as sandbox:
            result = sandbox.run_python(
                "import sys, time\nfrom datetime import datetime, timezone\nprint('notice', file=sys.stderr)\ntime.sleep(100)\ndatetime.now(timezone.utc).isoformat()"
            )
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.stderr, "notice\n")
            self.assertEqual(result.value, "2026-01-02T00:00:00+00:00")
            report = sandbox.kernel.capabilities()
            self.assertEqual(
                report["runtime"]["os_policy"]["datetime"], instant.isoformat()
            )
            json.dumps(report)
            sandbox.policy_digest()
            self.assertTrue(sandbox.checkpoint().ok)
            self.assertEqual(config.os_policy["random_start"], {"seed": b"42"})
            self.assertTrue(sandbox.run_python("import random\nrandom.random()").ok)

    def test_duration_resets_for_each_feed(self) -> None:
        """Keep a long-lived session after many separately bounded feeds."""
        config = KernelConfig(
            max_feed_duration_seconds=0.05, os_policy={"process_time": "elapsed"}
        )
        with PersistentMontyKernel(config=config) as kernel:
            for _ in range(8):
                result = kernel.execute(
                    "agent",
                    "import time\nwhile time.process_time() < 0.01:\n    pass\n42",
                )
                self.assertTrue(result["ok"], result["error"])

    def test_turn_timeout_and_system_sleep_timeout_discard_state(self) -> None:
        """Remove saved state after either kind of terminal TimeoutError."""
        cases = [
            (KernelConfig(max_turn_duration_seconds=0.01), "while True:\n    pass"),
            (KernelConfig(max_total_sleep_seconds=0), "import time\ntime.sleep(0.01)"),
        ]
        for config, source in cases:
            with (
                self.subTest(source=source),
                PersistentMontyKernel(config=config) as kernel,
            ):
                self.assertTrue(kernel.execute("agent", "answer = 42")["ok"])
                failed = kernel.execute("agent", source)
                self.assertFalse(failed["ok"])
                self.assertFalse(failed["state_saved"])
                self.assertEqual(failed["error"]["recovery"], "restart_session")
                self.assertIsNone(kernel.store.load("agent"))
                self.assertFalse(kernel.execute("agent", "answer")["ok"])

    def test_suspension_limit_keeps_consistent_state_and_resets(self) -> None:
        """Keep partial Python state and renew host-call allowance on restore."""
        config = KernelConfig(max_suspensions=2)
        with PersistentMontyKernel(
            config=config, external_lookup={"ping": lambda: 42}
        ) as kernel:
            failed = kernel.execute(
                "agent", "answer = 7\nfor i in range(10):\n    ping()"
            )
            self.assertEqual(failed["error"]["code"], "suspension_limit")
            self.assertTrue(failed["state_saved"])
            later = kernel.execute("agent", "[answer, ping(), ping()]")
            self.assertEqual(later["result"], [7, 42, 42])

    def test_restoring_checkout_caps_suspensions_but_not_other_limits(self) -> None:
        """Check native restore behavior separately from Arenero's policy guard."""
        with Monty() as pool:
            with pool.checkout(
                limits={"max_feed_duration_secs": 5, "max_suspensions": 10},
                os_policy={"process_time": "elapsed"},
            ) as session:
                session.feed_run("answer = 42")
                state = session.dump()
            with pool.checkout(
                limits={"max_feed_duration_secs": 0.001, "max_suspensions": 1}
            ) as session:
                session.load_session(state)
                result = session.feed_run(
                    "import time\nwhile time.process_time() < 0.01:\n    pass\nanswer"
                )
                self.assertEqual(result, 42)
                with self.assertRaises(MontyError) as failed:
                    session.feed_run(
                        "ping()\nping()", external_lookup={"ping": lambda: 1}
                    )
                self.assertIn("suspension limit", str(failed.exception).lower())

    def test_host_wait_is_outside_interpreter_duration(self) -> None:
        """Do not count a host callback's delay as interpreter work."""

        def slow_answer() -> int:
            """Wait in the host and return the test value.

            Returns:
                The integer 42.
            """
            time.sleep(0.03)
            return 42

        with PersistentMontyKernel(
            config=KernelConfig(max_feed_duration_seconds=0.01),
            external_lookup={"slow_answer": slow_answer},
        ) as kernel:
            self.assertEqual(kernel.execute("agent", "slow_answer()")["result"], 42)

    def test_native_modules_work_in_static_packages(self) -> None:
        """Pass newly available standard modules through the existing resolver."""
        files = OSAccess(
            files=[
                MemoryFile(
                    "/helper.py",
                    "import copy, random, time, base64\nanswer = copy.deepcopy([base64.b64decode('NDI=').decode()])",
                )
            ]
        )
        with PersistentMontyKernel(os_handler=files) as kernel:
            result = kernel.execute_module("agent", "/helper.py", "helper.answer")
        self.assertTrue(result["ok"], result["error"])
        self.assertEqual(result["result"], ["42"])

    def test_previous_version_state_requires_an_explicit_reset(self) -> None:
        """Reject the previous release before attempting to decode its dump."""
        store = InMemorySessionStore()
        store.save("agent", b"old dump", "0.0.21", expected_revision=None)
        with PersistentMontyKernel(store) as kernel:
            result = kernel.execute("agent", "42")
            self.assertEqual(result["error"]["code"], "session_version_error")
            self.assertEqual(store.load("agent").state, b"old dump")
            self.assertEqual(kernel.execute("agent", "42", restart=True)["result"], 42)


class AsyncMonty110Tests(unittest.IsolatedAsyncioTestCase):
    """Check new feature and limit parity through native async workers."""

    async def test_exec_native_sleep_and_random_survive_restore(self) -> None:
        """Use Monty exec with an awaited host function and a saved RNG."""

        async def answer() -> int:
            """Return the test answer through the async host boundary.

            Returns:
                The integer 42.
            """
            return 42

        config = KernelConfig(os_policy={"sleep": "zero", "random_start": {"seed": 42}})
        async with AsyncPersistentMontyKernel(
            config=config, external_lookup={"answer": answer}
        ) as kernel:
            first = await kernel.execute(
                "agent",
                "import asyncio, random\nexec('def draw():\\n    return random.random()')\nawait asyncio.sleep(100)\nawait answer()",
            )
            await kernel.fork_session("agent", "child")
            parent = await kernel.execute("agent", "draw()")
            child = await kernel.execute("child", "draw()")
        self.assertEqual(first["result"], 42)
        self.assertEqual(parent["result"], child["result"])

    async def test_native_directory_and_relative_files_survive_restore(self) -> None:
        """Set a workspace directory before async startup and later file reads."""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "answer.txt").write_text("42")
            async with AsyncPersistentMontyKernel(
                workspace=WorkspaceConfig(directory)
            ) as kernel:
                first = await kernel.execute("agent", "import os\nos.getcwd()")
                later = await kernel.execute("agent", "open('answer.txt').read()")
        self.assertEqual(first["result"], "/workspace")
        self.assertEqual(later["result"], "42")

    async def test_async_suspensions_and_sleep_are_bounded(self) -> None:
        """Apply the shared limits in async operations and preserve safe state."""

        async def ping() -> int:
            """Return a value through an awaited host call.

            Returns:
                The integer 42.
            """
            return 42

        config = KernelConfig(max_suspensions=2, max_total_sleep_seconds=0)
        async with AsyncPersistentMontyKernel(
            config=config, external_lookup={"ping": ping}
        ) as kernel:
            failed = await kernel.execute(
                "agent", "value = 7\nfor i in range(10):\n    await ping()"
            )
            self.assertEqual(failed["error"]["code"], "suspension_limit")
            self.assertTrue(failed["state_saved"])
            self.assertEqual((await kernel.execute("agent", "value"))["result"], 7)
            slept = await kernel.execute(
                "agent", "import asyncio\nawait asyncio.sleep(0.01)"
            )
            self.assertFalse(slept["ok"])
            self.assertEqual(slept["error"]["code"], "sleep_limit")
            self.assertEqual(slept["error"]["recovery"], "restart_session")
            self.assertIsNone(kernel.store.load("agent"))
