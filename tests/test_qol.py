import base64
import unittest

from pydantic_monty import OSAccess

from arenero import (
    AreneroRuntime,
    AsyncPersistentMontyKernel,
    KernelConfig,
    NetworkConfig,
    PersistentMontyKernel,
    execute_repl_cell,
    run_doctor,
)


class QualityOfLifeTests(unittest.TestCase):
    """Test the public quality-of-life APIs as one small integration surface."""

    def test_capabilities_and_bound_handle_use_active_configuration(self) -> None:
        """Report exact policy data and avoid repeated session IDs."""
        config = KernelConfig(allowed_shell_commands=frozenset({"echo"}))
        network = NetworkConfig(enabled=False)
        with PersistentMontyKernel(config=config, network_config=network) as kernel:
            report = kernel.capabilities()
            session = kernel.session("agent-1")
            first = session.execute("value = 21")
            second = session.execute("value * 2")

        self.assertEqual(report["shell"]["commands"], ["echo"])
        self.assertTrue(report["shell"]["supports_persistent_variables"])
        self.assertTrue(report["shell"]["supports_bounded_glob_expansion"])
        self.assertIn("command substitution", report["shell"]["does_not_support"])
        self.assertEqual(report["shell"]["redirections"], ["<", ">", ">>", "<<"])
        self.assertFalse(report["network"]["enabled"])
        self.assertIn("error.code", kernel.instructions())
        self.assertTrue(first["state_saved"])
        self.assertEqual(second["result"], 42)
        self.assertEqual(session.session_id, "agent-1")

    def test_errors_and_statistics_are_stable_and_actionable(self) -> None:
        """Return machine codes, hints, timing, state, and original sizes."""
        with PersistentMontyKernel() as kernel:
            result = kernel.execute("agent", "import numpy")

        self.assertEqual(result["error"]["code"], "unsupported_import")
        self.assertIn("execute_file", result["error"]["hint"])
        self.assertTrue(result["stats"]["execution_id"])
        self.assertGreaterEqual(result["stats"]["wall_time_seconds"], 0)
        self.assertGreater(result["stats"]["original_output_chars"]["error"], 0)

    def test_state_policies_control_python_snapshots_not_files(self) -> None:
        """Save always, only on success, or never as requested."""
        with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
            kernel.execute("always", "x = 1\nraise ValueError('stop')")
            kept = kernel.execute("always", "x")
            kernel.execute(
                "success",
                "x = 2\nfs_write('/kept.txt', 'file')\nraise ValueError('stop')",
                state_policy="success",
            )
            not_kept = kernel.execute("success", "x")
            file_kept = kernel.download_file("/kept.txt")
            kernel.execute("never", "x = 3", state_policy="never")
            never_kept = kernel.execute("never", "x")

        self.assertEqual(kept["result"], 1)
        self.assertFalse(not_kept["ok"])
        self.assertEqual(file_kept, b"file")
        self.assertFalse(never_kept["ok"])

    def test_bootstrap_fork_and_small_repl_commands(self) -> None:
        """Start new sessions, fork snapshots, and dispatch percent commands."""
        with PersistentMontyKernel(
            os_handler=OSAccess(),
            bootstrap_code="seed = 20",
            bootstrap_files=("/helper.py",),
        ) as kernel:
            kernel.upload_file(
                "/helper.py", b"def twice(value):\n    return value * 2\n"
            )
            source = kernel.session("source")
            self.assertEqual(source.execute("twice(seed + 1)")["result"], 42)
            source.execute("seed = 10")
            fork = source.fork("experiment")
            self.assertEqual(fork.execute("seed")["result"], 10)
            self.assertEqual(source.execute("seed")["result"], 10)
            kernel.upload_file("/run.py", b"answer = 6 * 7")
            run = execute_repl_cell(kernel, "magic", "%run /run.py")
            limits = execute_repl_cell(kernel, "magic", "%limits")
            sessions = execute_repl_cell(kernel, "magic", "%sessions")
            reset = execute_repl_cell(kernel, "magic", "%reset")

        self.assertTrue(run["ok"])
        self.assertEqual(run["mode"], "command")
        self.assertIn("max_state_bytes", limits["result"]["limits"])
        self.assertEqual(sessions["result"]["current_session"]["session_id"], "magic")
        self.assertTrue(reset["result"]["reset"])

    def test_failed_bootstrap_is_not_saved(self) -> None:
        """Return guided startup failure and retry bootstrap on the next call."""
        with PersistentMontyKernel(
            bootstrap_code="raise ValueError('bad startup')"
        ) as kernel:
            first = kernel.execute("agent", "1 + 1")
            second = kernel.execute("agent", "1 + 1")

        self.assertEqual(first["error"]["code"], "session_bootstrap_error")
        self.assertFalse(first["state_saved"])
        self.assertFalse(second["state_restored"])

    def test_runtime_events_doctor_and_artifacts(self) -> None:
        """Own lifecycle, observe safe events, diagnose, and write MCP media."""
        events = []
        kernel = PersistentMontyKernel(
            os_handler=OSAccess(),
            external_lookup={
                "get_media": lambda: {
                    "content": [
                        {
                            "type": "image",
                            "data": base64.b64encode(b"automatic").decode("ascii"),
                            "mimeType": "image/png",
                        }
                    ]
                }
            },
            event_sink=events.append,
            mcp_artifact_directory="/artifacts",
        )
        with AreneroRuntime(kernel) as runtime:
            value = {
                "ok": True,
                "content": [
                    {
                        "type": "image",
                        "data": base64.b64encode(b"image-bytes").decode("ascii"),
                        "mimeType": "image/png",
                    }
                ],
            }
            materialized, paths = kernel.materialize_mcp_artifacts(value, "/manual")
            self.assertEqual(kernel.download_file(paths[0]), b"image-bytes")
            self.assertNotIn("data", materialized["content"][0])
            result = runtime.session("agent").execute("1 + 1")
            automatic = runtime.session("media").execute("get_media()")
            automatic_bytes = kernel.download_file(automatic["mcp_artifacts"][0])
            doctor = run_doctor(kernel)

        self.assertEqual(result["result"], 2)
        self.assertEqual(len(automatic["mcp_artifacts"]), 1)
        self.assertEqual(automatic_bytes, b"automatic")
        self.assertTrue(doctor["ok"])
        self.assertTrue(any(event.category == "execution" for event in events))
        self.assertTrue(any(event.category == "filesystem" for event in events))
        self.assertTrue(kernel._closed)


class AsyncQualityOfLifeTests(unittest.IsolatedAsyncioTestCase):
    """Test native async parity for state and bound-session helpers."""

    async def test_async_bound_bootstrap_policy_fork_and_stats(self) -> None:
        """Use the same state controls on the native asynchronous kernel."""
        async with AsyncPersistentMontyKernel(bootstrap_code="seed = 21") as kernel:
            source = kernel.session("source")
            first = await source.execute("seed * 2")
            await source.execute("seed = 10")
            fork = await source.fork("fork")
            copied = await fork.execute("seed")
            await kernel.execute("temporary", "x = 1", state_policy="never")
            missing = await kernel.execute("temporary", "x")

        self.assertEqual(first["result"], 42)
        self.assertEqual(copied["result"], 10)
        self.assertFalse(missing["ok"])
        self.assertTrue(first["stats"]["execution_id"])
        capabilities = kernel.capabilities()
        self.assertEqual(capabilities["runtime"]["mode"], "native-async")
        self.assertFalse(capabilities["shell"]["enabled"])
        self.assertEqual(capabilities["shell"]["operators"], [])
