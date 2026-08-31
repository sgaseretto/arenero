import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic_monty import OSAccess

from arenero import AreneroSandbox, NetworkConfig, PersistentMontyKernel


class _Handler(BaseHTTPRequestHandler):
    """Serve fixed local responses for network policy tests."""

    def do_GET(self) -> None:
        """Return text, binary, redirect, error, or large test responses."""
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/text")
            self.end_headers()
            return
        if self.path == "/auth":
            body = (
                b"authorized"
                if self.headers.get("Authorization") == "Bearer host-secret"
                else b"denied"
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status = 404 if self.path == "/missing" else 200
        bodies = {
            "/binary": b"\x00\xffarenero",
            "/large": b"x" * 20,
            "/text": b"alpha\nbeta\n",
        }
        body = bodies.get(self.path, b"missing")
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        """Return one header-only successful response."""
        self.send_response(200)
        self.send_header("X-Arenero", "test")
        self.end_headers()

    def do_POST(self) -> None:
        """Echo one bounded request body."""
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Disable HTTP server log output during tests.

        Args:
            format: Server log format string.
            *args: Values for the format string.
        """


class NetworkTests(unittest.TestCase):
    """Test curl access and its host network policy."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start one local HTTP server for this test class."""
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop the local HTTP server after all tests."""
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def make_sandbox(self, config: NetworkConfig | None) -> AreneroSandbox:
        """Create one sandbox with an optional network policy.

        Args:
            config: Network policy, or ``None`` to keep curl disabled.

        Returns:
            Open sandbox facade. Test cleanup closes its kernel.
        """
        kernel = PersistentMontyKernel(
            os_handler=OSAccess(),
            network_config=config,
        )
        self.addCleanup(kernel.close)
        return AreneroSandbox(kernel, session_id="network-test")

    def local_config(self, **changes: object) -> NetworkConfig:
        """Create a local-only policy for deterministic HTTP tests.

        Args:
            **changes: NetworkConfig values that replace the test defaults.

        Returns:
            Network policy for the local test server.
        """
        values = {
            "allowed_url_prefixes": (self.origin,),
            "allowed_methods": frozenset({"GET", "HEAD", "POST"}),
            "deny_non_global_addresses": False,
        }
        values.update(changes)
        return NetworkConfig(**values)

    def test_curl_is_available_without_network_config(self) -> None:
        """Allow public curl requests when no network policy is supplied."""
        sandbox = self.make_sandbox(None)

        result = sandbox.execute(f"curl {self.origin}/text")

        self.assertEqual(result.exit_code, 1)
        self.assertIn("not public", result.output)
        self.assertIn("curl", sandbox.execute("help").output.split())

    def test_curl_can_be_disabled_explicitly(self) -> None:
        """Remove curl when the host sets the network policy to disabled."""
        sandbox = self.make_sandbox(NetworkConfig(enabled=False))

        result = sandbox.execute(f"curl {self.origin}/text")

        self.assertEqual(result.exit_code, 127)
        self.assertNotIn("curl", sandbox.execute("help").output.split())

    def test_curl_fetches_pipes_and_saves_binary_virtual_files(self) -> None:
        """Use allowed responses as text pipes or raw virtual file bytes."""
        sandbox = self.make_sandbox(self.local_config())

        piped = sandbox.execute(f"curl {self.origin}/text | grep beta")
        saved = sandbox.execute(f"curl -o /download.bin {self.origin}/binary")
        downloaded = sandbox.download_files(["/download.bin"])[0]

        self.assertEqual(piped.output, "beta\n")
        self.assertEqual(saved.exit_code, 0, saved.output)
        self.assertEqual(downloaded.content, b"\x00\xffarenero")
        self.assertEqual(sandbox.execute("help curl").exit_code, 0)

    def test_curl_supports_bounded_methods_headers_and_redirects(self) -> None:
        """Use the supported request flags and checked redirect flow."""
        sandbox = self.make_sandbox(self.local_config())

        posted = sandbox.execute(f"curl -d hello {self.origin}/echo")
        headed = sandbox.execute(f"curl -i -I {self.origin}/text")
        redirected = sandbox.execute(f"curl -L {self.origin}/redirect")
        failed = sandbox.execute(f"curl -f {self.origin}/missing")

        self.assertEqual(posted.output, "hello")
        self.assertIn("HTTP 200 OK", headed.output)
        self.assertIn("X-Arenero: test", headed.output)
        self.assertEqual(redirected.output, "alpha\nbeta\n")
        self.assertEqual(failed.exit_code, 22)

    def test_curl_rejects_methods_and_invalid_timeout_options(self) -> None:
        """Apply the default method policy and validate timeout syntax."""
        sandbox = self.make_sandbox(
            self.local_config(allowed_methods=frozenset({"GET", "HEAD"}))
        )

        method = sandbox.execute(f"curl -d hello {self.origin}/echo")
        timeout = sandbox.execute(f"curl --max-time 0 {self.origin}/text")

        self.assertEqual(method.exit_code, 1)
        self.assertIn("method is not allowed", method.output)
        self.assertEqual(timeout.exit_code, 2)
        self.assertIn("greater than 0", timeout.output)

    def test_network_policy_rejects_private_hosts_and_large_responses(self) -> None:
        """Reject a local address by default and enforce the response cap."""
        private = self.make_sandbox(NetworkConfig(allowed_url_prefixes=(self.origin,)))
        limited = self.make_sandbox(self.local_config(max_response_bytes=5))

        rejected = private.execute(f"curl {self.origin}/text")
        too_large = limited.execute(f"curl {self.origin}/large")

        self.assertEqual(rejected.exit_code, 1)
        self.assertIn("not public", rejected.output)
        self.assertEqual(too_large.exit_code, 1)
        self.assertIn("network limit", too_large.output)

    def test_trusted_host_headers_override_agent_headers(self) -> None:
        """Add a host secret without putting it in the agent curl command."""

        def trusted_headers(url: str) -> dict[str, str]:
            """Return the test authorization header for one checked URL.

            Args:
                url: Policy-checked request URL.

            Returns:
                Host-owned authorization header.
            """
            return {"authorization": "Bearer host-secret"}

        sandbox = self.make_sandbox(
            self.local_config(trusted_header_provider=trusted_headers)
        )

        result = sandbox.execute(
            f"curl -H 'Authorization: agent-value' {self.origin}/auth"
        )

        self.assertEqual(result.output, "authorized")

    def test_redirect_target_must_match_the_url_prefix_allowlist(self) -> None:
        """Check a redirect target against the allowlist before fetching it."""
        sandbox = self.make_sandbox(
            self.local_config(allowed_url_prefixes=(f"{self.origin}/redirect",))
        )

        result = sandbox.execute(f"curl -L {self.origin}/redirect")

        self.assertEqual(result.exit_code, 1)
        self.assertIn("allowed_url_prefixes", result.output)

    def test_network_config_rejects_ambiguous_policy_values(self) -> None:
        """Accept the public default but reject query prefixes and bad methods."""
        self.assertEqual(NetworkConfig().allowed_url_prefixes, ())
        with self.assertRaises(ValueError):
            NetworkConfig(allowed_url_prefixes=(f"{self.origin}/?token=value",))
        with self.assertRaises(ValueError):
            NetworkConfig(
                allowed_url_prefixes=(self.origin,),
                allowed_methods=frozenset({"get"}),
            )


if __name__ == "__main__":
    unittest.main()
