import importlib.resources
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    """Test lightweight imports and release configuration."""

    def test_core_import_does_not_load_langchain(self) -> None:
        """Keep the framework integration lazy for core users."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import arenero, sys; assert not any("
                    "name.startswith(('langchain', 'mcp', 'httpx2')) "
                    "for name in sys.modules)"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_integrations_are_optional_development_dependencies(self) -> None:
        """Keep LangChain and MCP out of the installed core dependency set."""
        project = tomllib.loads(Path("pyproject.toml").read_text())
        dependencies = project["project"]["dependencies"]

        self.assertFalse(any(item.startswith("langchain") for item in dependencies))
        self.assertIn("langchain", project["project"]["optional-dependencies"])
        self.assertIn("mcp", project["project"]["optional-dependencies"])
        self.assertIn("langchain>=1.3.18", project["dependency-groups"]["dev"])
        self.assertIn("mcp>=2,<3", project["dependency-groups"]["dev"])
        self.assertIn("pydantic-monty>=0.0.21,<0.0.22", dependencies)

    def test_package_marks_inline_types_and_has_ci(self) -> None:
        """Ship the typing marker and keep a repository validation workflow."""
        marker = importlib.resources.files("arenero").joinpath("py.typed")

        self.assertTrue(marker.is_file())
        self.assertTrue(Path(".github/workflows/ci.yml").is_file())


if __name__ == "__main__":
    unittest.main()
