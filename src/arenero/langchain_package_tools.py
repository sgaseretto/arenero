"""LangChain tools for dependency-free Monty source packages."""

from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from .kernel import PersistentMontyKernel
from .packages import MontyPackageManager, PackageConfig


def create_package_tools(
    kernel: PersistentMontyKernel,
    config: PackageConfig | None = None,
) -> list[BaseTool]:
    """Create opt-in LangChain tools for the Monty package workflow.

    Args:
        kernel: Synchronous kernel with a writable virtual filesystem.
        config: Optional package roots and hard limits.

    Returns:
        Package initialization, checking, testing, and export tools.
    """

    manager = MontyPackageManager(kernel, config)
    guidance = manager.instructions()

    @tool(
        "package_init",
        description=(
            "Create a dependency-free Monty package scaffold. This does not create "
            "a wheel or install dependencies.\n\n" + guidance
        ),
    )
    def package_init(
        name: Annotated[
            str,
            Field(description="Python package identifier, such as mathkit."),
        ],
        root: Annotated[
            str | None,
            Field(description="Optional absolute project path below /workspace."),
        ] = None,
        version: Annotated[
            str,
            Field(description="Simple source-package version label."),
        ] = "0.1.0",
    ) -> dict[str, Any]:
        """Create a working package source scaffold.

        Args:
            name: Python package identifier.
            root: Optional absolute project root.
            version: Simple source version label.

        Returns:
            Structured initialization result and next steps.
        """

        return manager.init(name, root, version)

    @tool(
        "package_check",
        description=(
            "Validate a dependency-free package manifest, file limits, UTF-8 "
            "source, and every static Monty module graph.\n\n" + guidance
        ),
    )
    def package_check(
        root: Annotated[
            str,
            Field(description="Absolute package project path below /workspace."),
        ],
        expected_revision: Annotated[
            str | None,
            Field(description="Optional revision returned by an earlier check."),
        ] = None,
    ) -> dict[str, Any]:
        """Validate one package source tree.

        Args:
            root: Absolute package project path.
            expected_revision: Optional expected source revision.

        Returns:
            Structured inventory, manifest, limits, and revision.
        """

        return manager.check(root, expected_revision)

    @tool(
        "package_test",
        description=(
            "Check a package and execute verify.py in a fresh temporary Monty "
            "session. verify.py must export passed = True.\n\n" + guidance
        ),
    )
    def package_test(
        root: Annotated[
            str,
            Field(description="Absolute package project path below /workspace."),
        ],
        expected_revision: Annotated[
            str | None,
            Field(description="Optional revision returned by package_check."),
        ] = None,
    ) -> dict[str, Any]:
        """Run one package verification entry in a fresh session.

        Args:
            root: Absolute package project path.
            expected_revision: Optional expected source revision.

        Returns:
            Structured check and Monty verification result.
        """

        return manager.test(root, expected_revision)

    @tool(
        "package_export",
        description=(
            "Check and test a package, then create a bounded deterministic source "
            "ZIP. The archive is not a wheel or standard Python sdist.\n\n" + guidance
        ),
    )
    def package_export(
        root: Annotated[
            str,
            Field(description="Absolute package project path below /workspace."),
        ],
        expected_revision: Annotated[
            str | None,
            Field(description="Optional revision returned by package_check."),
        ] = None,
    ) -> dict[str, Any]:
        """Export one checked package source archive.

        Args:
            root: Absolute package project path.
            expected_revision: Optional expected source revision.

        Returns:
            Structured archive path, size, and source revision.
        """

        return manager.export(root, expected_revision)

    return [package_init, package_check, package_test, package_export]
