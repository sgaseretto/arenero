"""Create fixed-session LangChain tools for the managed sandbox API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain.tools import BaseTool, tool

if TYPE_CHECKING:
    from .easy import ManagedSandbox, SandboxFeature


def create_managed_sandbox_tools(
    sandbox: ManagedSandbox,
    include: frozenset[SandboxFeature],
) -> list[BaseTool]:
    """Create one selected fixed-session tool bundle.

    Args:
        sandbox: Bound managed sandbox used by every tool.
        include: Feature groups to expose.

    Returns:
        LangChain tools with common result mappings.
    """

    unknown = include - {"python", "files", "shell", "packages"}
    if unknown:
        raise ValueError(f"unknown managed tool groups: {sorted(unknown)}")
    tools: list[BaseTool] = []

    if "python" in include:

        @tool(
            "python_repl",
            description=(
                "Run Monty-compatible Python in the bound persistent session. "
                "State is reused by later calls. This is not CPython. Imports, "
                "syntax, and standard-library support are limited."
            ),
        )
        def python_repl(
            code: str,
            output_limit_chars: int | None = None,
            restart: bool = False,
            state_policy: Literal["always", "success", "never"] = "always",
        ) -> dict[str, Any]:
            """Run one persistent Monty Python feed.

            Args:
                code: Monty-compatible Python source.
                output_limit_chars: Optional lower model-facing output limit.
                restart: Reset saved Python state before this feed.
                state_policy: Save state always, on success, or never.

            Returns:
                Common JSON-compatible sandbox result.
            """

            return sandbox.run_python(
                code,
                output_limit_chars,
                restart=restart,
                state_policy=state_policy,
            ).to_dict()

        tools.append(python_repl)

    if "shell" in include:

        @tool(
            "execute",
            description=(
                "Run the small Arenero virtual shell in the bound workspace. "
                "This is not an operating-system shell: no processes, pip, pytest, "
                "full Bash control flow, command substitution, descriptor redirection, "
                "or arbitrary binaries. It supports persistent virtual variables, "
                "bounded globs, and straight-line source files. "
                "Use the active sandbox instructions for commands and limits."
            ),
        )
        def execute(
            command: str,
            output_limit_chars: int | None = None,
        ) -> dict[str, Any]:
            """Run one virtual-shell command list.

            Args:
                command: Supported virtual-shell text.
                output_limit_chars: Optional lower output limit.

            Returns:
                Common JSON-compatible sandbox result.
            """

            return sandbox.run_shell(command, output_limit_chars).to_dict()

        tools.append(execute)

    if "files" in include:
        tools.extend(_file_tools(sandbox))
    if "packages" in include:
        tools.extend(_package_tools(sandbox))
    return tools


def _file_tools(sandbox: ManagedSandbox) -> list[BaseTool]:
    """Create relative-path file tools for one sandbox.

    Args:
        sandbox: Bound managed sandbox.

    Returns:
        Read, write, edit, delete, list, glob, and grep tools.
    """

    @tool(
        "read_file",
        description="Read one UTF-8 file relative to the sandbox working directory.",
    )
    def read_file(path: str) -> dict[str, Any]:
        """Read one text file.

        Args:
            path: Relative or workspace-absolute virtual path.

        Returns:
            Content, size, and revision in the common result contract.
        """

        return sandbox.read(path).to_dict()

    @tool("write_file", description="Create or replace one UTF-8 sandbox file.")
    def write_file(
        path: str,
        content: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Write one text file with an optional revision check.

        Args:
            path: Relative or workspace-absolute virtual path.
            content: Complete replacement text.
            expected_revision: Optional revision returned by read_file.

        Returns:
            New path, size, and revision in the common result contract.
        """

        return sandbox.write(
            path, content, expected_revision=expected_revision
        ).to_dict()

    @tool("edit_file", description="Replace exact text in one UTF-8 sandbox file.")
    def edit_file(
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Edit one text file with exact replacement.

        Args:
            path: Relative or workspace-absolute virtual path.
            old: Exact text to replace.
            new: Replacement text.
            replace_all: Replace each match when true.
            expected_revision: Optional revision returned by read_file.

        Returns:
            New revision and replacement count.
        """

        return sandbox.edit(
            path,
            old,
            new,
            replace_all=replace_all,
            expected_revision=expected_revision,
        ).to_dict()

    @tool("delete_file", description="Delete one sandbox file or directory.")
    def delete_file(path: str, recursive: bool = False) -> dict[str, Any]:
        """Delete one virtual path.

        Args:
            path: Relative or workspace-absolute virtual path.
            recursive: Delete a non-empty directory when true.

        Returns:
            Common JSON-compatible sandbox result.
        """

        return sandbox.delete(path, recursive=recursive).to_dict()

    @tool("list_files", description="List bounded entries in one sandbox directory.")
    def list_files(path: str = ".", recursive: bool = False) -> dict[str, Any]:
        """List virtual directory entries.

        Args:
            path: Relative or workspace-absolute directory.
            recursive: Include nested entries when true.

        Returns:
            Common result with bounded entry records.
        """

        return sandbox.ls(path, recursive=recursive).to_dict()

    @tool(
        "glob_files", description="Find sandbox paths with *, ?, and ** glob patterns."
    )
    def glob_files(pattern: str, path: str = ".") -> dict[str, Any]:
        """Find virtual paths by glob.

        Args:
            pattern: Supported virtual glob pattern.
            path: Relative or workspace-absolute search root.

        Returns:
            Common result with bounded matching paths.
        """

        return sandbox.glob(pattern, path).to_dict()

    @tool("grep_files", description="Find literal text in bounded UTF-8 sandbox files.")
    def grep_files(
        query: str,
        path: str = ".",
        pattern: str = "**/*",
    ) -> dict[str, Any]:
        """Find literal text in virtual files.

        Args:
            query: Literal text, not a regular expression.
            path: Relative or workspace-absolute search root.
            pattern: Glob used to select files.

        Returns:
            Common result with bounded line matches.
        """

        return sandbox.grep(query, path, pattern).to_dict()

    return [
        read_file,
        write_file,
        edit_file,
        delete_file,
        list_files,
        glob_files,
        grep_files,
    ]


def _package_tools(sandbox: ManagedSandbox) -> list[BaseTool]:
    """Create the fixed-root Monty package workflow tools.

    Args:
        sandbox: Bound managed sandbox.

    Returns:
        Package init, check, test, and export tools.
    """

    @tool(
        "package_init",
        description="Create a dependency-free Monty source-package scaffold.",
    )
    def package_init(
        name: str,
        root: str | None = None,
        version: str = "0.1.0",
    ) -> dict[str, Any]:
        """Create one source package.

        Args:
            name: Python package identifier.
            root: Optional relative or workspace-absolute project directory.
            version: Simple source-package version.

        Returns:
            Common package initialization result.
        """

        return sandbox.package_init(name, root, version).to_dict()

    @tool(
        "package_check",
        description="Validate one Monty source package and its static modules.",
    )
    def package_check(
        root: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Check one source package.

        Args:
            root: Relative or workspace-absolute package directory.
            expected_revision: Optional earlier package revision.

        Returns:
            Common package validation result.
        """

        return sandbox.package_check(root, expected_revision).to_dict()

    @tool(
        "package_test",
        description="Run verify.py for one package in a fresh Monty session.",
    )
    def package_test(
        root: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Test one source package.

        Args:
            root: Relative or workspace-absolute package directory.
            expected_revision: Optional earlier package revision.

        Returns:
            Common package test result.
        """

        return sandbox.package_test(root, expected_revision).to_dict()

    @tool(
        "package_export",
        description="Test and export one bounded deterministic Arenero source ZIP.",
    )
    def package_export(
        root: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Export one source package.

        Args:
            root: Relative or workspace-absolute package directory.
            expected_revision: Optional earlier package revision.

        Returns:
            Common package export result with a virtual archive path.
        """

        return sandbox.package_export(root, expected_revision).to_dict()

    return [package_init, package_check, package_test, package_export]
