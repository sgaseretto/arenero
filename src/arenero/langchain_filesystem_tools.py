from collections.abc import Callable
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import Field

from .integrations import _missing_thread_id
from .kernel import PersistentMontyKernel

SessionIdResolver = Callable[[ToolRuntime], str | None]


def create_filesystem_tools(
    kernel: PersistentMontyKernel,
    session_id_resolver: SessionIdResolver,
) -> list[BaseTool]:
    """Create virtual filesystem tools for LangChain or LangGraph.

    Args:
        kernel: Kernel that owns the shared virtual filesystem.
        session_id_resolver: Function that maps runtime data to a kernel session.

    Returns:
        Read, write, edit, delete, list, glob, and grep tools.
    """

    @tool("read_file")
    def read_file(
        path: Annotated[
            str,
            Field(min_length=1, description="Virtual POSIX path to a text file."),
        ],
        runtime: ToolRuntime,
        offset: Annotated[
            int,
            Field(ge=0, description="Zero-based character offset."),
        ] = 0,
        limit: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional characters to read. The application enforces a hard "
                    "maximum."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Read a bounded segment from a virtual text file.

        Args:
            path: Virtual POSIX file path.
            runtime: LangChain runtime that identifies the current thread.
            offset: Zero-based character offset.
            limit: Optional requested character count.

        Returns:
            A structured result containing the text segment.
        """
        session_id = session_id_resolver(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.call_filesystem(
                session_id,
                "read",
                {"path": path, "offset": offset, "limit": limit},
                output_limit_chars=limit,
            )
        )

    @tool("write_file")
    def write_file(
        path: Annotated[
            str,
            Field(min_length=1, description="Virtual POSIX path to write."),
        ],
        content: Annotated[str, Field(description="Complete text to write or append.")],
        runtime: ToolRuntime,
        append: Annotated[
            bool,
            Field(description="Append when true; replace the file when false."),
        ] = False,
        create_parents: Annotated[
            bool,
            Field(description="Create missing parent directories when true."),
        ] = False,
    ) -> dict[str, Any]:
        """Write or append bounded text in the virtual filesystem.

        Args:
            path: Virtual POSIX file path.
            content: Text to write.
            runtime: LangChain runtime that identifies the current thread.
            append: Append when true; replace when false.
            create_parents: Create missing parent directories when true.

        Returns:
            A structured result containing the written character count.
        """
        session_id = session_id_resolver(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.call_filesystem(
                session_id,
                "write",
                {
                    "path": path,
                    "content": content,
                    "append": append,
                    "create_parents": create_parents,
                },
            )
        )

    @tool("edit_file")
    def edit_file(
        path: Annotated[
            str,
            Field(min_length=1, description="Virtual POSIX path to edit."),
        ],
        old_text: Annotated[
            str,
            Field(min_length=1, description="Exact text that must already exist."),
        ],
        new_text: Annotated[str, Field(description="Replacement text.")],
        runtime: ToolRuntime,
        replace_all: Annotated[
            bool,
            Field(
                description=(
                    "Replace every occurrence when true. When false, old_text must "
                    "occur exactly once."
                )
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Replace exact text in a virtual file.

        Args:
            path: Virtual POSIX file path.
            old_text: Exact text to replace.
            new_text: Replacement text.
            runtime: LangChain runtime that identifies the current thread.
            replace_all: Replace every occurrence when true.

        Returns:
            A structured result containing replacement metadata.
        """
        session_id = session_id_resolver(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.call_filesystem(
                session_id,
                "edit",
                {
                    "path": path,
                    "old_text": old_text,
                    "new_text": new_text,
                    "replace_all": replace_all,
                },
            )
        )

    @tool("delete_file")
    def delete_file(
        path: Annotated[
            str,
            Field(min_length=1, description="Virtual POSIX path to delete."),
        ],
        runtime: ToolRuntime,
        recursive: Annotated[
            bool,
            Field(
                description=(
                    "Delete directory contents when true. The default only deletes "
                    "files and empty directories."
                )
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Delete a virtual file or directory.

        Args:
            path: Virtual POSIX path to delete.
            runtime: LangChain runtime that identifies the current thread.
            recursive: Delete directory contents when true.

        Returns:
            A structured result containing the deleted path count.
        """
        session_id = session_id_resolver(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.call_filesystem(
                session_id,
                "delete",
                {"path": path, "recursive": recursive},
            )
        )

    @tool("list_files")
    def list_files(
        runtime: ToolRuntime,
        path: Annotated[
            str,
            Field(min_length=1, description="Virtual POSIX directory to list."),
        ] = "/",
        recursive: Annotated[
            bool,
            Field(description="Include nested entries when true."),
        ] = False,
        limit: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional maximum entries. The application enforces a hard maximum."
                ),
            ),
        ] = None,
        output_limit_chars: Annotated[
            int | None,
            Field(
                ge=0,
                description="Optional character limit for the structured result.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """List bounded virtual directory entries.

        Args:
            runtime: LangChain runtime that identifies the current thread.
            path: Virtual POSIX directory path.
            recursive: Include nested entries when true.
            limit: Optional requested entry count.
            output_limit_chars: Optional result character limit.

        Returns:
            A structured result containing entries and scan metadata.
        """
        session_id = session_id_resolver(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.call_filesystem(
                session_id,
                "ls",
                {"path": path, "recursive": recursive, "limit": limit},
                output_limit_chars,
            )
        )

    @tool("glob_files")
    def glob_files(
        pattern: Annotated[
            str,
            Field(
                min_length=1,
                description="POSIX glob pattern with *, ?, or **.",
            ),
        ],
        runtime: ToolRuntime,
        root: Annotated[
            str,
            Field(min_length=1, description="Virtual directory to scan."),
        ] = "/",
        limit: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional maximum matches. The application enforces a hard maximum."
                ),
            ),
        ] = None,
        output_limit_chars: Annotated[
            int | None,
            Field(
                ge=0,
                description="Optional character limit for the structured result.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Find bounded virtual paths with a glob pattern.

        Args:
            pattern: Relative or absolute POSIX glob pattern.
            runtime: LangChain runtime that identifies the current thread.
            root: Virtual directory to scan.
            limit: Optional requested match count.
            output_limit_chars: Optional result character limit.

        Returns:
            A structured result containing matches and scan metadata.
        """
        session_id = session_id_resolver(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.call_filesystem(
                session_id,
                "glob",
                {"pattern": pattern, "root": root, "limit": limit},
                output_limit_chars,
            )
        )

    @tool("grep_files")
    def grep_files(
        query: Annotated[
            str,
            Field(min_length=1, description="Literal text or regular expression."),
        ],
        runtime: ToolRuntime,
        root: Annotated[
            str,
            Field(min_length=1, description="Virtual directory to scan."),
        ] = "/",
        pattern: Annotated[
            str,
            Field(min_length=1, description="Glob pattern used to select files."),
        ] = "**/*",
        regex: Annotated[
            bool,
            Field(description="Treat query as a regular expression when true."),
        ] = False,
        case_sensitive: Annotated[
            bool,
            Field(description="Preserve letter case when true."),
        ] = True,
        limit: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "Optional maximum matches. The application enforces a hard maximum."
                ),
            ),
        ] = None,
        output_limit_chars: Annotated[
            int | None,
            Field(
                ge=0,
                description="Optional character limit for the structured result.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Search bounded virtual text files.

        Args:
            query: Literal text or regular expression to find.
            runtime: LangChain runtime that identifies the current thread.
            root: Virtual directory to scan.
            pattern: Glob pattern used to select files.
            regex: Treat query as a regular expression when true.
            case_sensitive: Preserve letter case when true.
            limit: Optional requested match count.
            output_limit_chars: Optional result character limit.

        Returns:
            A structured result containing line matches and scan metadata.
        """
        session_id = session_id_resolver(runtime)
        if session_id is None:
            return _missing_thread_id()
        return dict(
            kernel.call_filesystem(
                session_id,
                "grep",
                {
                    "query": query,
                    "root": root,
                    "pattern": pattern,
                    "regex": regex,
                    "case_sensitive": case_sensitive,
                    "limit": limit,
                },
                output_limit_chars,
            )
        )

    return [
        read_file,
        write_file,
        edit_file,
        delete_file,
        list_files,
        glob_files,
        grep_files,
    ]
