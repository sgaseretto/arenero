from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

FilesystemOperation = Literal[
    "read",
    "write",
    "edit",
    "delete",
    "ls",
    "glob",
    "grep",
]
type FilesystemValue = str | int | bool | None


@dataclass(frozen=True)
class FilesystemConfig:
    """Set hard limits for sandbox filesystem helper functions.

    Attributes:
        default_read_limit_chars: Default characters returned by ``fs_read``.
        max_read_limit_chars: Maximum characters returned by ``fs_read``.
        max_write_chars: Maximum characters accepted by write and edit operations.
        default_result_limit: Default entries returned by list, glob, and grep.
        max_result_limit: Maximum entries returned by list, glob, and grep.
        max_scan_entries: Maximum paths visited by one recursive operation.
        max_grep_chars: Maximum file characters searched by one grep operation.
        max_depth: Maximum directory depth visited by recursive operations.
        max_transfer_bytes: Maximum bytes uploaded or downloaded in one file.
    """

    default_read_limit_chars: int = 8_000
    max_read_limit_chars: int = 200_000
    max_write_chars: int = 1_000_000
    default_result_limit: int = 100
    max_result_limit: int = 1_000
    max_scan_entries: int = 5_000
    max_grep_chars: int = 2_000_000
    max_depth: int = 20
    max_transfer_bytes: int = 10_000_000

    def __post_init__(self) -> None:
        """Validate all filesystem limits."""
        values = (
            self.default_read_limit_chars,
            self.max_read_limit_chars,
            self.max_write_chars,
            self.default_result_limit,
            self.max_result_limit,
            self.max_scan_entries,
            self.max_grep_chars,
            self.max_depth,
            self.max_transfer_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("filesystem limits must be greater than 0")
        if self.default_read_limit_chars > self.max_read_limit_chars:
            raise ValueError("the default read limit must not exceed its maximum")
        if self.default_result_limit > self.max_result_limit:
            raise ValueError("the default result limit must not exceed its maximum")


def build_filesystem_prelude(config: FilesystemConfig) -> str:
    """Build the Monty source that defines bounded filesystem helpers.

    Args:
        config: Hard limits embedded in the helper source.

    Returns:
        Monty-compatible Python source code.
    """
    limits = (
        f"_WEBMONTY_FS_DEFAULT_READ_LIMIT = {config.default_read_limit_chars}\n"
        f"_WEBMONTY_FS_MAX_READ_LIMIT = {config.max_read_limit_chars}\n"
        f"_WEBMONTY_FS_MAX_WRITE = {config.max_write_chars}\n"
        f"_WEBMONTY_FS_DEFAULT_RESULTS = {config.default_result_limit}\n"
        f"_WEBMONTY_FS_MAX_RESULTS = {config.max_result_limit}\n"
        f"_WEBMONTY_FS_MAX_SCAN = {config.max_scan_entries}\n"
        f"_WEBMONTY_FS_MAX_GREP_CHARS = {config.max_grep_chars}\n"
        f"_WEBMONTY_FS_MAX_DEPTH = {config.max_depth}\n"
    )
    return limits + _FILESYSTEM_PRELUDE


def build_filesystem_call(
    operation: FilesystemOperation,
    arguments: Mapping[str, FilesystemValue],
) -> str:
    """Build a safe call to one sandbox filesystem helper.

    Args:
        operation: One of ``read``, ``write``, ``edit``, ``delete``, ``ls``,
            ``glob``, or ``grep``.
        arguments: Simple keyword arguments accepted by the selected helper.

    Returns:
        Monty-compatible call source.
    """
    function_name, allowed_arguments = {
        "read": ("fs_read", {"path", "offset", "limit"}),
        "write": (
            "fs_write",
            {"path", "content", "append", "create_parents"},
        ),
        "edit": (
            "fs_edit",
            {"path", "old_text", "new_text", "replace_all"},
        ),
        "delete": ("fs_delete", {"path", "recursive"}),
        "ls": ("fs_ls", {"path", "recursive", "limit"}),
        "glob": ("fs_glob", {"pattern", "root", "limit"}),
        "grep": (
            "fs_grep",
            {
                "query",
                "root",
                "pattern",
                "regex",
                "case_sensitive",
                "limit",
            },
        ),
    }[operation]
    unexpected = set(arguments) - allowed_arguments
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"unexpected {operation} arguments: {names}")
    serialized = ", ".join(
        f"{name}={_filesystem_literal(value)}" for name, value in arguments.items()
    )
    return f"{function_name}({serialized})"


def _filesystem_literal(value: FilesystemValue) -> str:
    """Convert one supported tool value to Python source.

    Args:
        value: String, integer, Boolean, or ``None`` value.

    Returns:
        A safe Python literal.
    """
    if type(value) not in (str, int, bool) and value is not None:
        raise TypeError(f"unsupported filesystem argument: {type(value).__name__}")
    return repr(value)


_FILESYSTEM_PRELUDE = r'''
from pathlib import Path as _WebMontyPath
import re as _webmonty_re


def _webmonty_fs_limit(value, default, maximum):
    """Return a nonnegative request bounded by a hard maximum.

    Args:
        value: Requested limit or None.
        default: Value used when no limit is requested.
        maximum: Hard application maximum.

    Returns:
        Effective limit.
    """
    effective = default if value is None else value
    if effective < 0:
        raise ValueError("limit must be at least 0")
    return min(effective, maximum)


def _webmonty_fs_walk(root, recursive):
    """Collect bounded path metadata below one virtual directory.

    Args:
        root: Virtual directory to inspect.
        recursive: Include nested entries when true.

    Returns:
        Entries, truncation status, and scanned entry count.
    """
    pending = [(_WebMontyPath(root), 0)]
    entries = []
    scanned = 0
    truncated = False

    while pending:
        current, depth = pending.pop()
        for child in current.iterdir():
            if scanned >= _WEBMONTY_FS_MAX_SCAN:
                return entries, True, scanned

            scanned += 1
            if child.is_symlink():
                kind = "symlink"
            elif child.is_dir():
                kind = "directory"
            else:
                kind = "file"

            entries.append({"path": str(child), "type": kind})
            if recursive and kind == "directory":
                if depth < _WEBMONTY_FS_MAX_DEPTH:
                    pending.append((child, depth + 1))
                else:
                    truncated = True

    return entries, truncated, scanned


def _webmonty_fs_relative(path, root):
    """Return a path relative to one virtual root.

    Args:
        path: Virtual path to shorten.
        root: Virtual root prefix.

    Returns:
        Relative virtual path.
    """
    prefix = str(_WebMontyPath(root)).rstrip("/")
    if prefix:
        return path[len(prefix):].lstrip("/")
    return path.lstrip("/")


def _webmonty_fs_glob_regex(pattern):
    """Translate a small POSIX glob pattern to a regular expression.

    Args:
        pattern: Pattern containing *, ?, or ** wildcards.

    Returns:
        Anchored regular expression text.
    """
    output = "^"
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                if index < len(pattern) and pattern[index] == "/":
                    output += "(?:.*/)?"
                    index += 1
                else:
                    output += ".*"
                continue
            output += "[^/]*"
        elif character == "?":
            output += "[^/]"
        elif character in ".^$+{}[]|()\\":
            output += "\\" + character
        else:
            output += character
        index += 1
    return output + "$"


def _webmonty_fs_match(path, pattern):
    """Return whether a virtual path matches a glob pattern.

    Args:
        path: Virtual path to test.
        pattern: Pattern containing *, ?, or ** wildcards.

    Returns:
        True when the path matches.
    """
    return _webmonty_re.fullmatch(_webmonty_fs_glob_regex(pattern), path) is not None


def fs_read(path, offset=0, limit=None):
    """Read a bounded text segment from a virtual file.

    Args:
        path: Virtual file path.
        offset: Zero-based character offset.
        limit: Requested characters, capped by the application maximum.

    Returns:
        Requested text segment.
    """
    if offset < 0:
        raise ValueError("offset must be at least 0")
    effective = _webmonty_fs_limit(
        limit,
        _WEBMONTY_FS_DEFAULT_READ_LIMIT,
        _WEBMONTY_FS_MAX_READ_LIMIT,
    )
    text = _WebMontyPath(path).read_text()
    return text[offset:offset + effective]


def fs_write(path, content, append=False, create_parents=False):
    """Write bounded text to a virtual file.

    Args:
        path: Virtual file path.
        content: Text to write.
        append: Append when true; replace the file when false.
        create_parents: Create missing parent directories when true.

    Returns:
        Number of written characters.
    """
    if len(content) > _WEBMONTY_FS_MAX_WRITE:
        raise ValueError("content exceeds the filesystem write limit")
    target = _WebMontyPath(path)
    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    if append:
        return target.append_text(content)
    return target.write_text(content)


def fs_edit(path, old_text, new_text, replace_all=False):
    """Replace exact text in a virtual file.

    Args:
        path: Virtual file path.
        old_text: Exact text that must already exist.
        new_text: Replacement text.
        replace_all: Replace every occurrence when true. Otherwise exactly one
            occurrence must exist.

    Returns:
        Replacement count and final character count.
    """
    if old_text == "":
        raise ValueError("old_text must not be empty")
    target = _WebMontyPath(path)
    text = target.read_text()
    occurrences = text.count(old_text)
    if occurrences == 0:
        raise ValueError("old_text was not found")
    if not replace_all and occurrences != 1:
        raise ValueError("old_text must occur exactly once unless replace_all is true")

    if replace_all:
        updated = text.replace(old_text, new_text)
        replacements = occurrences
    else:
        updated = text.replace(old_text, new_text, 1)
        replacements = 1

    if len(updated) > _WEBMONTY_FS_MAX_WRITE:
        raise ValueError("edited content exceeds the filesystem write limit")
    target.write_text(updated)
    return {"replacements": replacements, "characters": len(updated)}


def fs_delete(path, recursive=False):
    """Delete a virtual file or directory.

    Args:
        path: Virtual path to delete. The virtual root cannot be deleted.
        recursive: Delete directory contents when true.

    Returns:
        Number of deleted paths.
    """
    target = _WebMontyPath(path)
    if str(target) == "/":
        raise ValueError("the virtual root cannot be deleted")
    if not target.is_dir() or target.is_symlink():
        target.unlink()
        return 1
    if not recursive:
        target.rmdir()
        return 1

    entries, truncated, scanned = _webmonty_fs_walk(target, True)
    if truncated:
        raise ValueError("recursive delete exceeds the filesystem scan limit")

    index = len(entries) - 1
    while index >= 0:
        entry = entries[index]
        child = _WebMontyPath(entry["path"])
        if entry["type"] == "directory":
            child.rmdir()
        else:
            child.unlink()
        index -= 1
    target.rmdir()
    return len(entries) + 1


def fs_ls(path="/", recursive=False, limit=None):
    """List bounded virtual directory entries.

    Args:
        path: Virtual directory path.
        recursive: Include nested entries when true.
        limit: Requested entries, capped by the application maximum.

    Returns:
        Entries, scan count, and truncation status.
    """
    effective = _webmonty_fs_limit(
        limit,
        _WEBMONTY_FS_DEFAULT_RESULTS,
        _WEBMONTY_FS_MAX_RESULTS,
    )
    entries, scan_truncated, scanned = _webmonty_fs_walk(path, recursive)
    return {
        "entries": entries[:effective],
        "scanned": scanned,
        "truncated": scan_truncated or len(entries) > effective,
    }


def fs_glob(pattern, root="/", limit=None):
    """Find bounded virtual paths with a POSIX glob pattern.

    Args:
        pattern: Relative or absolute pattern with ``*``, ``?``, and ``**``.
        root: Virtual directory to scan.
        limit: Requested matches, capped by the application maximum.

    Returns:
        Matching paths, scan count, and truncation status.
    """
    effective = _webmonty_fs_limit(
        limit,
        _WEBMONTY_FS_DEFAULT_RESULTS,
        _WEBMONTY_FS_MAX_RESULTS,
    )
    entries, scan_truncated, scanned = _webmonty_fs_walk(root, True)
    matches = []
    truncated = scan_truncated
    for entry in entries:
        path = entry["path"]
        candidate = path if pattern.startswith("/") else _webmonty_fs_relative(path, root)
        if _webmonty_fs_match(candidate, pattern):
            if len(matches) >= effective:
                truncated = True
                break
            matches.append(path)
    return {"matches": matches, "scanned": scanned, "truncated": truncated}


def fs_grep(
    query,
    root="/",
    pattern="**/*",
    regex=False,
    case_sensitive=True,
    limit=None,
):
    """Search bounded virtual text files.

    Args:
        query: Literal text or regular expression to find.
        root: Virtual directory to scan.
        pattern: Glob pattern used to select files.
        regex: Treat query as a regular expression when true.
        case_sensitive: Preserve case when true.
        limit: Requested matches, capped by the application maximum.

    Returns:
        Line matches and scan metadata.
    """
    effective = _webmonty_fs_limit(
        limit,
        _WEBMONTY_FS_DEFAULT_RESULTS,
        _WEBMONTY_FS_MAX_RESULTS,
    )
    entries, scan_truncated, scanned = _webmonty_fs_walk(root, True)
    matcher = None
    if regex:
        flags = 0 if case_sensitive else _webmonty_re.IGNORECASE
        matcher = _webmonty_re.compile(query, flags)

    matches = []
    files_scanned = 0
    characters_scanned = 0
    truncated = scan_truncated
    for entry in entries:
        if entry["type"] != "file":
            continue
        path = entry["path"]
        relative = _webmonty_fs_relative(path, root)
        if not _webmonty_fs_match(relative, pattern):
            continue
        if characters_scanned >= _WEBMONTY_FS_MAX_GREP_CHARS:
            truncated = True
            break

        text = _WebMontyPath(path).read_text()
        remaining = _WEBMONTY_FS_MAX_GREP_CHARS - characters_scanned
        if len(text) > remaining:
            text = text[:remaining]
            truncated = True
        files_scanned += 1
        characters_scanned += len(text)

        line_number = 0
        for line in text.splitlines():
            line_number += 1
            if regex:
                found = matcher.search(line) is not None
            elif case_sensitive:
                found = query in line
            else:
                found = query.lower() in line.lower()
            if found:
                if len(matches) >= effective:
                    return {
                        "matches": matches,
                        "files_scanned": files_scanned,
                        "characters_scanned": characters_scanned,
                        "truncated": True,
                    }
                matches.append({"path": path, "line": line_number, "text": line})

    return {
        "matches": matches,
        "files_scanned": files_scanned,
        "characters_scanned": characters_scanned,
        "truncated": truncated,
    }
'''
