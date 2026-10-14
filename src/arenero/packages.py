"""Create, validate, test, and export small Monty-compatible source packages."""

from __future__ import annotations

import hashlib
import io
import json
import keyword
import posixpath
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from pydantic_monty import MontyError

from .hosted_tools import HostedToolSet
from .kernel import PersistentMontyKernel
from .modules import VirtualModuleConfig, VirtualModuleError, VirtualModuleResolver

PACKAGE_MANIFEST = "arenero-package.json"
PACKAGE_FORMAT = 1
_SOURCE_SUFFIXES = {".json", ".md", ".py", ".txt"}
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

MONTY_PACKAGE_INSTRUCTIONS = """Create dependency-free Monty source packages only.
Use package_init first, then edit files with the normal virtual filesystem tools.
Keep arenero-package.json dependencies empty. Use only Monty's Python subset and
static imports. Run package_check after edits, then package_test. Package tests run
verify.py in a fresh temporary Monty session, so they cannot use REPL state. The
file must set the public variable passed to True after its assertions. Use
package_export only after checks and tests pass. It creates a source ZIP, not a
wheel or Python sdist. Arenero does not run pip, pytest, setuptools, build
backends, or third-party packages.
Binary non-Python assets are allowed only under assets/ when the host enables
allow_binary_assets. Assets use the existing file, byte, and ZIP limits; they
do not add importlib.resources or a package installer.
"""


@dataclass(frozen=True, slots=True)
class PackageConfig:
    """Set package roots and hard source limits.

    Attributes:
        allowed_root: Absolute virtual directory that contains package projects.
        max_files: Maximum included files, including assets.
        max_file_bytes: Maximum bytes in one included file.
        max_total_bytes: Maximum total bytes in included files.
        max_archive_bytes: Maximum exported ZIP size.
        allow_binary_assets: Allow non-Python binary files under ``assets/`` when true.
    """

    allowed_root: str = "/workspace"
    max_files: int = 100
    max_file_bytes: int = 256_000
    max_total_bytes: int = 2_000_000
    max_archive_bytes: int = 2_500_000
    allow_binary_assets: bool = False

    def __post_init__(self) -> None:
        """Validate package paths and limits."""

        normalized = posixpath.normpath(self.allowed_root)
        if not normalized.startswith("/") or normalized == "/":
            raise ValueError("allowed_root must be an absolute path other than /")
        if normalized != self.allowed_root.rstrip("/"):
            raise ValueError("allowed_root must be a normalized virtual path")
        limits = (
            self.max_files,
            self.max_file_bytes,
            self.max_total_bytes,
            self.max_archive_bytes,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("package limits must be greater than 0")


class MontyPackageManager:
    """Manage one package workflow through a synchronous Arenero kernel."""

    def __init__(
        self,
        kernel: PersistentMontyKernel,
        config: PackageConfig | None = None,
    ) -> None:
        """Create a package manager.

        Args:
            kernel: Kernel that owns the virtual filesystem and Monty workers.
            config: Optional package path and size limits.
        """

        self.kernel = kernel
        self.config = config or PackageConfig()

    def init(
        self,
        name: str,
        root: str | None = None,
        version: str = "0.1.0",
    ) -> dict[str, Any]:
        """Create a working dependency-free package scaffold.

        Args:
            name: Python identifier used for the package directory.
            root: Optional absolute project root below the configured root.
            version: Small source-package version label.

        Returns:
            Structured initialization result with created paths.
        """

        operation = "init"
        if not name.isidentifier() or keyword.iskeyword(name):
            return _error(
                operation,
                "invalid_package_name",
                "package name must be a non-keyword Python identifier",
                "Use letters, digits, and underscores, and do not start with a digit.",
            )
        if _VERSION_PATTERN.fullmatch(version) is None:
            return _error(
                operation,
                "invalid_package_version",
                "version must be 1 to 64 simple ASCII characters",
                "Use letters, digits, dots, underscores, or hyphens.",
            )
        try:
            project_root = self._root(root or f"{self.config.allowed_root}/{name}")
        except ValueError as exc:
            return _error(
                operation,
                "invalid_package_root",
                str(exc),
                f"Choose a project below {self.config.allowed_root}.",
            )
        manifest_path = f"{project_root}/{PACKAGE_MANIFEST}"
        if self._exists(manifest_path):
            return _error(
                operation,
                "package_exists",
                f"package manifest already exists at {manifest_path}",
                "Choose another root or edit the existing package.",
                root=project_root,
            )

        manifest = {
            "format": PACKAGE_FORMAT,
            "name": name,
            "version": version,
            "entry": f"{name}/__init__.py",
            "verify": "verify.py",
            "dependencies": [],
        }
        files = {
            PACKAGE_MANIFEST: json.dumps(manifest, indent=2) + "\n",
            f"{name}/__init__.py": "from .core import identity\n",
            f"{name}/core.py": (
                "def identity(value):\n"
                '    """Return the supplied value."""\n'
                "    return value\n"
            ),
            "verify.py": (
                f"from {name} import identity\n\n"
                "assert identity(42) == 42\n"
                "passed = True\n"
            ),
            "README.md": (
                f"# {name}\n\n"
                "A dependency-free source package for Pydantic Monty and Arenero.\n"
            ),
        }
        created: list[str] = []
        try:
            for relative, content in files.items():
                path = f"{project_root}/{relative}"
                self.kernel.upload_file(path, content.encode())
                created.append(path)
        except (MontyError, TypeError, ValueError) as exc:
            return _error(
                operation,
                "package_write_failed",
                str(exc),
                "Inspect created_files, fix filesystem access, and retry in a new root.",
                root=project_root,
                created_files=created,
            )
        checked = self.check(project_root)
        if not checked["ok"]:
            return _error(
                operation,
                "scaffold_check_failed",
                "the generated package did not pass package_check",
                "Inspect check and fix the configured filesystem or limits.",
                root=project_root,
                created_files=created,
                check=checked,
            )
        return {
            "ok": True,
            "operation": operation,
            "root": project_root,
            "manifest": manifest,
            "created_files": created,
            "revision": checked["revision"],
            "next": ["edit files", "package_check", "package_test", "package_export"],
        }

    def check(
        self,
        root: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Validate a package manifest, source tree, and static module graphs.

        Args:
            root: Absolute virtual package root.
            expected_revision: Optional source revision that must still match.

        Returns:
            Structured validation result, file inventory, and source revision.
        """

        operation = "check"
        try:
            project_root = self._root(root)
            inventory = self._inventory(project_root)
            manifest = self._manifest(project_root)
            entry = self._relative_path(project_root, manifest.get("entry"), "entry")
            verify = self._relative_path(project_root, manifest.get("verify"), "verify")
            self._validate_manifest(manifest, inventory)
            self._validate_modules(project_root, inventory["python_files"])
            if entry not in inventory["files"]:
                raise PackageValidationError(
                    "missing_entry", f"entry file does not exist: {entry}"
                )
            if verify not in inventory["files"]:
                raise PackageValidationError(
                    "missing_verify", f"verification file does not exist: {verify}"
                )
        except PackageValidationError as exc:
            return _error(
                operation,
                exc.code,
                str(exc),
                exc.hint,
                root=root,
            )
        except (MontyError, TypeError, UnicodeDecodeError, ValueError) as exc:
            return _error(
                operation,
                "package_check_failed",
                str(exc),
                "Check the package root, virtual filesystem, and source limits.",
                root=root,
            )

        revision = inventory["revision"]
        if expected_revision is not None and revision != expected_revision:
            return _error(
                operation,
                "package_revision_conflict",
                "package files changed after the expected revision",
                "Run package_check again before testing or exporting.",
                root=project_root,
                expected_revision=expected_revision,
                revision=revision,
            )
        return {
            "ok": True,
            "operation": operation,
            "root": project_root,
            "manifest": manifest,
            "entry_path": entry,
            "verify_path": verify,
            "files": inventory["files"],
            "python_files": inventory["python_files"],
            "file_count": len(inventory["files"]),
            "total_bytes": inventory["total_bytes"],
            "revision": revision,
        }

    def test(
        self,
        root: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Run verify.py in a new temporary Monty session.

        Args:
            root: Absolute virtual package root.
            expected_revision: Optional source revision that must still match.

        Returns:
            Structured validation and Monty execution results.
        """

        operation = "test"
        checked = self.check(root, expected_revision)
        if not checked["ok"]:
            return _error(
                operation,
                checked["error"]["code"],
                checked["error"]["message"],
                checked["error"]["hint"],
                root=root,
                check=checked,
            )
        session_id = f"_arenero_package_test_{uuid.uuid4().hex}"
        verified: dict[str, Any] = {}

        def _arenero_package_capture(passed: Any) -> Any:
            """Read the verifier flag before model output trimming.

            Args:
                passed: Public value exported by verify.py.

            Returns:
                The same value, to keep the public execution result unchanged.
            """
            verified["passed"] = passed
            return passed

        try:
            execution = self.kernel.execute_module(
                session_id,
                checked["verify_path"],
                code="_arenero_package_capture(passed=passed)",
                module_config=VirtualModuleConfig(roots=(checked["root"],)),
                state_policy="never",
                restart=True,
                hosted_tools=HostedToolSet.from_callables([_arenero_package_capture]),
            )
        finally:
            self.kernel.reset(session_id)
        if not execution["ok"]:
            return _error(
                operation,
                "package_verification_failed",
                "verify.py failed in a fresh Monty session",
                "Inspect execution.error, edit the package, then run package_check again.",
                root=checked["root"],
                revision=checked["revision"],
                execution=dict(execution),
            )
        if verified.get("passed") is not True:
            return _error(
                operation,
                "invalid_verification_result",
                "verify.py must export passed with the boolean value True",
                "Set passed = True after all assertions.",
                root=checked["root"],
                revision=checked["revision"],
                execution=dict(execution),
            )
        return {
            "ok": True,
            "operation": operation,
            "root": checked["root"],
            "revision": checked["revision"],
            "passed": True,
            "execution": dict(execution),
        }

    def export(
        self,
        root: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Create a deterministic source ZIP in the package dist directory.

        Args:
            root: Absolute virtual package root.
            expected_revision: Optional source revision that must still match.

        Returns:
            Structured archive path, size, and source revision.
        """

        operation = "export"
        tested = self.test(root, expected_revision)
        if not tested["ok"]:
            return _error(
                operation,
                tested["error"]["code"],
                tested["error"]["message"],
                tested["error"]["hint"],
                root=root,
                test=tested,
            )
        checked = self.check(root, tested["revision"])
        manifest = checked["manifest"]
        prefix = f"{manifest['name']}-{manifest['version']}"
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            for path in checked["files"]:
                relative = posixpath.relpath(path, checked["root"])
                if relative == "dist" or relative.startswith("dist/"):
                    continue
                content = self.kernel.download_file(
                    path, max_bytes=self.config.max_file_bytes
                )
                info = zipfile.ZipInfo(
                    f"{prefix}/{relative}", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                output.writestr(info, content)
        data = archive.getvalue()
        if len(data) > self.config.max_archive_bytes:
            return _error(
                operation,
                "package_archive_too_large",
                "source ZIP exceeds max_archive_bytes",
                "Remove files or increase the host-configured archive limit.",
                root=checked["root"],
                size_bytes=len(data),
            )
        confirmed = self.check(checked["root"], checked["revision"])
        if not confirmed["ok"]:
            return _error(
                operation,
                confirmed["error"]["code"],
                confirmed["error"]["message"],
                confirmed["error"]["hint"],
                root=checked["root"],
                check=confirmed,
            )
        archive_path = f"{checked['root']}/dist/{prefix}-arenero.zip"
        try:
            self.kernel.upload_file(archive_path, data)
        except (MontyError, TypeError, ValueError) as exc:
            return _error(
                operation,
                "package_export_failed",
                str(exc),
                "Check virtual filesystem write access and transfer limits.",
                root=checked["root"],
            )
        return {
            "ok": True,
            "operation": operation,
            "root": checked["root"],
            "revision": checked["revision"],
            "archive_path": archive_path,
            "size_bytes": len(data),
            "format": "arenero-source-zip",
        }

    def instructions(self) -> str:
        """Return package workflow guidance for a human or model.

        Returns:
            Prompt-ready package instructions and active limits.
        """

        return (
            MONTY_PACKAGE_INSTRUCTIONS
            + "\nActive package limits:\n"
            + f"- allowed_root: {self.config.allowed_root}\n"
            + f"- max_files: {self.config.max_files}\n"
            + f"- max_file_bytes: {self.config.max_file_bytes}\n"
            + f"- max_total_bytes: {self.config.max_total_bytes}\n"
            + f"- max_archive_bytes: {self.config.max_archive_bytes}\n"
        )

    def _root(self, root: str) -> str:
        """Normalize and confine one package root.

        Args:
            root: Requested absolute virtual root.

        Returns:
            Normalized root below the configured allowed root.

        Raises:
            ValueError: If the root is invalid or outside the allowed directory.
        """

        if not isinstance(root, str) or "\x00" in root or not root.startswith("/"):
            raise ValueError("package root must be an absolute virtual path")
        normalized = posixpath.normpath(root)
        allowed = self.config.allowed_root
        if normalized == allowed or not normalized.startswith(allowed + "/"):
            raise ValueError(f"package root must be below {allowed}")
        return normalized

    def _relative_path(self, root: str, value: Any, field: str) -> str:
        """Resolve one safe manifest path below the project root.

        Args:
            root: Normalized package root.
            value: Relative manifest value.
            field: Manifest field name used in errors.

        Returns:
            Absolute virtual path.

        Raises:
            PackageValidationError: If the value escapes the package root.
        """

        if not isinstance(value, str) or not value or value.startswith("/"):
            raise PackageValidationError(
                f"invalid_{field}", f"manifest {field} must be a relative path"
            )
        relative = PurePosixPath(value)
        if ".." in relative.parts or relative.suffix != ".py":
            raise PackageValidationError(
                f"invalid_{field}", f"manifest {field} must name a .py file"
            )
        return f"{root}/{relative.as_posix()}"

    def _exists(self, path: str) -> bool:
        """Return whether one virtual file is readable.

        Args:
            path: Absolute virtual file path.

        Returns:
            True when the file exists and is readable.
        """

        try:
            self.kernel.download_file(path, max_bytes=self.config.max_file_bytes)
        except (MontyError, TypeError, ValueError):
            return False
        return True

    def _inventory(self, root: str) -> dict[str, Any]:
        """Read and hash one bounded source tree.

        Args:
            root: Normalized package root.

        Returns:
            File paths, Python paths, total bytes, and deterministic revision.

        Raises:
            PackageValidationError: If file or tree limits are exceeded.
        """

        try:
            result = self.kernel._call_filesystem_raw(
                "ls",
                {
                    "path": root,
                    "recursive": True,
                    "limit": self.config.max_files * 4 + 1,
                },
            )
        except (MontyError, TypeError, ValueError) as exc:
            raise PackageValidationError(
                "package_not_found",
                f"cannot list package root: {root}",
                "Run package_init or choose an existing package root.",
            ) from exc
        if not isinstance(result, dict):
            raise PackageValidationError(
                "invalid_inventory", "filesystem listing returned invalid data"
            )
        entries = result.get("entries", [])
        if result.get("truncated"):
            raise PackageValidationError(
                "too_many_package_files",
                "package directory listing exceeds its bounded scan limit",
            )
        files: list[str] = []
        python_files: list[str] = []
        total_bytes = 0
        digest = hashlib.sha256()
        for entry in sorted(entries, key=lambda item: str(item.get("path", ""))):
            path = entry.get("path")
            kind = entry.get("type")
            if not isinstance(path, str):
                raise PackageValidationError(
                    "invalid_inventory", "filesystem entry has no valid path"
                )
            if kind == "directory":
                continue
            if kind != "file":
                raise PackageValidationError(
                    "unsupported_package_file",
                    f"package contains unsupported {kind}: {path}",
                    "Use regular source files and directories only.",
                )
            relative = posixpath.relpath(path, root)
            if relative == "dist" or relative.startswith("dist/"):
                continue
            if any(part.startswith(".") for part in PurePosixPath(relative).parts):
                raise PackageValidationError(
                    "hidden_package_file",
                    f"hidden package paths are not supported: {path}",
                )
            suffix = PurePosixPath(path).suffix
            binary_asset = (
                self.config.allow_binary_assets
                and relative.startswith("assets/")
                and suffix != ".py"
            )
            if suffix not in _SOURCE_SUFFIXES and not binary_asset:
                raise PackageValidationError(
                    "unsupported_package_file",
                    f"unsupported source file type: {path}",
                    "Use .py, .json, .md, or .txt source files.",
                )
            if len(files) >= self.config.max_files:
                raise PackageValidationError(
                    "too_many_package_files",
                    f"package exceeds the {self.config.max_files}-file limit",
                )
            try:
                content = self.kernel.download_file(
                    path, max_bytes=self.config.max_file_bytes
                )
            except (MontyError, TypeError, ValueError) as exc:
                raise PackageValidationError(
                    "package_file_too_large",
                    f"cannot read bounded source file {path}: {exc}",
                    "Reduce the file below max_file_bytes.",
                ) from exc
            if not binary_asset:
                try:
                    content.decode()
                except UnicodeDecodeError as exc:
                    raise PackageValidationError(
                        "non_utf8_package_file",
                        f"source file is not UTF-8: {path}",
                    ) from exc
            total_bytes += len(content)
            if total_bytes > self.config.max_total_bytes:
                raise PackageValidationError(
                    "package_source_too_large",
                    "package exceeds max_total_bytes",
                )
            digest.update(relative.encode())
            digest.update(b"\x00")
            digest.update(content)
            digest.update(b"\x00")
            files.append(path)
            if suffix == ".py":
                python_files.append(path)
        if not files:
            raise PackageValidationError(
                "empty_package", "package source tree contains no files"
            )
        return {
            "files": files,
            "python_files": python_files,
            "total_bytes": total_bytes,
            "revision": digest.hexdigest(),
        }

    def _manifest(self, root: str) -> dict[str, Any]:
        """Read one package manifest.

        Args:
            root: Normalized package root.

        Returns:
            Parsed manifest mapping.

        Raises:
            PackageValidationError: If the manifest is absent or invalid.
        """

        path = f"{root}/{PACKAGE_MANIFEST}"
        try:
            raw = self.kernel.download_file(path, max_bytes=self.config.max_file_bytes)
            value = json.loads(raw)
        except (MontyError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise PackageValidationError(
                "invalid_package_manifest",
                f"cannot read {PACKAGE_MANIFEST}: {exc}",
                "Create valid UTF-8 JSON with package_init or fix the manifest.",
            ) from exc
        if not isinstance(value, dict):
            raise PackageValidationError(
                "invalid_package_manifest", "package manifest must be a JSON object"
            )
        return value

    def _validate_manifest(
        self,
        manifest: dict[str, Any],
        inventory: dict[str, Any],
    ) -> None:
        """Validate manifest values and dependency policy.

        Args:
            manifest: Parsed package manifest.
            inventory: Current bounded source inventory.

        Raises:
            PackageValidationError: If required manifest data is invalid.
        """

        if (
            type(manifest.get("format")) is not int
            or manifest.get("format") != PACKAGE_FORMAT
        ):
            raise PackageValidationError(
                "unsupported_package_format",
                f"manifest format must be {PACKAGE_FORMAT}",
            )
        name = manifest.get("name")
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
        ):
            raise PackageValidationError(
                "invalid_package_name", "manifest name must be a Python identifier"
            )
        version = manifest.get("version")
        if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
            raise PackageValidationError(
                "invalid_package_version", "manifest version is invalid"
            )
        dependencies = manifest.get("dependencies")
        if dependencies != []:
            raise PackageValidationError(
                "package_dependencies_not_supported",
                "Monty-compatible packages must have an empty dependencies list",
                "Remove dependencies and use only Monty's documented subset.",
            )
        if manifest.get("entry") != f"{name}/__init__.py":
            raise PackageValidationError(
                "invalid_entry",
                f"manifest entry must be {name}/__init__.py",
            )
        if manifest.get("verify") != "verify.py":
            raise PackageValidationError(
                "invalid_verify",
                "manifest verify must be verify.py",
            )
        package_init = next(
            (
                path
                for path in inventory["python_files"]
                if path.endswith(f"/{name}/__init__.py")
            ),
            None,
        )
        if package_init is None:
            raise PackageValidationError(
                "missing_package_init",
                f"package directory {name} must contain __init__.py",
            )

    def _validate_modules(self, root: str, python_files: list[str]) -> None:
        """Resolve every Python source file as a bounded static graph.

        Args:
            root: Normalized package project root.
            python_files: Absolute package Python paths.

        Raises:
            PackageValidationError: If any syntax or import graph is unsupported.
        """

        def load(path: str) -> str | None:
            """Load one UTF-8 virtual source candidate.

            Args:
                path: Absolute virtual source path.

            Returns:
                Source text, or None when the file is unavailable.
            """

            try:
                return self.kernel.download_file(
                    path, max_bytes=self.config.max_file_bytes
                ).decode()
            except (MontyError, TypeError, UnicodeDecodeError, ValueError):
                return None

        resolver = VirtualModuleResolver(load, VirtualModuleConfig(roots=(root,)))
        for path in python_files:
            try:
                resolver.bundle(path)
            except (SyntaxError, VirtualModuleError, ValueError) as exc:
                raise PackageValidationError(
                    "unsupported_package_module",
                    f"cannot resolve {path}: {exc}",
                    "Use supported static imports and remove cycles or missing modules.",
                ) from exc


class PackageValidationError(ValueError):
    """Report one stable package validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        hint: str = "Fix the package source and run package_check again.",
    ) -> None:
        """Create a package validation error.

        Args:
            code: Stable machine-readable error code.
            message: Human-readable failure description.
            hint: Suggested next action.
        """

        super().__init__(message)
        self.code = code
        self.hint = hint


def _error(
    operation: str,
    code: str,
    message: str,
    hint: str,
    **data: Any,
) -> dict[str, Any]:
    """Build one structured package operation error.

    Args:
        operation: Package operation that failed.
        code: Stable machine-readable error code.
        message: Human-readable failure description.
        hint: Suggested next action.
        **data: Additional safe operation metadata.

    Returns:
        Structured non-retryable package error.
    """

    return {
        "ok": False,
        "operation": operation,
        **data,
        "error": {
            "type": "MontyPackageError",
            "code": code,
            "message": message,
            "hint": hint,
            "retryable": False,
            "recovery": "revise_code",
        },
    }
