"""Small static virtual-module resolver for Monty source files."""

from __future__ import annotations

import ast
import hashlib
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

MONTY_STDLIB_MODULES = frozenset(
    {
        "asyncio",
        "base64",
        "binascii",
        "collections",
        "dataclasses",
        "datetime",
        "functools",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "sys",
        "typing",
        "unicodedata",
    }
)


class VirtualModuleError(ValueError):
    """Report an invalid or unsupported virtual module graph."""


@dataclass(frozen=True)
class VirtualModuleConfig:
    """Set limits and roots for static virtual modules.

    Attributes:
        roots: Absolute virtual directories allowed for module files.
        aliases: Module-name prefixes mapped to absolute virtual directories.
        max_modules: Maximum files in one module graph.
        max_depth: Maximum import depth, including the entry file.
        max_source_bytes: Maximum combined UTF-8 source size.
    """

    roots: tuple[str, ...] = ("/",)
    aliases: Mapping[str, str] = field(default_factory=dict)
    max_modules: int = 32
    max_depth: int = 8
    max_source_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate roots, aliases, and positive limits."""

        if not self.roots or any(not path.startswith("/") for path in self.roots):
            raise ValueError("virtual module roots must be absolute")
        if any(not name.isidentifier() for name in self.aliases):
            raise ValueError("virtual module aliases must be Python identifiers")
        if any(not path.startswith("/") for path in self.aliases.values()):
            raise ValueError("virtual module alias paths must be absolute")
        if min(self.max_modules, self.max_depth, self.max_source_bytes) < 1:
            raise ValueError("virtual module limits must be greater than 0")


@dataclass(frozen=True)
class ModuleBundle:
    """Hold generated Monty code and resolved graph metadata.

    Attributes:
        code: Complete source submitted as one Monty feed.
        entry_path: Normalized entry file path.
        entry_name: Global module-object name made available to follow-up code.
        files: Dependency-first resolved virtual file paths.
        exports: Public entry-module names copied into the REPL globals.
    """

    code: str
    entry_path: str
    entry_name: str
    files: tuple[str, ...]
    exports: tuple[str, ...]


SourceLoader = Callable[[str], str | None]


class VirtualModuleResolver:
    """Bundle a limited static module graph from a virtual filesystem."""

    def __init__(
        self,
        loader: SourceLoader,
        config: VirtualModuleConfig | None = None,
    ) -> None:
        """Create a resolver.

        Args:
            loader: Function that returns UTF-8 source or ``None`` for no file.
            config: Optional roots, aliases, and graph limits.
        """

        self.loader = loader
        self.config = config or VirtualModuleConfig()
        self._sources: dict[str, str] = {}
        self._compiled: dict[str, str] = {}
        self._exports: dict[str, tuple[str, ...]] = {}
        self._order: list[str] = []
        self._active: list[str] = []
        self._total_bytes = 0

    def bundle(self, entry_path: str, followup_code: str | None = None) -> ModuleBundle:
        """Resolve, isolate, and combine one virtual module graph.

        Args:
            entry_path: Absolute virtual ``.py`` file.
            followup_code: Optional code that uses entry exports in the same feed.

        Returns:
            Generated code plus graph metadata.

        Raises:
            VirtualModuleError: If resolution, syntax, or limits fail.
        """

        self._sources.clear()
        self._compiled.clear()
        self._exports.clear()
        self._order.clear()
        self._active.clear()
        self._total_bytes = 0
        entry = self._normalize_path(entry_path)
        if not entry.endswith(".py"):
            raise VirtualModuleError("entry module must use a .py file")
        self._require_allowed(entry)
        self._compile(entry, 1)
        entry_symbol = self._symbol(entry)
        entry_name = PurePosixPath(entry).stem
        if entry_name == "__init__":
            entry_name = PurePosixPath(entry).parent.name or "module"
        if not entry_name.isidentifier():
            entry_name = "module"
        exports = self._exports[entry]
        lines = [
            "class _AreneroVirtualModule:",
            "    pass",
            "",
            *(self._compiled[path] for path in self._order),
            f"{entry_name} = {entry_symbol}",
            *(f"{name} = {entry_symbol}.{name}" for name in exports),
        ]
        if followup_code is not None:
            lines.extend(("", followup_code))
        return ModuleBundle(
            code="\n".join(lines),
            entry_path=entry,
            entry_name=entry_name,
            files=tuple(self._order),
            exports=exports,
        )

    def _compile(self, path: str, depth: int) -> None:
        """Compile one module and its dependencies once.

        Args:
            path: Normalized virtual source path.
            depth: Current graph depth.
        """

        if path in self._compiled:
            return
        if path in self._active:
            cycle = " -> ".join([*self._active, path])
            raise VirtualModuleError(f"circular virtual import: {cycle}")
        if depth > self.config.max_depth:
            raise VirtualModuleError("virtual module import depth limit exceeded")
        if len(self._sources) >= self.config.max_modules:
            raise VirtualModuleError("virtual module count limit exceeded")
        source = self._sources.get(path)
        if source is None:
            source = self.loader(path)
        if source is None:
            raise VirtualModuleError(f"virtual module file does not exist: {path}")
        size = len(source.encode())
        self._total_bytes += size
        if self._total_bytes > self.config.max_source_bytes:
            raise VirtualModuleError("virtual module source byte limit exceeded")
        self._sources[path] = source
        self._active.append(path)
        tree = ast.parse(source, filename=path)
        self._reject_nested_imports(tree)
        body: list[ast.stmt] = []
        imported_names: list[str] = []
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                replacement, names = self._rewrite_import(statement, path, depth)
                body.extend(replacement)
                imported_names.extend(names)
            elif isinstance(statement, ast.ImportFrom):
                replacement, names = self._rewrite_from_import(statement, path, depth)
                body.extend(replacement)
                imported_names.extend(names)
            else:
                body.append(statement)
        exports = self._public_names(body, imported_names)
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        function_name = f"_arenero_load_{self._digest(path)}"
        module_symbol = self._symbol(path)
        source_lines = [f"def {function_name}():"]
        if body:
            for statement in body:
                rendered = ast.unparse(statement)
                source_lines.extend(f"    {line}" for line in rendered.splitlines())
        source_lines.append("    _module = _AreneroVirtualModule()")
        source_lines.extend(f"    _module.{name} = {name}" for name in exports)
        source_lines.append("    return _module")
        source_lines.append(f"{module_symbol} = {function_name}()")
        self._compiled[path] = "\n".join(source_lines) + "\n"
        self._exports[path] = exports
        self._order.append(path)
        self._active.pop()

    def _rewrite_import(
        self,
        statement: ast.Import,
        importer: str,
        depth: int,
    ) -> tuple[list[ast.stmt], list[str]]:
        """Keep Monty imports or replace simple virtual imports.

        Args:
            statement: Top-level import statement.
            importer: Importing virtual file.
            depth: Importer depth.

        Returns:
            Replacement statements and imported local names.
        """

        replacements: list[ast.stmt] = []
        names: list[str] = []
        for alias in statement.names:
            if alias.name.split(".", 1)[0] in MONTY_STDLIB_MODULES:
                replacements.append(ast.Import(names=[alias]))
                names.append(alias.asname or alias.name.split(".", 1)[0])
                continue
            if "." in alias.name and alias.asname is None:
                raise VirtualModuleError(
                    "dotted virtual imports require 'as NAME' in the minimal resolver"
                )
            dependency = self._resolve(importer, alias.name, 0)
            self._compile(dependency, depth + 1)
            local = alias.asname or alias.name
            replacements.append(
                ast.Assign(
                    targets=[ast.Name(id=local, ctx=ast.Store())],
                    value=ast.Name(id=self._symbol(dependency), ctx=ast.Load()),
                )
            )
            names.append(local)
        return replacements, names

    def _rewrite_from_import(
        self,
        statement: ast.ImportFrom,
        importer: str,
        depth: int,
    ) -> tuple[list[ast.stmt], list[str]]:
        """Keep Monty imports or bind names from one virtual module.

        Args:
            statement: Top-level from-import statement.
            importer: Importing virtual file.
            depth: Importer depth.

        Returns:
            Replacement statements and imported local names.
        """

        module = statement.module
        if (
            module
            and statement.level == 0
            and module.split(".", 1)[0] in MONTY_STDLIB_MODULES
        ):
            return [statement], [
                alias.asname or alias.name for alias in statement.names
            ]
        if module is None:
            raise VirtualModuleError("'from . import name' is not supported")
        if any(alias.name == "*" for alias in statement.names):
            raise VirtualModuleError("wildcard virtual imports are not supported")
        dependency = self._resolve(importer, module, statement.level)
        self._compile(dependency, depth + 1)
        available = set(self._exports[dependency])
        replacements: list[ast.stmt] = []
        names: list[str] = []
        for alias in statement.names:
            if alias.name not in available:
                raise VirtualModuleError(
                    f"{alias.name!r} is not a public export of {dependency}"
                )
            local = alias.asname or alias.name
            replacements.append(
                ast.Assign(
                    targets=[ast.Name(id=local, ctx=ast.Store())],
                    value=ast.Attribute(
                        value=ast.Name(id=self._symbol(dependency), ctx=ast.Load()),
                        attr=alias.name,
                        ctx=ast.Load(),
                    ),
                )
            )
            names.append(local)
        return replacements, names

    def _resolve(self, importer: str, name: str, level: int) -> str:
        """Resolve one module name to a readable virtual file.

        Args:
            importer: Importing file path.
            name: Dotted module name.
            level: Relative import level, or zero for absolute.

        Returns:
            Normalized readable ``.py`` or ``__init__.py`` path.
        """

        parts = name.split(".")
        if any(not part.isidentifier() for part in parts):
            raise VirtualModuleError(f"invalid virtual module name: {name}")
        bases: list[str]
        alias_root = self.config.aliases.get(parts[0])
        if level == 0 and alias_root is not None:
            bases = [posixpath.join(alias_root, *parts[1:])]
        elif level == 0:
            relative = posixpath.join(*parts)
            bases = [posixpath.join(root, relative) for root in self.config.roots]
        else:
            base = posixpath.dirname(importer)
            for _ in range(level - 1):
                base = posixpath.dirname(base)
            bases = [posixpath.join(base, *parts)]
        for base in bases:
            for candidate in (f"{base}.py", posixpath.join(base, "__init__.py")):
                normalized = self._normalize_path(candidate)
                self._require_allowed(normalized)
                source = self.loader(normalized)
                if source is not None:
                    self._sources[normalized] = source
                    return normalized
        raise VirtualModuleError(
            f"cannot resolve virtual module {name!r} from {importer}"
        )

    def _reject_nested_imports(self, tree: ast.Module) -> None:
        """Reject imports that are not direct module statements.

        Args:
            tree: Parsed Python module.
        """

        top_level = {id(statement) for statement in tree.body}
        if any(
            isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top_level
            for node in ast.walk(tree)
        ):
            raise VirtualModuleError(
                "imports inside functions or blocks are not supported"
            )

    def _public_names(
        self, body: list[ast.stmt], imported_names: list[str]
    ) -> tuple[str, ...]:
        """Collect public names made by direct module statements.

        Args:
            body: Rewritten direct statements.
            imported_names: Local names created by import statements.

        Returns:
            Stable public export names.
        """

        names = list(imported_names)
        for statement in body:
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.append(statement.name)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                names.extend(
                    target.id for target in targets if isinstance(target, ast.Name)
                )
        return tuple(dict.fromkeys(name for name in names if not name.startswith("_")))

    def _normalize_path(self, path: str) -> str:
        """Normalize one absolute POSIX path without allowing traversal.

        Args:
            path: Virtual path to normalize.

        Returns:
            Normalized absolute path.
        """

        if not path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise VirtualModuleError(
                "virtual module paths must be absolute and confined"
            )
        return posixpath.normpath(path)

    def _require_allowed(self, path: str) -> None:
        """Require a path to remain under one configured root.

        Args:
            path: Normalized virtual path.
        """

        roots = (*self.config.roots, *self.config.aliases.values())
        if not any(
            path == root or path.startswith(root.rstrip("/") + "/") for root in roots
        ):
            raise VirtualModuleError(
                f"virtual module path is outside configured roots: {path}"
            )

    def _symbol(self, path: str) -> str:
        """Build a collision-resistant generated module variable.

        Args:
            path: Normalized module path.

        Returns:
            Valid generated Python identifier.
        """

        return f"_arenero_module_{self._digest(path)}"

    def _digest(self, path: str) -> str:
        """Build a short stable path digest.

        Args:
            path: Normalized module path.

        Returns:
            Twelve hexadecimal characters.
        """

        return hashlib.sha256(path.encode()).hexdigest()[:12]
