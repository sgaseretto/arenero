# Static virtual modules

Arenero's pinned Monty `1.1.0` has no `sys.path`, site-packages, file import
loader, or user module namespace.
Arenero therefore provides a small host-side resolver for user-owned Python
files in the virtual filesystem. It is useful for a few helper files. It is not
a CPython compatibility layer.

Released Monty `1.1.0` adds native `exec`, but does not add a source-file loader.
Development `main` adds explicitly supplied host modules. Neither change replaces
this resolver in the current project. See [the upstream review](monty-updates.md).

The resolver passes supported built-in modules through, including `base64`,
`binascii`, `copy`, `functools`, `random`, and `time`. Their APIs remain partial.
It does not enable development-only `hashlib`, `os.path`, or host module providers.

Use `execute_file()` for one flat helper. Use `execute_module()` only when files
need explicit static imports.

```python
from pydantic_monty import MemoryFile, OSAccess

from arenero import PersistentMontyKernel

filesystem = OSAccess(
    files=[
        MemoryFile(
            "/helper.py",
            "FACTOR = 2\ndef double(value):\n    return value * FACTOR\n",
        ),
        MemoryFile(
            "/main.py",
            "from helper import double\nANSWER = double(21)\n",
        ),
    ]
)

with PersistentMontyKernel(os_handler=filesystem) as kernel:
    result = kernel.execute_module(
        "conversation-42",
        "/main.py",
        code="ANSWER",
    )
    assert result["result"] == 42

    later = kernel.execute("conversation-42", "double(5)")
    assert later["result"] == 10
```

The async kernel and both LangChain adapters provide the same operation. Deep
Agents middleware exposes it as `python_interpreter_module`.

## Resolution flow

```mermaid
flowchart TD
    Entry["Absolute entry .py path"] --> Read["Read bounded UTF-8 source"]
    Read --> Parse["Parse with host Python AST"]
    Parse --> Imports{"Import kind"}
    Imports -->|Monty standard library| Keep["Keep normal import"]
    Imports -->|Top-level virtual module| Resolve["Resolve .py or __init__.py"]
    Imports -->|Explicit relative sibling| Resolve
    Imports -->|Nested virtual module or wildcard| Reject["Reject before execution"]
    Resolve --> Limits["Check root, count, depth, bytes, and cycles"]
    Limits --> Dependency["Build dependency-first module loader"]
    Dependency --> Namespace["Create isolated simple module object"]
    Keep --> Namespace
    Namespace --> Exports["Module path/name metadata; bound public exports"]
    Exports --> Followup["Copy only bound exports; run same-call follow-up"]
    Followup --> Snapshot["Save normal persistent session snapshot"]
```

Each file runs inside a generated loader function. Its functions close over that
file's local values. A simple generated object holds public names. Import
statements become references to these objects. This avoids the unsafe flat-file
behavior where two modules can overwrite each other's globals.

The entry module object is available under its filename stem. Public entry names
are also copied into the REPL globals. For `/jobs/main.py`, follow-up code can use
both `main.run()` and `run()` when `run` is public.

## Supported forms

These forms work when files exist under an allowed root:

```python
import helper
import helper as h
from helper import calculate
from helper import calculate as run
from .helper import calculate
from . import helper
from . import helper as h, other
from .. import shared
from package.values import VALUE
from pathlib import Path
```

Bundled standard-module imports also work inside functions or supported blocks:

```python
def distance(x, y):
    import math
    from functools import reduce

    return math.sqrt(reduce(lambda a, b: a + b, [x * x, y * y]))
```

These imports stay unchanged for Monty to run. They do not load virtual files.
Monty's syntax and partial module APIs still apply: passing a bundled root name
does not make an unsupported submodule such as `os.path` available in 1.1.0.
Imports in a class body remain unsupported by Monty.

Packages resolve through `package/__init__.py`. Absolute virtual names search
configured roots. Relative imports start at the importing file. A configured
alias maps its first name segment to a virtual directory:

```python
from arenero import VirtualModuleConfig

config = VirtualModuleConfig(
    roots=("/app",),
    aliases={"shared": "/libraries/shared"},
    max_modules=16,
    max_depth=6,
    max_source_bytes=500_000,
)

result = kernel.execute_module(
    "agent-1",
    "/app/main.py",
    module_config=config,
)
```

With this configuration, `from shared.math import add` reads
`/libraries/shared/math.py` or `/libraries/shared/math/__init__.py`.

`from . import helper` binds a module object, not one exported value. It resolves
an explicit sibling `.py` file or `__init__.py` directory. Aliases and several
explicit names work; each counts toward the same graph limits. Parent levels
still must remain inside allowed roots. There is no implicit parent-package
execution, runtime module lookup, or import cache.

Inside each static module, `__file__` is its absolute virtual source path.
`__name__` is the filename stem, or the parent directory name for `__init__.py`.
Both are also readable on the generated module object. A function keeps the
file-local path in its closure and can use `Path(__file__).parent` for assets.
Metadata is not copied into the REPL's public entry exports. The name is a small
label, not a fully qualified import name; there is no `__package__`, `__spec__`,
`__path__`, implicit parent package initialization, or import runtime.

Flat `execute_file()` binds absolute `__file__` and `__name__="__main__"` through
Monty's native `inputs` before the file feed. Source and follow-up remain checked
when configured. Metadata overrides those two input keys and persists with flat
REPL globals. A later flat file replaces it; flat helper functions use those
shared globals. Static module functions keep their separate file-local closure.
Monty makes module-level metadata read-only for Python assignments; the host
uses the supported input-binding path instead of rewriting user source.

### Compound exports

Public assignments, tuple/list/starred unpacking, and definitions inside supported
`if`, `for`, `while`, `try`, and `with` statements can be exports. Collection stops
at function/class scopes. Generated code tests whether a name was actually bound
before copying it. An unexecuted branch or an annotation-only name does not make
loading fail. Comprehension-local and nested function/class-local names do not
leak. The bundle's export metadata lists static candidates, not proof that every
branch ran. Importing an unbound candidate still fails at runtime.

```python
A, B = (20, 22)
if True:
    TOTAL = A + B
if False:
    UNUSED = 0
```

The module exposes `A`, `B`, and `TOTAL`; it has no `UNUSED` attribute. Re-running
a bundle does not delete old REPL bindings for candidates absent on the new run.
Private names and `__all__` remain outside the export contract. Wildcard, nested
virtual-file, and circular imports remain rejected.

Both kernels accept `inputs=` and `output_file=` for module execution. Inputs
bind once after bootstrap, before the generated bundle. Full bounded overflow
uses the same JSON format as direct Python. The generated bundle still skips
type checking.

The default limits are:

| Limit | Default | Meaning |
| --- | ---: | --- |
| `roots` | `("/",)` | Allowed virtual source roots. |
| `max_modules` | 32 | Maximum files in one graph. |
| `max_depth` | 8 | Maximum entry-plus-import depth. |
| `max_source_bytes` | 1,000,000 | Combined UTF-8 bytes across resolved files. |

The filesystem transfer limit is also enforced on each file.

## Deliberate limitations

The resolver rejects:

- `from module import *`;
- virtual-file imports inside functions, classes, `if`, loops, or other blocks;
- circular module graphs;
- a dotted `import package.helper` without an `as` name;
- paths outside configured roots or any path with `..`;
- non-UTF-8 source and files without a `.py` entry path;
- unresolved names that are not public exports;
- third-party packages, native extensions, dynamic imports, reload, custom import
  hooks, namespace packages, and `sys.path` changes.

Only Monty's documented partial standard-library modules remain normal imports.
A module name that is not in that set is treated as a virtual module.
For example, this still fails, even if `/helper.py` exists:

```python
def run():
    from helper import calculate  # Nested virtual import: rejected.

    return calculate()
```

Move that virtual import to the file's top level. Keep standard imports nested
only where Monty accepts the scope.

Generated support code uses dynamic attributes on a simple Monty class. Arenero
therefore disables Monty type checking for the complete generated module feed.
Direct `execute()` and `execute_file()` calls still follow `KernelConfig.type_check`.
Validate module files before an untrusted user can write them, or use direct
typed cells when pre-execution type checks are required.

AST parsing and source rewriting happen in the trusted host process. Source code
still executes only in Monty. The resolver does not execute user code on the
host.

## Persistence and side effects

The generated module objects, functions, closures, and public entry exports are
normal Monty values. They remain in a saved session and can be used by later
cells. Re-running the module creates new objects and overwrites the entry exports.
There is no import cache or automatic reload check.

Module source files stay in the separate virtual filesystem. A session reset
does not delete them. A graph checkpoint restores Python state but does not
restore old file bytes. File or hosted-tool changes made while a module loads are
not transactional and are not rolled back by `state_policy`.
