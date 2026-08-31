# Monty constraints and operating details

Use this reference when an implementation decision depends on compatibility, security, persistence, or files. Verify version-sensitive details against the official sources linked from `SKILL.md`.

## Runtime and isolation

- Monty parses Python with Ruff and runs its own bytecode VM. It does not embed CPython and has no FFI or C dependencies.
- The Python `Monty` and `AsyncMonty` pools run sessions in worker subprocesses. A worker crash does not crash the host. In-process Rust and some WebAssembly environments lose this crash boundary.
- Workers start with an empty environment, except for the minimum Windows requirement. The sandbox sees only values the host supplies.
- A remote WebSocket worker can use a different runtime. Local Monty security guarantees do not automatically apply to it.
- Pin the worker binary for security-sensitive use. Binary resolution can otherwise use `MONTY_BIN`, the environment scripts directory, or `PATH`.

## Supported Python shape

Monty supports ordinary functions, closures, lambdas, function and class decorators, simple classes, dataclasses, comprehensions, control flow, exceptions, synchronous context managers, f-strings, and basic async functions with `await`.

It rejects or omits:

- Class bases, inheritance, metaclasses, method decorators, descriptors, `super()`, and most dunder protocols.
- Generator functions, `yield`, `match`, `del`, `async with`, `async for`, async comprehensions, `try*`, PEP 695 aliases, wildcard imports, complex literals, and t-strings.
- Dynamic execution and namespace inspection: `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`, `vars`, and `dir`.
- Function metadata such as `fn.__name__`, `fn.__doc__`, and `fn.__dict__`.
- A real module namespace, packages, relative imports, `sys.path`, and site-packages.

Arenero's optional `execute_module()` does not change this Monty limitation. The
host parses and rewrites a bounded static file graph into one normal Monty feed.
It gives each file a simple generated object, but it does not provide importlib,
reload, a module cache, dynamic imports, or third-party packages. Generated
module feeds disable type checking because the simple objects need dynamic
attributes.

Generator expressions and `enumerate`, `zip`, `map`, `filter`, and `reversed` are eager lists. Avoid them on large or unbounded inputs.

## Standard library

The importable modules are `asyncio`, `base64`, `binascii`, `collections`, `dataclasses`, `datetime`, `functools`, `itertools`, `json`, `math`, `os`, `pathlib`, `re`, `sys`, `typing`, and `unicodedata`. Each exposes only part of CPython's API.

Notable unavailable modules include `csv`, `hashlib`, `io`, `logging`, `pickle`, `random`, `socket`, `subprocess`, `tempfile`, `threading`, `multiprocessing`, `time`, `traceback`, `urllib`, `uuid`, and `zipfile`.

Type checking can resolve some stub-support modules that the runtime cannot import. A clean type check is useful but is not proof that every runtime path exists.

## Boundary values and host functions

- `inputs` bind eagerly. `external_lookup` resolves lazily when a name is read. Callables become host functions only through `external_lookup`.
- Sync `Monty` cannot drive coroutine host functions. Use `AsyncMonty` or manual suspension handling.
- Boundary values have nesting limits and a 256 MiB wire-frame limit.
- Sandbox values without a faithful host representation can become plain `repr` strings when passed out.
- An unsupported host return value raises a `TypeError` in the sandbox. Unsupported initial `inputs` or `external_lookup` values raise a host conversion error.
- Host exceptions cross by supported exception type name. Monty has a fixed exception set and no user-defined exception subclasses.
- Host functions and OS callbacks run with host authority and can suspend outside the interpreter time budget.

## Resource limits

- `max_memory` counts allocator requests, not RSS. Fragmentation, allocation overhead, stacks, mapped images, and direct memory mappings are outside this count. Add host OS or cgroup limits when process memory must be strictly bounded.
- `max_duration_secs` is cumulative interpreter time for the session. It pauses during host calls and between feeds, and it survives snapshots.
- `request_timeout` is a per-protocol-turn wall deadline. A program that makes frequent host calls can reset it. Use it together with `max_duration_secs`.
- Compilation time is not charged to duration. Compilation has structural caps but should still run behind the worker boundary.
- The default recursion ceiling is 1000 and cannot be disabled. `gc_interval` also cannot disable collection.
- Print collectors and mount memory have separate limits.
- After memory or time failure, discard the session. The pool does not do this automatically. A memory failure can leave invalid heap state, and a spent cumulative time budget fails every later feed.

## Filesystems and files

Without a mount or OS callback, all file I/O is denied. Virtual paths are POSIX on all hosts.

`MountDir` modes:

- `read-only`: host files can be read; writes fail.
- `read-write`: changes go to the real host directory. Treat all output as untrusted and never mount a directory on an import or configuration path.
- `overlay`: reads fall through to the host and writes stay in memory for one feed. Each feed starts with a clean overlay.

Mounts are descriptor-confined. `..`, path replacement, and symlinks cannot escape the mount. Overlay mounts refuse symlinks. Direct mounts accept only safe relative symlinks. Virtual paths have a 4096-byte limit, 255-byte component limit, and 64-component limit.

Only regular files can be opened, read, or written. FIFOs, sockets, and device files are denied.

File behavior differs from CPython:

- There are no live file descriptors. Each host operation is one-shot.
- `+` and `x` modes are not supported.
- Text is UTF-8. Most `open()` options are unavailable.
- File iteration is not supported. Use `readlines()`.
- The first read caches the whole file in the Monty heap. Later external changes are not visible through that file object.
- Write positioning does not provide normal random-access overwrite semantics.
- `pathlib` and a small part of `os` are available. There is no working directory, `os.path`, shell, process, or network API.

`OSAccess` with `MemoryFile` provides a host-memory virtual filesystem. `CallbackFile` executes host callbacks and is only as safe as those callbacks.

## Binary data and Base64

Monty supports `bytes`, binary file modes, `Path.read_bytes()`, and `Path.write_bytes()`. Current `main` documentation also lists Base64, Base32, Base16, Base85, Z85, MIME Base64 helpers, and common `binascii` operations. This is version-sensitive: `pydantic-monty 0.0.21` does not import `base64`, so use a host-side encoder for that release.

Base64 encoders accept `bytes`. Decoders accept `bytes` or ASCII `str`. There is no `bytearray` or `memoryview`. `a85encode`, `a85decode`, file-object Base64 helpers, and the Base64 command-line interface are not available.

Use bytes for file transfer and storage. Use Base64 only across text-only interfaces. An LLM cannot see an image from a path or Base64 string by magic: the host must read the bytes, determine the media type, and create the model provider's image content block or data URL.

## Persistence and snapshots

Monty can dump an idle session or a suspended mid-feed snapshot to opaque bytes.

- Idle session dumps contain globals, functions, classes, and the remaining time budget.
- Suspended snapshots also contain the paused call stack.
- Load only into a fresh session, before any feed.
- Snapshots do not contain host functions, mount configurations, callbacks, or live host resources. Supply these again when execution resumes.
- A suspension can be announced again after restore. Make side-effecting callbacks idempotent.
- Store the Monty version with every dump and reject or reset incompatible state.
- Apply a comparable memory limit while restoring large dumps.
- Files in host memory or host storage have a separate lifecycle from the interpreter dump unless the application stores them together.

Arenero RLM recursion uses a new temporary snapshot for each child. It does not
copy the parent namespace. The parent checkout stays active during the child, so
each active depth needs another worker. Recursive children omit direct mounts,
filesystem callbacks, and `bootstrap_files` to avoid the parent's filesystem
lock. Trusted `bootstrap_code` and explicitly inherited hosted file tools remain
available.

## Type checking pattern

Enable `type_check=True` and define host tools in `type_check_stubs`. Give the model the same declarations. Successful snippets accumulate into the type-checking context; failed snippets do not execute or accumulate. Type checking is a strong retry signal for model-generated code, but documented runtime gaps still require tests.
