# Monty constraints and operating details

Use this reference when an implementation decision depends on compatibility, security, persistence, or files. Verify version-sensitive details against the official sources linked from `SKILL.md`.

Reviewed and migrated on **2026-10-14**. Arenero pins `pydantic-monty==1.1.0`.
The previous installed release was `0.0.21`.
Development `main` was inspected at
`4246aa75724b133e55f8a82e2cf5dfc0120e62e8`.
Read [the project review](../../../../docs/monty-updates.md) for release changes,
migration choices, and evidence from 48 direct runtime probes per version.
Do not use development documentation as an installed-runtime contract.

## Runtime and isolation

- Monty parses Python with Ruff and runs its own bytecode VM. It does not embed CPython and has no FFI or C dependencies.
- The Python `Monty` and `AsyncMonty` pools run sessions in worker subprocesses. A worker crash does not crash the host. In-process Rust and some WebAssembly environments lose this crash boundary.
- Workers start with an empty environment, except for the minimum Windows requirement. The sandbox sees only values the host supplies.
- A remote WebSocket worker can use a different runtime. Local Monty security guarantees do not automatically apply to it.
- Pin the worker binary for security-sensitive use. Binary resolution can otherwise use `MONTY_BIN`, the environment scripts directory, or `PATH`.

## Supported Python shape

Monty supports ordinary functions, closures, lambdas, function and class decorators, simple classes, dataclasses, comprehensions, control flow, exceptions, synchronous context managers, f-strings, and basic async functions with `await`.

The installed `1.1.0` rejects or omits:

- Class bases, inheritance, metaclasses, method decorators, descriptors, `super()`, and most dunder protocols.
- Generator functions, `yield`, `match`, `del`, `async with`, `async for`, async comprehensions, `try*`, PEP 695 aliases, wildcard imports, and t-strings.
- `compile`, `__import__`, `globals`, `vars`, and `dir`. Native `eval`, `exec`,
  and `locals` are available.
- Function metadata such as `fn.__name__`, `fn.__doc__`, and `fn.__dict__`.
- A real module namespace, packages, relative imports, `sys.path`, and site-packages.

Arenero's optional `execute_module()` does not change this Monty limitation. The
host parses and rewrites a bounded static file graph into one normal Monty feed.
It gives each file a simple generated object, but it does not provide importlib,
reload, a module cache, dynamic imports, or third-party packages. Generated
module feeds disable type checking because the simple objects need dynamic
attributes.

Arenero passes nested bundled-standard-module imports through in supported
Monty scopes. Nested virtual-file imports and wildcard imports remain rejected.
This does not enable unsupported submodules or class-body imports.
Explicit `from . import helper` binds sibling module objects through the same
bounded static resolver. Each loader has a virtual `__file__` for nearby asset
reads; it is not exported as general module metadata.

Arenero can wrap this static graph in its own dependency-free source package
format. The host checks the fixed JSON manifest and bounded UTF-8 files, then runs
`verify.py` in a fresh Monty session. The file must export `passed = True` after
its assertions. Export creates a deterministic source ZIP. It does not add an
installer, build backend, wheel, standard sdist, third-party dependency, or new
import behavior.
Optional non-Python bytes below project-level `assets/` use the same quotas,
digest, and ZIP. They require host `allow_binary_assets=True`; `.py` files stay
UTF-8. Inventory and verifier decisions must not depend on public output trimming.

Generator expressions and `enumerate`, `zip`, `map`, `filter`, and `reversed` are eager lists. Avoid them on large or unbounded inputs.

This eager behavior still applies to `1.1.0` and reviewed `main`. Ordinary
`iter`, `next`, and class `__iter__` already worked in `0.0.21`.

Released `1.1.0` adds complex literals and values, `format()`, string and bytes
percent formatting, `eval`, `exec`, and `locals`. Class inheritance, method
decorators, generators, `match`, `del`, `async with`, `async for`, and wildcard
imports remain unavailable. Sandbox classes dispatch selected dunders only;
ordinary `__len__`, `__getitem__`, `__call__`, and arithmetic overloads are absent.

## Dynamic source execution by version

- `0.0.21`: `eval`, `exec`, and `locals` do not exist. Use a new host-driven feed.
- `1.1.0`: `eval` and `exec` accept string or UTF-8 bytes source and run it inside
  Monty. Namespace arguments must be dictionaries; code objects and non-`None`
  closure arguments are unsupported. `compile`, `globals`, and `__import__` stay
  absent. Module `locals()` is a fresh snapshot, not a writable namespace.
- Runtime-compiled source bypasses static type checking. Its compilation time
  counts against feed and turn duration. Once execution starts, its compilation
  products remain allocated for the session lifetime, including after an error.
- A type-checked later feed does not know names defined only in an exec string.
  Use a checked source feed (`execute_file`), stubs, or keep definition and use in
  dynamic source. Reset alone does not fix this static namespace gap.
- Under an explicit globals dictionary, names resolve from that dictionary and
  builtins; unknown names do not trigger lazy host lookup.
- Native dynamic source execution does not add a virtual-file import loader or
  change Arenero's current `execute_module` and source-package contracts.

## Standard library

The previous `0.0.21` had `asyncio`, `collections`, `dataclasses`, `datetime`,
`itertools`, `json`, `math`, `os`, `pathlib`, `re`, `sys`, `typing`, and
`unicodedata`. That runtime did not import `base64`, `binascii`, or `functools`;
the previous reference incorrectly combined development docs with this version.

Released `1.1.0` adds `base64`, `binascii`, `copy`, `functools`, `random`, and
`time`, and implements the complete `itertools` surface. `functools` has only
`reduce` and `partial`; `copy` has only `copy` and `deepcopy`.
`dataclass` supports `eq` and `frozen`, but not `field` or `__post_init__` in this
release. `asyncio` has only `run`, `gather`, and `sleep`; no exposed event loop,
task creation, or asyncio synchronization objects.

Development `main` also adds `hashlib`, `os.path`, and `posixpath`.
Guaranteed hash algorithms are present; `file_digest`, `scrypt`, and a separate
`hmac` module are absent. It adds dataclass fields, factories, `MISSING`, and
post-init hooks; these hooks cannot suspend. These are not `1.1.0` features.

Still unavailable as bundled modules: `csv`, `io`, `logging`, `pickle`, `socket`,
`subprocess`, `tempfile`, `threading`, `multiprocessing`, `traceback`, `urllib`,
`uuid`, and `zipfile`. There is no site-packages or `sys.path` loader.

Development `ExternalModule` can serve explicitly selected host attributes,
functions, and nested modules for unknown imports. Supply `external_modules` at
checkout and `.pyi` stubs for checked code. It is not automatic filesystem import
or arbitrary host-module access. The mapping is live, but a module's attributes
and its first retained stub have separate lifetimes. Reattach providers after
restore. Do not replace Arenero's static resolver until a released native path
has passed integration tests.

Type checking can resolve some stub-support modules that the runtime cannot import. A clean type check is useful but is not proof that every runtime path exists.

## Boundary values and host functions

- `inputs` bind eagerly. `external_lookup` resolves lazily when a name is read. Callables become host functions only through `external_lookup`.
- Monty's static checker needs input names in stubs; binding alone does not
  declare them. Arenero forwards native inputs after bootstrap without type
  inference or automatic checking bypass.
  Direct, file, module, and Python/%run cell APIs bind inputs once before first
  user source. Trusted file reads skip checking, while user file feeds follow
  the configured type policy. Follow-up feeds do not rebind input values.
- Sync `Monty` cannot drive coroutine host functions. Use `AsyncMonty` or manual suspension handling.
- Payloads and dumps have a 256 MiB wire-frame limit. Export depth is bounded by
  the remaining recursion limit; deep values can become placeholder strings.
  In `1.1.0`, the flat wire representation itself has no nesting ceiling, and
  protobuf decoding has a separate 1 GiB allocation budget per frame.
- Sandbox values without a faithful host representation can become plain `repr` strings when passed out.
- An unsupported host return value raises a `TypeError` in the sandbox. Unsupported initial `inputs` or `external_lookup` values raise a host conversion error.
- Host exceptions cross by supported exception type name. Monty has a fixed exception set and no user-defined exception subclasses.
- Host functions and OS callbacks run with host authority and can suspend outside the interpreter time budget.

`1.1.0` adds `ClassInstance`, `ClassType`, `MontyClassProxy`, and related type
proxies. Explicit wrappers grant selected eager values, lazy reads, method calls,
and optionally host class construction. Sandbox writes change its copy only.
Host wrapper state does not survive dump/restore; method calls and lazy reads
then fail unless reattached. This matters because Arenero restores before later
operations. Host wrapper retention is outside `max_memory`.

Arenero's public result converter emits Base64 bytes records, complex records,
ISO date text, and Monty instance class/attribute records. This does not grant
host-object access, preserve identity or methods, or decode inputs automatically.
Raw file transfers still use bytes. Monty's deep/cyclic export placeholders
remain possible.

Most builtin methods are call-only: `items.append` cannot be kept as a callback.
Synchronous nested interpreter callbacks such as sort keys, iterator predicates,
copy hooks, and implicit dunders cannot suspend to a host function. Use an
explicit loop for tool calls.

## Resource limits

- `max_memory` counts allocator requests, not RSS. Fragmentation, allocation overhead, stacks, mapped images, and direct memory mappings are outside this count. Add host OS or cgroup limits when process memory must be strictly bounded.
- In previous `0.0.21`, `max_duration_secs` was cumulative interpreter time for
  the session. It pauses during host calls and between feeds and survives snapshots.
- Released `1.1.0` rejects that key. Use `max_feed_duration_secs` and
  `max_turn_duration_secs`. They reset per feed and per host answer; neither is a
  session-lifetime budget. Arenero now uses `max_feed_duration_seconds` (5 s),
  optional `max_turn_duration_seconds` (off), `max_suspensions` (1,000), and
  `max_total_sleep_seconds` (10 s). The old duration field was removed.
- `request_timeout` is a per-protocol-turn wall deadline. Frequent host calls can
  reset it. Keep an interpreter duration limit and an outer host-work deadline.
- `1.1.0` adds `max_suspensions` (default 1,000 per checkout),
  `max_total_sleep_secs`, and pool `feed_duration_limit_grace` /
  `turn_duration_limit_grace` options. Arenero keeps both 1-second grace defaults.
  Restores reset suspension and used system-sleep counts.
- Initial feed compilation time is not charged to duration. Runtime eval/exec
  compilation is charged. Both use the worker boundary and structural caps.
- The default recursion ceiling is 1000 and cannot be disabled. `gc_interval` also cannot disable collection.
- Print collectors and mount memory have separate limits.
- After memory or time failure, discard the session. The pool does not do this
  automatically. In `1.1.0`, a later feed's time clocks reset, but the old heap
  still cannot be trusted. Suspension-limit failures end cleanly; recursion
  failures are catchable.

`1.1.0` has `OSPolicy`: system UTC wall time and system random seeding by default,
system sleep capped at ten seconds per call, and zero process time. Configure a
fixed instant, explicit seed, and zero sleep for deterministic tests. System
sleep is charged to its own budget; call-host sleep needs a host deadline.

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
- File iteration is not supported in `0.0.21` or `1.1.0`. Use `readlines()`.
- The first read caches the whole file in the Monty heap. Later external changes are not visible through that file object.
- Write positioning does not provide normal random-access overwrite semantics.
- `0.0.21` has no native working directory or `os.path`. `1.1.0` has a native
  virtual directory with `os.getcwd`, `os.chdir`, and relative file paths.
  Arenero uses that native directory for Python and shell calls. New mounted
  workspace sessions start there before bootstrap. The host process directory
  is unchanged. No version reviewed adds a real shell or process API.

`OSAccess` with `MemoryFile` provides a host-memory virtual filesystem. `CallbackFile` executes host callbacks and is only as safe as those callbacks.

Development `main` adds file iteration and the `newline` argument. An unread
file can load through a `for` loop or `next`; `list(f)` and similar consumers need
the buffer loaded first (for example, `f.seek(0)`). All reads still cache the
complete file. It adds `Path.glob`, `rglob`, `walk`, `os.scandir`, `os.walk`, and
`os.path`. Native scans collect a bounded subtree before iteration; pruning a
walk does not avoid host reads. They do not use Arenero's simpler glob grammar or
scan limits. Read-write mounts still require a dedicated host workspace.

## Binary data and Base64

Monty supports `bytes`, binary file modes, `Path.read_bytes()`, and
`Path.write_bytes()`. `1.1.0` provides Base64, Base32, Base16, Base85, Z85, ASCII85,
MIME Base64 helpers, and `binascii` conversions. The previous `0.0.21` imported
neither `base64` nor `binascii`. Host-side transfer helpers still avoid text
encoding when transferring bytes from an application.

Base64 encoders accept `bytes`. Decoders accept `bytes` or ASCII `str`. There is
no `bytearray` or `memoryview`. `a85encode` and `a85decode` do work in `1.1.0`;
file-object Base64 helpers and the Base64 command-line interface do not.

Use bytes for file transfer and storage. Use Base64 only across text-only interfaces. An LLM cannot see an image from a path or Base64 string by magic: the host must read the bytes, determine the media type, and create the model provider's image content block or data URL.

## Persistence and snapshots

Monty can dump an idle session or a suspended mid-feed snapshot to opaque bytes.

- Idle session dumps contain globals, functions, classes, and duration state.
  In `0.0.21` that is a cumulative session budget. In `1.1.0` it is the current
  feed's budget; a new feed restarts its clock.
- Suspended snapshots also contain the paused call stack.
- Load only into a fresh session, before any feed.
- Snapshots do not contain host functions, mount configurations, callbacks, or live host resources. Supply these again when execution resumes.
- A suspension can be announced again after restore. Make side-effecting callbacks idempotent.
- Store the Monty version with every dump and reject or reset incompatible state.
- Apply a comparable memory limit while restoring large dumps.
- Files in host memory or host storage have a separate lifecycle from the interpreter dump unless the application stores them together.

Arenero reuses filesystem helpers from idle dumps and checks a private source
hash after restore. A changed helper source or helper limits refreshes them once.
Read-only name/type and managed-directory inspection load a disposable dump,
attach no host/file callbacks, and save nothing. They read the last commit, need
a worker, and must not re-enter an active parent interpreter.
Native async percent cells reuse the shared parser/result format. Raw uploads and
overflow output use isolated checkouts without bootstrap or saved state. Overflow
starts after the execution checkout and async filesystem lock end; a failed write
does not roll back Python state. No native async shell is added.

Released `1.1.0` uses CBOR and an upstream dump-format version. Arenero still
requires an exact package-version match. Do not infer that old `0.0.21` snapshots
can be loaded by a new wheel. In `1.1.0`, dump resource limits replace checkout
limits apart from the suspension cap. Reviewed `main` caps every restored limit
at the restoring checkout's limit; this is another unreleased change.

Arenero RLM recursion uses a new temporary snapshot for each child. It does not
copy the parent namespace. The parent checkout stays active during the child, so
each active depth needs another worker. Recursive children omit direct mounts,
filesystem callbacks, and `bootstrap_files` to avoid the parent's filesystem
lock. Trusted `bootstrap_code` and explicitly inherited hosted file tools remain
available.

Arenero speculative programmatic tool calling uses the same snapshot property
for a temporary shadow of the real session. Host-side `ast` processing only
inspects syntax and evaluates a fixed side-effect-free expression allowlist. It
must never execute generated source in CPython. The forked Monty shadow has no
direct mount or filesystem callbacks. Its state is discarded. Only explicitly
named hosted tools can run early, and `sequential=True` tools are excluded.
Nondeterministic identical calls need separate FIFO entries. Deterministic reuse
requires an explicit host policy. Failed or unused work can still consume an
external provider request, so speculate read-only or idempotent functions.

## Type checking pattern

Enable `type_check=True` and define host tools in `type_check_stubs`. Give the model the same declarations. Successful snippets accumulate into the type-checking context; failed snippets do not execute or accumulate. Type checking is a strong retry signal for model-generated code, but documented runtime gaps still require tests.

In `1.1.0`, runtime `eval`/`exec` strings bypass checking. In development `main`,
external-module stubs are requested once, cached, and saved in dumps; later
imports can use a new runtime provider but retain the old stub. Conditional
imports can also pass a later check even when their branch never executed.
General upstream summary pages can lag specific features: runtime probes confirm
`1.1.0` stderr printing despite older print restrictions in some summary notes.
