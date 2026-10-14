---
name: monty-sandbox
description: Build, review, debug, or integrate Pydantic Monty sandboxes and Monty-based agent REPLs. Use for Monty execution, host functions, resource limits, snapshots, virtual filesystems, mounts, or CPython-compatibility decisions; do not assume normal CPython behavior.
---

# Monty Sandbox

Treat Monty as a bounded Python sandbox for model-written glue code. Use its
documented language subset and explicit host capabilities.

## Before changing Monty code

1. Check the installed `pydantic-monty` version and its Python API. Monty reached
   `1.0.0` on 2026-09-25; older releases and newer development source have
   different language, limit, and snapshot contracts.
2. Read [references/constraints.md](references/constraints.md) when a task depends on Python syntax, imports, files, host functions, resource limits, security, or persistence.
3. Prefer a short runtime probe over a CPython assumption. Turn on Monty type checking when the integration can support it.

Arenero pins `pydantic-monty==1.1.0`. The 2026-10-14 review compared previous
`0.0.21`, released `1.1.0`, and upstream `main` at
`4246aa75724b133e55f8a82e2cf5dfc0120e62e8`.
Read [the migration guide](../../../docs/monty-updates.md) before a change to
runtime behavior. Use tagged docs for releases.
Public docs can include unreleased host modules, file iteration, directory scans,
dataclass factories, and hashlib. The project's
`scripts/monty_feature_probe.py` runs the same bounded checks in separate runtime
environments; [saved evidence](../../../docs/monty-compatibility.json) records
the last comparison. Do not remove the dependency pin to run that comparison.

## Choose Monty only for the right workload

Use Monty for short, model-written programs that filter, join, transform, and calculate over plain values or call a small host tool surface.

Use a container, VM, or managed sandbox when code needs any of these:

- Third-party packages or most of the Python standard library.
- Shell commands, subprocesses, Git, package installation, or arbitrary executables.
- Native extensions, threads, multiprocessing, sockets, or direct network access.
- Full Python compatibility, notebooks, data-science libraries, or application builds.

Arenero has one narrow exception to general package building. Use
`MontyPackageManager` or the opt-in package tools to create a dependency-free
Monty source package. Keep the fixed manifest and layout, run `package_check`,
run `package_test` in its fresh session, and export only its source ZIP. This is
not a wheel, Python sdist, package install, or build backend.

Do not describe a Monty-backed `execute` operation as shell execution. Name it `execute_python`, `python_repl`, or clearly state that it accepts only Monty's Python subset.

Arenero also has a host-side virtual shell. It is not part of Monty and does not
start operating-system processes. Its supported subset includes allowlisted file
and text commands, UTF-8 pipes and redirection, persistent virtual variables,
`PWD`, `OLDPWD`, `$?`, `~`, bounded `*`, `?`, and `**` expansion, here documents,
and straight-line files through `source`. It does not support full Bash control
flow, command substitution, field or brace expansion, descriptor redirection,
binary pipes, arbitrary commands, or host environment variables. Read
`docs/virtual-shell.md` before changing or describing the shell.

For a new Arenero application, start with `open_sandbox()` and a
`SandboxConfig` preset. Use `open_sandbox_manager()` for several bound session
IDs on one worker pool. These APIs already own lifecycle, relative workspace
paths, common results, total file quotas, revisions, tree transfer, checkpoints,
and selected LangChain tools. Use low-level kernel assembly only when the managed
surface cannot express the required host boundary.

The `batch` preset has no filesystem by default so distinct sessions can overlap.
Adding a shared filesystem makes filesystem-capable kernel calls serialize. Use
separate managers and workspaces for concurrent file workloads.

Managed async handles use `asyncio.to_thread()` over the synchronous managed
surface. They are complete wrappers, not the native `AsyncMonty` path. Use
`AsyncPersistentMontyKernel` when native async host calls are required.

## Design the host boundary

- Monty has no ambient host filesystem, environment, network, or process access.
  Released `1.1.0` does expose wall time and random entropy by default through
  `OSPolicy`; select a fixed clock, seed, and zero sleep for deterministic runs.
- Arenero's shell `curl` runs on the host and allows bounded public HTTP and HTTPS
  requests by default. It does not add sockets or network APIs to Monty. Use
  `NetworkConfig` to narrow URL prefixes and methods or to disable curl. Keep
  non-global address blocking, redirect checks, and hard byte limits.
- Put ordinary input values in `inputs`. Put tools and lazy values in `external_lookup`.
- A callable in `external_lookup` runs in the host process with full host authority. Validate every argument as untrusted input.
- Use `HostedToolSet` when tools need generated signatures, selection, per-operation
  call limits, sequencing, value conversion, and a bounded `host_calls` ledger.
- Hosted tools accept keyword arguments only. Their JSON Schema property names
  must be valid Python identifiers. Missing schema types become `Any`.
- Keep framework calls on their public path: `BaseTool.invoke` or `ainvoke` for
  LangChain, the current `ToolManager` for Pydantic AI, and bounded hub calls for
  promoted MCP tools.
- Do not persist framework run contexts or tool callbacks in Monty. Reattach them
  for each operation. Tool errors can hide completed side effects; do not retry
  non-idempotent work without deduplication.
- Monty snapshots can keep type-check state. Configure stable hosted tools before
  a session starts. A newly added tool can require `restart=True` before an older
  type-checked snapshot knows its name.
- Do not make a hosted shell function re-enter the active persistent session. For
  notebook behavior, dispatch `!command` or `%%shell` before starting the Monty
  feed, then run Python or the virtual shell with the same session ID.
- For Arenero RLMs, use `llm_query` for a plain model call and `rlm_query` for a
  new temporary child session. Never call the active interpreter tool from
  itself. A recursive chain needs at least one free worker per active level.
- Recursive RLM child sessions disable direct kernel mounts and filesystem
  callbacks because the parent feed holds the filesystem lock. Give children
  explicit hosted file tools when they must share files.
- Keep speculative execution optional and host-approved. Parse streamed source
  only to inspect it. Run state-dependent generated statements in a forked
  Monty session, never with CPython `exec`, `eval`, or `compile`.
- Disable direct filesystem access in a speculative shadow. Reattach only the
  hosted tools named in the speculation policy. Do not speculate sequential
  tools. Prefer read-only or idempotent calls because wrong-path work cannot be
  rolled back.
- Preserve call multiplicity for nondeterministic speculative tools. Reuse one
  matching result only when the host marks the tool deterministic. A failed bet
  must fall back to normal execution, and pending unused work must have bounded
  cleanup.
- Materialize MCP binary blocks only after the active Monty checkout ends. Scan
  the final returned value or use an explicit host-side MCP result. Intermediate
  results and data removed by MCP result limits are unavailable.
- Start with an empty `McpHub()` when an agent can need tools that are not known at
  startup. The fixed `mcp_add_server`, `mcp_replace_server`, and
  `mcp_remove_server` host functions let Monty change stdio and Streamable HTTP
  servers without changing an outer agent tool schema. The new catalog is shared
  by all sessions on that hub and lasts only until the hub closes. It is not part
  of a Monty snapshot or SQLite session state. A running kernel created without
  a hub cannot attach its first MCP from inside Monty.
- Use generic `mcp_search`, `mcp_describe`, and `mcp_call` after a hot load.
  These fixed names work in existing type-checked snapshots. A promoted direct
  function can still require `restart=True` before an older snapshot knows its
  Python name.
- Prefer narrow domain tools such as `read_customer(id)` over broad primitives such as `read_file(path)`, `fetch(url)`, or `run(command)`.
- Prefer `MountDir` for host-directory access because it provides descriptor-based confinement and path checks. Use `OSAccess` with `MemoryFile` for an in-memory virtual filesystem.
- Use Arenero's `WorkspaceConfig` for a kernel-owned persistent read-write mount.
  It stores files in a dedicated host directory and exposes them at `/workspace`
  by default. SQLite still stores interpreter snapshots, not file content.
- Do not place secrets in sandbox inputs, files, mounts, or environment mappings. Keep authenticated work in narrow host functions.
- Return only Monty-supported boundary values. Complex sandbox-only objects sent to a host function can degrade to their `repr` string.
- On `1.1.0`, use explicit `ClassInstance` or `ClassType` policies for host
  objects and `MontyClassProxy` for returned sandbox instances. Their live host
  registrations do not survive dump/restore. Arenero restores between operations,
  so test later calls before adopting an object API. Do not expose a session
  handle whose methods re-enter its own locked checkout.
- Development `ExternalModule` can put approved tools under Python import names,
  including a live module mapping and module stubs. It is not in `1.1.0` and does
  not automatically load virtual source files. Prefer it only after a released
  implementation passes Arenero's persistence and tool integration tests.

## Execute safely

- Use `KernelConfig.max_feed_duration_seconds`, optional
  `max_turn_duration_seconds`, `max_suspensions`, and `max_total_sleep_seconds`.
  The old `max_duration_seconds` name was removed, not silently redefined.
- Configure `max_memory`, `max_recursion_depth`, output collection limits, filesystem limits, and persisted snapshot size.
- Duration resets per feed and per host answer. Waiting for host functions does
  not count. There is no session-lifetime duration budget.
- A checkout can contain helper, startup, file-read, file-source, and follow-up
  feeds. Duration resets for each; suspension and system-sleep use are shared
  within the checkout and reset on restore.
- Initial feed compilation is outside duration. Runtime `eval`/`exec`
  compilation counts. Keep compilation behind the worker boundary.
- A restored dump supplies its resource limits except for the restoring
  checkout's stricter suspension cap. Reset low-level sessions after changing
  limits or OS policy; managed checkpoints also check the policy digest.
- Treat `MemoryError`, `TimeoutError`, and worker crashes as terminal for that session. Delete its saved state and start a fresh session. `RecursionError` is not terminal.
- Keep one execution at a time per persistent session. Use an optimistic revision check as a second guard when state is shared across processes.
- Use `state_policy="always"` to keep valid changes before normal errors,
  `"success"` to save only successful code, and `"never"` for temporary Python
  state. None of these policies rolls back files or external side effects.
- A session fork copies the opaque Monty snapshot, native directory, and RNG
  state. Its next feed gets a fresh feed-duration clock. It does not copy files,
  mounts, host functions, object registrations, or connections.
- Run bootstrap files as source feeds for new sessions only. They are not imports.
  Do not save a partly completed bootstrap.
- Arenero serializes filesystem-capable calls across sessions in one kernel. This
  does not coordinate separate processes that share one host directory.
- Do not restore snapshots from untrusted sources into a worker you cannot discard.

## Work within the Python subset

- Expect only Monty's documented modules and partial APIs. There is no `sys.path`, site-packages, package system, or dynamic import.
- `eval`, `exec`, and `locals` run within Monty, without CPython. Use
  `execute_file` for a host-managed helper feed and same-call follow-up result.
  Runtime source strings bypass
  static checking and retain started compilation products for the session
  lifetime. `compile`, `globals`, and `__import__` remain absent.
- With `type_check=True`, names defined only by `exec` remain unknown to the
  checker. Use a checked `execute_file` feed, stubs, or definition and use within
  dynamic source. Reset does not solve this static namespace gap.
- In Arenero, use `execute_module()` only for a small static virtual graph. It is
  host-side AST rewriting, not a new Monty import system. It supports top-level
  `.py` and `__init__.py` imports, relative imports, roots, and aliases. It rejects
  dynamic, wildcard, nested, circular, and third-party imports. The generated
  module feed skips Monty type checking.
- For an Arenero source package, keep `arenero-package.json` dependencies empty.
  Use `NAME/__init__.py` as `entry` and `verify.py` as `verify`. Write plain
  assertions in `verify.py`, then set public `passed = True`. Always test in the
  package manager's new temporary session. Do not use a persistent REPL as proof
  that current source passes.
- Simple classes work. Inheritance, method decorators, descriptors, `super()`, and user-defined exception classes do not.
- Generator functions, `match`, `del`, `async with`, `async for`, exception groups,
  and wildcard imports do not run. Complex literals and values do work.
- Python `os.getcwd()` and `os.chdir()` share the persistent virtual directory
  with the shell. New mounted workspace sessions start in that workspace before
  bootstrap. No host process directory is changed.
- Some familiar operations are eager. In particular, `enumerate`, `zip`, `map`, `filter`, `reversed`, and generator expressions materialize lists and can consume large memory.
- Files have no live OS descriptor. Do not use file iteration. Use `read()`, `readline()`, or `readlines()`.
- The file-iteration rule applies to `0.0.21` and `1.1.0`. Development `main`
  supports `for` and `next` on unread files; other consumers need the full-file
  buffer loaded first. Do not call this released support until verified.

## Make model use effective

- Put the exact supported tool signatures and Monty limits in the model instructions.
- Put RLM depth, call, context, prompt, response, transcript, token, cost,
  concurrency, wall-time, and worker rules in model instructions. Token and cost
  limits enforce only usage reported by the model adapter.
- Put the active speculation mode, eligible tool names, call limits, shadow
  rules, and non-transactional side-effect warning in model instructions.
- Tell the model to write small direct snippets, use `pathlib`, use `readlines()` instead of iterating a file, and avoid unavailable imports.
- For Deep Agents, use `AreneroInterpreterMiddleware` with `ainvoke()` or
  `astream()`. Treat its PTC calls as nested calls that bypass the parent
  ToolNode. Parent per-tool approval and `Command.update` handling do not apply.
  Use its `thread`, `turn`, or `call` mode for Python state only; virtual files
  keep their separate lifecycle.
- Provide type-check stubs for every host function and give the same signatures to the model.
- Enable `KernelConfig.type_check` when early agent feedback is worth the extra
  check. Put host declarations in `type_check_stubs`; Arenero supplies filesystem
  helper declarations.
- Return structured results with separate value, stdout, stderr, error, retryable, truncated, and state fields.
- Let callers request a smaller output limit, but always enforce an application hard cap.
- For persistent sessions, key state by the agent thread or explicit workspace ID. Keep virtual files separate from interpreter snapshots unless their lifecycle must be identical.
- For binary artifacts, use bytes at the application transfer boundary. Use Base64 only when a text-only model or tool interface requires it.

## Verify behavior

Test the actual Monty runtime for:

- Denied ambient file, environment, network, and process access.
- Every allowed host function and mount mode.
- Duration, request timeout, memory, recursion, stream, filesystem, and saved-state limits.
- Session deletion after terminal resource failures.
- Snapshot restore with the same Monty version and comparable checkout limits.
- Thread isolation, shared-file behavior, and concurrent calls.
- Unsupported syntax and imports that the model is likely to produce.
- Binary file round trips if the application accepts images or other artifacts.
- The complete package init, edit, check, fresh test, revision, and export flow
  when the application enables Monty source package tools.

## Authoritative sources

Use versioned `docs/limitations/` in the upstream repository for CPython
differences (`limitations/` is now a symlink). Use the current runtime's release
tag, and check the specific feature page and runtime when summary notes disagree:

- [Python quickstart](https://pydantic.dev/docs/monty/quickstart/python/)
- [Python subset and limitations](https://pydantic.dev/docs/monty/limitations/)
- [Security model](https://pydantic.dev/docs/monty/concepts/security/)
- [Filesystem access](https://pydantic.dev/docs/monty/concepts/filesystem/)
- [Resource limits](https://pydantic.dev/docs/monty/concepts/resource-limits/)
- [Host functions](https://pydantic.dev/docs/monty/concepts/host-functions/)
- [Snapshots](https://pydantic.dev/docs/monty/concepts/snapshots/)
- [Released 1.1.0 limitations](https://github.com/pydantic/monty/tree/v1.1.0/docs/limitations)
- [Current development limitations](https://github.com/pydantic/monty/tree/main/docs/limitations)
- [Release history](https://github.com/pydantic/monty/releases)
