# WebMonty sandbox compatibility

This report compares WebMonty with the utilities described in the LangChain
[Deep Agents sandbox documentation](https://docs.langchain.com/oss/python/deepagents/sandboxes).
WebMonty does not implement a Deep Agents backend. It provides the independent
parts needed to build an adapter later.

## Implemented

| Capability | WebMonty support |
| --- | --- |
| Isolated execution | Monty runs in worker subprocesses and has no ambient host access. |
| Persistent execution | Memory, SQLite, and stateless session modes are available. |
| Thread or assistant scoping | The application selects the persistent `session_id`. |
| Shared files | Sessions can share one `OSAccess` object or mount set. |
| `read`, `write`, `edit`, `delete`, `ls`, `glob`, `grep` | Available inside Monty, as LangChain tools, and through `WebMontySandbox`. |
| Binary upload and download | `upload_file`, `download_file`, `upload_files`, and `download_files` preserve bytes. |
| Image input bridge | `read_media_file` creates a Base64 data URL or LangChain image block. |
| Output bounds | Kernel, stream, filesystem, search, state, and binary transfer limits are configurable. |
| Stable sandbox identifier | `WebMontySandbox.id` is available. |
| Secrets outside execution | Narrow host functions can use credentials without exposing them to Monty. |
| Network blocking | Network access is absent unless a host function provides it. |

## Limited execution

Deep Agents defines `execute` as arbitrary shell execution. Monty cannot provide
that contract because it has no shell, processes, system Python, third-party
packages, or direct network access.

`WebMontySandbox.execute` provides a smaller contract:

- `python -c '<Monty-compatible Python code>'`
- `python /absolute/virtual/file.py`

It returns combined output, an exit code, and a truncation flag. It rejects every
other command. The application must give `MONTY_SANDBOX_INSTRUCTIONS` to the model
so the model does not plan shell commands.

This limited form is useful for calculations, short transformation programs, and
virtual helper files. It is not suitable for coding agents that build packages,
run test commands, install dependencies, use Git, or call command-line programs.

## Feasible next adapters

These additions are practical if direct Deep Agents compatibility becomes useful:

| Addition | Difficulty | Notes |
| --- | --- | --- |
| Deep Agents result conversion | Low | Convert WebMonty result dataclasses to the matching Deep Agents classes. |
| Async method wrappers | Low | Run the synchronous facade through worker threads. |
| Deep Agents tool names and schemas | Low | Wrap the existing facade methods without changing filesystem behavior. |
| File metadata | Low to medium | Add byte size and timestamps where the selected filesystem backend can supply them. |
| Large-result offloading | Medium | Save oversized tool output to a bounded virtual file and return its path. |
| Per-call lower timeout | Medium | Requires a new disposable checkout policy or a worker request deadline per call. It must not restore unsafe state. |
| More complete glob rules | Medium | Add brace expansion and the exact hidden-file rules used by Deep Agents. |
| Permission or approval hooks | Medium | Add callbacks before mutation or execution, or use agent middleware. |

Do not subclass a shell-based `BaseSandbox` and let it build filesystem operations
through shell scripts. Those generated commands cannot run in Monty. A future
adapter must implement the filesystem methods directly.

## Not feasible without changing the product

These capabilities need a container, VM, managed sandbox, or another runtime:

- Arbitrary shell commands and executables.
- Git, package managers, compilers, system test runners, and Docker.
- Full CPython and third-party Python packages such as NumPy or pandas.
- Agent-in-sandbox deployment, because LangChain and the model client cannot run
  inside the Monty VM.
- General outbound network access.
- Process, thread, socket, device, or operating-system APIs.

Adding these through broad host functions would remove the main security boundary.
For example, a `run_shell(command)` hosted function would give model-written code
the host process's full command authority. Do not use hosted functions to imitate a
general shell.

## Security notes

- Treat every file and value produced by Monty as untrusted input.
- Keep credentials out of files, mounts, inputs, and environment mappings.
- Use both Monty's cumulative duration limit and the pool request timeout.
- A time limit, memory limit, or worker crash invalidates the affected persistent
  session. WebMonty deletes its saved state in these cases.
- Read-write mounts change real host files. Mount only a dedicated data directory
  that is outside import, configuration, and execution paths.
- The current Monty release is experimental. Store the Monty version with every
  snapshot and reset incompatible sessions.
