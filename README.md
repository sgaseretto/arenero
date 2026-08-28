# WebMonty

WebMonty is a persistent Python tool built on
[Monty](https://github.com/pydantic/monty). It can run as a terminal REPL or as a
LangChain and LangGraph tool.

Each session can use in-memory, SQLite, or stateless storage. In-memory storage is
the default kernel mode. It keeps variables and functions across agent loops and
separate human turns while the application process stays active.

## Install and test

```console
uv sync
uv run python -m unittest discover -s tests -v
```

## Terminal REPL

```console
uv run webmonty
```

The CLI uses SQLite by default, with `.webmonty/sessions.db` as its database. Its
default session name is `cli`.

```console
uv run webmonty --session research
uv run webmonty --store memory --session research
uv run webmonty --store none
uv run webmonty --session research --output-limit 2000
uv run webmonty --session research --reset
```

The storage modes are:

| Mode | Survives agent calls | Survives human turns | Survives process restart |
| --- | --- | --- | --- |
| `memory` | Yes | Yes | No |
| `sqlite` | Yes | Yes | Yes |
| `none` | No | No | No |

## Kernel API

```python
from webmonty import KernelConfig, PersistentMontyKernel

config = KernelConfig(
    default_output_limit_chars=8_000,
    max_output_limit_chars=50_000,
    max_stream_bytes=1_000_000,
    max_state_bytes=5_000_000,
    max_duration_seconds=5,
    max_memory_bytes=64 * 1024 * 1024,
    max_recursion_depth=100,
    request_timeout_seconds=10,
    session_ttl_seconds=24 * 60 * 60,
)

with PersistentMontyKernel(config=config) as kernel:
    kernel.execute("thread-1", "x = 21")
    execution = kernel.execute("thread-1", "x * 2")
    print(execution["result"])
    # 42
```

The kernel uses `InMemorySessionStore` when no store is given. Select another mode
with an explicit store:

```python
from webmonty import (
    InMemorySessionStore,
    PersistentMontyKernel,
    SQLiteSessionStore,
    StatelessSessionStore,
)

# Default: state remains while this kernel process is active.
memory_kernel = PersistentMontyKernel()

# The same in-memory state can also be shared by replacement kernel objects.
memory_store = InMemorySessionStore()
memory_kernel = PersistentMontyKernel(memory_store)

# State remains after a process restart.
sqlite_kernel = PersistentMontyKernel(SQLiteSessionStore("sessions.db"))

# Every execution starts with empty state.
stateless_kernel = PersistentMontyKernel(StatelessSessionStore())
```

`execute()` returns a dictionary with these main fields:

```python
{
    "ok": True,
    "result": 42,
    "stdout": "",
    "stderr": "",
    "error": None,
    "truncated": {
        "result": False,
        "stdout": False,
        "stderr": False,
        "error": False,
    },
    "state_saved": True,
    "state_restored": True,
}
```

## LangChain and LangGraph

The adapter creates these Python tools:

- `python_repl` runs code in the current graph thread.
- `python_repl_execute_file` runs a virtual Python file and optional follow-up
  code in one tool call.
- `reset_python_repl` removes that thread's Python state.

It also creates shared virtual filesystem tools:

- `read_file`
- `write_file`
- `edit_file`
- `delete_file`
- `list_files`
- `glob_files`
- `grep_files`

```python
from langchain.agents import create_agent

from webmonty import (
    PersistentMontyKernel,
    SQLiteSessionStore,
    create_python_repl_tools,
)

# In-memory state survives separate human turns in this running process.
kernel = PersistentMontyKernel()
tools = create_python_repl_tools(kernel)

agent = create_agent(
    model="your-model",
    tools=tools,
    system_prompt=(
        "Use python_repl for calculations and temporary data. Use "
        "python_repl_execute_file when a known virtual Python file must be loaded. "
        "Use the file tools to inspect and change virtual files."
    ),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Calculate 19 * 23."}]},
    {"configurable": {"thread_id": "conversation-123"}},
)

kernel.close()
```

Use `PersistentMontyKernel(SQLiteSessionStore("sessions.db"))` when the agent must
recover its Python state after the application process restarts.

The adapter reads `thread_id` from `ToolRuntime`. The model cannot select another
thread. You can provide a custom `session_id_resolver` when a session key must also
contain a user or tenant ID.

## Configurable output limits

The application controls two limits:

- `default_output_limit_chars` is used when a tool call does not request a limit.
- `max_output_limit_chars` is the hard maximum that a tool call cannot exceed.

The `python_repl` and `python_repl_execute_file` tools expose
`output_limit_chars` to the model. The model can use a smaller value when it only
needs a short answer:

```json
{
  "code": "large_result[:10]",
  "output_limit_chars": 1000
}
```

The limit applies separately to `result`, `stdout`, `stderr`, and the error message.
Set it to `0` to omit the contents of these fields. The response reports which
fields were truncated and whether the requested value was capped.

`max_stream_bytes` is separate. It stops code that prints too much before trimming
happens. This protects the host process.

## Approved host functions

Monty code has no direct host file, environment, or network access. You can expose
specific synchronous functions:

```python
def lookup_price(product_id: str) -> float:
    return 12.50


kernel = PersistentMontyKernel(
    external_lookup={"lookup_price": lookup_price},
)

kernel.execute("thread-1", 'lookup_price("item-42")')
```

Only functions in `external_lookup` are available inside the sandbox.

## Filesystem access

Filesystem access is off by default. Enable it with an in-memory filesystem, one
or more directory mounts, or both. The same filesystem settings apply when the
kernel is used through the LangChain and LangGraph tools.

### In-memory files

Pass an `OSAccess` object to `os_handler`. Keep the object alive for as long as
its files must remain available:

```python
from pydantic_monty import MemoryFile, OSAccess

from webmonty import PersistentMontyKernel

filesystem = OSAccess(
    files=[MemoryFile("/data/input.txt", "21")],
)

with PersistentMontyKernel(os_handler=filesystem) as kernel:
    kernel.execute(
        "thread-1",
        "from pathlib import Path\n"
        "value = int(Path('/data/input.txt').read_text())\n"
        "Path('/data/result.txt').write_text(str(value * 2))",
    )
    result = kernel.execute("thread-1", "open('/data/result.txt').read()")
    print(result["result"])
    # 42
```

The same `OSAccess` object is shared by every session that uses that kernel. Use a
different object per user or thread when files must be isolated. These files are
not part of a Monty session dump, so SQLite does not save them.

### Host directory mounts

Pass one `MountDir` or a list of mounts to `mounts`. Virtual paths always use POSIX
syntax, including on Windows.

```python
from pathlib import Path

from pydantic_monty import MountDir

from webmonty import PersistentMontyKernel

workspace = Path("agent-workspace")
workspace.mkdir(exist_ok=True)

with (
    MountDir(
        host_path=workspace,
        virtual_path="/workspace",
        mode="read-write",
        write_bytes_limit=1_000_000,
        memory_usage_limit=10_000_000,
    ) as mount,
    PersistentMontyKernel(mounts=mount) as kernel,
):
    kernel.execute(
        "thread-1",
        "open('/workspace/result.txt', 'w').write('saved')",
    )
```

Mount modes are:

- `read-only`: code can read host files but cannot change them.
- `read-write`: changes go directly to the host directory and survive later calls
  and process restarts.
- `overlay`: writes are temporary and are discarded when the Monty feed that made
  them ends.

`MountDir` holds a host directory open. The caller owns it and must close it, as in
the example. A read-write mount can change or delete host files. Use dedicated
directories when agents must be isolated. `write_bytes_limit` limits writes in one
call, and `memory_usage_limit` limits memory used by the mount.

`reset()` removes Python state only. It does not clear an `OSAccess` object or
delete files in a mounted directory. Filesystem lifecycle, per-thread workspace
creation, cleanup, and SQLite file storage are not part of this basic support.

### Filesystem helpers and agent tools

WebMonty loads these bounded helper functions into every REPL session:

- `fs_read(path, offset=0, limit=None)`
- `fs_write(path, content, append=False, create_parents=False)`
- `fs_edit(path, old_text, new_text, replace_all=False)`
- `fs_delete(path, recursive=False)`
- `fs_ls(path="/", recursive=False, limit=None)`
- `fs_glob(pattern, root="/", limit=None)`
- `fs_grep(query, root="/", pattern="**/*", regex=False,
  case_sensitive=True, limit=None)`

They run inside Monty and use the same `OSAccess` or `MountDir` configuration as
normal `open()` and `Path` calls. Hosted functions are not used, so these helpers
do not bypass Monty's mount path checks.

```python
with PersistentMontyKernel(mounts=mount) as kernel:
    result = kernel.execute(
        "thread-1",
        "fs_write('/workspace/notes.txt', 'Alpha\\nBeta\\n')\n"
        "fs_edit('/workspace/notes.txt', 'Beta', 'Gamma')\n"
        "fs_grep('gamma', root='/workspace', case_sensitive=False)",
    )
```

The LangChain and LangGraph file tools call these same helpers through the kernel.
The runtime thread ID selects the Python session, but files remain shared when the
kernel uses one shared `OSAccess` object or mount set.

Host application code can call the same helper path without building Python source:

```python
result = kernel.call_filesystem(
    "thread-1",
    "grep",
    {
        "query": "PersistentMontyKernel",
        "root": "/workspace",
        "pattern": "**/*.py",
        "regex": False,
        "case_sensitive": True,
        "limit": 20,
    },
)
```

`edit_file` uses exact text replacement. By default, the old text must occur once.
Set `replace_all=true` to replace every occurrence. Recursive deletion is off by
default, and the virtual root `/` cannot be deleted.

Configure hard filesystem limits separately from normal kernel limits:

```python
from webmonty import FilesystemConfig, PersistentMontyKernel

filesystem_config = FilesystemConfig(
    default_read_limit_chars=8_000,
    max_read_limit_chars=200_000,
    max_write_chars=1_000_000,
    default_result_limit=100,
    max_result_limit=1_000,
    max_scan_entries=5_000,
    max_grep_chars=2_000_000,
    max_depth=20,
    max_transfer_bytes=10_000_000,
)

kernel = PersistentMontyKernel(
    mounts=mount,
    filesystem_config=filesystem_config,
)
```

The model can request smaller read and result limits. The configured maximums are
hard caps. Kernel output trimming still applies after a helper returns its result.

### Execute a virtual Python file

Monty does not import Python files from the virtual filesystem and does not expose
`eval()` or `exec()`. WebMonty can still read a file and submit its source directly
to the persistent Monty session:

```python
filesystem = OSAccess(
    files=[
        MemoryFile(
            "/helper.py",
            "def double(value):\n    return value * 2\n",
        )
    ],
)

with PersistentMontyKernel(os_handler=filesystem) as kernel:
    result = kernel.execute_file(
        "thread-1",
        "/helper.py",
        code="double(21)",
    )
    print(result["result"])
    # 42

    # The helper remains in the persistent Python session.
    kernel.execute("thread-1", "double(5)")
```

The optional `code` runs only after the complete file succeeds. The result and
captured output cover both the file and the follow-up code. The operation saves
the session once, after the final feed.

Agents receive the same feature as `python_repl_execute_file`:

```json
{
  "file_path": "/helper.py",
  "code": "double(21)",
  "output_limit_chars": 1000
}
```

The file source stays between WebMonty and Monty. It is not returned to the model,
and the output limit does not shorten it. This is file execution, not normal Python
module importing. Names defined by the file enter the current REPL namespace.

## Persistence and recovery

- In-memory storage is the default and stays active between human turns.
- SQLite storage survives application restarts.
- Stateless storage starts clean for every tool call.
- In-memory and SQLite writes use revision checks.
- SQLite writes use transactions.
- Calls for one session run in order inside one process.
- Different sessions can run independently.
- A Monty crash, time limit, or memory limit resets the affected session.
- Code errors save valid changes made before the error.
- Old sessions expire after `session_ttl_seconds`.
- Dumps made by another Monty version require a session reset.
- Oversized session states do not replace the last saved state.

Monty supports a useful subset of Python. See the
[Monty project](https://github.com/pydantic/monty) for its current language limits.

## Binary files and model images

The text tools remain text-only. Application code can move binary files through a
separate transfer plane:

```python
from pathlib import Path

from pydantic_monty import OSAccess

from webmonty import PersistentMontyKernel, read_media_file

filesystem = OSAccess()
with PersistentMontyKernel(os_handler=filesystem) as kernel:
    image = Path("ant-image.png").read_bytes()
    kernel.upload_file("/images/ant-image.png", image)

    downloaded = kernel.download_file("/images/ant-image.png")
    assert downloaded == image

    media = read_media_file(kernel, "/images/ant-image.png")
    image_block = media.as_langchain_image(detail="low")
```

`image_block` is an OpenAI-style LangChain content block for a `HumanMessage`
content list. WebMonty reads the bytes from the virtual filesystem and Base64-
encodes them on the host. The model does not gain direct filesystem access.

The binary transfer hard cap is `FilesystemConfig.max_transfer_bytes`. A caller
can request a lower download cap but cannot exceed this application limit.

## Limited sandbox facade

`WebMontySandbox` groups execution, file utilities, binary transfer, and a stable
sandbox ID behind one framework-neutral object:

```python
from pydantic_monty import OSAccess

from webmonty import (
    MONTY_SANDBOX_INSTRUCTIONS,
    PersistentMontyKernel,
    WebMontySandbox,
)

with PersistentMontyKernel(os_handler=OSAccess()) as kernel:
    sandbox = WebMontySandbox(kernel, session_id="thread-1")
    sandbox.write("/helper.py", "value = 21\n")
    sandbox.execute("python /helper.py")
    result = sandbox.execute("python -c 'value * 2'")
    print(result.output)
    # 42
```

The `execute` method is deliberately not a shell. It accepts only:

- `python -c '<Monty-compatible Python code>'`
- `python /absolute/virtual/file.py`

It rejects shell operators, pipes, redirection, Git, package installation,
network commands, and all other executables. Per-call timeout overrides are not
available; configure time and request limits on `KernelConfig`.

Give `MONTY_SANDBOX_INSTRUCTIONS` to an LLM whenever it can call this limited
execute operation. See [the sandbox compatibility report](docs/sandbox-compatibility.md)
for the supported Deep Agents concepts and the remaining gaps.
