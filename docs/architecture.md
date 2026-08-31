# Arenero architecture and components

Arenero gives an application one small persistent execution service. It adds
session storage, virtual files, a limited shell, agent adapters, and application
limits around Monty. Monty remains the only Python runtime.

## System map

```mermaid
flowchart TB
    Caller["Human, agent, CLI, or application"]

    subgraph PublicAPI["Public Arenero APIs"]
        CLI["Terminal REPL"]
        Tools["LangChain and LangGraph tools"]
        Deep["Deep Agents interpreter middleware"]
        Facade["AreneroSandbox and AsyncAreneroSandbox"]
        Runtime["AreneroRuntime lifecycle owner"]
        Recursive["Recursive model runner"]
        Handle["Bound session handle"]
        Direct["PersistentMontyKernel or AsyncPersistentMontyKernel"]
    end

    Cell["Python or shell cell dispatcher"]
    Shell["Allowlisted virtual shell"]
    Kernel["Persistent kernel"]
    Store["Session store"]
    Files["Virtual filesystem"]
    Network["Bounded HTTP and HTTPS"]
    Hosted["Hosted tool guard and call ledger"]
    Modules["Static virtual-module resolver"]
    Frameworks["Application, LangChain, or Pydantic AI tools"]
    McpHub["Optional MCP hub"]
    McpServers["Configured MCP servers"]
    Models["Configured async model adapters"]

    subgraph Worker["Monty worker pool"]
        VM["One or more Monty worker processes"]
    end

    Caller --> CLI
    Caller --> Tools
    Caller --> Deep
    Caller --> Facade
    Caller --> Direct
    Caller --> Runtime
    Caller --> Recursive
    Runtime --> Direct
    Direct --> Handle
    Recursive --> Kernel
    Recursive <--> Models
    Handle --> Kernel
    CLI --> Cell
    Tools --> Cell
    Deep --> Kernel
    Deep --> Modules
    Facade --> Cell
    Cell -->|Python| Kernel
    Cell -->|Shell cell| Shell
    Direct --> Kernel
    Shell --> Kernel
    Kernel <--> Store
    Kernel <--> Files
    Modules --> Files
    Modules --> Kernel
    Kernel --> VM
    VM -->|Selected function calls| Hosted
    Hosted <--> Frameworks
    VM -->|Approved file operations| Files
    Shell --> Files
    Shell -->|curl only| Network
    Kernel -->|Fixed host functions| McpHub
    McpHub -->|Promoted function| Hosted
    McpHub <--> McpServers
```

The cell dispatcher chooses Python or virtual shell execution before a Monty feed
starts. This prevents a hosted function from entering the same persistent session
again.

## Components

| Component | Main module | Responsibility |
| --- | --- | --- |
| Terminal REPL | `cli.py` | Parse CLI options, select a store, and run or administer a session. |
| Cell dispatcher | `cell.py` | Route percent commands, normal Python, `!COMMAND`, and `%%shell` cells and return common result fields. |
| Capability report | `capabilities.py` | Report active limits, commands, storage, network, MCP, bootstrap, and sync or async rules and generate model instructions. |
| Bound session | `session.py` | Keep one trusted session ID with execute, file, reset, info, and fork operations. |
| Static virtual modules | `modules.py` | Resolve bounded top-level imports, build isolated module objects, and generate one Monty feed. |
| Runtime owner | `runtime.py` | Close a kernel and its optional MCP hub in dependency order. |
| Event records | `events.py` | Send safe execution, session, file, network, MCP, RLM, speculation, and runtime metadata to one callback. |
| Hosted tools | `hosted_tools.py` | Generate function catalogs, apply per-operation budgets and sequencing, convert boundary values, and record nested calls. |
| Framework adapters | `langchain_hosted_tools.py` and `pydantic_ai_hosted_tools.py` | Keep public framework validation and callbacks, inject current runtime values, and document which parent dispatch middleware is bypassed. |
| Recursive model runner | `rlm.py` | Keep context in Monty, run bounded root-model loops, expose plain and recursive model calls, isolate child sessions, and report shared usage. |
| Speculative call coordinator | `speculation.py` | Inspect streamed code, run selected calls early, use a forked Monty shadow, and let real execution claim matching results. |
| LangChain RLM adapter | `langchain_rlm.py` | Call or stream chat models through `ainvoke()` and `astream()`, then normalize text and reported usage. |
| Doctor | `doctor.py` | Check Monty, worker startup, optional packages, workspace access, and MCP status. |
| Persistent sync kernel | `kernel.py` | Restore, execute, limit, save, reset, and inspect Monty sessions. |
| Persistent async kernel | `async_kernel.py` | Run persistent Python, virtual `fs_*` helpers, file transfer for resolution, `execute_file`, and `execute_module` on native `AsyncMonty` workers. |
| Session stores | `store.py` | Keep opaque snapshots in memory, SQLite, or nowhere. Apply revision checks and expiry. |
| Filesystem helpers | `filesystem.py` | Implement bounded `read`, `write`, `edit`, `mkdir`, `delete`, `ls`, `glob`, and `grep`. |
| Persistent workspace | `workspace.py` | Own one confined read-write host directory mount. |
| Virtual shell | `shell.py` | Parse and dispatch the documented command allowlist without starting host processes. |
| Network policy | `network.py` | Check URLs, addresses, methods, redirects, time, headers, and byte limits for virtual `curl`. |
| Sandbox facade | `sandbox.py` | Group execution, files, binary transfer, and a stable sandbox ID in one framework-neutral API. |
| Media bridge | `media.py` | Read virtual binary files and create data URLs or image content blocks. |
| Agent adapters | `langchain_tools.py` and `langchain_filesystem_tools.py` | Build thread-scoped LangChain and LangGraph tools. |
| Deep Agents middleware | `deepagents.py` | Add async interpreter tools, live prompts, PTC runtime injection, three state modes, subagent calls, and private checkpoint snapshots. |
| Optional integration loader | `integrations.py` | Keep LangChain out of the core import path until tools are requested. |
| MCP hub | `mcp.py` | Keep configured async MCP clients on a host event loop and expose bounded sync or async host functions to Monty. |
| MCP artifacts | `mcp_artifacts.py` | Decode supported MCP Base64 blocks and write them through a supplied virtual-file writer. |

## Python execution lifecycle

```mermaid
sequenceDiagram
    participant C as Caller
    participant K as Kernel
    participant S as Session store
    participant M as Monty checkout
    participant F as Virtual files or host tools
    participant T as Hosted tool guard
    participant H as Optional MCP hub

    C->>K: execute(session_id, code, state_policy)
    K->>K: Lock the session
    K->>S: Load snapshot and revision
    alt Saved compatible state exists
        K->>M: Restore a fresh checkout
    else New, expired, or stateless session
        K->>M: Create a fresh checkout
        K->>M: Feed configured bootstrap files and code
    end
    K->>M: Feed user code
    M->>F: Use only configured access
    F-->>M: Return bounded values
    opt Selected hosted tool call
        M->>T: Keyword arguments
        T->>T: Reserve call and time budget
        T-->>M: Plain bounded value
    end
    opt MCP function call
        M->>H: Search, describe, refresh, or call
        H-->>M: Plain bounded data
    end
    M-->>K: Result, streams, or error
    alt State policy allows save
        K->>M: Dump safe final state
        K->>S: Save with expected revision
    else State policy blocks save
        K->>K: Keep the previous snapshot
    end
    K-->>C: Structured result, error guidance, and statistics
```

`execute_file()` reads one virtual Python file, feeds its complete source, then
feeds optional follow-up code. It saves the session once after the last successful
feed. It does not perform a Python import.

`execute_module()` reads and parses a bounded static file graph on the host. It
rewrites supported top-level imports into simple isolated module objects and runs
the complete generated source as one Monty feed. The generated feed is not type
checked because its module objects need dynamic attributes. User source never
executes in the host parser. See [Static virtual modules](virtual-modules.md).

Normal code errors can leave useful changes. Arenero saves the valid state that
exists after the error. A time limit, memory limit, or worker failure makes the
session unsafe, so Arenero removes its saved state.

The `success` state policy keeps the previous snapshot after a normal code error.
The `never` policy does not save the operation. Neither policy rolls back file,
network, or MCP side effects. Session forking copies the opaque Monty snapshot,
including its remaining duration budget, but does not copy files or host tools.

## State ownership

Python state and file state are intentionally separate:

| State | Owner | Persistence |
| --- | --- | --- |
| Python globals, functions, classes, and remaining duration budget | Monty snapshot in the selected `SessionStore` | Follows memory, SQLite, or stateless mode. |
| Virtual shell working directory | Reserved value in the Monty session | Follows the Python session. |
| `OSAccess` memory files | The caller-owned `OSAccess` object | Lives while that object lives. It is not in SQLite. |
| `WorkspaceConfig` files | Dedicated host directory through `MountDir` | Survives process restarts. It is not in SQLite. |
| Caller-supplied mounts | Caller-owned `MountDir` objects | Follows the selected mount mode and host directory. |
| Host functions | `external_lookup` on each kernel checkout | Never stored in a snapshot. The application supplies them again. |
| Hosted tool catalog, callbacks, and call budgets | Host process and one execution scope | Catalog configuration is process memory. Call budgets and records reset for each operation. |
| Deep Agents graph snapshot | Private LangGraph state when enabled | Contains full Monty bytes and optional HMAC. It does not contain files or host resources. |
| Network policy | `NetworkConfig` in the host process | Never enters Monty state. |
| MCP clients, credentials, and catalog | Caller-owned `McpHub` in the host process | Live while the hub is active. Reconnect after restart. Never enters a snapshot. |

```mermaid
flowchart LR
    SessionID["session_id"] --> Snapshot["Monty snapshot"]
    Snapshot --> PythonState["Python values and functions"]
    Snapshot --> Cwd["Virtual shell directory"]
    Snapshot --> Budget["Remaining cumulative duration"]

    FileConfig["Filesystem configuration"] --> MemoryFiles["OSAccess memory files"]
    FileConfig --> Workspace["Host-backed workspace"]
    FileConfig --> Mounts["Caller-supplied mounts"]

    Snapshot -. does not contain .-> FileConfig
```

Resetting a session removes Python state and the shell directory. It does not
delete shared files. Applications must define file cleanup separately.

## Deep Agents execution path

```mermaid
sequenceDiagram
    participant G as Deep Agents graph
    participant A as Arenero middleware
    participant K as Async kernel
    participant M as Monty
    participant T as LangChain tools

    G->>A: abefore_agent(state, thread_id)
    opt checkpoint_state
        A->>K: Restore private snapshot to working store
    end
    G->>A: awrap_model_call(live tools)
    A-->>G: Dynamic limits and async function catalog
    G->>A: python_interpreter(code)
    A->>K: execute(session_id, operation tools)
    K->>M: Run Monty feed
    opt Programmatic tool call
        M->>A: await selected_tool(...)
        A->>T: Inject derived ToolRuntime and call ainvoke
        T-->>A: Plain value or task response text
        A-->>M: Boundary-safe value
    end
    M-->>K: Result and snapshot
    K-->>A: Structured result and host_calls
    A-->>G: ToolMessage
    G->>A: aafter_agent
    opt checkpoint_state
        A-->>G: Private full snapshot and HMAC
    end
```

Nested PTC calls bypass the parent ToolNode. Parent per-tool approval middleware
does not run for them. The outer interpreter call, the called tool, or the
subagent must enforce the required approval policy.

## Recursive model execution path

```mermaid
flowchart TD
    Root["RLM root model"] --> Code["Generated Monty code"]
    Root -. "optional text deltas" .-> Planner["Safe speculative planner"]
    Planner --> Early["Selected early host calls"]
    Planner --> Shadow["Forked Monty shadow"]
    Shadow --> Early
    Code --> Context["Inspect external context variable"]
    Code --> Plain{"Subquery type"}
    Plain -->|llm_query| Model["Plain async model call"]
    Plain -->|rlm_query below max depth| Child["New temporary child session"]
    Plain -->|rlm_query at max depth| Model
    Child --> ChildRoot["Child root-model loop"]
    ChildRoot --> ChildCode["Child Monty code"]
    ChildCode --> Answer["Child answer text"]
    Model --> Answer
    Early -. "matching result claim" .-> Code
    Answer --> Code
    Code --> Final["answer content and ready flag"]
    Final --> Cleanup["Delete root and child snapshots"]
```

The model adapter runs on the host. It receives only the prompt assembled by the
RLM. The full external context enters Monty through a plain feed input and then
becomes persistent session state for that temporary RLM node.

A parent Monty checkout remains active while `rlm_query` runs. Each recursive
level therefore needs another worker. Child sessions use different session IDs
and the same shared model-call ledger. Direct kernel filesystem access is disabled
for recursive children because the parent holds the kernel filesystem lock. An
application can explicitly inherit hosted file tools when children must share
files.

## Storage modes

| Store | Later calls | Later human turns | Process restart | Main use |
| --- | --- | --- | --- | --- |
| `InMemorySessionStore` | Yes | Yes, while the application remains active | No | Services with short-lived process memory. |
| `SQLiteSessionStore` | Yes | Yes | Yes | Durable local agents and terminal sessions. |
| `StatelessSessionStore` | No | No | No | Independent calculations and low-state tasks. |

The store holds opaque Monty bytes, the Monty version, a revision, and an update
time. It does not understand user variables. Snapshots from a different Monty
version require a reset.

## Concurrency and consistency

Arenero uses these controls:

1. A per-session thread lock or async lock keeps calls for one session in order.
2. The configured Monty worker pool lets different sessions run at the same time.
3. `worker_checkout_timeout_seconds` applies bounded backpressure when all
   workers are busy.
4. A process-local filesystem lock serializes filesystem-capable calls when one
   kernel shares files between sessions.
5. A store revision check detects a save based on stale state. SQLite also uses a
   transaction.

```mermaid
flowchart LR
    Batch["Concurrent agent batch"] --> A1["agent-1 session lock"]
    Batch --> A2["agent-2 session lock"]
    Batch --> A3["agent-3 session lock"]
    A1 --> Pool["Shared AsyncMonty pool"]
    A2 --> Pool
    A3 --> Pool
    Pool --> W1["Worker 1"]
    Pool --> W2["Worker 2"]
    Pool --> Queue["Bounded checkout wait"]
    W1 --> Store["Shared session store"]
    W2 --> Store
```

`min_worker_processes`, `max_worker_processes`,
`worker_checkout_timeout_seconds`, and `max_checkouts_per_worker` configure both
kernel types. The native async kernel must stay on one application event loop.
Its async host functions do not block that loop while they wait.

The shared filesystem lock is still a batch bottleneck by design. A kernel with
`OSAccess`, mounts, or `WorkspaceConfig` serializes complete filesystem-capable
operations. Use separate filesystems or kernels when agents need independent file
throughput.

These controls do not create a distributed lock. Two application processes that
write the same host workspace still need external coordination. A custom store
also needs shared revision semantics when many processes use it.

## Security boundary

Monty has no ambient filesystem, environment, socket, subprocess, package, or host
object access. Access enters through explicit configuration:

- `inputs` for ordinary eager values.
- `external_lookup` for narrow lazy values and sync or async host functions.
  Async functions require `AsyncPersistentMontyKernel` and `await` in Monty code.
- `HostedToolSet` for schema-aware selected tools with call budgets, generated
  declarations, boundary conversion, and a bounded nested-call ledger.
- `OSAccess` or `MountDir` for virtual files.
- The host-side virtual shell for its fixed command handlers.
- `NetworkConfig` for the virtual `curl` handler.
- An optional `McpHub` for six fixed discovery, promotion, and tool-call functions.

Hosted functions execute with host authority. Validate all arguments and return
only supported plain values. Keep secrets in narrow host functions or trusted
header providers. Do not place secrets in sandbox inputs, virtual files, or
prompts.

MCP tools keep the authority of their configured server. Monty limits model code,
but it does not limit what an approved remote tool can do. The application selects
server commands, URLs, credentials, and qualified tool allowlists outside model
control. See [MCP tools inside the REPL](mcp.md).

Hosted tools keep the authority of their source framework. Arenero does not make
a broad application tool safe. It keeps the source framework's public invoke
path and adds operation limits around it. See
[Hosted tools and code-mode composition](hosted-tools.md).

The default network policy permits bounded requests to public HTTP and HTTPS URLs.
It blocks private and non-global addresses. Use narrow URL prefixes when an agent
can access sensitive data because it can send data that it knows to a public
service.

## Extension points

| Need | Extension point | Rule |
| --- | --- | --- |
| Another persistence backend | Implement `SessionStore` | Keep Monty version data, revisions, expiry, and safe metadata listing. |
| Domain actions | Add `HostedToolSet` or narrow `external_lookup` callables | Prefer hosted tools when schemas, budgets, records, or framework adapters are useful. |
| Tenant-aware state | Set `session_id` or provide `session_id_resolver` | Include a trusted tenant key outside model control. |
| Tenant-aware files | Create one `OSAccess` or workspace per tenant | Do not use a shared mount when file isolation is required. |
| A smaller shell | Set `KernelConfig.allowed_shell_commands` | Include aliases separately, such as `python3`. |
| Restricted network access | Set `NetworkConfig` | Prefer exact public origins and path prefixes; disable `curl` when unused. |
| Another agent framework | Wrap `AreneroSandbox` and adapt its tools to `HostedTool` | Keep its result, path, persistence, nested-call, and execution limits visible to the model. |
| Another model image format | Convert `MediaFile.content` on the host | The model cannot read a path or Base64 text as an image without a content block. |
| Configured external tools | Add `McpHub` with the `mcp` extra | Keep server configuration outside model control and set call, schema, argument, result, and concurrency limits. |

Do not add a broad `run_shell(command)` host function. It would give model-written
code the host process's command authority and remove the main product boundary.

## Useful products to build on top

- A conversation-scoped calculation notebook.
- A report agent that keeps intermediate text and JSON artifacts.
- A constrained HTTP data collector with an exact endpoint allowlist.
- A domain assistant that combines Monty calculations with narrow business APIs.
- A shared virtual workspace for collaborating agents in one trusted tenant.
- A framework adapter that maps another agent SDK to `AreneroSandbox`.
- A local assistant that recovers state and files after an application restart.

Use a real execution sandbox for repository coding agents, package builds, system
tests, browser automation, native libraries, or arbitrary command-line tools.

## Related references

- [Usage and capabilities](usage-and-capabilities.md)
- [Virtual shell reference](virtual-shell.md)
- [Sandbox compatibility](sandbox-compatibility.md)
- [Design decisions](design-decisions.md)
- [MCP tools inside the REPL](mcp.md)
- [Hosted tools and code-mode composition](hosted-tools.md)
- [Quality-of-life APIs](quality-of-life.md)
