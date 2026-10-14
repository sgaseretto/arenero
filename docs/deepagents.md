# Deep Agents interpreter middleware

`AreneroInterpreterMiddleware` gives a Deep Agents or LangChain agent one
asynchronous Monty interpreter. The model gets two tools:

- `python_interpreter` runs one Monty Python cell.
- `python_interpreter_module` resolves and runs a bounded static virtual module
  graph, with optional follow-up code in the same call.

The middleware also adds current limits and hosted-function signatures to every
model request. It reads the live tool list after other middleware changes it, so
the instructions match the functions that the next interpreter call can use.

Install the optional integration:

```console
uv add "arenero[deepagents]"
```

## Minimal setup

```python
from arenero import AsyncPersistentMontyKernel, create_deepagents_middleware
from deepagents import create_deep_agent

kernel = AsyncPersistentMontyKernel()
interpreter = create_deepagents_middleware(kernel=kernel, mode="thread")
agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[interpreter],
)

async with kernel:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Calculate 37 squared."}]},
        {"configurable": {"thread_id": "conversation-42"}},
    )
```

Use `ainvoke()` or `astream()`. Async Monty can suspend while it waits for a host
tool, and different session IDs can use different worker processes. A synchronous
agent call can see the tools, but an interpreter call returns
`AsyncInvocationRequired` with a hint to use the async API.

The caller owns a supplied kernel and must close it. If the middleware creates
its default kernel, call `await interpreter.aclose()` or use the middleware as an
async context manager.

## State lifetime modes

| Mode | Same interpreter call | Later call in one agent turn | Later human turn with the same thread ID |
| --- | --- | --- | --- |
| `thread` | Yes | Yes | Yes, while its selected store remains available |
| `turn` | Yes | Yes | No; `aafter_agent` deletes the turn session |
| `call` | Yes | No | No; the tool deletes state in `finally` |

`thread` is the default. A stable hash of `configurable.thread_id` becomes the
kernel session key. `session_prefix` separates two graphs that share one store.
When no thread ID exists, one middleware-instance fallback is used. Production
agents should always supply a thread ID.

```mermaid
flowchart LR
    Request["Agent request"] --> Before["abefore_agent"]
    Before --> Model["Model sees live Arenero instructions"]
    Model --> Eval["python_interpreter"]
    Eval --> Session{"mode"}
    Session -->|thread| Thread["Stable thread session"]
    Session -->|turn| Turn["Temporary turn session"]
    Session -->|call| Call["Fresh call session"]
    Thread --> Pool["Shared AsyncMonty pool"]
    Turn --> Pool
    Call --> Pool
    Pool --> After["aafter_agent"]
    After -->|turn or call| Delete["Delete Python snapshot"]
    After -->|thread| Keep["Keep in store or graph checkpoint"]
```

Virtual files do not follow these modes. Files belong to the configured virtual
filesystem or workspace and can be shared by sessions and agents. Resetting or
restoring Python state does not undo file, network, MCP, or application-tool side
effects.

## Programmatic tool composition

Use `ptc` as an allowlist of current tool names or explicit `BaseTool` objects:

```python
interpreter = create_deepagents_middleware(
    kernel=kernel,
    ptc=("search", "lookup_customer"),
)
```

The model receives exact async declarations, such as:

```python
async def search(*, query: str) -> list[dict]: ...
```

It can compose calls in one cell:

```python
found = await search(query="Arenero")
names = [item["name"] for item in found]
names[:5]
```

The adapter supplies the same hidden values that LangGraph ToolNode normally
injects: `ToolRuntime`, state, store, context, configuration, stream writer, and
a synthetic child tool-call ID. It still uses each tool's public `invoke()` or
`ainvoke()` path for validation and callbacks.

Programmatic calls do not pass through the parent ToolNode because Monty calls
them from inside one interpreter tool execution. This has important effects:

- Parent per-tool human approval or `interrupt_on` middleware does not run for
  each nested call. Approve the outer interpreter, put approval inside the
  called tool or subagent, or do not expose that tool.
- A returned LangGraph `Command` is not applied to parent graph state. For the
  Deep Agents `task` tool, Arenero returns the final subagent message to Monty.
- Side effects are real and are not transactional.
- `HostedToolConfig` limits calls, total call time, overlap, and returned records
  for each interpreter operation.
- Interpreter tools and other code-execution tools are always excluded from the
  nested catalog. This prevents recursive interpreter calls.

## Recursive model calls

Configure an `RLM` when Monty code must call a model or create child RLMs:

```python
from langchain.chat_models import init_chat_model
from arenero import RLM, RLMConfig, adapt_langchain_model

model = init_chat_model("openai:gpt-5.5")
rlm = RLM(
    adapt_langchain_model(model),
    config=RLMConfig(max_depth=1, max_model_calls=32),
)
interpreter = create_deepagents_middleware(
    kernel=kernel,
    rlm=rlm,
    mode="thread",
)
```

The interpreter receives five async functions:

```python
summary = await llm_query(prompt="Summarize: " + context_part)
summaries = await llm_query_batched(prompts=prompts)
answer = await rlm_query(prompt=child_task)
answers = await rlm_query_batched(prompts=child_tasks)
usage = await rlm_usage()
```

Plain model calls do not create a REPL. Recursive calls create separate temporary
Monty sessions. At `max_depth`, a recursive call becomes a plain call. The active
agent interpreter never calls itself and its Python state is not copied into
children.

Each interpreter operation gets a new shared RLM call, token, cost, concurrency,
and wall-time ledger. Token and cost limits depend on adapter-reported usage.
Configure enough Monty workers for the parent and recursive children. A chain of
depth `N` needs at least `N + 1` workers.

Direct kernel filesystem mounts are disabled in recursive child sessions because
the parent feed holds the shared filesystem lock. Child RLMs can use hosted file
tools inherited through normal `ptc` selection. See
[Recursive Language Models](rlm.md) for the full API and limits.

The direct interpreter functions do not speculate because the parent agent owns
model streaming. An application can use `SpeculativeTurn` in that outer stream,
or use speculation in the separate `RLM.complete()` controller. See
[Speculative programmatic tool calling](speculative-tool-calling.md).

## Subagent bridge

When `subagents=True`, the middleware detects a live Deep Agents `task` tool with
`description` and `subagent_type` fields. It exposes this async function even
when normal PTC is disabled:

```python
answer = await task(
    description="Check the calculation independently.",
    subagent_type="reviewer",
)
```

The subagent receives a derived runtime with a child tool-call ID. Arenero
returns the task tool's final `ToolMessage.content`. It does not merge the task
tool's `Command.update` into the parent graph. Set `subagents=False` when each
dispatch must use the normal parent tool path.

## Checkpoint-backed Python snapshots

Arenero normally stores snapshots in the kernel's `SessionStore`. Set
`checkpoint_state=True` to put the full opaque Monty snapshot in private
LangGraph state instead:

```python
interpreter = create_deepagents_middleware(
    kernel=kernel,
    mode="thread",
    checkpoint_state=True,
    max_snapshot_bytes=5_000_000,
    snapshot_signing_key=application_secret,
)
```

Before the agent starts, the middleware restores private snapshot bytes into the
working kernel store. After the agent ends, it copies the current bytes back to
private state and removes the working copy. A LangGraph checkpointer can then
persist and time-travel the Python snapshot with the graph state.

Set a high-entropy HMAC key when checkpoint storage is not fully trusted. The
signature includes the stable Arenero session ID, so a snapshot cannot be moved
to a different thread. A missing, changed, or oversized signed snapshot is
rejected and cleared.

This feature stores full snapshots. It does not implement snapshot deltas.
Monty's resource settings, native directory, and RNG state are part of the
snapshot. A new feed resets the duration clock; it is not a session-lifetime
budget. Checkpoint time travel does not restore virtual files or undo external
tool effects. Old Monty-version snapshots require a reset.

## Dynamic instructions

`interpreter.instructions(tools)` returns the same content added before a model
call. It includes:

- active kernel limits, persistence, filesystem, MCP, and async rules;
- exact selected hosted-function signatures;
- optional RLM function signatures, recursion rules, and limits;
- current persistence mode;
- virtual-module limits;
- explicit warnings about CPython gaps, ToolNode bypass, approvals, and side
  effects.

This report is generated from configuration. Do not copy a stale prompt into an
agent when the kernel or tool list can change.

## Limits and non-goals

- This middleware is async-only for execution.
- It is not a replacement for a container, VM, or full CPython sandbox.
- It does not add subprocesses, sockets, package installation, native modules,
  or ambient host access.
- Host calls pause Monty's interpreter-time budget. The hosted-tool wall-time
  budget limits those waits separately.
- One session runs one operation at a time. Different session IDs can run in
  parallel up to the worker-pool limit.
- Recursive child sessions need free workers and do not receive direct kernel
  filesystem mounts.
- The full runtime injection uses LangGraph's current injection metadata. Keep
  the pinned LangChain, LangGraph, and Deep Agents versions covered by tests.
