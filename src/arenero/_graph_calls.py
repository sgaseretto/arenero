"""Keep graph control signals outside Monty's guest exception conversion."""

import asyncio
from dataclasses import replace

from langgraph.errors import GraphBubbleUp

from .hosted_tools import HostedToolSet, _invoke_tool


class GraphCalls:
    """Cancel the outer feed, then restore an intercepted graph signal."""

    def __init__(self) -> None:
        """Record the task that owns the current interpreter operation."""
        self.owner = asyncio.current_task()
        self.signal: GraphBubbleUp | None = None
        self.tasks: set[asyncio.Task] = set()

    def wrap(self, tools: HostedToolSet) -> HostedToolSet:
        """Wrap async host calls without changing their schemas or budgets.

        Args:
            tools: Operation-scoped tool definitions.

        Returns:
            Tool definitions that stop the guest on a graph control signal.
        """
        wrapped = []
        for tool in tools.tools:

            async def invoke(arguments, active=tool):
                """Call one tool or cancel the feed on a graph signal.

                Args:
                    arguments: Validated guest arguments.
                    active: Tool captured for this wrapper.

                Returns:
                    The tool result on normal completion.
                """
                task = asyncio.current_task()
                assert task is not None
                self.tasks.add(task)
                try:
                    return await _invoke_tool(active, arguments)
                except GraphBubbleUp as exc:
                    self.signal = exc
                    assert self.owner is not None
                    self.owner.cancel()
                    # Do not return a catchable guest exception or permit more work.
                    await asyncio.Future()
                finally:
                    self.tasks.discard(task)

            wrapped.append(replace(tool, ainvoke=invoke))
        return HostedToolSet(tuple(wrapped), tools.config)

    async def close(self) -> None:
        """Cancel and drain host callbacks left behind by a cancelled feed."""
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def reraise(self) -> None:
        """Raise the saved signal after the cancelled checkout has ended."""
        if self.signal is not None:
            assert self.owner is not None
            self.owner.uncancel()
            raise self.signal
