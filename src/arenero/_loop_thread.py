"""One event loop for synchronous and asynchronous adapter calls."""

import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
from threading import Thread, current_thread
from typing import Any


class LoopThread:
    """Own a loop so an adapter can serve both graph invocation styles."""

    def __init__(self) -> None:
        """Start the loop and wait until it can accept work."""
        ready: Future[asyncio.AbstractEventLoop] = Future()

        def run() -> None:
            """Run the loop and finish its pending tasks on shutdown."""
            with asyncio.Runner() as runner:
                loop = runner.get_loop()
                ready.set_result(loop)
                loop.run_forever()

        self.thread = Thread(target=run, name="arenero-interpreter", daemon=True)
        self.thread.start()
        self.loop = ready.result()

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future[Any]:
        """Submit work with the caller's context.

        Args:
            coroutine: Adapter operation to run on the owned loop.

        Returns:
            Thread-safe future for its result.
        """
        if self.loop.is_closed() or current_thread() is self.thread:
            coroutine.close()
            raise RuntimeError("the interpreter is closed or re-entered its own loop")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def stop(self) -> None:
        """Stop the loop after the adapter has closed its resources."""
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
