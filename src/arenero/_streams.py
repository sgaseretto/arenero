"""Bounded console collection for the agent compatibility profile."""


class BoundedStreams:
    """Keep a UTF-8 byte prefix per stream and count discarded characters.

    Args:
        max_bytes: Retained byte limit for each stream; excess is discarded.
    """

    def __init__(self, max_bytes: int) -> None:
        """Create output and counters for both streams.

        Args:
            max_bytes: Maximum retained UTF-8 bytes for each stream.
        """
        self.max_bytes = max_bytes
        self.output: list[tuple[str, str]] = []
        self.original_chars = {"stdout": 0, "stderr": 0}
        self._bytes = {"stdout": 0, "stderr": 0}

    def __call__(self, stream: str, text: str) -> None:
        """Keep only the available prefix without raising on excess output.

        Args:
            stream: stdout or stderr.
            text: Monty output chunk.
        """
        self.original_chars[stream] += len(text)
        available = self.max_bytes - self._bytes[stream]
        if available > 0:
            kept = (
                text[:available].encode("utf-8")[:available].decode("utf-8", "ignore")
            )
            self._bytes[stream] += len(kept.encode("utf-8"))
            if kept:
                self.output.append((stream, kept))
            if len(kept) < len(text):
                self._bytes[stream] = self.max_bytes
