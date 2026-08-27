import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ConcurrentSessionUpdate(RuntimeError):
    """The session changed after it was loaded."""


@dataclass(frozen=True)
class StoredSession:
    """Hold one saved Monty session revision.

    Attributes:
        state: Serialized Monty session bytes.
        monty_version: Monty version that created the state.
        revision: Monotonic storage revision.
        updated_at: Last update time as a Unix timestamp.
    """

    state: bytes
    monty_version: str
    revision: int
    updated_at: float


class SessionStore(Protocol):
    """Storage used for Monty session snapshots."""

    stores_state: bool

    def load(
        self,
        session_id: str,
        max_age_seconds: float | None = None,
    ) -> StoredSession | None:
        """Load current session state.

        Args:
            session_id: Session key to load.
            max_age_seconds: Optional maximum state age.

        Returns:
            Saved session, or ``None`` when absent or expired.
        """
        ...

    def save(
        self,
        session_id: str,
        state: bytes,
        monty_version: str,
        expected_revision: int | None,
    ) -> int | None:
        """Save session state with a revision check.

        Args:
            session_id: Session key to save.
            state: Serialized Monty state.
            monty_version: Monty version that created the state.
            expected_revision: Revision loaded before execution.

        Returns:
            New revision, or ``None`` when state is discarded.
        """
        ...

    def delete(self, session_id: str) -> bool:
        """Delete one session.

        Args:
            session_id: Session key to delete.

        Returns:
            ``True`` when saved state existed.
        """
        ...

    def delete_expired(self, max_age_seconds: float) -> int:
        """Delete sessions older than a maximum age.

        Args:
            max_age_seconds: Maximum retained state age.

        Returns:
            Number of deleted sessions.
        """
        ...


class StatelessSessionStore:
    """Discard session state after every execution."""

    stores_state = False

    def load(
        self,
        session_id: str,
        max_age_seconds: float | None = None,
    ) -> None:
        """Return no state for every session.

        Args:
            session_id: Ignored session key.
            max_age_seconds: Ignored maximum age.

        Returns:
            Always ``None``.
        """
        return

    def save(
        self,
        session_id: str,
        state: bytes,
        monty_version: str,
        expected_revision: int | None,
    ) -> None:
        """Discard supplied session state.

        Args:
            session_id: Ignored session key.
            state: Ignored serialized state.
            monty_version: Ignored Monty version.
            expected_revision: Ignored revision.

        Returns:
            Always ``None``.
        """
        return

    def delete(self, session_id: str) -> bool:
        """Report that no saved session exists.

        Args:
            session_id: Ignored session key.

        Returns:
            Always ``False``.
        """
        return False

    def delete_expired(self, max_age_seconds: float) -> int:
        """Report that no expired sessions exist.

        Args:
            max_age_seconds: Ignored maximum age.

        Returns:
            Always zero.
        """
        return 0


class InMemorySessionStore:
    """Keep session state while this store object remains in memory."""

    stores_state = True

    def __init__(self) -> None:
        """Create an empty thread-safe in-memory store."""
        self._sessions: dict[str, StoredSession] = {}
        self._lock = threading.Lock()

    def load(
        self,
        session_id: str,
        max_age_seconds: float | None = None,
    ) -> StoredSession | None:
        """Load current, unexpired in-memory state.

        Args:
            session_id: Session key to load.
            max_age_seconds: Optional maximum state age.

        Returns:
            Saved session, or ``None`` when absent or expired.
        """
        with self._lock:
            stored = self._sessions.get(session_id)
            if stored is None:
                return None
            if (
                max_age_seconds is not None
                and time.time() - stored.updated_at > max_age_seconds
            ):
                del self._sessions[session_id]
                return None
            return stored

    def save(
        self,
        session_id: str,
        state: bytes,
        monty_version: str,
        expected_revision: int | None,
    ) -> int:
        """Save in-memory state with a revision check.

        Args:
            session_id: Session key to save.
            state: Serialized Monty state.
            monty_version: Monty version that created the state.
            expected_revision: Revision loaded before execution.

        Returns:
            New session revision.
        """
        with self._lock:
            stored = self._sessions.get(session_id)
            current_revision = stored.revision if stored else None
            if current_revision != expected_revision:
                raise ConcurrentSessionUpdate(session_id)

            revision = 1 if expected_revision is None else expected_revision + 1
            self._sessions[session_id] = StoredSession(
                state=state,
                monty_version=monty_version,
                revision=revision,
                updated_at=time.time(),
            )
            return revision

    def delete(self, session_id: str) -> bool:
        """Delete one in-memory session.

        Args:
            session_id: Session key to delete.

        Returns:
            ``True`` when saved state existed.
        """
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def delete_expired(self, max_age_seconds: float) -> int:
        """Delete old in-memory sessions.

        Args:
            max_age_seconds: Maximum retained state age.

        Returns:
            Number of deleted sessions.
        """
        cutoff = time.time() - max_age_seconds
        with self._lock:
            expired = [
                session_id
                for session_id, stored in self._sessions.items()
                if stored.updated_at < cutoff
            ]
            for session_id in expired:
                del self._sessions[session_id]
            return len(expired)


class SQLiteSessionStore:
    """Store Monty session dumps in SQLite."""

    stores_state = True

    def __init__(self, path: str | Path) -> None:
        """Create a SQLite session store.

        Args:
            path: SQLite database path.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a configured SQLite connection.

        Returns:
            New database connection.
        """
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Manage one transactional SQLite connection.

        Yields:
            Open database connection.
        """
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        """Create the SQLite session table when needed."""
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state BLOB NOT NULL,
                    monty_version TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def load(
        self,
        session_id: str,
        max_age_seconds: float | None = None,
    ) -> StoredSession | None:
        """Load current, unexpired SQLite state.

        Args:
            session_id: Session key to load.
            max_age_seconds: Optional maximum state age.

        Returns:
            Saved session, or ``None`` when absent or expired.
        """
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT state, monty_version, revision, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                return None

            state, monty_version, revision, updated_at = row
            if (
                max_age_seconds is not None
                and time.time() - updated_at > max_age_seconds
            ):
                connection.execute(
                    "DELETE FROM sessions WHERE session_id = ? AND revision = ?",
                    (session_id, revision),
                )
                return None

            return StoredSession(
                state=state,
                monty_version=monty_version,
                revision=revision,
                updated_at=updated_at,
            )

    def save(
        self,
        session_id: str,
        state: bytes,
        monty_version: str,
        expected_revision: int | None,
    ) -> int:
        """Save SQLite state with an atomic revision check.

        Args:
            session_id: Session key to save.
            state: Serialized Monty state.
            monty_version: Monty version that created the state.
            expected_revision: Revision loaded before execution.

        Returns:
            New session revision.
        """
        updated_at = time.time()

        with self._connection() as connection:
            if expected_revision is None:
                try:
                    connection.execute(
                        """
                        INSERT INTO sessions (
                            session_id, state, monty_version, revision, updated_at
                        ) VALUES (?, ?, ?, 1, ?)
                        """,
                        (session_id, state, monty_version, updated_at),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ConcurrentSessionUpdate(session_id) from exc
                return 1

            next_revision = expected_revision + 1
            cursor = connection.execute(
                """
                UPDATE sessions
                SET state = ?, monty_version = ?, revision = ?, updated_at = ?
                WHERE session_id = ? AND revision = ?
                """,
                (
                    state,
                    monty_version,
                    next_revision,
                    updated_at,
                    session_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ConcurrentSessionUpdate(session_id)
            return next_revision

    def delete(self, session_id: str) -> bool:
        """Delete one SQLite session.

        Args:
            session_id: Session key to delete.

        Returns:
            ``True`` when saved state existed.
        """
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount == 1

    def delete_expired(self, max_age_seconds: float) -> int:
        """Delete old SQLite sessions.

        Args:
            max_age_seconds: Maximum retained state age.

        Returns:
            Number of deleted sessions.
        """
        cutoff = time.time() - max_age_seconds
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (cutoff,),
            )
            return cursor.rowcount
