import tempfile
import unittest
from pathlib import Path

from arenero.store import (
    ConcurrentSessionUpdate,
    InMemorySessionStore,
    SQLiteSessionStore,
    StatelessSessionStore,
)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteSessionStore(Path(self.directory.name) / "sessions.db")

    def test_revision_prevents_a_lost_update(self) -> None:
        revision = self.store.save(
            "thread",
            b"one",
            "1.0",
            expected_revision=None,
        )

        with self.assertRaises(ConcurrentSessionUpdate):
            self.store.save(
                "thread",
                b"stale",
                "1.0",
                expected_revision=None,
            )

        next_revision = self.store.save(
            "thread",
            b"two",
            "1.0",
            expected_revision=revision,
        )
        stored = self.store.load("thread")

        self.assertEqual(next_revision, 2)
        self.assertEqual(stored.state, b"two")

    def test_delete_expired_removes_old_sessions(self) -> None:
        self.store.save("thread", b"state", "1.0", expected_revision=None)

        deleted = self.store.delete_expired(max_age_seconds=0)

        self.assertEqual(deleted, 1)
        self.assertIsNone(self.store.load("thread"))

    def test_memory_store_uses_the_same_revision_rules(self) -> None:
        store = InMemorySessionStore()
        revision = store.save("thread", b"one", "1.0", expected_revision=None)

        with self.assertRaises(ConcurrentSessionUpdate):
            store.save("thread", b"stale", "1.0", expected_revision=None)

        next_revision = store.save(
            "thread",
            b"two",
            "1.0",
            expected_revision=revision,
        )

        self.assertEqual(next_revision, 2)
        self.assertEqual(store.load("thread").state, b"two")

    def test_stateless_store_discards_everything(self) -> None:
        store = StatelessSessionStore()

        revision = store.save("thread", b"state", "1.0", expected_revision=None)

        self.assertIsNone(revision)
        self.assertIsNone(store.load("thread"))
        self.assertFalse(store.delete("thread"))
        self.assertEqual(store.list_sessions(), [])

    def test_stores_list_safe_session_metadata(self) -> None:
        """List sorted metadata without returning snapshot content."""
        self.store.save("two", b"22", "1.0", expected_revision=None)
        self.store.save("one", b"1", "1.0", expected_revision=None)

        sqlite_sessions = self.store.list_sessions()
        memory = InMemorySessionStore()
        memory.save("one", b"abc", "1.0", expected_revision=None)
        memory_sessions = memory.list_sessions()

        self.assertEqual(
            [session.session_id for session in sqlite_sessions],
            ["one", "two"],
        )
        self.assertEqual(sqlite_sessions[0].state_size_bytes, 1)
        self.assertEqual(sqlite_sessions[1].state_size_bytes, 2)
        self.assertEqual(memory_sessions[0].state_size_bytes, 3)


if __name__ == "__main__":
    unittest.main()
