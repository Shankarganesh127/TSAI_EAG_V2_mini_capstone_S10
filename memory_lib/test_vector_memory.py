import json
import tempfile
import unittest
from pathlib import Path

from memory_lib import MemoryConfig, SessionVectorMemory


class FakeEmbedder:
    def embed(self, text):
        normalized = text.lower()
        if "capital" in normalized or "france" in normalized:
            return [[1.0, 0.0, 0.0]]
        return [[0.0, 1.0, 0.0]]


class TestSessionVectorMemory(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def write_session(path: Path, answer: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "session_id": "session-1",
            "query": "interactive session",
            "started_at": "2026-07-16T10:00:00",
            "stages": [{
                "stage": "agent_result",
                "timestamp": "2026-07-16T10:01:00",
                "query": "What is the capital of France?",
                "answer": answer,
            }],
        }), encoding="utf-8")

    async def test_stores_and_finds_similar_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            config = MemoryConfig(directory, similarity_threshold=0.9, top_k=3)
            memory = SessionVectorMemory(FakeEmbedder(), config)
            await memory.add(
                "What is the capital of France?", "Paris", "session-1", "session.json"
            )

            hits = await memory.search("Tell me France's capital")

            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].answer, "Paris")
            self.assertEqual(hits[0].session_id, "session-1")
            self.assertEqual(hits[0].log_path, "session.json")

    async def test_persists_across_instances_and_rejects_dissimilar_query(self):
        with tempfile.TemporaryDirectory() as directory:
            config = MemoryConfig(directory, similarity_threshold=0.9, top_k=3)
            first = SessionVectorMemory(FakeEmbedder(), config)
            await first.add("Capital of France", "Paris", "session-1")

            reloaded = SessionVectorMemory(FakeEmbedder(), config)

            self.assertEqual((await reloaded.search("France capital"))[0].answer, "Paris")
            self.assertEqual(await reloaded.search("How do I bake bread?"), [])

    async def test_rebuilds_only_when_session_database_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            config = MemoryConfig(str(root / "vectors"), similarity_threshold=0.9, top_k=3)
            log_path = logs / "2026" / "07" / "16" / "session-1.json"
            self.write_session(log_path, "Paris")
            memory = SessionVectorMemory(FakeEmbedder(), config)

            first = await memory.sync_session_logs(logs)
            unchanged = await memory.sync_session_logs(logs)
            self.write_session(log_path, "Paris, France")
            changed = await memory.sync_session_logs(logs)

            self.assertTrue(first.changed)
            self.assertFalse(unchanged.changed)
            self.assertTrue(changed.changed)
            self.assertEqual(changed.indexed_records, 1)
            self.assertEqual((await memory.search("France capital"))[0].answer, "Paris, France")


if __name__ == "__main__":
    unittest.main()
