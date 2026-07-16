import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mcp_lib import mcp_server_documents as documents


def fake_embedding(text: str) -> np.ndarray:
    if "france" in text.lower() or "paris" in text.lower():
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


class TestDocumentRAG(unittest.TestCase):
    def test_rebuilds_on_add_edit_and_delete_and_searches_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "Documents"
            index = root / "index"
            docs.mkdir()
            source = docs / "facts.txt"
            source.write_text("Paris is the capital of France.", encoding="utf-8")

            with (
                patch.object(documents, "DOC_DIR", docs),
                patch.object(documents, "INDEX_DIR", index),
                patch.object(documents, "get_embedding", side_effect=fake_embedding),
            ):
                first = documents.process_documents()
                unchanged = documents.process_documents()
                results = documents.search_documents("What is France's capital?")
                source.write_text("Paris is the capital and largest city of France.", encoding="utf-8")
                edited = documents.process_documents()
                source.unlink()
                deleted = documents.process_documents()

            self.assertTrue(first["changed"])
            self.assertFalse(unchanged["changed"])
            self.assertIn("Paris", results[0])
            self.assertIn("Source: facts.txt", results[0])
            self.assertTrue(edited["changed"])
            self.assertEqual(deleted["chunks"], 0)
            self.assertFalse((index / "index.bin").exists())


if __name__ == "__main__":
    unittest.main()
