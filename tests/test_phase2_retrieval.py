import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever, SlotStatus
from neurograph3.store import Graph3Store


class Phase2RetrievalTests(unittest.TestCase):
    def test_store_is_idempotent_and_numeric_search_returns_citation(self):
        source = """# Talk\n\n## Slide 1\n\n### PPT 视觉提取\n\nGeoDose dose calculation runtime was 70.1 ms.\n"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture", created_at=datetime(2026, 8, 14, tzinfo=timezone.utc))
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                first_counts = store.counts()
                store.put_ingest_result(result)
                self.assertEqual(first_counts, store.counts())
                pack = Graph3Retriever(store).retrieve("GeoDose 70.1 ms", limit=5)

            self.assertEqual(pack.slot_status["direct_evidence"], SlotStatus.SUPPORTED)
            self.assertTrue(any("70.1 ms" in item.value for item in pack.evidence))
            self.assertTrue(pack.citations[0]["locator"]["slide"] == 1)
            self.assertFalse(pack.retrieval_trace["completion_model_called"])


if __name__ == "__main__":
    unittest.main()
