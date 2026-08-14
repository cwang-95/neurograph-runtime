import tempfile
import unittest
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store
from neurograph3.vector_index import ANNIndex, ANNUnavailable


class Phase10ANNTests(unittest.TestCase):
    def _fixture(self, temp_dir: str):
        source_path = Path(temp_dir) / "talk.md"
        source_path.write_text(
            "# Talk\n\n## Slide 1\n\n### PPT 视觉提取\n\n"
            "GeoDose calculates dose in 70.1 ms.\n\n"
            "## Slide 2\n\n### PPT 视觉提取\n\n"
            "DREME reconstructs anatomy for adaptive therapy.\n",
            encoding="utf-8",
        )
        result = ingest_markdown(
            source_path,
            storage_root=Path(temp_dir) / "assets",
            dataset="fixture",
        )
        return result

    def test_ann_adapter_is_optional_and_explicit_when_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index = ANNIndex(Path(temp_dir) / "index")
            if index.available:
                self.skipTest("an optional ANN backend is installed in this environment")
            with self.assertRaises(ANNUnavailable):
                index.rebuild(
                    [{"observation_id": "obs", "model": "fake", "dimensions": 2, "vector": [1.0, 0.0]}]
                )

    def test_retriever_uses_external_index_and_loads_full_evidence_hits(self):
        class FakeEmbedder:
            model = "fake-embedding-v1"

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        class RecordingIndex:
            backend_name = "test-index"

            def __init__(self):
                self.records = []

            def rebuild(self, records):
                self.records = records

            def search(self, query_vector, limit):
                return [{"observation_id": self.records[0]["observation_id"], "vector_score": 0.97}]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._fixture(temp_dir)
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                index = RecordingIndex()
                retriever = Graph3Retriever(store, embedder=FakeEmbedder(), vector_index=index)
                self.assertEqual(retriever.index_vectors(), len(result.observations))
                pack = retriever.retrieve("GeoDose", limit=5)

            self.assertEqual(len(index.records), len(result.observations))
            self.assertEqual(pack.retrieval_trace["vector_backend"], "test-index")
            self.assertTrue(any("vector" in item.retrieval_routes for item in pack.evidence))
            self.assertTrue(any(item.vector_score == 0.97 for item in pack.evidence))

    def test_retriever_falls_back_to_sqlite_when_ann_search_fails(self):
        class FakeEmbedder:
            model = "fake-embedding-v1"

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        class BrokenIndex:
            backend_name = "broken-index"

            def search(self, query_vector, limit):
                raise ANNUnavailable("index file is unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._fixture(temp_dir)
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_embeddings(
                    "fake-embedding-v1",
                    {item.observation_id: [1.0, 0.0] for item in result.observations},
                )
                pack = Graph3Retriever(
                    store,
                    embedder=FakeEmbedder(),
                    vector_index=BrokenIndex(),
                ).retrieve("GeoDose", limit=5)

            self.assertEqual(pack.retrieval_trace["vector_fallback"], "sqlite_bruteforce")
            self.assertIn("vector", pack.retrieval_trace["routes"])
            self.assertTrue(pack.evidence)


if __name__ == "__main__":
    unittest.main()
