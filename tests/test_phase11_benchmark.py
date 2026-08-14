import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store


class Phase11BenchmarkTests(unittest.TestCase):
    def _store(self, temp_dir: str) -> Path:
        source_path = Path(temp_dir) / "talk.md"
        source_path.write_text(
            "# Talk\n\n## Slide 1\n\n### PPT 视觉提取\n\n"
            "GeoDose runtime was 70.1 ms.\n\n"
            "## Slide 2\n\n### PPT 视觉提取\n\n"
            "DREME supports adaptive radiotherapy.\n",
            encoding="utf-8",
        )
        result = ingest_markdown(
            source_path,
            storage_root=Path(temp_dir) / "assets",
            dataset="fixture",
        )
        storage_root = Path(temp_dir) / "db"
        with Graph3Store(storage_root) as store:
            store.put_ingest_result(result)
        return storage_root

    def test_incremental_embedding_index_skips_existing_model_rows(self):
        class CountingEmbedder:
            model = "fake-embedding-v1"
            batch_size = 2

            def __init__(self):
                self.calls = []

            def embed(self, texts):
                self.calls.append(list(texts))
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_root = self._store(temp_dir)
            embedder = CountingEmbedder()
            with Graph3Store(storage_root) as store:
                retriever = Graph3Retriever(store, embedder=embedder)
                self.assertEqual(retriever.index_vectors(only_missing=True), 2)
                self.assertEqual(retriever.index_vectors(only_missing=True), 0)
                self.assertEqual([len(call) for call in embedder.calls], [2])
                self.assertEqual(store.counts()["observation_embeddings"], 2)

    def test_benchmark_cli_emits_latency_and_route_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_root = self._store(temp_dir)
            queries_path = Path(temp_dir) / "queries.txt"
            queries_path.write_text("GeoDose 70.1 ms\n\n# ignored\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/graph3_benchmark",
                    "--storage-root",
                    str(storage_root),
                    "--queries-file",
                    str(queries_path),
                    "--rounds",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)

            self.assertEqual(report["query_count"], 1)
            self.assertEqual(report["total_queries"], 2)
            self.assertGreaterEqual(report["latency_ms"]["p95"], 0.0)
            self.assertGreaterEqual(report["mean_evidence_count"], 1.0)
            self.assertEqual(report["route_counts"]["lexical"], 2)
            self.assertFalse(report["embedding_enabled"])


if __name__ == "__main__":
    unittest.main()
