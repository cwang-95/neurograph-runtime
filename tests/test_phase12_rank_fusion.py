import math
import tempfile
import unittest
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store


class Phase12RankFusionTests(unittest.TestCase):
    def test_evidence_uses_rrf_route_contributions_instead_of_raw_score_addition(self):
        class FakeEmbedder:
            model = "fixture-embedding-v1"

            def embed(self, texts):
                return [[1.0, 0.0] if "GeoDose" in text else [0.0, 1.0] for text in texts]

        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose calculates dose.

## Slide 2

### PPT 视觉提取

DREME reconstructs anatomy.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(
                source_path,
                storage_root=Path(temp_dir) / "assets",
                dataset="fixture",
            )
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                retriever = Graph3Retriever(store, embedder=FakeEmbedder())
                retriever.index_vectors()
                pack = retriever.retrieve("GeoDose", limit=2)

            item = next(item for item in pack.evidence if "GeoDose" in item.value)
            self.assertEqual(pack.retrieval_trace["fusion"]["method"], "rrf")
            self.assertIn("lexical", item.route_contributions)
            self.assertIn("vector", item.route_contributions)
            self.assertAlmostEqual(item.fusion_score, sum(item.route_contributions.values()))
            self.assertFalse(
                math.isclose(
                    item.combined_score,
                    item.lexical_score + (item.vector_score or 0.0),
                    rel_tol=1e-6,
                )
            )


if __name__ == "__main__":
    unittest.main()
