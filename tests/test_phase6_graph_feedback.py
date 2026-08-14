import tempfile
import unittest
from pathlib import Path

from neurograph3.entities import extract_entities, extract_semantic_relations
from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store
from neurograph3.zenbrain import ZenBrainEventLedger, ZenBrainEventType


class Phase6GraphFeedbackTests(unittest.TestCase):
    def test_graph_paths_and_relations_get_only_explicit_feedback_prior(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME provides anatomy to GeoDose for dose calculation.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            observation = result.observations[0]
            entities = extract_entities(observation)
            relations = extract_semantic_relations(observation, entities)
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_graph(entities, relations)
                ledger = ZenBrainEventLedger(store)
                retriever = Graph3Retriever(store, zenbrain_prior=ledger)

                first_pack = retriever.retrieve("GeoDose", limit=5)
                self.assertTrue(first_pack.graph_paths)
                self.assertTrue(all(path["path_id"] for path in first_pack.graph_paths))
                self.assertIn("zenbrain_graph_prior", first_pack.retrieval_trace["routes"])
                self.assertEqual(store.counts()["zenbrain_events"], 0)
                self.assertTrue(all(item.zenbrain_path_prior == 0 for item in first_pack.evidence))

                feedback_count = ledger.record_pack_graph_feedback(
                    first_pack,
                    ZenBrainEventType.CITED,
                    caller="answer-layer",
                )
                self.assertGreater(feedback_count, 0)
                path_id = first_pack.graph_paths[0]["path_id"]
                relation_id = first_pack.graph_paths[0]["path_edges"][0]["relation_id"]
                self.assertGreater(ledger.score_targets("path", [path_id])[path_id], 0)
                self.assertGreater(ledger.score_targets("relation", [relation_id])[relation_id], 0)

                second_pack = retriever.retrieve("GeoDose", limit=5)

            self.assertTrue(any(path["zenbrain_path_prior"] > 0 for path in second_pack.graph_paths))
            self.assertTrue(any(item.zenbrain_edge_prior > 0 for item in second_pack.evidence))


if __name__ == "__main__":
    unittest.main()
