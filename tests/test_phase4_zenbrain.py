import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from neurograph3.entities import extract_entities, extract_semantic_relations
from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store
from neurograph3.zenbrain import ZenBrainEventLedger, ZenBrainEventType


class Phase4ZenBrainTests(unittest.TestCase):
    def test_retrieval_reads_prior_without_implicit_reinforcement(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME provides anatomy to GeoDose.
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

                before = store.counts()["zenbrain_events"]
                first_pack = retriever.retrieve("GeoDose", limit=5)
                self.assertEqual(store.counts()["zenbrain_events"], before)
                self.assertIn("zenbrain_prior", first_pack.retrieval_trace["routes"])
                self.assertEqual(first_pack.evidence[0].zenbrain_prior, 0.0)

                selected_count = ledger.record_pack_event(
                    first_pack,
                    ZenBrainEventType.SELECTED,
                    caller="test",
                )
                self.assertEqual(selected_count, len(first_pack.evidence))
                second_pack = retriever.retrieve("GeoDose", limit=5)

            self.assertTrue(any(item.zenbrain_prior > 0 for item in second_pack.evidence))

    def test_correction_can_suppress_but_not_delete_cold_evidence(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            observation = result.observations[0]
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                ledger = ZenBrainEventLedger(store)
                now = datetime.now(timezone.utc)
                ledger.record_event(observation.observation_id, ZenBrainEventType.USER_CONFIRMED, created_at=now)
                ledger.record_event(
                    observation.observation_id,
                    ZenBrainEventType.CORRECTED,
                    created_at=now + timedelta(seconds=1),
                )
                scores = ledger.score([observation.observation_id], now=now + timedelta(seconds=2))
                pack = Graph3Retriever(store, zenbrain_prior=ledger).retrieve("GeoDose 70.1 ms", limit=5)

            self.assertLess(scores[observation.observation_id], 0.0)
            self.assertTrue(pack.evidence)
            self.assertAlmostEqual(pack.evidence[0].zenbrain_prior, scores[observation.observation_id], places=5)


if __name__ == "__main__":
    unittest.main()
