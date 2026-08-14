import tempfile
import unittest
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store
from neurograph3.zenbrain import ZenBrainEventLedger, ZenBrainEventType


class FakeFSRS:
    def __init__(self):
        self.recall_calls = []
        self.decay_calls = 0

    def new_memory(self):
        return {"schedulers": {"fsrs": {"stability": 7.0}}}

    def recall(self, memory, quality=4, after_days=7.0):
        self.recall_calls.append((quality, after_days))
        updated = {"schedulers": {"fsrs": dict(memory["schedulers"]["fsrs"])}}
        updated["schedulers"]["fsrs"]["stability"] += 1.0
        return updated

    def decay(self, memory):
        self.decay_calls += 1
        return 0.8


class Phase5FSRSTests(unittest.TestCase):
    def test_retrieval_does_not_update_fsrs_but_feedback_does(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

Adaptive radiotherapy uses daily imaging.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            observation_id = result.observations[0].observation_id
            fsrs = FakeFSRS()
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                ledger = ZenBrainEventLedger(store, fsrs=fsrs)
                retriever = Graph3Retriever(store, zenbrain_prior=ledger)

                retriever.retrieve("daily imaging", limit=5)
                self.assertEqual(fsrs.recall_calls, [])
                self.assertIsNone(store.get_zenbrain_scheduler("observation", observation_id))

                ledger.record_feedback(
                    [observation_id],
                    ZenBrainEventType.CITED,
                    caller="answer-layer",
                    query="daily imaging",
                )
                self.assertEqual(fsrs.recall_calls, [(4, 7.0)])
                self.assertEqual(store.counts()["zenbrain_nodes"], 1)

                score = ledger.score([observation_id])[observation_id]
                self.assertGreater(score, 0.0)
                self.assertEqual(fsrs.decay_calls, 1)

                ledger.record_feedback([observation_id], ZenBrainEventType.CORRECTED)
                self.assertEqual(len(fsrs.recall_calls), 1)
                self.assertLess(ledger.score([observation_id])[observation_id], score)


if __name__ == "__main__":
    unittest.main()
