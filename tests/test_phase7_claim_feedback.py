import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from neurograph3.extract import extract_numeric_claims
from neurograph3.ingest import ingest_markdown
from neurograph3.models import ClaimStatus, ObservationKind
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store
from neurograph3.zenbrain import ZenBrainEventLedger, ZenBrainEventType


class Phase7ClaimFeedbackTests(unittest.TestCase):
    def test_claim_feedback_is_scoped_and_does_not_delete_source_evidence(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(
                source_path,
                storage_root=Path(temp_dir) / "assets",
                dataset="fixture",
                created_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            )
            observation = next(item for item in result.observations if item.kind is ObservationKind.VISION)
            claims = extract_numeric_claims(observation, created_at=observation.created_at)
            claim_id = claims[0][0].claim_version_id
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_claims(claims)
                ledger = ZenBrainEventLedger(store)
                retriever = Graph3Retriever(store, zenbrain_prior=ledger)

                first_pack = retriever.retrieve("GeoDose 70.1 ms", limit=5)
                item = next(item for item in first_pack.evidence if item.claim_version_ids)
                self.assertIn(claim_id, item.claim_version_ids)
                self.assertEqual(store.counts()["zenbrain_events"], 0)

                ledger.record_claim_feedback(
                    [claim_id],
                    ZenBrainEventType.USER_CONFIRMED,
                    query=first_pack.query,
                    caller="answer-layer",
                )
                self.assertGreater(ledger.score_targets("claim", [claim_id])[claim_id], 0)
                second_pack = retriever.retrieve("GeoDose 70.1 ms", limit=5)
                confirmed_item = next(item for item in second_pack.evidence if claim_id in item.claim_version_ids)
                self.assertGreater(confirmed_item.zenbrain_claim_prior, 0)

                before_correction_events = store.counts()["zenbrain_events"]
                ledger.record_claim_feedback([claim_id], ZenBrainEventType.CORRECTED)
                self.assertEqual(store.counts()["zenbrain_events"], before_correction_events + 1)
                self.assertLess(ledger.score_targets("claim", [claim_id])[claim_id], 0)
                self.assertFalse(retriever.retrieve("GeoDose 70.1 ms", limit=5).evidence)
                self.assertTrue(store.observation_hits([observation.observation_id]))

                ledger.record_claim_feedback([claim_id], ZenBrainEventType.USER_CONFIRMED)
                self.assertTrue(retriever.retrieve("GeoDose 70.1 ms", limit=5).evidence)

    def test_positive_claim_feedback_can_explicitly_propagate_to_source(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            observation = next(item for item in result.observations if item.kind is ObservationKind.VISION)
            claims = extract_numeric_claims(observation)
            claim_id = claims[0][0].claim_version_id
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_claims(claims)
                ledger = ZenBrainEventLedger(store)
                count = ledger.record_claim_feedback(
                    [claim_id],
                    ZenBrainEventType.CITED,
                    caller="answer-layer",
                    propagate_to_observations=True,
                )
                self.assertEqual(count, 2)
                self.assertEqual(store.counts()["zenbrain_events"], 2)

    def test_static_rejected_claim_is_not_presented_but_source_remains(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            observation = next(item for item in result.observations if item.kind is ObservationKind.VISION)
            claim, link = extract_numeric_claims(observation)[0]
            rejected_claim = claim.model_copy(update={"status": ClaimStatus.REJECTED})
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_claims([(rejected_claim, link)])
                pack = Graph3Retriever(store).retrieve("GeoDose 70.1 ms", limit=5)
                self.assertFalse(pack.evidence)
                self.assertEqual(pack.retrieval_trace["claim_suppressed_count"], 1)
                self.assertTrue(store.observation_hits([observation.observation_id]))

    def test_unresolved_versions_are_reported_as_conflict_not_auto_deleted(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.

## Slide 2

### PPT 视觉提取

GeoDose runtime was 72.5 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            claims = []
            for observation in result.observations:
                if observation.kind is ObservationKind.VISION:
                    claims.extend(extract_numeric_claims(observation))
            self.assertEqual(len(claims), 2)
            self.assertEqual(claims[0][0].claim_id, claims[1][0].claim_id)
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_claims(claims)
                pack = Graph3Retriever(store).retrieve("GeoDose runtime", limit=5)

            self.assertTrue(pack.evidence)
            self.assertEqual(len(pack.conflicts), 1)
            self.assertEqual(pack.conflicts[0]["status"], "conflicted")


if __name__ == "__main__":
    unittest.main()
