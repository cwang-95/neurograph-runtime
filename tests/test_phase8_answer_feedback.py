import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from neurograph3.answer_feedback import AnswerFeedbackRecorder, FeedbackRequest
from neurograph3.extract import extract_numeric_claims
from neurograph3.ingest import ingest_markdown
from neurograph3.models import ObservationKind
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store
from neurograph3.zenbrain import ZenBrainEventLedger, ZenBrainEventType


class Phase8AnswerFeedbackTests(unittest.TestCase):
    def test_answer_feedback_is_idempotent_and_pack_scoped(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                pack = Graph3Retriever(store).retrieve("GeoDose 70.1 ms", limit=5)
                observation_id = pack.evidence[0].observation_id
                request = FeedbackRequest.from_pack(
                    pack,
                    ZenBrainEventType.CITED,
                    observation_ids=[observation_id],
                    caller="openclaw",
                    feedback_id="answer-001",
                )
                recorder = AnswerFeedbackRecorder(store)
                first = recorder.record(request)
                second = recorder.record(request)

                self.assertEqual(first.recorded_events, 1)
                self.assertEqual(second.recorded_events, 0)
                self.assertEqual(second.feedback_id, "answer-001")
                self.assertEqual(store.counts()["zenbrain_events"], 1)
                with self.assertRaises(ValidationError):
                    FeedbackRequest(
                        context=request.context,
                        event_type=ZenBrainEventType.CITED,
                        observation_ids=["not-in-this-pack"],
                    )

    def test_claim_feedback_can_propagate_only_within_pack_sources(self):
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
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_claims(claims)
                pack = Graph3Retriever(store).retrieve("GeoDose 70.1 ms", limit=5)
                request = FeedbackRequest.from_pack(
                    pack,
                    ZenBrainEventType.USER_CONFIRMED,
                    claim_version_ids=[claims[0][0].claim_version_id],
                    caller="codex",
                    feedback_id="answer-002",
                    propagate_to_observations=True,
                )
                recorder = AnswerFeedbackRecorder(store)
                result_payload = recorder.record(request)

                self.assertEqual(result_payload.claim_events, 1)
                self.assertEqual(result_payload.observation_events, 1)
                self.assertEqual(store.counts()["zenbrain_events"], 2)


if __name__ == "__main__":
    unittest.main()
