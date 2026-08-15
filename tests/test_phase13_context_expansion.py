import tempfile
import unittest
from pathlib import Path

from neurograph3.answer_feedback import AnswerEvidenceContext, FeedbackRequest
from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store
from neurograph3.zenbrain import ZenBrainEventType


class Phase13ContextExpansionTests(unittest.TestCase):
    def test_ingest_links_adjacent_slides_and_retrieval_adds_context_only(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

Background anatomy context.

## Slide 2

### PPT 视觉提取

GeoDose runtime was 70.1 ms.

## Slide 3

### PPT 视觉提取

TransFM result was 30.6 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(
                source_path,
                storage_root=Path(temp_dir) / "assets",
                dataset="fixture",
            )
            slides = [element for element in result.elements if element.element_type.value == "slide"]
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                pack = Graph3Retriever(store).retrieve("70.1 ms", limit=1)

                primary_ids = {item.observation_id for item in pack.evidence}
                context_slides = {item.locator["slide"] for item in pack.context_evidence}
                context = pack.context_evidence[0]
                answer_context = AnswerEvidenceContext.from_pack(pack)
                request = FeedbackRequest.from_pack(
                    pack,
                    ZenBrainEventType.CITED,
                    observation_ids=[context.observation_id],
                )

            self.assertIsNone(slides[0].previous_id)
            self.assertEqual(slides[0].next_id, slides[1].element_id)
            self.assertEqual(slides[1].previous_id, slides[0].element_id)
            self.assertEqual(slides[1].next_id, slides[2].element_id)
            self.assertEqual(slides[2].previous_id, slides[1].element_id)
            self.assertEqual(len(pack.evidence), 1)
            self.assertEqual(context_slides, {1, 3})
            self.assertTrue(all(item.observation_id not in primary_ids for item in pack.context_evidence))
            self.assertTrue(all(item.context_relation == "adjacent_slide" for item in pack.context_evidence))
            self.assertEqual(pack.retrieval_trace["context_expansion"]["context_count"], 2)
            self.assertIn(context.observation_id, answer_context.observation_ids)
            self.assertEqual(request.observation_ids, [context.observation_id])


if __name__ == "__main__":
    unittest.main()
