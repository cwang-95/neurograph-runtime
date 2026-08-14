import tempfile
import unittest
from pathlib import Path

from neurograph3.answer import assemble_answer
from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store


class Phase17AnswerTests(unittest.TestCase):
    def _pack(self, query: str):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose architecture framework receives delivered beam and recorded dynamic anatomy.

## Slide 2

### PPT 视觉提取

GeoDose dose calculation runtime was 70.1 ms.
"""
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        source_path = root / "talk.md"
        source_path.write_text(source, encoding="utf-8")
        result = ingest_markdown(source_path, storage_root=root / "assets", dataset="fixture")
        store = Graph3Store(root / "db")
        store.__enter__()
        store.put_ingest_result(result)
        pack = Graph3Retriever(store).retrieve(query, limit=5)
        return temp_dir, store, pack

    def test_answer_contains_sections_and_traceable_citations(self):
        temp_dir, store, pack = self._pack("GeoDose 的机制和结果")
        try:
            draft = assemble_answer(pack)
            self.assertEqual(draft.status, "answer")
            self.assertIn("机制与流程", {section.title for section in draft.sections})
            self.assertIn("量化结果", {section.title for section in draft.sections})
            self.assertIn("70.1 ms", draft.response_markdown)
            self.assertTrue(draft.citations)
            self.assertEqual(
                {citation.citation_id for citation in draft.citations},
                {item.citation_id for section in draft.sections for item in section.evidence},
            )
            self.assertTrue(all(citation.locator for citation in draft.citations))
        finally:
            store.close()
            temp_dir.cleanup()

    def test_follow_up_does_not_present_unverified_answer(self):
        temp_dir, store, pack = self._pack("这个方法的效果如何？")
        try:
            draft = assemble_answer(pack)
            self.assertEqual(draft.status, "follow_up")
            self.assertIn("需要进一步确认", draft.response_markdown)
            self.assertEqual(draft.used_observation_ids, [])
        finally:
            store.close()
            temp_dir.cleanup()

    def test_conflict_status_blocks_single_conclusion(self):
        temp_dir, store, pack = self._pack("GeoDose 的机制和结果")
        try:
            pack.conflicts = [{"claim_id": "claim-1", "status": "conflicted"}]
            draft = assemble_answer(pack)
            self.assertEqual(draft.status, "conflict")
            self.assertIn("未裁决冲突", draft.response_markdown)
        finally:
            store.close()
            temp_dir.cleanup()

    def test_one_source_runtime_breakdown_is_not_marked_as_conflict(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

Runtime breakdown: DREME 1.5 ms, GeoDose 70.1 ms, TransFM 30.6 ms, total 100 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=root / "assets", dataset="fixture")
            with Graph3Store(root / "db") as store:
                store.put_ingest_result(result)
                pack = Graph3Retriever(store).retrieve("GeoDose 的机制和结果", limit=5)

            self.assertEqual(pack.conflicts, [])


if __name__ == "__main__":
    unittest.main()
