import tempfile
import unittest
from pathlib import Path

from neurograph3.answer_feedback import AnswerEvidenceContext
from neurograph3.ingest import ingest_markdown
from neurograph3.retrieval import Graph3Retriever
from neurograph3.store import Graph3Store


class Phase14CrossDocumentDedupTests(unittest.TestCase):
    def test_duplicate_facts_use_one_representative_and_keep_supporting_sources(self):
        source_template = """# {title}

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.md"
            second_path = root / "second.md"
            first_path.write_text(source_template.format(title="Talk A"), encoding="utf-8")
            second_path.write_text(source_template.format(title="Talk B"), encoding="utf-8")
            first = ingest_markdown(first_path, storage_root=root / "assets", dataset="fixture")
            second = ingest_markdown(second_path, storage_root=root / "assets", dataset="fixture")
            with Graph3Store(root / "db") as store:
                store.put_ingest_result(first)
                store.put_ingest_result(second)
                pack = Graph3Retriever(store).retrieve("GeoDose 70.1 ms", limit=2)
                context = AnswerEvidenceContext.from_pack(pack)

            self.assertEqual(len(pack.evidence), 1)
            representative = pack.evidence[0]
            self.assertEqual(len(representative.supporting_observation_ids), 1)
            self.assertEqual(len(representative.supporting_citations), 1)
            self.assertEqual(
                len([citation for citation in pack.citations if citation["role"] == "supporting"]),
                1,
            )
            self.assertIn(representative.supporting_observation_ids[0], context.observation_ids)
            self.assertGreaterEqual(representative.source_quality, 0.0)
            self.assertLessEqual(representative.source_quality, 1.0)


if __name__ == "__main__":
    unittest.main()
