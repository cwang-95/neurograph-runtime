import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from neurograph3.entities import extract_cooccurrence_relations, extract_entities
from neurograph3.ingest import ingest_markdown
from neurograph3.models import ObservationKind
from neurograph3.store import Graph3Store


class Phase3bGraphTests(unittest.TestCase):
    def test_closed_vocabulary_entities_and_weak_relation_persist(self):
        source = """# Talk\n\n## Slide 1\n\n### PPT 视觉提取\n\nDREME provides anatomy to GeoDose for dose calculation.\n"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture", created_at=datetime(2026, 8, 14, tzinfo=timezone.utc))
            observation = next(item for item in result.observations if item.kind is ObservationKind.VISION)
            entities = extract_entities(observation)
            relations = extract_cooccurrence_relations(observation, entities)
            self.assertEqual({item.canonical_name for item in entities}, {"DREME", "GeoDose"})
            self.assertEqual(len(relations), 1)
            self.assertEqual(relations[0].predicate, "co_occurs_in_observation")
            self.assertLess(relations[0].confidence, 0.5)
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_graph(entities, relations)
                first_counts = store.counts()
                store.put_graph(entities, relations)
                self.assertEqual(first_counts, store.counts())
                self.assertEqual(first_counts["entities"], 2)
                self.assertEqual(first_counts["relations"], 1)


if __name__ == "__main__":
    unittest.main()
