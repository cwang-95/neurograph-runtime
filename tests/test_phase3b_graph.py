import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from neurograph3.entities import extract_cooccurrence_relations, extract_entities, extract_semantic_relations
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

    def test_explicit_relation_is_directional_and_graph_traversable(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME provides anatomy to GeoDose for dose calculation.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            observation = next(item for item in result.observations if item.kind is ObservationKind.VISION)
            entities = extract_entities(observation)
            relations = extract_semantic_relations(observation, entities)

            self.assertEqual(len(relations), 1)
            self.assertEqual(relations[0].predicate, "provides_input_to")
            self.assertEqual(
                relations[0].source_entity_id,
                next(item.entity_id for item in entities if item.canonical_name == "DREME"),
            )
            self.assertGreater(relations[0].confidence, 0.7)

            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_graph(entities, relations)
                paths, hits = store.expand_graph([relations[0].source_entity_id])

            self.assertEqual(paths[0]["predicate"], "provides_input_to")
            self.assertEqual(paths[0]["extraction_method"], "explicit-pattern-semantic-v1")
            self.assertEqual(hits[0]["observation_id"], observation.observation_id)

    def test_repeated_relation_merges_all_supporting_observations(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME provides anatomy to GeoDose.

## Slide 2

### PPT 视觉提取

DREME provides real-time anatomy to GeoDose.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            all_entities = {}
            relations = []
            for observation in result.observations:
                extracted = extract_entities(observation)
                all_entities.update({item.entity_id: item for item in extracted})
                relations.extend(extract_semantic_relations(observation, extracted))

            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_graph(list(all_entities.values()), relations)
                paths, _ = store.expand_graph([next(iter(all_entities.values())).entity_id])
                counts = store.counts()

            self.assertEqual(counts["relations"], 1)
            self.assertEqual(len(paths[0]["observation_ids"]), 2)

    def test_multi_hop_beam_search_returns_full_typed_path(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME provides anatomy to GeoDose.

## Slide 2

### PPT 视觉提取

GeoDose uses TransFM for adaptive planning.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            all_entities = {}
            relations = []
            for observation in result.observations:
                extracted = extract_entities(observation)
                all_entities.update({item.entity_id: item for item in extracted})
                relations.extend(extract_semantic_relations(observation, extracted))

            dreme_id = next(item.entity_id for item in all_entities.values() if item.canonical_name == "DREME")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_graph(list(all_entities.values()), relations)
                paths, _ = store.expand_graph([dreme_id], max_hops=2, beam_width=10)

            two_hop = next(path for path in paths if path["hop"] == 2 and path["end_name"] == "TransFM")
            self.assertEqual(len(two_hop["path_edges"]), 2)
            self.assertEqual(
                [edge["predicate"] for edge in two_hop["path_edges"]],
                ["provides_input_to", "uses"],
            )
            self.assertEqual(len(two_hop["observation_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
