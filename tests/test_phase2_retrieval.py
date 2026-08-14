import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.entities import extract_cooccurrence_relations, extract_entities
from neurograph3.embedding import OpenAICompatibleEmbedder
from neurograph3.retrieval import Graph3Retriever, SlotStatus
from neurograph3.store import Graph3Store


class Phase2RetrievalTests(unittest.TestCase):
    def test_embedding_client_batches_large_requests(self):
        class RecordingEmbedder(OpenAICompatibleEmbedder):
            def __init__(self):
                super().__init__(batch_size=2)
                self.calls = []

            def _embed_batch(self, texts):
                self.calls.append(list(texts))
                return [[float(index)] for index, _ in enumerate(texts)]

        embedder = RecordingEmbedder()
        vectors = embedder.embed(["one", "two", "three", "four", "five"])

        self.assertEqual([len(call) for call in embedder.calls], [2, 2, 1])
        self.assertEqual(len(vectors), 5)

    def test_store_is_idempotent_and_numeric_search_returns_citation(self):
        source = """# Talk\n\n## Slide 1\n\n### PPT 视觉提取\n\nGeoDose dose calculation runtime was 70.1 ms.\n"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture", created_at=datetime(2026, 8, 14, tzinfo=timezone.utc))
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                first_counts = store.counts()
                store.put_ingest_result(result)
                self.assertEqual(first_counts, store.counts())
                pack = Graph3Retriever(store).retrieve("GeoDose 70.1 ms", limit=5)

            self.assertEqual(pack.slot_status["direct_evidence"], SlotStatus.SUPPORTED)
            self.assertTrue(any("70.1 ms" in item.value for item in pack.evidence))
            self.assertTrue(pack.citations[0]["locator"]["slide"] == 1)
            self.assertFalse(pack.retrieval_trace["completion_model_called"])

    def test_slot_coverage_requests_follow_up_when_required_evidence_is_missing(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose dose calculation runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                pack = Graph3Retriever(store).retrieve("GeoDose 的机制和结果", limit=5)

            self.assertEqual(pack.slot_status["direct_evidence"], SlotStatus.SUPPORTED)
            self.assertEqual(pack.slot_status["quantitative_result"], SlotStatus.SUPPORTED)
            self.assertEqual(pack.slot_status["mechanism"], SlotStatus.MISSING)
            self.assertEqual(pack.missing, ["mechanism"])
            self.assertTrue(pack.follow_up_required)
            self.assertIn("机制或流程", pack.follow_up_questions[0].question)
            self.assertEqual(
                pack.query_plan.evidence_slots[1].status,
                SlotStatus.MISSING,
            )

    def test_evidence_selection_fills_missing_required_slot_from_candidate_pool(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose architecture framework has three modules.

## Slide 2

### PPT 视觉提取

GeoDose architecture framework receives real-time anatomy.

## Slide 3

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                pack = Graph3Retriever(store).retrieve("GeoDose 的机制和结果", limit=2)

            self.assertEqual(pack.slot_status["mechanism"], SlotStatus.SUPPORTED)
            self.assertEqual(pack.slot_status["quantitative_result"], SlotStatus.SUPPORTED)
            self.assertFalse(pack.follow_up_required)
            self.assertTrue(any("70.1 ms" in item.value for item in pack.evidence))

    def test_vector_and_graph_routes_are_merged(self):
        source = """# Talk\n\n## Slide 1\n\n### PPT 视觉提取\n\nDREME provides anatomy to GeoDose.\n"""

        class FakeEmbedder:
            model = "fake-embedding-v1"

            def embed(self, texts):
                return [[1.0, 0.0] if "GeoDose" in text else [0.0, 1.0] for text in texts]

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            observation = result.observations[0]
            entities = extract_entities(observation)
            relations = extract_cooccurrence_relations(observation, entities)
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_graph(entities, relations)
                retriever = Graph3Retriever(store, embedder=FakeEmbedder())
                self.assertEqual(retriever.index_vectors(), len(result.observations))
                pack = retriever.retrieve("GeoDose", limit=5)

            self.assertIn("vector", pack.retrieval_trace["routes"])
            self.assertTrue(pack.evidence)
            self.assertTrue(any("vector" in item.retrieval_routes for item in pack.evidence))
            self.assertTrue(pack.graph_paths)

    def test_vector_index_falls_back_to_single_items_after_batch_failure(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose dose calculation runtime was 70.1 ms.

## Slide 2

### PPT 视觉提取

DREME supports adaptive radiotherapy.
"""

        class BatchFailingEmbedder:
            model = "batch-failing-fake-v1"
            batch_size = 2

            def embed(self, texts):
                if len(texts) > 1:
                    raise RuntimeError("fake batch failure")
                return [[1.0, 0.0]]

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                retriever = Graph3Retriever(store, embedder=BatchFailingEmbedder())
                self.assertEqual(retriever.index_vectors(), len(result.observations))
                self.assertEqual(store.counts()["observation_embeddings"], len(result.observations))

    def test_ambiguous_multi_direction_query_returns_clarification_options(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME method supports real-time anatomy reconstruction.

## Slide 2

### PPT 视觉提取

GeoDose method supports dose calculation.
"""

        class BroadEmbedder:
            model = "broad-fake-v1"

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                entities = []
                relations = []
                for observation in result.observations:
                    extracted = extract_entities(observation)
                    entities.extend(extracted)
                    relations.extend(extract_cooccurrence_relations(observation, extracted))
                store.put_graph(entities, relations)
                retriever = Graph3Retriever(store, embedder=BroadEmbedder())
                retriever.index_vectors()
                pack = retriever.retrieve("AI 方法的效果", limit=5)

            self.assertTrue(pack.follow_up_required)
            self.assertTrue(pack.follow_up_questions[0].options)
            self.assertIn("DREME", pack.follow_up_questions[0].options)
            self.assertIn("GeoDose", pack.follow_up_questions[0].options)

    def test_explicit_framework_subject_does_not_trigger_ambiguity_follow_up(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

AI-driven in-treatment adaptation framework uses DREME for anatomy reconstruction and GeoDose for dose accumulation.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                pack = Graph3Retriever(store).retrieve(
                    "AI-driven in-treatment adaptation framework 的完整流程",
                    limit=5,
                )

            self.assertFalse(pack.follow_up_required)
            self.assertEqual(pack.slot_status["mechanism"], SlotStatus.SUPPORTED)

    def test_runtime_words_require_quantitative_slot(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose dose accumulation takes 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                pack = Graph3Retriever(store).retrieve(
                    "GeoDose 的机制、输入、输出和运行时间",
                    limit=5,
                )

            self.assertIn("quantitative_result", pack.slot_status)
            self.assertEqual(pack.slot_status["quantitative_result"], SlotStatus.SUPPORTED)


if __name__ == "__main__":
    unittest.main()
