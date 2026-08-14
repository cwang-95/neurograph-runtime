import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from neurograph3.entities import extract_entities
from neurograph3.ingest import ingest_markdown
from neurograph3.llm_relations import (
    DeepSeekRelationClient,
    load_deepseek_api_key,
    validate_relation_proposals,
)
from neurograph3.models import ObservationKind
from neurograph3.pipeline import build_relation_graph
from neurograph3.store import Graph3Store


class Phase3cLLMRelationTests(unittest.TestCase):
    def _observation(self, temp_dir):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME provides anatomy to GeoDose for dose calculation.
"""
        source_path = Path(temp_dir) / "talk.md"
        source_path.write_text(source, encoding="utf-8")
        result = ingest_markdown(
            source_path,
            storage_root=Path(temp_dir) / "assets",
            dataset="fixture",
            created_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        )
        observation = next(item for item in result.observations if item.kind is ObservationKind.VISION)
        return result, observation, extract_entities(observation)

    def test_validation_accepts_only_allowlisted_supported_relation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result, observation, entities = self._observation(temp_dir)
            output = validate_relation_proposals(
                {
                    "relations": [
                        {
                            "source": "DREME",
                            "predicate": "provides_input_to",
                            "target": "GeoDose",
                            "confidence": 0.91,
                            "rationale": "explicit provides ... to wording",
                        },
                        {
                            "source": "DREME",
                            "predicate": "co_occurs_in_observation",
                            "target": "GeoDose",
                            "confidence": 0.99,
                        },
                        {
                            "source": "DREME",
                            "predicate": "uses",
                            "target": "Unknown",
                            "confidence": 0.99,
                        },
                    ]
                },
                observation,
                entities,
                model="fake-deepseek",
            )

            self.assertEqual(len(output.accepted), 1)
            self.assertEqual(output.accepted[0].predicate, "provides_input_to")
            self.assertEqual(output.accepted[0].extraction_method, "deepseek-structured-v1")
            self.assertEqual(len(output.rejected), 2)

            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                self.assertEqual(store.put_relation_extraction_results(entities, [output]), 1)
                self.assertEqual(store.counts()["relations"], 1)

    def test_deepseek_client_parses_fenced_json_without_exposing_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result, observation, entities = self._observation(temp_dir)
            response_payload = {
                "choices": [
                    {
                        "message": {
                            "content": "```json\n{\"relations\":[{\"source\":\"DREME\",\"predicate\":\"provides_input_to\",\"target\":\"GeoDose\",\"confidence\":0.88}]}\n```"
                        }
                    }
                ]
            }

            class FakeHTTPResponse:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self):
                    return json.dumps(response_payload).encode("utf-8")

            with patch("urllib.request.urlopen", return_value=FakeHTTPResponse()):
                client = DeepSeekRelationClient(api_key="test-secret", model="fake-model")
                output = client.extract(observation, entities)

            self.assertEqual(output.model, "fake-model")
            self.assertEqual(len(output.accepted), 1)

    def test_deepseek_key_loader_reads_existing_openclaw_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "providers": {
                                "custom-api-deepseek-com": {"apiKey": "local-test-key"}
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=False):
                self.assertEqual(load_deepseek_api_key(config_path), "local-test-key")

    def test_relation_pipeline_is_opt_in_and_respects_llm_budget(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

DREME provides anatomy to GeoDose.

## Slide 2

### PPT 视觉提取

DREME provides real-time anatomy to GeoDose.
"""

        class FakeRelationClient:
            def __init__(self):
                self.calls = 0

            def extract(self, observation, entities):
                self.calls += 1
                return validate_relation_proposals(
                    {
                        "relations": [
                            {
                                "source": "DREME",
                                "predicate": "provides_input_to",
                                "target": "GeoDose",
                                "confidence": 0.9,
                            }
                        ]
                    },
                    observation,
                    entities,
                    model="fake",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture")
            client = FakeRelationClient()
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                stats = build_relation_graph(
                    store,
                    result.observations,
                    relation_client=client,
                    use_deepseek=True,
                    max_llm_calls=1,
                    include_cooccurrence=False,
                )

                self.assertEqual(client.calls, 1)
                self.assertEqual(stats.llm_calls, 1)
                self.assertEqual(stats.llm_accepted, 1)
                self.assertTrue(stats.llm_budget_exhausted)
                self.assertEqual(store.counts()["relations"], 1)

                no_llm_client = FakeRelationClient()
                no_llm_stats = build_relation_graph(
                    store,
                    result.observations,
                    relation_client=no_llm_client,
                    use_deepseek=False,
                )
                self.assertEqual(no_llm_client.calls, 0)
                self.assertEqual(no_llm_stats.llm_calls, 0)


if __name__ == "__main__":
    unittest.main()
