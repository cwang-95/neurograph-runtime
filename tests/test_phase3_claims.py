import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from neurograph3.extract import extract_numeric_claims
from neurograph3.ingest import ingest_markdown
from neurograph3.models import ObservationKind
from neurograph3.store import Graph3Store


class Phase3ClaimTests(unittest.TestCase):
    def test_numeric_candidate_keeps_source_link_and_persists_idempotently(self):
        source = """# Talk\n\n## Slide 1\n\n### PPT 视觉提取\n\nGeoDose delivered dose calculation took 70.1 ms.\n"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=Path(temp_dir) / "assets", dataset="fixture", created_at=datetime(2026, 8, 14, tzinfo=timezone.utc))
            observation = next(item for item in result.observations if item.kind is ObservationKind.VISION)
            claims = extract_numeric_claims(observation, created_at=datetime(2026, 8, 14, tzinfo=timezone.utc))
            self.assertEqual(len(claims), 1)
            claim, link = claims[0]
            self.assertEqual(claim.subject, "GeoDose")
            self.assertEqual(claim.predicate, "runtime")
            self.assertEqual(claim.object_value, 70.1)
            self.assertEqual(claim.unit, "ms")
            self.assertEqual(link.observation_id, observation.observation_id)
            with Graph3Store(Path(temp_dir) / "db") as store:
                store.put_ingest_result(result)
                store.put_claims(claims)
                first_counts = store.counts()
                store.put_claims(claims)
                self.assertEqual(first_counts, store.counts())
                self.assertEqual(first_counts["claim_versions"], 1)
                self.assertEqual(first_counts["evidence_links"], 1)


if __name__ == "__main__":
    unittest.main()
