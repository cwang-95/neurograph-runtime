import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.models import ElementType, ObservationKind


class Phase1IngestTests(unittest.TestCase):
    def test_markdown_ingest_preserves_locations_and_deduplicates_observations(self):
        source = """# Talk\n\n## Slide 1\n\n### 讲者语音\n\nRepeated explanation.\n\n### PPT 视觉提取\n\nGeoDose: 70.1 ms\n\n## Slide 2\n\n### 讲者语音\n\nRepeated explanation.\n\n### PPT 视觉提取\n\nGeoDose: 70.1 ms\n"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(
                source_path,
                storage_root=Path(temp_dir) / "store",
                dataset="fixture",
                created_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            )

            self.assertEqual(len(result.observations), 2)
            asr = next(item for item in result.observations if item.kind is ObservationKind.ASR)
            vision = next(item for item in result.observations if item.kind is ObservationKind.VISION)
            self.assertEqual(len(asr.aligned_element_ids), 1)
            self.assertEqual(len(vision.aligned_element_ids), 1)
            self.assertEqual({item.locator.slide for item in result.elements if item.element_type is ElementType.FIGURE}, {1, 2})
            self.assertTrue(Path(result.asset.storage_uri).exists())


if __name__ == "__main__":
    unittest.main()
