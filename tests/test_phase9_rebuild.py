import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurograph3.pipeline import ingest_and_build_graph
from neurograph3.store import Graph3Store


class Phase9RebuildTests(unittest.TestCase):
    def test_rebuild_pipeline_persists_claims_and_is_idempotent(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.

## Slide 2

### PPT 视觉提取

DREME provides anatomy to GeoDose.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            storage_root = Path(temp_dir) / "db"

            _, first_stats = ingest_and_build_graph(source_path, storage_root=storage_root, dataset="fixture")
            with Graph3Store(storage_root) as store:
                first_counts = store.counts()
            _, second_stats = ingest_and_build_graph(source_path, storage_root=storage_root, dataset="fixture")
            with Graph3Store(storage_root) as store:
                second_counts = store.counts()

            self.assertEqual(first_stats.claim_candidates, 1)
            self.assertEqual(first_stats.evidence_links_submitted, 1)
            self.assertEqual(first_counts, second_counts)
            self.assertEqual(second_stats.claim_candidates, 1)
            self.assertEqual(second_counts["claim_versions"], 1)
            self.assertEqual(second_counts["evidence_links"], 1)
            self.assertEqual(second_counts["relations"], 2)

    def test_batch_rebuild_emits_auditable_report(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            storage_root = root / "db"
            report_path = root / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/graph3_rebuild",
                    str(source_path),
                    "--storage-root",
                    str(storage_root),
                    "--report",
                    str(report_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["source_count"], 1)
            self.assertEqual(report["totals"]["claim_candidates"], 1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
