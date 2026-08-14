import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurograph3.ingest import ingest_markdown
from neurograph3.store import Graph3Store


class Phase15EvaluationTests(unittest.TestCase):
    def test_eval_cli_reports_gold_recall_slots_and_traceability(self):
        source = """# Talk

## Slide 1

### PPT 视觉提取

GeoDose architecture framework has three modules.

## Slide 2

### PPT 视觉提取

GeoDose runtime was 70.1 ms.
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "talk.md"
            source_path.write_text(source, encoding="utf-8")
            result = ingest_markdown(source_path, storage_root=root / "assets", dataset="fixture")
            storage_root = root / "db"
            with Graph3Store(storage_root) as store:
                store.put_ingest_result(result)
            gold_path = root / "gold.json"
            gold_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "mechanism-and-number",
                            "query": "GeoDose 的机制和结果",
                            "expected_terms": ["architecture", "70.1 ms"],
                            "required_slots": ["mechanism", "quantitative_result"],
                            "must_follow_up": False,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/graph3_eval",
                    "--storage-root",
                    str(storage_root),
                    "--gold-file",
                    str(gold_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)

            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["passed_cases"], 1)
            self.assertEqual(report["pass_rate"], 1.0)
            self.assertEqual(report["citation_traceability_rate"], 1.0)
            self.assertEqual(report["results"][0]["slot_status"]["mechanism"], "supported")


if __name__ == "__main__":
    unittest.main()
