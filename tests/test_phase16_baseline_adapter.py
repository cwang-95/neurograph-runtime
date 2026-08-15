import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from importlib.machinery import SourceFileLoader


def load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Phase16BaselineAdapterTests(unittest.TestCase):
    def test_small_sample_p95_uses_nearest_rank_max(self):
        self.assertEqual(load_script("graph3_eval")._p95_latency([1.0, 2.0, 100.0]), 100.0)
        self.assertEqual(load_script("cognee_eval")._p95_latency([1.0, 2.0, 100.0]), 100.0)

    def test_cognee_adapter_parses_pretty_json_after_logs(self):
        adapter = load_script("cognee_eval")
        payload = {"mode": "graph-evidence", "query": "q", "graph_context": ["70.1 ms"]}
        stdout = "log line\nwarning {not json}\n" + json.dumps(payload, indent=2)
        self.assertEqual(adapter._parse_payload(stdout), payload)

    def test_ab_compare_marks_reports_non_comparable_without_corpus_match(self):
        compare = load_script("graph3_ab_compare")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph3 = root / "graph3.json"
            baseline = root / "baseline.json"
            graph3.write_text(
                json.dumps({"corpus_label": "aapm", "corpus_fingerprint": "same", "isolated_storage": True, "pass_rate": 1.0, "mean_latency_ms": 2.0, "p95_latency_ms": 3.0}),
                encoding="utf-8",
            )
            baseline.write_text(
                json.dumps({"corpus_label": "wiki", "corpus_fingerprint": "other", "isolated_storage": True, "pass_rate": 0.0, "mean_latency_ms": 5.0, "p95_latency_ms": 6.0}),
                encoding="utf-8",
            )
            report = json.loads(
                subprocess.check_output(
                    [
                        str(Path(__file__).resolve().parent.parent / "scripts" / "graph3_ab_compare"),
                        "--graph3-report",
                        str(graph3),
                        "--baseline-report",
                        str(baseline),
                    ],
                    text=True,
                )
            )
            self.assertFalse(report["comparable"])
            self.assertIsNone(report["delta_graph3_minus_baseline"])

    def test_ab_compare_accepts_matching_isolated_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph3 = root / "graph3.json"
            baseline = root / "baseline.json"
            payload = {
                "corpus_label": "aapm",
                "corpus_fingerprint": "same",
                "isolated_storage": True,
                "pass_rate": 1.0,
                "mean_latency_ms": 2.0,
                "p95_latency_ms": 3.0,
            }
            graph3.write_text(json.dumps(payload), encoding="utf-8")
            payload["pass_rate"] = 0.5
            payload["mean_latency_ms"] = 5.0
            payload["p95_latency_ms"] = 6.0
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            report = json.loads(
                subprocess.check_output(
                    [
                        str(Path(__file__).resolve().parent.parent / "scripts" / "graph3_ab_compare"),
                        "--graph3-report",
                        str(graph3),
                        "--baseline-report",
                        str(baseline),
                        "--corpus-match",
                    ],
                    text=True,
                )
            )
            self.assertTrue(report["comparable"])
            self.assertEqual(report["delta_graph3_minus_baseline"]["pass_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
