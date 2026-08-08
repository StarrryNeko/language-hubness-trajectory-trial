import sys
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_v2.common import lexical_script_features, settings, split_natural_completion
from behavior_v2.annotation import create_sample, validate_annotations
from behavior_v2.common import ensure_paths
from behavior_v2.prepare_tasks import build_tasks, role_order
from common import load_config


class BehaviorV2ProtocolTests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parents[1]
        return load_config(root / "configs" / "behavior_v2" / "xglm_1b7.json")

    def test_protocol_is_mean_pool_only_and_non_latin_targeted(self):
        protocol = settings(self.config())
        self.assertEqual(protocol["representation_protocol"], "mean_pool_v1")
        self.assertNotIn("en", protocol["source_languages"])
        self.assertNotIn("en", protocol["target_languages"])
        self.assertTrue(all(
            protocol["language_metadata"][lang]["script"] != "Latin"
            for lang in protocol["target_languages"]
        ))

    def test_english_detector_requires_a_span_and_function_word(self):
        positive = lexical_script_features("这是 the system is broken 今天", minimum_run=3)
        proper_name = lexical_script_features("访问 New York City 之后", minimum_run=3)
        ambiguous_single = lexical_script_features("访问 casa a Madrid 之后", minimum_run=3)
        single = lexical_script_features("这是 system 测试", minimum_run=3)
        self.assertEqual(positive["english_lexical_leakage"], 1)
        self.assertEqual(proper_name["has_latin_span"], 1)
        self.assertEqual(proper_name["english_lexical_leakage"], 0)
        self.assertEqual(ambiguous_single["english_lexical_leakage"], 0)
        self.assertEqual(single["has_latin_span"], 0)

    def test_natural_completion_excludes_termination_control_id(self):
        content, reason, token = split_natural_completion([101, 102, 2, 0], {2}, 4)
        self.assertEqual(content, [101, 102])
        self.assertEqual(reason, "natural_stop")
        self.assertEqual(token, 2)

    def test_ceiling_requires_exact_length(self):
        content, reason, token = split_natural_completion([1, 2, 3], {9}, 3)
        self.assertEqual((content, reason, token), ([1, 2, 3], "token_budget", None))
        with self.assertRaisesRegex(ValueError, "did not consume the token ceiling"):
            split_natural_completion([1, 2], {9}, 3)

    def test_role_assignment_is_deterministic(self):
        ids = ["00001", "00002", "00003", "00004"]
        self.assertEqual(role_order(ids, 17), role_order(reversed(ids), 17))
        self.assertNotEqual(role_order(ids, 17), role_order(ids, 18))

    def test_full_frozen_task_design_is_balanced(self):
        cfg = self.config()
        languages = list(cfg["dataset"]["languages"])
        rows = [
            {"id": f"{semantic_id:05d}", "lang": language,
             "text": f"text {semantic_id} {language}"}
            for semantic_id in range(804)
            for language in languages
        ]
        tasks, demos, evaluation_ids, summary = build_tasks(cfg, rows)
        self.assertEqual(len(demos), 4)
        self.assertEqual(len(evaluation_ids), 800)
        self.assertEqual(len(tasks), 4000)
        self.assertEqual(summary["semantic_ids"], 800)
        self.assertTrue(all(task["source_lang"] != "en" for task in tasks))
        self.assertTrue(all(task["target_lang"] != "en" for task in tasks))

    def test_detector_audit_is_blinded_and_bound_to_item_results(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            cfg["output_dir"] = str(Path(temporary) / "output")
            paths = ensure_paths(cfg)
            records = []
            for target in settings(cfg)["target_languages"]:
                records.append({
                    "task_id": f"positive-{target}", "semantic_id": "1",
                    "source_lang": "es", "target_lang": target,
                    "generated_text": "the system is broken",
                    "english_lexical_leakage": 1,
                })
                for index in range(5):
                    records.append({
                        "task_id": f"negative-{target}-{index}", "semantic_id": str(index + 2),
                        "source_lang": "es", "target_lang": target,
                        "generated_text": "target text",
                        "english_lexical_leakage": 0,
                    })
            pd.DataFrame(records).to_csv(
                paths.metrics / "behavior_v2_item_results.csv", index=False
            )
            audit = paths.measurement / "audit.csv"
            report = paths.measurement / "report.json"
            create_sample(cfg, audit, 60)
            frame = pd.read_csv(audit)
            self.assertNotIn("automatic_english_leakage", frame.columns)
            frame["human_english_leakage"] = frame.task_id.str.startswith("positive-").astype(int)
            frame.to_csv(audit, index=False)
            validate_annotations(cfg, audit, report)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["precision"], 1.0)
            self.assertEqual(payload["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
