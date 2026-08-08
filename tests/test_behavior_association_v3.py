import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_association_v3.common import classify_finish, settings, trim_completion
from behavior_association_v3.prepare_tasks import build
from common import load_config
from structure_v2.geometry import bootstrap_interval


class BehaviorAssociationV3Tests(unittest.TestCase):
    @classmethod
    def config(cls):
        root = Path(__file__).resolve().parents[1]
        return load_config(root / "configs" / "behavior_association_v3" / "xglm_1b7.json")

    def test_frozen_semantic_partition_is_disjoint_and_complete(self):
        cfg = self.config()
        languages = list(cfg["dataset"]["languages"])
        rows = [
            {"id": f"{semantic_id:05d}", "lang": language, "text": f"{semantic_id} {language}"}
            for semantic_id in range(804) for language in languages
        ]
        demos, split_ids, tasks = build(cfg, rows)
        self.assertEqual(len(demos), 4)
        self.assertEqual(len(split_ids["calibration"]), 80)
        self.assertEqual(len(split_ids["formal"]), 720)
        self.assertFalse(set(demos) & set(split_ids["calibration"]))
        self.assertFalse(set(demos) & set(split_ids["formal"]))
        self.assertFalse(set(split_ids["calibration"]) & set(split_ids["formal"]))
        self.assertEqual(len(tasks["calibration"]), 400)
        self.assertEqual(len(tasks["formal"]), 3600)

    def test_text_boundary_is_not_a_token_ceiling(self):
        text, stopped, marker = trim_completion("正确翻译\nChinese: continuation", ["\n"])
        self.assertEqual(text, "正确翻译")
        self.assertTrue(stopped)
        self.assertEqual(marker, "\n")
        text, stopped, marker = trim_completion("正常原生长度", ["\n"])
        self.assertEqual(text, "正常原生长度")
        self.assertFalse(stopped)
        self.assertIsNone(marker)

    def test_finish_reason_uses_the_earliest_real_boundary(self):
        self.assertEqual(classify_finish(12, True, 13, 192), "text_boundary")
        self.assertEqual(classify_finish(12, False, 13, 192), "native_eos")
        self.assertEqual(classify_finish(None, False, 192, 192), "token_ceiling")
        with self.assertRaisesRegex(ValueError, "recognized V3 boundary"):
            classify_finish(None, False, 18, 192)

    def test_dynamic_batch_and_target_inventory_are_frozen(self):
        protocol = settings(self.config())
        self.assertEqual(protocol["target_languages"], ["zh", "ar", "hi", "ru", "ja"])
        self.assertIn("\n\n", protocol["decoding"]["stop_strings"])
        self.assertIn("\nEnglish:", protocol["decoding"]["stop_strings"])
        self.assertLessEqual(protocol["runtime"]["initial_batch_size"], protocol["runtime"]["maximum_batch_size"])
        self.assertEqual(protocol["counts"], {"demonstration": 4, "calibration": 80, "formal": 720})

    def test_semantic_bootstrap_is_deterministic(self):
        values = np.asarray([-1.0, 0.0, 1.0, 2.0])
        first = bootstrap_interval(values, 17, 100, 0.95)
        second = bootstrap_interval(values, 17, 100, 0.95)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.5)


if __name__ == "__main__":
    unittest.main()
