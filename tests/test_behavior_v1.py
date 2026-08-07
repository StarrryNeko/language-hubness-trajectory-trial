import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import select_semantic_indices_excluding
from compute_behavior_association import benjamini_hochberg
from evaluate_behavior_outputs import LanguageIdentifier
from export_behavior_predictors import align_semantic_ids, similarity_matrices
from prepare_behavior_tasks import build_tasks


class BehaviorV1Tests(unittest.TestCase):
    def make_config(self):
        languages = ["en", "zh", "ar", "hi", "es"]
        return {
            "seed": 9,
            "output_dir": "unused",
            "dataset": {"languages": {language: language for language in languages}},
            "behavior_v1": {
                "enabled": True,
                "seed": 9,
                "evaluation_languages": languages[1:],
                "language_names": {language: language for language in languages},
                "demonstration_semantic_ids": 2,
                "evaluation_semantic_ids": 5,
                "prompt_template": "{demonstrations}\n\n{source_name}: {source_text}\n{target_name}:",
                "decoding": {"do_sample": False, "num_beams": 1},
                "primary_layer": 2,
                "analysis_layers": [1, 2, 3],
                "language_id": {"backend": "script_heuristic"},
            },
        }

    def test_exclusion_sampling_is_disjoint_and_deterministic(self):
        first = select_semantic_indices_excluding(100, 20, "random_without_replacement", 7, [1, 2, 3])
        second = select_semantic_indices_excluding(100, 20, "random_without_replacement", 7, [1, 2, 3])
        self.assertEqual(first, second)
        self.assertFalse(set(first) & {1, 2, 3})

    def test_task_builder_keeps_demos_out_of_evaluation(self):
        cfg = self.make_config()
        rows = [
            {"id": semantic_id, "lang": language, "text": f"{language}-{semantic_id}"}
            for semantic_id in range(7)
            for language in cfg["dataset"]["languages"]
        ]
        tasks, demos, evaluation, summary = build_tasks(cfg, rows)
        self.assertEqual(len(tasks), 5 * 4 * 3)
        self.assertFalse(set(demos) & set(evaluation))
        self.assertEqual(summary["semantic_ids"], 5)
        self.assertTrue(all("-0" in task["prompt"] for task in tasks))

    def test_script_lid_is_explicitly_smoke_only(self):
        identifier = LanguageIdentifier(self.make_config()["behavior_v1"])
        self.assertEqual(identifier.predict("这是中文句子")[0], "zh")
        self.assertEqual(identifier.predict("هذا نص عربي")[0], "ar")

    def test_bh_adjustment_is_monotone_in_p_value_order(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
        self.assertTrue(np.isnan(adjusted[-1]))
        ordered = adjusted[np.argsort([0.01, 0.04, 0.03, 1.0])[:3]]
        self.assertTrue(np.all(np.diff(ordered) >= -1e-12))

    def test_cpu_similarity_backend_matches_cosine_contract(self):
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(4, 5, 7)).astype(np.float32)
        cosine, scaled = similarity_matrices(vectors, 2, {
            "geometry_device": "cpu", "geometry_dtype": "float32", "allow_tf32": False,
        })
        normalized = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)
        expected = normalized @ np.swapaxes(normalized, 1, 2)
        np.testing.assert_allclose(cosine, expected, atol=1e-6)
        self.assertEqual(scaled.shape, (4, 5, 5))
        np.testing.assert_allclose(np.diagonal(scaled, axis1=1, axis2=2), 1.0)

    def test_behavior_ids_align_after_csv_strips_leading_zeroes(self):
        task_ids, hidden_ids = align_semantic_ids(
            ["00040", "00049", "00102"], ["40", "49", "102"]
        )
        self.assertEqual(task_ids, ["00040", "00049", "00102"])
        self.assertEqual(hidden_ids, ["40", "49", "102"])


if __name__ == "__main__":
    unittest.main()
