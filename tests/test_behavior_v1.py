import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import select_semantic_indices_excluding
from compute_behavior_association import benjamini_hochberg
from evaluate_behavior_outputs import LanguageIdentifier
from evaluate_behavior_outputs import classify_language_behavior, load_frozen_calibration_report
from export_behavior_predictors import align_semantic_ids, similarity_matrices
from prepare_behavior_tasks import build_tasks


class BehaviorV1Tests(unittest.TestCase):
    def test_low_confidence_english_label_is_not_leakage(self):
        retention, leakage = classify_language_behavior(
            "en", 0.42, "sw", 0.0, 0.70, 0.15
        )
        self.assertEqual(retention, 0)
        self.assertEqual(leakage, 0)

    def test_confident_english_label_or_span_is_leakage(self):
        _, confident_label = classify_language_behavior(
            "en", 0.91, "sw", 0.0, 0.70, 0.15
        )
        _, confident_span = classify_language_behavior(
            "sw", 0.91, "sw", 0.20, 0.70, 0.15
        )
        self.assertEqual(confident_label, 1)
        self.assertEqual(confident_span, 1)

    def test_english_target_has_no_unnecessary_leakage_outcome(self):
        retention, leakage = classify_language_behavior(
            "en", 0.91, "en", 1.0, 0.70, 0.15
        )
        self.assertEqual(retention, 1)
        self.assertTrue(np.isnan(leakage))

    def test_evaluation_requires_configured_threshold_to_match_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "model.json"
            config_path.write_text("{}", encoding="utf-8")
            report_path = root / "calibration_lid_report.json"
            report_path.write_text(json.dumps({
                "mode": "calibration",
                "threshold_selection_permitted": True,
                "selected_threshold_by_frozen_rule": 0.70,
                "required_accuracy": 0.95,
                "threshold_sweep": [{"threshold": 0.70}],
                "threshold_selection_rule": "highest_candidate_with_overall_accuracy_at_least_minimum",
            }), encoding="utf-8")
            settings = {"language_id": {
                "confidence_threshold": 0.70,
                "calibration": {
                    "report_path": "calibration_lid_report.json",
                    "candidate_thresholds": [0.70],
                    "selection_rule": "highest_candidate_with_overall_accuracy_at_least_minimum",
                },
            }}
            cfg = {"behavior_v1": {"minimum_reference_lid_accuracy": 0.95}}
            resolved, _ = load_frozen_calibration_report(config_path, cfg, settings)
            self.assertEqual(resolved, report_path)
            settings["language_id"]["confidence_threshold"] = 0.65
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_frozen_calibration_report(config_path, cfg, settings)

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
                "language_id": {
                    "backend": "script_heuristic",
                    "reference_gate": "top1_label_accuracy",
                    "english_span_threshold_status": "frozen_before_behavior_results",
                    "calibration": {
                        "report_path": "calibration_lid_report.json",
                        "candidate_thresholds": [0.7],
                        "selection_rule": "highest_candidate_with_overall_accuracy_at_least_minimum",
                    },
                },
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

    def test_pc1_projection_mean_is_scalar_convertible(self):
        centered = np.arange(12, dtype=float).reshape(4, 3)
        direction = np.array([0.2, -0.1, 0.5])
        projection_mean = float((centered @ direction).mean())
        self.assertIsInstance(projection_mean, float)


if __name__ == "__main__":
    unittest.main()
