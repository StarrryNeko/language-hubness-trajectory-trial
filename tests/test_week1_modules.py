import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import portable_model_directory_name, resolve_model_source
from compute_norm_trajectory import (
    descending_average_rank,
    leave_one_out_centroid_distances,
    radial_tangential_update,
)
from compute_similarity_competition import candidate_competition
from run_extraction_suite import validate_frozen_suite


class WeekOneModuleTests(unittest.TestCase):
    def test_hard_margin_exposes_mean_dilution(self):
        # Source 0 sees English (candidate 1) beat most languages, but candidate 2 is stronger.
        matrix = np.eye(5, dtype=float)[None, :, :]
        matrix[0, 0, 1] = matrix[0, 1, 0] = 0.8
        matrix[0, 0, 2] = matrix[0, 2, 0] = 0.9
        matrix[0, 0, 3] = matrix[0, 3, 0] = 0.2
        matrix[0, 0, 4] = matrix[0, 4, 0] = 0.1
        result = candidate_competition(matrix, english_index=1)
        source_column = list(result["source_indices"]).index(0)
        self.assertAlmostEqual(result["hard_margin"][0, source_column], -0.1)
        self.assertEqual(result["best_non_english_index"][0, source_column], 2)
        self.assertGreater(result["pairwise_win_rate"][0, source_column], 0.5)

    def test_competition_ties_receive_half_credit_and_average_rank(self):
        matrix = np.zeros((1, 4, 4), dtype=float)
        np.fill_diagonal(matrix[0], 1.0)
        matrix[0, 0, 1:] = [0.7, 0.7, 0.2]
        matrix[0, 1:, 0] = matrix[0, 0, 1:]
        result = candidate_competition(matrix, english_index=1)
        source_column = list(result["source_indices"]).index(0)
        self.assertAlmostEqual(result["hard_margin"][0, source_column], 0.0)
        self.assertAlmostEqual(result["pairwise_win_rate"][0, source_column], 0.75)
        self.assertAlmostEqual(result["english_rank"][0, source_column], 1.5)

    def test_leave_one_out_centroid_excludes_current_vector(self):
        groups = np.array([[[0.0], [2.0], [4.0]]])
        distances = leave_one_out_centroid_distances(groups)
        self.assertTrue(np.allclose(distances, [[3.0, 0.0, 3.0]]))

    def test_radial_and_tangential_updates_are_separated(self):
        previous = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        radial_current = np.array([[[2.0, 0.0], [0.0, 3.0]]])
        radial, tangential = radial_tangential_update(previous, radial_current)
        self.assertTrue(np.allclose(radial, [[1.0, 2.0]]))
        self.assertTrue(np.allclose(tangential, 0.0))
        turning_current = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        radial, tangential = radial_tangential_update(previous, turning_current)
        self.assertTrue(np.allclose(radial, 0.0))
        self.assertTrue(np.allclose(tangential, 1.0))

    def test_descending_rank_handles_ties(self):
        ranks = descending_average_rank(np.array([[4.0, 2.0, 2.0, 1.0]]))
        self.assertTrue(np.allclose(ranks, [[1.0, 2.5, 2.5, 4.0]]))

    def test_portable_model_root_preserves_canonical_id(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / portable_model_directory_name("org/model")
            target.mkdir()
            old = os.environ.get("LHT_MODEL_ROOT")
            os.environ["LHT_MODEL_ROOT"] = folder
            try:
                canonical, source, is_local = resolve_model_source("org/model")
            finally:
                if old is None:
                    os.environ.pop("LHT_MODEL_ROOT", None)
                else:
                    os.environ["LHT_MODEL_ROOT"] = old
            self.assertEqual(canonical, "org/model")
            self.assertEqual(Path(source), target.resolve())
            self.assertTrue(is_local)

    def test_frozen_suite_rejects_sample_drift(self):
        suite = {
            "sample_size_per_language": 200,
            "sample_selection_strategy": "random_without_replacement",
            "sample_selection_seed": 20260801,
            "model_list_frozen_before_results": True,
        }
        configs = [
            {
                "experiment_name": "a",
                "output_dir": "out/a",
                "model": {"name_or_path": "org/a"},
                "dataset": {
                    "languages": {"en": "eng", "zh": "zho"},
                    "sample_size_per_language": 200,
                    "sample_selection": {
                        "strategy": "random_without_replacement", "seed": 20260801
                    },
                },
            },
            {
                "experiment_name": "b",
                "output_dir": "out/b",
                "model": {"name_or_path": "org/b"},
                "dataset": {
                    "languages": {"en": "eng", "zh": "zho"},
                    "sample_size_per_language": 100,
                    "sample_selection": {
                        "strategy": "random_without_replacement", "seed": 20260801
                    },
                },
            },
        ]
        with self.assertRaisesRegex(ValueError, "sample_size_per_language"):
            validate_frozen_suite(suite, configs)


if __name__ == "__main__":
    unittest.main()
