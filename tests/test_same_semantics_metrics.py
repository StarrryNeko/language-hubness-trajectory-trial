import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import configured_representations, load_config, validate_language_inventory
from compute_metrics import bootstrap_mean_ci, group_statistics, locally_scaled_similarity, rank_percentiles


class SameSemanticMetricTests(unittest.TestCase):
    def test_official_config_has_only_two_representations_and_24_languages(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "qwen25_1_5b_mvp.json")
        self.assertEqual(configured_representations(cfg), ["mean_pool", "sentinel_eos"])
        self.assertEqual(len(validate_language_inventory(cfg)), 24)

    def test_group_knn_conserves_total_occurrence(self):
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(24, 16))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = vectors @ vectors.T
        selected, occurrence, *_ = group_statistics(similarity, k=5)
        self.assertEqual(int(selected.sum()), 24 * 5)
        self.assertEqual(int(occurrence.sum()), 24 * 5)
        self.assertTrue(np.all(np.diag(selected) == 0))

    def test_engineered_english_center_becomes_reverse_knn_hub(self):
        rng = np.random.default_rng(9)
        center = np.ones(32)
        center /= np.linalg.norm(center)
        others = center + rng.normal(scale=0.35, size=(23, 32))
        vectors = np.vstack([center, others])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        _, occurrence, centrality, percentile, medoid = group_statistics(vectors @ vectors.T, k=3)
        self.assertGreater(occurrence[0], 3)
        self.assertEqual(percentile[0], 1.0)
        self.assertEqual(medoid[0], 1.0)
        self.assertEqual(centrality.argmax(), 0)

    def test_local_scaling_is_symmetric_and_rank_percentiles_are_bounded(self):
        similarity = np.array([
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ])
        adjusted = locally_scaled_similarity(similarity, density_k=1)
        self.assertTrue(np.allclose(adjusted, adjusted.T))
        ranks = rank_percentiles([3.0, 2.0, 1.0])
        self.assertTrue(np.allclose(ranks, [1.0, 0.5, 0.0]))

    def test_topk_boundary_ties_are_fractional_not_language_ordered(self):
        similarity = np.ones((24, 24), dtype=float)
        selected, occurrence, *_ = group_statistics(similarity, k=5)
        self.assertTrue(np.allclose(selected.sum(axis=1), 5))
        self.assertTrue(np.allclose(occurrence, 5))
        self.assertTrue(np.allclose(np.diag(selected), 0))

    def test_bootstrap_rejects_nonfinite_observations(self):
        rng = np.random.default_rng(2)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bootstrap_mean_ci([0.1, np.nan], rng)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            bootstrap_mean_ci([], rng)

    def test_similarity_and_k_are_validated(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            locally_scaled_similarity(np.array([[1.0, np.nan], [np.nan, 1.0]]), 1)
        with self.assertRaisesRegex(ValueError, "k must be"):
            group_statistics(np.eye(3), k=0)
        with self.assertRaisesRegex(ValueError, "k must be"):
            group_statistics(np.eye(3), k=3)


if __name__ == "__main__":
    unittest.main()
