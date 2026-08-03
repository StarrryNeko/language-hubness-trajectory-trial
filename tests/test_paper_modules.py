import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compute_alignment import retrieval_hits
from compute_geometry_controls import fit_common_directions, remove_common_directions
from compute_hubness_paper import candidate_measure_samples, max_target_null
from compute_language_structure import fractional_topk_purity, layer_windows
from paper_common import build_semantic_splits, derangement, load_or_create_splits
from run_language_probe import semantic_split_masks


class PaperModuleTests(unittest.TestCase):
    def test_derangement_has_no_fixed_points(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            permutation = derangement(rng, 30)
            self.assertTrue(np.all(permutation != np.arange(30)))

    def test_retrieval_candidates_are_target_language_rows_only(self):
        vectors = np.eye(6)
        hits = retrieval_hits(vectors, vectors)
        self.assertTrue(hits[1].all())
        shuffled = vectors[[1, 2, 3, 4, 5, 0]]
        self.assertFalse(retrieval_hits(vectors, shuffled)[1].any())

    def test_semantic_split_never_separates_parallel_translations(self):
        split = build_semantic_splits([str(i) for i in range(20)], 11)
        meta = pd.DataFrame([
            {"row_idx": row, "id": semantic_id, "lang": lang}
            for row, (semantic_id, lang) in enumerate(
                (item for semantic_id in map(str, range(20)) for item in [(semantic_id, "en"), (semantic_id, "zh")])
            )
        ])
        masks = semantic_split_masks(meta, split)
        for semantic_id in map(str, range(20)):
            row_mask = meta.sort_values("row_idx").id.astype(str).to_numpy() == semantic_id
            self.assertEqual(sum(bool(np.any(mask & row_mask)) for mask in masks.values()), 1)

    def test_frozen_split_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            load_or_create_splits(path, [str(i) for i in range(10)], 4)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_or_create_splits(path, [str(i) for i in range(11)], 4)

    def test_purity_excludes_parallel_group_and_fractionally_handles_ties(self):
        similarities = np.array([1.0, 0.9, 0.9, 0.9, 0.1])
        eligible = np.array([False, False, True, True, True])
        same_language = np.array([True, True, True, False, False])
        self.assertAlmostEqual(
            fractional_topk_purity(similarities, eligible, same_language, k=2),
            0.5,
        )

    def test_relative_layer_windows_exclude_embedding_layer(self):
        windows = layer_windows(13)
        self.assertNotIn(0, sum(windows.values(), []))
        self.assertEqual(sorted(sum(windows.values(), [])), list(range(1, 13)))

    def test_common_direction_is_estimated_without_eval_rows(self):
        rng = np.random.default_rng(12)
        direction = np.zeros(8)
        direction[0] = 1
        train = rng.normal(scale=0.05, size=(40, 8)) + 4 * direction
        evaluation = rng.normal(scale=0.05, size=(10, 8)) + 4 * direction
        mean, components, _ = fit_common_directions(train, 3, 5)
        transformed = remove_common_directions(evaluation, mean, components, 1)
        self.assertEqual(transformed.shape, evaluation.shape)
        self.assertTrue(np.isfinite(transformed).all())

    def test_rotated_candidate_measures_are_symmetric(self):
        occurrence = np.array([[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
        centrality = occurrence / 10
        percentile = occurrence / 3
        medoid = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        measures = candidate_measure_samples(occurrence, centrality, percentile, medoid, k=2)
        self.assertEqual(set(measures), {
            "k_occurrence_excess", "centrality_advantage",
            "rank_percentile_advantage", "medoid_rate_excess",
        })
        self.assertTrue(np.allclose(measures["k_occurrence_excess"].sum(axis=1), 0))

    def test_max_target_null_controls_target_selection(self):
        samples = np.array([[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]])
        permutations = np.array([
            [[0, 1, 2], [0, 1, 2]],
            [[1, 0, 2], [2, 1, 0]],
        ])
        null = max_target_null(samples, permutations)
        self.assertEqual(null.shape, (2,))
        self.assertTrue(np.all(null >= samples.mean(axis=0).min()))


if __name__ == "__main__":
    unittest.main()
