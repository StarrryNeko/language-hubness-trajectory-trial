import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numerical_validation import (
    require_finite,
    require_nonzero_row_norms,
    validate_representation_array,
    validate_similarity_matrix,
)


class NumericalValidationTests(unittest.TestCase):
    def test_non_finite_values_are_rejected_with_context(self):
        with self.assertRaisesRegex(ValueError, "model=xglm.*NaN/Inf"):
            require_finite([1.0, np.nan], "model=xglm")

    def test_zero_norm_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero/near-zero"):
            require_nonzero_row_norms([[1.0, 0.0], [0.0, 0.0]], "semantic=s1")

    def test_representation_shape_and_rows_are_validated(self):
        with self.assertRaisesRegex(ValueError, "must be 3D"):
            validate_representation_array(np.ones((2, 3)), 2, "mean_pool")
        with self.assertRaisesRegex(ValueError, "row count"):
            validate_representation_array(np.ones((2, 3, 4)), 3, "mean_pool")

    def test_similarity_must_be_square_finite_and_symmetric(self):
        with self.assertRaisesRegex(ValueError, "square"):
            validate_similarity_matrix(np.ones((2, 3)), "cosine")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_similarity_matrix([[1.0, np.nan], [np.nan, 1.0]], "cosine")
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            validate_similarity_matrix([[1.0, 0.2], [0.3, 1.0]], "cosine")

    def test_valid_inputs_are_returned(self):
        values = np.ones((2, 3, 4), dtype=np.float32)
        self.assertEqual(validate_representation_array(values, 2, "valid").shape, values.shape)
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.assertTrue(np.array_equal(validate_similarity_matrix(matrix, "valid"), matrix))


if __name__ == "__main__":
    unittest.main()
