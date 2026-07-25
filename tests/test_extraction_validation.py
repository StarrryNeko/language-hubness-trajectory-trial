import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import configured_representations, load_config
from numerical_validation import require_nonzero_row_norms, validate_representation_array


class ExtractionValidationTests(unittest.TestCase):
    def test_extraction_context_survives_in_numeric_error(self):
        context = "model=xglm row=7 semantic_id=s3 lang=zh layer=2 representation=mean_pool"
        with self.assertRaisesRegex(ValueError, "model=xglm row=7 semantic_id=s3 lang=zh layer=2"):
            require_nonzero_row_norms(np.zeros(4), context)

    def test_post_storage_overflow_is_rejected(self):
        values = np.array([1e10], dtype=np.float32).astype(np.float16)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_representation_array(values.reshape(1, 1, 1), 1, "after_storage")

    def test_xglm_forces_float32_compute_and_storage(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "xglm_1b7_24lang.json")
        self.assertEqual(cfg["dtype"], "float32")
        self.assertEqual(cfg["storage_dtype"], "float32")

    def test_qwen_forces_float32_compute_and_storage(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "qwen25_1_5b_mvp.json")
        self.assertEqual(cfg["dtype"], "float32")
        self.assertEqual(cfg["storage_dtype"], "float32")

    def test_active_protocol_extracts_only_mean_pool_without_eos(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "qwen25_1_5b_mvp.json")
        self.assertEqual(configured_representations(cfg), ["mean_pool"])
        self.assertNotIn("validation_representation", cfg["metrics"])
        self.assertNotIn("append_sentinel_eos", cfg.get("representation_controls", {}))


if __name__ == "__main__":
    unittest.main()
