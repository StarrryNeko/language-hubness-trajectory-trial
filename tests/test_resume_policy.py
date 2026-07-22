import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_model_suite import extraction_reusable


class ResumePolicyTests(unittest.TestCase):
    def make_config(self, root, storage_dtype):
        return {
            "output_dir": str(root),
            "storage_dtype": storage_dtype,
            "model": {"name_or_path": "facebook/xglm-1.7B"},
            "metrics": {
                "representations": ["mean_pool", "sentinel_eos"],
                "primary_representation": "mean_pool",
                "validation_representation": "sentinel_eos",
            },
        }

    def write_extraction(self, root, storage_dtype):
        (root / "hidden").mkdir(parents=True)
        for name in ("metadata.csv", "sentence_layer_mean_pool.npy", "sentence_layer_sentinel_eos.npy"):
            (root / "hidden" / name).write_bytes(b"present")
        (root / "extraction_manifest.json").write_text(json.dumps({
            "model": "facebook/xglm-1.7B",
            "storage_dtype": storage_dtype,
            "representations": ["mean_pool", "sentinel_eos"],
            "truncated_inputs": 0,
        }), encoding="utf-8")

    def test_old_fp16_xglm_extraction_is_not_reused_for_fp32_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_extraction(root, "float16")
            self.assertFalse(extraction_reusable(self.make_config(root, "float32")))

    def test_matching_extraction_can_skip_expensive_model_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_extraction(root, "float32")
            self.assertTrue(extraction_reusable(self.make_config(root, "float32")))


if __name__ == "__main__":
    unittest.main()
