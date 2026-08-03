import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_model_suite import extraction_reusable, prepared_data_reusable


class ResumePolicyTests(unittest.TestCase):
    def test_random_sample_data_requires_matching_strategy_seed_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            (data / "parallel_samples.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = {
                "languages": ["en", "zh"],
                "semantic_groups": 200,
                "sample_selection_strategy": "random_without_replacement",
                "sample_selection_seed": 17,
                "selected_semantic_indices_sha256": "abc",
            }
            (data / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            cfg = {
                "output_dir": str(root),
                "dataset": {
                    "languages": {"en": "eng", "zh": "zho"},
                    "sample_size_per_language": 200,
                    "sample_selection": {"strategy": "random_without_replacement", "seed": 17},
                },
            }
            self.assertTrue(prepared_data_reusable(cfg))
            cfg["dataset"]["sample_selection"]["seed"] = 18
            self.assertFalse(prepared_data_reusable(cfg))

    def make_config(self, root, storage_dtype):
        return {
            "output_dir": str(root),
            "storage_dtype": storage_dtype,
            "model": {"name_or_path": "facebook/xglm-1.7B"},
            "metrics": {
                "representations": ["mean_pool"],
                "primary_representation": "mean_pool",
            },
        }

    def write_extraction(self, root, storage_dtype):
        (root / "hidden").mkdir(parents=True)
        for name in ("metadata.csv", "sentence_layer_mean_pool.npy"):
            (root / "hidden" / name).write_bytes(b"present")
        (root / "extraction_manifest.json").write_text(json.dumps({
            "protocol_version": "mean_pool_no_eos_v1",
            "model": "facebook/xglm-1.7B",
            "storage_dtype": storage_dtype,
            "representations": ["mean_pool"],
            "appended_eos": False,
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

    def test_old_protocol_manifest_cannot_reuse_hidden_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_extraction(root, "float32")
            manifest_path = root / "extraction_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol_version"] = "mean_pool_plus_eos_v0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertFalse(extraction_reusable(self.make_config(root, "float32")))


if __name__ == "__main__":
    unittest.main()
