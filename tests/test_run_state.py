import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspect_run_state import classify_run_state


class RunStateTests(unittest.TestCase):
    def config(self, root):
        return {
            "output_dir": str(root),
            "storage_dtype": "float32",
            "model": {"name_or_path": "org/model"},
            "metrics": {"representations": ["mean_pool"], "primary_representation": "mean_pool"},
            "paper_v1": {"result_directory": "paper_v1"},
        }

    def test_partial_hidden_is_never_called_reusable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "hidden").mkdir()
            (root / "hidden" / "sentence_layer_mean_pool.npy").write_bytes(b"partial")
            state, _ = classify_run_state(self.config(root))
            self.assertEqual(state, "PARTIAL_OR_INCOMPATIBLE_EXTRACTION")

    def test_paper_summary_has_highest_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = root / "paper_v1" / "validation" / "paper_validation_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"status": "done"}), encoding="utf-8")
            state, _ = classify_run_state(self.config(root))
            self.assertEqual(state, "PAPER_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    unittest.main()

