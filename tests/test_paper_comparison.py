import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compare_paper_models import compare_suite


class PaperComparisonTests(unittest.TestCase):
    def test_comparison_preserves_negative_and_density_sensitive_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = []
            for index, status in enumerate(("NOT_SUPPORTED", "DENSITY_SENSITIVE")):
                output = root / f"output_{index}"
                paper = output / "paper_v1"
                (paper / "validation").mkdir(parents=True)
                (paper / "metrics" / "hubness").mkdir(parents=True)
                (paper / "validation" / "paper_validation_summary.json").write_text(
                    json.dumps({
                        "overall_assessment": "READY_FOR_MODEL_COMPARISON",
                        "claim_level": status,
                        "statuses": {
                            "alignment_status": "SUPPORTED",
                            "english_specificity_status": "NOT_SUPPORTED",
                            "geometry_robustness_status": status,
                            "sample_robustness_status": "REPLICATED",
                        },
                    }), encoding="utf-8"
                )
                pd.DataFrame([
                    {"similarity_method": method, "english_rank": rank, "layer": 1}
                    for method, rank in (("cosine", index + 1), ("local_scaled_cosine", index + 2))
                ]).to_csv(
                    paper / "metrics" / "hubness" / "target_rotation_summary.csv", index=False
                )
                config = {
                    "experiment_name": f"model_{index}",
                    "output_dir": str(output),
                    "model": {"name_or_path": f"org/model-{index}"},
                }
                config_path = root / f"model_{index}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                configs.append(config_path.name)
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps({
                "configs": configs,
                "comparison_output_dir": str(root / "comparison"),
            }), encoding="utf-8")
            frame, summary = compare_suite(suite_path)
            self.assertEqual(len(frame), 2)
            self.assertEqual(summary["cross_model_status"], "AVAILABLE")
            self.assertEqual(set(frame.geometry_robustness_status), {"NOT_SUPPORTED", "DENSITY_SENSITIVE"})


if __name__ == "__main__":
    unittest.main()

