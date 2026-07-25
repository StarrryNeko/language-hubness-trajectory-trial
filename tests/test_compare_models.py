import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compare_models import compare_suite
from evidence_rules import REQUIRED_EVIDENCE_METRICS


class CompareModelTests(unittest.TestCase):
    def write_model(
        self,
        root,
        number,
        status="ROBUST",
        invalid_value=None,
        density_layers=None,
    ):
        output = root / f"output_{number}"
        (output / "metrics").mkdir(parents=True)
        (output / "validation").mkdir()
        rows = [
            {
                "representation": "mean_pool",
                "similarity_method": "cosine",
                "layer": layer,
                "metric": metric,
                "mean": 0.2,
                "ci_lower": invalid_value if invalid_value is not None and layer == 0 and metric == REQUIRED_EVIDENCE_METRICS[0] else 0.1,
                "ci_upper": 0.3,
            }
            for layer in range(3)
            for metric in REQUIRED_EVIDENCE_METRICS
        ]
        pd.DataFrame(rows).to_csv(output / "metrics" / "english_hubness_evidence.csv", index=False)
        primary_layers = [0, 1] if status == "NOT_SUPPORTED" else [0, 1, 2]
        eos_layers = [0, 1] if status == "REPRESENTATION_SENSITIVE" else [0, 1, 2]
        density_layers = [0, 1, 2] if density_layers is None else density_layers
        (output / "validation" / "validation_summary.json").write_text(json.dumps({
            "model_status": status,
            "joint_evidence": {
                "status": status,
                "primary_joint_layers": primary_layers,
                "primary_joint_longest_run": len(primary_layers),
                "eos_joint_layers": eos_layers,
                "eos_joint_longest_run": len(eos_layers),
                "density_joint_layers": density_layers,
                "density_joint_longest_run": len(density_layers),
                "min_consecutive_layers": 3,
            },
        }), encoding="utf-8")
        (output / "extraction_manifest.json").write_text(json.dumps({"layers": 3}), encoding="utf-8")
        config = {
            "experiment_name": f"experiment_{number}",
            "model": {"name_or_path": f"model/{number}"},
            "output_dir": str(output),
            "metrics": {"primary_representation": "mean_pool"},
        }
        path = root / f"model_{number}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path.name

    def test_two_robust_models_replicate_while_invalid_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = [
                self.write_model(root, 1),
                self.write_model(root, 2),
                self.write_model(root, 3, invalid_value=float("nan")),
            ]
            suite = root / "suite.json"
            suite.write_text(json.dumps({
                "configs": configs,
                "comparison_output_dir": str(root / "comparison"),
            }), encoding="utf-8")
            verdict = compare_suite(suite)
            self.assertEqual(verdict["replication_status"], "REPLICATED")
            self.assertEqual(verdict["valid_model_count"], 2)
            self.assertEqual(verdict["model_statuses"][2]["status"], "INVALID")
            summary = pd.read_csv(root / "comparison" / "model_comparison_summary.csv")
            self.assertNotIn("model/3", set(summary.model))

    def test_eos_sensitive_model_can_replicate_when_density_is_robust(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = [
                self.write_model(root, 1),
                self.write_model(root, 2, status="REPRESENTATION_SENSITIVE"),
            ]
            suite = root / "suite.json"
            suite.write_text(json.dumps({
                "configs": configs,
                "comparison_output_dir": str(root / "comparison"),
            }), encoding="utf-8")
            verdict = compare_suite(suite)
            self.assertEqual(verdict["replication_status"], "REPLICATED")
            self.assertEqual(
                verdict["model_statuses"][1]["source_validation_status"],
                "REPRESENTATION_SENSITIVE",
            )
            self.assertEqual(verdict["model_statuses"][1]["status"], "ROBUST")
            self.assertEqual(
                verdict["evaluation_policy"]["eos_role"],
                "diagnostic_only_not_a_cross_model_gate",
            )

    def test_primary_only_replication_is_reported_separately_from_density_robustness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = [
                self.write_model(root, 1, density_layers=[0, 1]),
                self.write_model(root, 2, density_layers=[]),
            ]
            suite = root / "suite.json"
            suite.write_text(json.dumps({
                "configs": configs,
                "comparison_output_dir": str(root / "comparison"),
            }), encoding="utf-8")
            verdict = compare_suite(suite)
            self.assertEqual(verdict["primary_only_replication_status"], "REPLICATED")
            self.assertEqual(verdict["replication_status"], "NOT_REPLICATED")
            self.assertEqual(
                [row["status"] for row in verdict["model_statuses"]],
                ["DENSITY_SENSITIVE", "DENSITY_SENSITIVE"],
            )

    def test_missing_validation_and_duplicate_grid_are_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            missing_validation = self.write_model(root, 1)
            (root / "output_1" / "validation" / "validation_summary.json").unlink()
            duplicate_grid = self.write_model(root, 2)
            evidence_path = root / "output_2" / "metrics" / "english_hubness_evidence.csv"
            frame = pd.read_csv(evidence_path)
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(evidence_path, index=False)
            suite = root / "suite.json"
            suite.write_text(json.dumps({
                "configs": [missing_validation, duplicate_grid],
                "comparison_output_dir": str(root / "comparison"),
            }), encoding="utf-8")
            verdict = compare_suite(suite)
            self.assertEqual([row["status"] for row in verdict["model_statuses"]], ["INVALID", "INVALID"])
            self.assertEqual(verdict["valid_model_count"], 0)


if __name__ == "__main__":
    unittest.main()
