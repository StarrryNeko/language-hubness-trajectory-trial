import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_rules import (
    REQUIRED_EVIDENCE_METRICS,
    classify_model_status,
    joint_positive_layers,
    max_consecutive_layers,
    validate_evidence_grid,
)


def evidence_frame(layers, positive_by_metric=None):
    positive_by_metric = positive_by_metric or {
        metric: set(layers) for metric in REQUIRED_EVIDENCE_METRICS
    }
    return pd.DataFrame([
        {
            "layer": layer,
            "metric": metric,
            "mean": 0.2,
            "ci_lower": 0.1 if layer in positive_by_metric.get(metric, set()) else -0.1,
            "ci_upper": 0.3,
        }
        for layer in layers
        for metric in REQUIRED_EVIDENCE_METRICS
    ])


class EvidenceRuleTests(unittest.TestCase):
    def test_all_metrics_must_be_positive_on_the_same_layer(self):
        positive = {
            metric: {index} for index, metric in enumerate(REQUIRED_EVIDENCE_METRICS)
        }
        self.assertEqual(joint_positive_layers(evidence_frame(range(4), positive)), [])
        self.assertEqual(joint_positive_layers(evidence_frame(range(4))), [0, 1, 2, 3])

    def test_duplicate_missing_and_nonfinite_records_are_invalid(self):
        frame = evidence_frame(range(2))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_evidence_grid(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_evidence_grid(frame.iloc[:-1])
        frame.loc[0, "ci_lower"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_evidence_grid(frame)

    def test_missing_real_layer_breaks_a_run(self):
        self.assertEqual(max_consecutive_layers([0, 1, 3, 4]), 2)

    def test_model_status_requires_primary_breadth_and_both_controls(self):
        self.assertEqual(classify_model_status([0, 1], [0, 1], [0, 1], [0, 1], 3)["status"], "NOT_SUPPORTED")
        self.assertEqual(classify_model_status([0, 1, 2], [0, 1, 2], [0, 1], [0, 1, 2], 3)["status"], "REPRESENTATION_SENSITIVE")
        result = classify_model_status([0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 3, 4], 3)
        self.assertEqual(result["status"], "ROBUST")
        self.assertEqual(result["primary_joint_longest_run"], 3)


if __name__ == "__main__":
    unittest.main()
