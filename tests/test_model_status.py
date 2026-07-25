import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_rules import MODEL_STATUSES, classify_model_status, validate_model_status_payload


class ModelStatusTests(unittest.TestCase):
    def test_status_vocabulary_contains_exactly_four_states(self):
        self.assertEqual(MODEL_STATUSES, {
            "INVALID", "NOT_SUPPORTED", "DENSITY_SENSITIVE", "ROBUST",
        })

    def test_primary_requires_breadth_on_the_same_layers(self):
        result = classify_model_status(
            primary_layers=[0, 1, 2],
            breadth_layers=[3, 4, 5],
            density_layers=[0, 1, 2],
            min_run=3,
        )
        self.assertEqual(result["status"], "NOT_SUPPORTED")

    def test_density_must_overlap_primary_on_the_same_layers(self):
        result = classify_model_status(
            primary_layers=[0, 1, 2],
            breadth_layers=[0, 1, 2],
            density_layers=[1, 2, 3],
            min_run=3,
        )
        self.assertEqual(result["status"], "DENSITY_SENSITIVE")

    def test_status_payload_cannot_claim_robust_with_short_control_run(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_model_status_payload({
                "model_status": "ROBUST",
                "joint_evidence": {
                    "status": "ROBUST",
                    "primary_joint_longest_run": 3,
                    "density_joint_longest_run": 3,
                    "primary_density_overlap_longest_run": 2,
                    "min_consecutive_layers": 3,
                },
            })


if __name__ == "__main__":
    unittest.main()
