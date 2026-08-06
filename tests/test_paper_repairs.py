import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_checkpoint_identity import checkpoint_inventory
from common import json_dumps_strict, write_json
from repair_paper_outputs import strict_json_audit
from validate_paper import classify_paper_claim


class PaperRepairTests(unittest.TestCase):
    def test_negative_hubness_cannot_be_promoted_by_missing_competition_support(self):
        overall, claim = classify_paper_claim(
            blockers=[],
            alignment_status="SUPPORTED",
            english_specificity_status="SELECTION_CORRECTED",
            competition_status="NOT_SUPPORTED",
            hubness_status="NOT_SUPPORTED",
            sample_status="REPLICATED",
        )
        self.assertEqual(overall, "READY_FOR_MODEL_COMPARISON")
        self.assertEqual(claim, "ENGLISH_TARGET_BIAS_WITHOUT_ROBUST_HUBNESS")
        self.assertNotIn("ENGLISH_HUB_", claim)

    def test_density_sensitive_status_precedes_competition_claim(self):
        _, claim = classify_paper_claim(
            [], "SUPPORTED", "SELECTION_CORRECTED", "NOT_SUPPORTED",
            "DENSITY_SENSITIVE", "REPLICATED",
        )
        self.assertEqual(claim, "DENSITY_SENSITIVE_ENGLISH_PATTERN")

    def test_strict_json_converts_nested_nonfinite_values_to_null(self):
        encoded = json_dumps_strict({"a": math.nan, "b": [np.float32(np.inf)]})
        self.assertEqual(json.loads(encoded), {"a": None, "b": [None]})
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_strict_json_audit_rejects_legacy_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "legacy.json").write_text('{"value": NaN}', encoding="utf-8")
            self.assertEqual(len(strict_json_audit(root)), 1)
            write_json(root / "legacy.json", {"value": math.nan})
            self.assertEqual(strict_json_audit(root), [])

    def test_checkpoint_inventory_hash_changes_with_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weight = root / "model.safetensors"
            weight.write_bytes(b"first")
            first = checkpoint_inventory(root)
            weight.write_bytes(b"second")
            second = checkpoint_inventory(root)
            self.assertNotEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
            self.assertEqual(second["weight_file_count"], 1)


if __name__ == "__main__":
    unittest.main()
