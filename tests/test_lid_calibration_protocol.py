import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "xglm_1b7_behavior_v1.json"
BUILD = ROOT / "scripts" / "build_lid_calibration_tasks.py"
DIAGNOSE = ROOT / "scripts" / "diagnose_reference_lid.py"
LANGUAGES = ["en", "zh", "ar", "hi", "es", "ru", "sw", "tr", "ja"]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class LidCalibrationProtocolTests(unittest.TestCase):
    def build_fixture(self, folder, formal_ids):
        root = Path(folder)
        exclusion = root / "exclusions.json"
        formal = root / "formal_manifest.json"
        source = root / "parallel_samples.jsonl"
        output = root / "calibration.jsonl"
        write_json(exclusion, {
            "union_count": 2,
            "selected_semantic_indices": [1, 2],
        })
        write_json(formal, {"selected_semantic_indices": formal_ids})
        write_jsonl(source, [
            {"id": str(semantic_id), "lang": language, "text": f"{language}-{semantic_id}"}
            for semantic_id in (1, 2)
            for language in LANGUAGES
        ])
        return exclusion, formal, source, output

    def test_builder_writes_hashed_zero_overlap_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            exclusion, formal, source, output = self.build_fixture(folder, [3, 4])
            subprocess.run([
                sys.executable, str(BUILD), "--config", str(CONFIG),
                "--parallel-samples", str(source), "--exclusions", str(exclusion),
                "--formal-data-manifest", str(formal), "--output", str(output),
                "--min-complete-groups", "1",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads(
                Path(f"{output}.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["protocol_version"], "behavior_v1_lid_calibration_v1")
            self.assertEqual(manifest["calibration_semantic_id_count"], 2)
            self.assertEqual(manifest["calibration_formal_overlap_count"], 0)
            self.assertEqual(manifest["rows"], 18)
            self.assertEqual(set(manifest["rows_per_language"].values()), {2})
            self.assertEqual(len(manifest["task_file_sha256"]), 64)

    def test_builder_rejects_formal_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            exclusion, formal, source, output = self.build_fixture(folder, [2, 3])
            result = subprocess.run([
                sys.executable, str(BUILD), "--config", str(CONFIG),
                "--parallel-samples", str(source), "--exclusions", str(exclusion),
                "--formal-data-manifest", str(formal), "--output", str(output),
                "--min-complete-groups", "1",
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("overlap", result.stderr)

    def test_diagnostic_modes_guard_threshold_search_inputs(self):
        calibration = subprocess.run([
            sys.executable, str(DIAGNOSE), "--mode", "calibration",
            "--config", str(CONFIG), "--task-file", "missing.jsonl",
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(calibration.returncode, 0)
        self.assertIn("requires --task-file and --calibration-manifest", calibration.stderr)

        formal = subprocess.run([
            sys.executable, str(DIAGNOSE), "--mode", "formal-audit",
            "--config", str(CONFIG), "--task-file", "forbidden.jsonl",
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(formal.returncode, 0)
        self.assertIn("reads only the frozen behavior task file", formal.stderr)


if __name__ == "__main__":
    unittest.main()
