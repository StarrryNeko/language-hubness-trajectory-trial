import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavior_common import ensure_behavior_dirs
from common import write_json, write_jsonl
from export_behavior_predictors import main as export_main
from prepare_behavior_tasks import build_tasks


class BehaviorExportIntegrationTests(unittest.TestCase):
    def test_padded_task_ids_export_against_csv_inferred_hidden_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            languages = [
                "en", "zh", "ar", "hi", "ru", "de", "es", "fr", "ja", "ko",
                "sw", "tr", "vi", "th", "id", "fi", "el", "ta", "te", "bn",
            ]
            cfg = {
                "seed": 7,
                "output_dir": str(output),
                "storage_dtype": "float32",
                "dataset": {
                    "minimum_languages_per_semantic_group": 20,
                    "languages": {language: language for language in languages},
                },
                "metrics": {"representations": ["mean_pool"], "primary_representation": "mean_pool"},
                "behavior_v1": {
                    "enabled": True,
                    "seed": 7,
                    "evaluation_languages": ["zh", "ar", "hi", "ru"],
                    "language_names": {language: language for language in languages},
                    "demonstration_semantic_ids": 2,
                    "evaluation_semantic_ids": 5,
                    "primary_layer": 2,
                    "analysis_layers": [1, 2, 3],
                    "local_scaling_k": 2,
                    "decoding": {"do_sample": False, "num_beams": 1},
                    "language_id": {
                        "reference_gate": "top1_label_accuracy",
                        "english_span_threshold_status": "frozen_before_behavior_results",
                        "calibration": {
                            "report_path": "calibration_lid_report.json",
                            "candidate_thresholds": [0.7],
                            "selection_rule": "highest_candidate_with_overall_accuracy_at_least_minimum",
                        },
                    },
                    "resources": {
                        "cpu_threads": 2, "evaluation_workers": 2,
                        "geometry_device": "cpu", "geometry_dtype": "float32",
                        "allow_tf32": False,
                    },
                },
            }
            config_path = root / "config.json"
            write_json(config_path, cfg)
            parallel_rows = [
                {"id": f"{semantic_id:05d}", "lang": language, "text": f"{language}-{semantic_id}"}
                for semantic_id in range(40, 47)
                for language in languages
            ]
            tasks, _, _, _ = build_tasks(cfg, parallel_rows)
            paths = ensure_behavior_dirs(cfg)
            write_jsonl(paths.data / "behavior_tasks.jsonl", tasks)

            metadata = []
            for row_idx, row in enumerate(parallel_rows):
                metadata.append({
                    "row_idx": row_idx, "id": int(row["id"]), "lang": row["lang"],
                    "was_truncated": False, "sentence_num_tokens": 4,
                })
            hidden = output / "hidden"
            hidden.mkdir(parents=True)
            pd.DataFrame(metadata).to_csv(hidden / "metadata.csv", index=False)
            rng = np.random.default_rng(5)
            vectors = rng.normal(size=(len(metadata), 4, 8)).astype(np.float32)
            np.save(hidden / "sentence_layer_mean_pool.npy", vectors)
            write_json(output / "checkpoint_identity.json", {
                "checkpoint_sha256": "synthetic-checkpoint"
            })

            with patch.object(sys, "argv", ["export_behavior_predictors.py", "--config", str(config_path)]):
                export_main()

            frame = pd.read_csv(paths.metrics / "behavior_geometry_predictors.csv")
            self.assertEqual(len(frame), len(tasks) * 3)
            self.assertEqual(frame.task_id.nunique(), len(tasks))
            self.assertFalse(frame.source_sentence_token_count.isna().any())
            manifest = json.loads(
                (paths.metrics / "behavior_geometry_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["rows"], len(tasks) * 3)


if __name__ == "__main__":
    unittest.main()
