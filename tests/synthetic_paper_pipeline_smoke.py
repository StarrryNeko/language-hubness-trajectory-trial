"""Manual end-to-end smoke test for every offline paper_v1 module."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_config


def main():
    base = load_config(ROOT / "configs" / "base_24lang_same_semantics.json")
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "synthetic_output"
        hidden = output / "hidden"
        hidden.mkdir(parents=True)
        languages = list(base["dataset"]["languages"])
        semantic_ids = [str(index) for index in range(24)]
        rng = np.random.default_rng(19)
        rows = []
        vectors = []
        language_offsets = rng.normal(scale=0.08, size=(len(languages), 8))
        for semantic_index, semantic_id in enumerate(semantic_ids):
            semantic = rng.normal(size=8)
            for language_index, lang in enumerate(languages):
                rows.append({
                    "row_idx": len(rows), "id": semantic_id, "lang": lang,
                    "was_truncated": False, "text": f"{lang}-{semantic_id}",
                })
                layers = []
                for layer in range(4):
                    layers.append(
                        semantic + layer * language_offsets[language_index]
                        + rng.normal(scale=0.02, size=8)
                    )
                vectors.append(layers)
        pd.DataFrame(rows).to_csv(hidden / "metadata.csv", index=False)
        np.save(hidden / "sentence_layer_mean_pool.npy", np.asarray(vectors, dtype=np.float32))
        cfg = base
        cfg["output_dir"] = str(output)
        cfg["metrics"]["bootstrap_samples"] = 5
        cfg["metrics"]["min_consecutive_layers"] = 2
        cfg["paper_v1"].update({
            "bootstrap_samples": 5,
            "shuffled_permutations": 2,
            "label_permutations": 5,
            "probe_permutations": 1,
            "sample_robustness_subsets": 2,
            "sample_robustness_size": 20,
            "language_structure_k": 3,
            "similarity_block_size": 64,
        })
        config_path = Path(folder) / "synthetic.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        subprocess.run([
            sys.executable, str(ROOT / "src" / "run_paper_analysis.py"),
            "--config", str(config_path),
        ], check=True)
        summary = json.loads(
            (output / "paper_v1" / "validation" / "paper_validation_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if summary["overall_assessment"] == "NEEDS_REVISION":
            raise AssertionError(summary)
        print("Synthetic paper_v1 pipeline smoke test passed")


if __name__ == "__main__":
    main()
