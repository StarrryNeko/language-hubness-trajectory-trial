"""End-to-end smoke test for metrics, plots, and validation without downloading a model."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    languages = [
        "en", "zh", "de", "ar", "hi", "es", "fr", "ru", "ja", "ko", "sw", "tr",
        "vi", "th", "id", "fi", "el", "ta", "te", "bn", "ur", "bg", "it", "pt",
    ]
    language_map = {language: f"{language}_Test" for language in languages}
    config = {
        "experiment_name": "synthetic_smoke",
        "model": {"name_or_path": "synthetic/model"},
        "output_dir": str(output),
        "seed": 4,
        "dataset": {
            "source": "local_jsonl",
            "sample_size_per_language": 8,
            "minimum_languages_per_semantic_group": 20,
            "languages": language_map,
            "language_metadata": {
                lang: {"family": "test_family", "script": "test_script"} for lang in languages
            },
        },
        "metrics": {
            "english_language": "en",
            "nearest_neighbors_k": 3,
            "representations": ["mean_pool", "sentinel_eos"],
            "primary_representation": "mean_pool",
            "validation_representation": "sentinel_eos",
            "bootstrap_samples": 30,
            "confidence_level": 0.95,
        },
        "similarity_controls": {
            "methods": ["cosine", "local_scaled_cosine"],
            "local_scaling_k": 3,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for folder in ["data", "hidden", "metrics", "figures", "validation"]:
        (output / folder).mkdir(exist_ok=True)
    config_path = output / "synthetic_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rng = np.random.default_rng(13)
    rows = []
    mean_vectors = []
    eos_vectors = []
    layers, dimension = 4, 48
    for semantic_number in range(8):
        semantic_id = f"s{semantic_number:03d}"
        semantic = rng.normal(size=dimension)
        semantic /= np.linalg.norm(semantic)
        for language_index, language in enumerate(languages):
            row_idx = len(rows)
            rows.append({
                "row_idx": row_idx,
                "id": semantic_id,
                "lang": language,
                "text": f"synthetic parallel sentence {semantic_id} {language}",
                "was_truncated": False,
                "sentinel_eos_token_id": 2,
            })
            layer_vectors = []
            eos_layer_vectors = []
            language_offset = rng.normal(scale=0.35, size=dimension)
            if language == "en":
                language_offset *= 0.05
            for layer in range(layers):
                strength = 0.2 + 0.2 * layer
                vector = semantic + strength * language_offset
                layer_vectors.append(vector)
                eos_layer_vectors.append(vector + rng.normal(scale=0.02, size=dimension))
            mean_vectors.append(layer_vectors)
            eos_vectors.append(eos_layer_vectors)
    pd.DataFrame(rows).to_csv(output / "hidden" / "metadata.csv", index=False)
    np.save(output / "hidden" / "sentence_layer_mean_pool.npy", np.asarray(mean_vectors, dtype=np.float32))
    np.save(output / "hidden" / "sentence_layer_sentinel_eos.npy", np.asarray(eos_vectors, dtype=np.float32))
    (output / "data" / "dataset_manifest.json").write_text(json.dumps({
        "semantic_groups": 8,
        "languages_per_semantic_group": 24,
        "complete_parallel_groups": True,
        "candidate_scope": "same_semantic_id_only",
    }), encoding="utf-8")
    (output / "extraction_manifest.json").write_text(json.dumps({
        "layers": layers,
        "storage_dtype": "float32",
        "representations": ["mean_pool", "sentinel_eos"],
    }), encoding="utf-8")

    for script in ["compute_metrics.py", "plot_trajectories.py", "run_validations.py"]:
        subprocess.run([
            sys.executable, str(project / "src" / script), "--config", str(config_path)
        ], cwd=project, check=True, env=os.environ.copy())
    suite_configs = []
    for model_number in range(3):
        model_config = dict(config)
        model_config["experiment_name"] = f"synthetic_model_{model_number}"
        model_config["model"] = {"name_or_path": f"synthetic/model-{model_number}"}
        model_config_path = output / f"model_{model_number}.json"
        model_config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
        suite_configs.append(model_config_path.name)
    suite_path = output / "suite.json"
    suite_path.write_text(json.dumps({
        "configs": suite_configs,
        "comparison_output_dir": str(output / "model_comparison"),
    }, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable, str(project / "src" / "compare_models.py"), "--suite", str(suite_path)
    ], cwd=project, check=True, env=os.environ.copy())
    expected = [
        output / "metrics" / "english_hubness_evidence.csv",
        output / "metrics" / "metrics_manifest.json",
        output / "figures" / "english_hubness_evidence.png",
        output / "validation" / "validation_summary.md",
        output / "model_comparison" / "model_comparison_verdict.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Smoke pipeline missed outputs: {missing}")
    print("SYNTHETIC_PIPELINE_OK")


if __name__ == "__main__":
    main()
