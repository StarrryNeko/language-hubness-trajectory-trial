"""Export task-level competition predictors from frozen mean-pool hidden states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_association_v3.common import load_tasks, paths, read_checkpoint, settings, sha256_file, task_path, write_manifest
from behavior_v1.geometry import similarity_matrices
from common import load_config
from paper_common import group_vectors, load_hidden_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    dataset = load_hidden_dataset(cfg, representation="mean_pool")
    if dataset.languages != protocol["languages"]:
        raise ValueError("V3 predictor language order mismatch")
    tasks = load_tasks(cfg, "calibration") + load_tasks(cfg, "formal")
    grouped_tasks = {}
    for task in tasks:
        grouped_tasks.setdefault(str(task["semantic_id"]), []).append(task)
    dataset_positions = {str(value): index for index, value in enumerate(dataset.semantic_ids)}
    missing = set(grouped_tasks) - set(dataset_positions)
    if missing:
        raise ValueError(f"V3 tasks are absent from hidden states: {sorted(missing)[:5]}")
    semantic_ids = sorted(grouped_tasks, key=lambda value: dataset_positions[value])
    hidden_ids = [dataset.semantic_ids[dataset_positions[value]] for value in semantic_ids]
    lang_index = {lang: index for index, lang in enumerate(dataset.languages)}
    english = lang_index["en"]
    latin = [lang_index[x] for x in dataset.languages if protocol["metadata"][x]["script"] == "Latin" and x != "en"]
    non_latin = [lang_index[x] for x in dataset.languages if protocol["metadata"][x]["script"] != "Latin"]
    token_counts = {
        (str(row["id"]), str(row["lang"])): int(row.get("sentence_num_tokens", 0))
        for row in dataset.meta.to_dict("records")
    }
    records = []
    resources = cfg.get("structure_v2", {}).get("resources", cfg.get("behavior_v2", {}).get("resources", {}))
    for layer in protocol["analysis_layers"]:
        vectors = group_vectors(dataset, hidden_ids, layer, normalize=False)
        raw_matrices, local_matrices = similarity_matrices(vectors, protocol["local_scaling_k"], resources)
        for position, semantic_id in enumerate(semantic_ids):
            for task in grouped_tasks[semantic_id]:
                source = lang_index[task["source_lang"]]
                target = lang_index[task["target_lang"]]
                other_latin = [idx for idx in latin if idx != source]
                other_non_latin = [idx for idx in non_latin if idx not in {source, target}]
                record = {
                    "task_id": task["task_id"], "split": task["split"],
                    "semantic_id": semantic_id, "source_lang": task["source_lang"],
                    "target_lang": task["target_lang"], "layer": layer,
                    "source_token_count": token_counts[(str(hidden_ids[position]), task["source_lang"])],
                    "target_token_count": token_counts[(str(hidden_ids[position]), task["target_lang"])],
                }
                for scale, matrix in (("raw", raw_matrices[position]), ("local_scaled", local_matrices[position])):
                    latin_mean = float(matrix[source, other_latin].mean())
                    nonlatin_mean = float(matrix[source, other_non_latin].mean())
                    suffix = "" if scale == "raw" else "_local_scaled"
                    record[f"english_target_competition{suffix}"] = float(matrix[source, english] - matrix[source, target])
                    record[f"english_specific_advantage{suffix}"] = float(matrix[source, english] - latin_mean)
                    record[f"latin_attraction{suffix}"] = latin_mean - nonlatin_mean
                records.append(record)
    output = paths(cfg).measurement / "geometry_predictors.csv"
    pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8")
    write_manifest(paths(cfg).measurement / "predictor_manifest.json", {
        "config_path": str(config_path), "rows": len(records),
        "representation": "mean_pool", "representation_protocol": protocol["representation_protocol"],
        "layers": protocol["analysis_layers"], "primary_layer": protocol["primary_layer"],
        "calibration_task_sha256": sha256_file(task_path(cfg, "calibration")),
        "formal_task_sha256": sha256_file(task_path(cfg, "formal")),
        "checkpoint_sha256": read_checkpoint(cfg)["checkpoint_sha256"],
        "predictor_file_sha256": sha256_file(output),
        "primary_predictor": "english_target_competition",
        "activation_intervention": False,
    })
    print(f"Saved V3 predictors to {output}")


if __name__ == "__main__":
    main()
