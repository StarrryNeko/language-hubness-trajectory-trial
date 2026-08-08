"""Export full-language geometry and task predictors for behavior_v2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from behavior_v1.geometry import align_semantic_ids, similarity_matrices
from behavior_v2.common import (
    ensure_paths, load_tasks, read_checkpoint_identity, settings, sha256_file,
    stable_sha256, write_manifest,
)
from common import load_config
from paper_common import group_vectors, load_hidden_dataset, semantic_id_hash


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, mean - 1.96 * se, mean + 1.96 * se, se


def off_diagonal_mean(matrix, left, right, same_group=False):
    values = []
    for i in left:
        for j in right:
            if i == j:
                continue
            if same_group and j <= i:
                continue
            values.append(matrix[i, j])
    if not values:
        raise ValueError("empty script-comparison cell")
    return float(np.mean(values))


def k_occurrence(matrix, k):
    masked = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(masked, -np.inf)
    k = min(int(k), len(masked) - 1)
    neighbors = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    return np.bincount(neighbors.ravel(), minlength=len(masked)).astype(float)


def main():
    parser = argparse.ArgumentParser(description="Compute behavior_v2 language geometry")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    paths = ensure_paths(cfg)
    checkpoint = read_checkpoint_identity(cfg)
    tasks = load_tasks(cfg)
    dataset = load_hidden_dataset(cfg, representation="mean_pool")
    languages = dataset.languages
    language_index = {language: index for index, language in enumerate(languages)}
    if languages != protocol["languages"]:
        raise ValueError("hidden-state and V2 language orders differ")
    evaluation_ids, hidden_ids = align_semantic_ids(
        (task["semantic_id"] for task in tasks), dataset.semantic_ids
    )
    english = language_index["en"]
    latin = [language_index[value] for value in protocol["latin_languages"]]
    other_latin = [language_index[value] for value in protocol["other_latin_languages"]]
    non_latin = [language_index[value] for value in protocol["non_latin_languages"]]
    task_groups = {}
    for task in tasks:
        task_groups.setdefault(str(task["semantic_id"]), []).append(task)
    token_counts = {
        (str(row["id"]), str(row["lang"])): int(row.get("sentence_num_tokens", 0))
        for row in dataset.meta.to_dict("records")
    }
    predictor_records = []
    pair_records = []
    centrality_records = []
    script_records = []
    english_records = []
    centroid_records = []
    total_layers = int(dataset.vectors.shape[1])
    if min(protocol["analysis_layers"]) < 0 or max(protocol["analysis_layers"]) >= total_layers:
        raise ValueError(
            f"behavior_v2 analysis layers must be within 0..{total_layers - 1}"
        )
    for layer in protocol["analysis_layers"]:
        groups = group_vectors(dataset, hidden_ids, layer, normalize=False)
        cosine, local_scaled = similarity_matrices(
            groups, protocol["local_scaling_k"], protocol["resources"]
        )
        normalized_depth = layer / max(total_layers - 1, 1)
        mean_matrix = cosine.mean(axis=0)
        for left, left_language in enumerate(languages):
            for right, right_language in enumerate(languages):
                values = cosine[:, left, right]
                mean, low, high, se = mean_ci(values)
                pair_records.append({
                    "layer": layer,
                    "normalized_depth": normalized_depth,
                    "language_a": left_language,
                    "language_b": right_language,
                    "script_a": protocol["language_metadata"][left_language]["script"],
                    "script_b": protocol["language_metadata"][right_language]["script"],
                    "same_script": int(
                        protocol["language_metadata"][left_language]["script"]
                        == protocol["language_metadata"][right_language]["script"]
                    ),
                    "mean_cosine_similarity": mean,
                    "ci_lower": low,
                    "ci_upper": high,
                    "mean_cosine_distance": 1.0 - mean,
                    "distance_ci_lower": 1.0 - high,
                    "distance_ci_upper": 1.0 - low,
                    "standard_error": se,
                    "semantic_ids": len(evaluation_ids),
                })
        per_semantic_centrality = (
            (cosine.sum(axis=2) - 1.0) / (len(languages) - 1)
        )
        average_centrality = per_semantic_centrality.mean(axis=0)
        hubness_by_k = {
            k: np.stack([k_occurrence(matrix, k) for matrix in cosine])
            for k in protocol["k_sensitivity"]
        }
        for index, language in enumerate(languages):
            record = {
                "layer": layer,
                "normalized_depth": normalized_depth,
                "language": language,
                "script": protocol["language_metadata"][language]["script"],
                "family": protocol["language_metadata"][language]["family"],
                "mean_centrality": float(average_centrality[index]),
                "centrality_rank": int(
                    np.where(np.argsort(-average_centrality, kind="stable") == index)[0][0]
                ) + 1,
            }
            for k, values in hubness_by_k.items():
                record[f"k_occurrence_k{k}"] = float(values[:, index].mean())
            centrality_records.append(record)
        latin_within, nonlatin_within, between = [], [], []
        english_excess = []
        english_hubness_excess_by_k = {k: [] for k in protocol["k_sensitivity"]}
        for semantic_position, matrix in enumerate(cosine):
            latin_within.append(off_diagonal_mean(matrix, latin, latin, same_group=True))
            nonlatin_within.append(
                off_diagonal_mean(matrix, non_latin, non_latin, same_group=True)
            )
            between.append(off_diagonal_mean(matrix, latin, non_latin))
            semantic_centrality = per_semantic_centrality[semantic_position]
            english_excess.append(
                semantic_centrality[english] - semantic_centrality[other_latin].mean()
            )
            for k, values in hubness_by_k.items():
                semantic_hubness = values[semantic_position]
                english_hubness_excess_by_k[k].append(
                    semantic_hubness[english] - semantic_hubness[other_latin].mean()
                )
        for name, values in {
            "within_latin_similarity": latin_within,
            "within_non_latin_similarity": nonlatin_within,
            "latin_to_non_latin_similarity": between,
            "latin_concentration": np.asarray(latin_within) - np.asarray(between),
        }.items():
            mean, low, high, se = mean_ci(values)
            script_records.append({
                "layer": layer, "normalized_depth": normalized_depth,
                "metric": name, "mean": mean, "ci_lower": low,
                "ci_upper": high, "standard_error": se,
                "semantic_ids": len(evaluation_ids),
            })
        english_metrics = {
            "english_centrality_excess_over_other_latin": english_excess,
            **{
                f"english_hubness_excess_over_other_latin_k{k}": values
                for k, values in english_hubness_excess_by_k.items()
            },
        }
        for name, values in english_metrics.items():
            mean, low, high, se = mean_ci(values)
            english_records.append({
                "layer": layer, "normalized_depth": normalized_depth,
                "metric": name, "mean": mean, "ci_lower": low,
                "ci_upper": high, "standard_error": se,
                "english_rank_within_latin": int(
                    np.where(
                        np.argsort(-average_centrality[latin], kind="stable")
                        == latin.index(english)
                    )[0][0]
                ) + 1,
                "exact_rank_p": (
                    1 + int(np.sum(average_centrality[other_latin] >= average_centrality[english]))
                ) / len(latin),
                "semantic_ids": len(evaluation_ids),
            })
        semantic_centered = groups - groups.mean(axis=1, keepdims=True)
        language_centroids = semantic_centered.mean(axis=0)
        coordinates = PCA(n_components=2, svd_solver="full").fit_transform(language_centroids)
        for index, language in enumerate(languages):
            centroid_records.append({
                "layer": layer, "normalized_depth": normalized_depth,
                "language": language,
                "script": protocol["language_metadata"][language]["script"],
                "pc1": float(coordinates[index, 0]),
                "pc2": float(coordinates[index, 1]),
                "centroid_norm": float(np.linalg.norm(language_centroids[index])),
            })
        for semantic_position, semantic_id in enumerate(evaluation_ids):
            raw = cosine[semantic_position]
            scaled = local_scaled[semantic_position]
            primary_occurrence = hubness_by_k[protocol["local_scaling_k"]][semantic_position]
            hidden_id = hidden_ids[semantic_position]
            for task in task_groups[semantic_id]:
                source = language_index[task["source_lang"]]
                target = language_index[task["target_lang"]]
                latin_candidates = [index for index in other_latin if index != source]
                nonlatin_candidates = [index for index in non_latin if index != source]
                raw_other_latin = float(raw[source, latin_candidates].mean())
                raw_non_latin = float(raw[source, nonlatin_candidates].mean())
                scaled_other_latin = float(scaled[source, latin_candidates].mean())
                scaled_non_latin = float(scaled[source, nonlatin_candidates].mean())
                predictor_records.append({
                    "task_id": task["task_id"],
                    "semantic_id": semantic_id,
                    "condition": task["condition"],
                    "source_lang": task["source_lang"],
                    "target_lang": task["target_lang"],
                    "layer": layer,
                    "normalized_depth": normalized_depth,
                    "source_target_cosine": float(raw[source, target]),
                    "source_english_cosine": float(raw[source, english]),
                    "source_other_latin_mean_cosine": raw_other_latin,
                    "source_non_latin_mean_cosine": raw_non_latin,
                    "latin_attraction": raw_other_latin - raw_non_latin,
                    "english_specific_advantage": float(raw[source, english]) - raw_other_latin,
                    "english_minus_target_cosine": float(raw[source, english] - raw[source, target]),
                    "source_target_local_scaled": float(scaled[source, target]),
                    "latin_attraction_local_scaled": scaled_other_latin - scaled_non_latin,
                    "english_specific_advantage_local_scaled": (
                        float(scaled[source, english]) - scaled_other_latin
                    ),
                    "english_k_occurrence": float(primary_occurrence[english]),
                    "source_sentence_token_count": token_counts[(hidden_id, task["source_lang"])],
                    "target_sentence_token_count": token_counts[(hidden_id, task["target_lang"])],
                })
    outputs = {
        "language_pair_similarity.csv": pd.DataFrame(pair_records),
        "language_centrality.csv": pd.DataFrame(centrality_records),
        "script_concentration.csv": pd.DataFrame(script_records),
        "english_advantage.csv": pd.DataFrame(english_records),
        "language_centroid_pca.csv": pd.DataFrame(centroid_records),
        "behavior_v2_geometry_predictors.csv": pd.DataFrame(predictor_records),
    }
    for name, frame in outputs.items():
        frame.to_csv(paths.geometry / name, index=False, encoding="utf-8")
    predictor_path = paths.geometry / "behavior_v2_geometry_predictors.csv"
    write_manifest(paths.geometry / "geometry_manifest.json", {
        "config_path": str(config_path),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "task_file_sha256": sha256_file(paths.data / "behavior_v2_tasks.jsonl"),
        "representation": "mean_pool",
        "representation_protocol": protocol["representation_protocol"],
        "languages": languages,
        "language_order_sha256": stable_sha256(languages),
        "semantic_ids": len(evaluation_ids),
        "hidden_semantic_id_sha256": semantic_id_hash(hidden_ids),
        "evaluation_semantic_id_sha256": stable_sha256(evaluation_ids),
        "layers": protocol["analysis_layers"],
        "primary_layer": protocol["primary_layer"],
        "local_scaling_k": protocol["local_scaling_k"],
        "k_sensitivity": protocol["k_sensitivity"],
        "predictor_file_sha256": sha256_file(predictor_path),
        "semantic_centering_scope": "visualization-only language centroids",
        "candidate_scope": "same semantic ID only",
        "activation_intervention": False,
    })
    print(f"Saved behavior_v2 geometry to {paths.geometry}")


if __name__ == "__main__":
    main()
