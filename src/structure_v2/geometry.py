"""Compute semantic-ID bootstrap geometry without behavior-task dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from behavior_v1.geometry import similarity_matrices
from behavior_v2.geometry import k_occurrence, off_diagonal_mean
from common import load_config
from paper_common import group_vectors, load_hidden_dataset, semantic_id_hash
from structure_v2.common import paths, settings, sha256_file, stable_sha256, write_manifest


def bootstrap_interval(values, seed, n_boot, confidence):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("cannot bootstrap an empty statistic")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(n_boot), dtype=np.float64)
    chunk = 100
    for start in range(0, int(n_boot), chunk):
        count = min(chunk, int(n_boot) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start:start + count] = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(values.mean()), float(np.quantile(estimates, alpha)), float(
        np.quantile(estimates, 1.0 - alpha)
    )


def bootstrap_matrix(values, seed, n_boot, confidence):
    """Bootstrap the semantic axis in bounded chunks for a full 24x24 matrix."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty((int(n_boot), *values.shape[1:]), dtype=np.float32)
    chunk = 25
    for start in range(0, int(n_boot), chunk):
        count = min(chunk, int(n_boot) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start:start + count] = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        values.mean(axis=0), np.quantile(estimates, alpha, axis=0),
        np.quantile(estimates, 1.0 - alpha, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    destination = paths(cfg)
    dataset = load_hidden_dataset(cfg, representation="mean_pool")
    if dataset.languages != protocol["languages"]:
        raise ValueError("hidden-state language order differs from structure_v2 config")
    language_index = {lang: index for index, lang in enumerate(dataset.languages)}
    latin = [language_index[x] for x in protocol["latin"]]
    other_latin = [language_index[x] for x in protocol["other_latin"]]
    non_latin = [language_index[x] for x in protocol["non_latin"]]
    english = language_index["en"]
    records, pair_records, centrality_records, centroid_records = [], [], [], []
    total_layers = int(dataset.vectors.shape[1])
    for layer in protocol["analysis_layers"]:
        groups = group_vectors(dataset, dataset.semantic_ids, layer, normalize=False)
        cosine, local = similarity_matrices(groups, protocol["local_scaling_k"], protocol["resources"])
        centrality = (cosine.sum(axis=2) - 1.0) / (len(dataset.languages) - 1)
        pair_mean, pair_low, pair_high = bootstrap_matrix(
            cosine, protocol["seed"] + layer * 1000,
            protocol["bootstrap_samples"], protocol["confidence_level"],
        )
        for left, language_a in enumerate(dataset.languages):
            for right, language_b in enumerate(dataset.languages):
                pair_records.append({
                    "layer": layer, "language_a": language_a, "language_b": language_b,
                    "script_a": protocol["metadata"][language_a]["script"],
                    "script_b": protocol["metadata"][language_b]["script"],
                    "mean_cosine_similarity": float(pair_mean[left, right]),
                    "ci_lower": float(pair_low[left, right]),
                    "ci_upper": float(pair_high[left, right]),
                    "mean_cosine_distance": float(1.0 - pair_mean[left, right]),
                    "semantic_ids": len(cosine),
                })
        for index, language in enumerate(dataset.languages):
            mean, low, high = bootstrap_interval(
                centrality[:, index], protocol["seed"] + layer * 100 + index,
                protocol["bootstrap_samples"], protocol["confidence_level"],
            )
            centrality_records.append({
                "layer": layer, "language": language,
                "script": protocol["metadata"][language]["script"],
                "family": protocol["metadata"][language]["family"],
                "mean_centrality": mean, "ci_lower": low, "ci_upper": high,
                "centrality_rank": int(np.where(np.argsort(-centrality.mean(axis=0), kind="stable") == index)[0][0]) + 1,
                "semantic_ids": len(cosine),
            })
        semantic_centered = groups - groups.mean(axis=1, keepdims=True)
        language_centroids = semantic_centered.mean(axis=0)
        coordinates = PCA(n_components=2, svd_solver="full").fit_transform(language_centroids)
        for index, language in enumerate(dataset.languages):
            centroid_records.append({
                "layer": layer, "language": language,
                "script": protocol["metadata"][language]["script"],
                "pc1": float(coordinates[index, 0]), "pc2": float(coordinates[index, 1]),
                "centroid_norm": float(np.linalg.norm(language_centroids[index])),
                "interpretation": "visualization_only",
            })
        raw_hub = {
            k: np.stack([k_occurrence(matrix, k) for matrix in cosine])
            for k in protocol["k_sensitivity"]
        }
        local_hub = {
            k: np.stack([k_occurrence(matrix, k) for matrix in local])
            for k in protocol["k_sensitivity"]
        }
        metrics = {
            "latin_concentration": np.asarray([
                off_diagonal_mean(m, latin, latin, True) - off_diagonal_mean(m, latin, non_latin)
                for m in cosine
            ]),
            "english_centrality_excess": centrality[:, english] - centrality[:, other_latin].mean(axis=1),
        }
        for k in protocol["k_sensitivity"]:
            metrics[f"english_hubness_excess_raw_k{k}"] = (
                raw_hub[k][:, english] - raw_hub[k][:, other_latin].mean(axis=1)
            )
            metrics[f"english_hubness_excess_local_scaled_k{k}"] = (
                local_hub[k][:, english] - local_hub[k][:, other_latin].mean(axis=1)
            )
        for offset, (name, values) in enumerate(metrics.items()):
            mean, low, high = bootstrap_interval(
                values, protocol["seed"] + layer * 100 + offset,
                protocol["bootstrap_samples"], protocol["confidence_level"],
            )
            records.append({
                "layer": layer,
                "normalized_depth": layer / max(total_layers - 1, 1),
                "metric": name,
                "mean": mean,
                "ci_lower": low,
                "ci_upper": high,
                "semantic_ids": len(values),
                "direction_positive_rate": float(np.mean(values > 0)),
                "is_primary_layer": layer == protocol["primary_layer"],
            })
    outputs = {
        "confirmatory_geometry.csv": pd.DataFrame(records),
        "language_pair_similarity.csv": pd.DataFrame(pair_records),
        "language_centrality.csv": pd.DataFrame(centrality_records),
        "language_centroid_pca.csv": pd.DataFrame(centroid_records),
    }
    for name, frame in outputs.items():
        frame.to_csv(destination.geometry / name, index=False, encoding="utf-8")
    output = destination.geometry / "confirmatory_geometry.csv"
    checkpoint_path = Path(cfg["output_dir"]) / "checkpoint_identity.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    write_manifest(destination.geometry / "geometry_manifest.json", {
        "config_path": str(config_path),
        "representation": "mean_pool",
        "representation_protocol": protocol["representation_protocol"],
        "candidate_scope": "same semantic ID only",
        "semantic_ids": len(dataset.semantic_ids),
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "language_order_sha256": stable_sha256(dataset.languages),
        "layers": protocol["analysis_layers"],
        "primary_layer": protocol["primary_layer"],
        "k_sensitivity": protocol["k_sensitivity"],
        "bootstrap_unit": "semantic_id",
        "bootstrap_samples": protocol["bootstrap_samples"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "output_sha256": {
            name: sha256_file(destination.geometry / name) for name in outputs
        },
        "result_sha256": sha256_file(output),
        "activation_intervention": False,
    })
    print(f"Saved structure_v2 geometry to {output}")


if __name__ == "__main__":
    main()
