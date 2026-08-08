"""Create descriptive behavior_v2 geometry figures (never inferential evidence)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from behavior_v2.common import ensure_paths, settings, sha256_file, write_manifest
from common import load_config


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot descriptive behavior_v2 geometry")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    paths = ensure_paths(cfg)
    sns.set_theme(style="whitegrid", context="notebook")

    pair_path = paths.geometry / "language_pair_similarity.csv"
    pca_path = paths.geometry / "language_centroid_pca.csv"
    script_path = paths.geometry / "script_concentration.csv"
    english_path = paths.geometry / "english_advantage.csv"
    pair = pd.read_csv(pair_path)
    pca = pd.read_csv(pca_path)
    script = pd.read_csv(script_path)
    english = pd.read_csv(english_path)
    outputs = []

    for layer in protocol["analysis_layers"]:
        layer_pairs = pair.loc[pair.layer == layer]
        matrix = layer_pairs.pivot(
            index="language_a", columns="language_b", values="mean_cosine_distance"
        ).loc[protocol["languages"], protocol["languages"]]
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, cmap="mako", square=True, cbar_kws={"label": "cosine distance"})
        plt.title(f"All-language mean-pool distance, layer {layer}")
        destination = paths.figures / f"language_distance_heatmap_layer_{layer}.png"
        save_figure(destination)
        outputs.append(destination)

        layer_pca = pca.loc[pca.layer == layer].copy()
        plt.figure(figsize=(10, 8))
        for _, row in layer_pca.iterrows():
            is_english = row.language == "en"
            color = "#c0392b" if is_english else (
                "#2980b9" if row.script == "Latin" else "#f39c12"
            )
            plt.scatter(
                row.pc1, row.pc2, color=color, s=150 if is_english else 55,
                marker="*" if is_english else "o", zorder=3,
            )
            plt.annotate(str(row.language), (row.pc1, row.pc2), xytext=(4, 3),
                         textcoords="offset points", fontsize=8)
        plt.axhline(0, color="0.8", linewidth=0.8)
        plt.axvline(0, color="0.8", linewidth=0.8)
        plt.xlabel("PC1 (visualization only)")
        plt.ylabel("PC2 (visualization only)")
        plt.title(f"Semantic-centered language centroids, layer {layer}")
        destination = paths.figures / f"language_centroid_pca_layer_{layer}.png"
        save_figure(destination)
        outputs.append(destination)

    plt.figure(figsize=(9, 6))
    for metric, group in script.groupby("metric", sort=True):
        plt.plot(group.normalized_depth, group["mean"], marker="o", label=metric)
        plt.fill_between(
            group.normalized_depth.to_numpy(dtype=float),
            group.ci_lower.to_numpy(dtype=float),
            group.ci_upper.to_numpy(dtype=float), alpha=0.15
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("normalized layer depth")
    plt.ylabel("cosine similarity contrast")
    plt.title("Latin/non-Latin geometry trajectory")
    plt.legend(fontsize=8)
    destination = paths.figures / "script_concentration_trajectory.png"
    save_figure(destination)
    outputs.append(destination)

    metrics = list(english.metric.drop_duplicates())
    figure, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.5 * len(metrics)), squeeze=False)
    for axis, metric in zip(axes[:, 0], metrics):
        group = english.loc[english.metric == metric]
        x = group.normalized_depth.to_numpy(dtype=float)
        axis.plot(x, group["mean"].to_numpy(dtype=float), marker="o")
        axis.fill_between(
            x, group.ci_lower.to_numpy(dtype=float),
            group.ci_upper.to_numpy(dtype=float), alpha=0.15
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel(metric)
    axes[-1, 0].set_xlabel("normalized layer depth")
    figure.suptitle("English-specific geometry trajectory")
    destination = paths.figures / "english_advantage_trajectory.png"
    save_figure(destination)
    outputs.append(destination)

    write_manifest(paths.figures / "figure_manifest.json", {
        "config_path": str(config_path),
        "source_files": {
            path.name: sha256_file(path)
            for path in (pair_path, pca_path, script_path, english_path)
        },
        "figures": [path.name for path in outputs],
        "interpretation": "descriptive visualization only; PCA is not confirmatory evidence",
        "representation": "mean_pool",
        "activation_intervention": False,
    })
    print(f"Saved {len(outputs)} behavior_v2 figures to {paths.figures}")


if __name__ == "__main__":
    main()
