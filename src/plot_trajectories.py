import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ensure_dirs, load_config


def save_lineplot(df, y, title, ylabel, out_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="layer", y=y, hue="lang", marker="o")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)

    summary = pd.read_csv(Path(paths["metrics"]) / "layer_summary.csv")
    centrality = pd.read_csv(Path(paths["metrics"]) / "language_centrality.csv")
    shapes = pd.read_csv(Path(paths["metrics"]) / "trajectory_shapes.csv")

    sns.set_theme(style="whitegrid")

    save_lineplot(
        summary,
        "mean_drift_en",
        "English Drift Across Layers",
        "sim(x, English centroid) - sim(x, own-language centroid)",
        Path(paths["figures"]) / "english_drift_by_layer.png",
    )

    save_lineplot(
        summary,
        "mean_english_hub_attraction",
        "English Hub Attraction Across Layers",
        "Proportion of English top-k neighbors",
        Path(paths["figures"]) / "english_hub_attraction_by_layer.png",
    )

    save_lineplot(
        centrality,
        "centrality_rate",
        "Language Centrality Across Layers",
        "Share of all top-k neighbor slots",
        Path(paths["figures"]) / "language_centrality_by_layer.png",
    )

    shape_counts = shapes.groupby(["lang", "drift_shape"]).size().reset_index(name="count")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=shape_counts, x="lang", y="count", hue="drift_shape")
    plt.title("Drift Trajectory Shape Distribution")
    plt.xlabel("Language")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(Path(paths["figures"]) / "drift_shape_distribution.png", dpi=180)
    plt.close()

    print(f"Saved figures to {paths['figures']}")


if __name__ == "__main__":
    main()

