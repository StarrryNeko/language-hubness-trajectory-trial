import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ensure_dirs, load_config


def lineplot(data, y, hue, title, ylabel, output, baseline=None, style=None):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="layer", y=y, hue=hue, style=style, errorbar=None)
    if baseline is not None:
        plt.axhline(baseline, color="black", linestyle="--", linewidth=1, label="uniform baseline")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    metrics = Path(paths["metrics"])
    figures = Path(paths["figures"])
    primary = cfg["metrics"].get("primary_representation", "last_token")
    english = cfg["metrics"].get("english_language", "en")
    sns.set_theme(style="whitegrid")

    alignment = pd.read_csv(metrics / "alignment_gain.csv")
    alignment["pair"] = alignment["lang_a"] + "-" + alignment["lang_b"]
    generated = []

    path = figures / "alignment_gain_by_layer.png"
    lineplot(alignment[alignment.representation == primary], "alignment_gain", "pair",
             "Cross-lingual Semantic Alignment", "Parallel cosine - shuffled cosine",
             path)
    generated.append(path)

    specificity = pd.read_csv(metrics / "anchor_specificity.csv")
    spec_plot = specificity[specificity.representation == primary].groupby(
        ["layer", "anchor_lang"], as_index=False
    )["mean_specificity"].mean()
    path = figures / "anchor_specificity_by_layer.png"
    lineplot(spec_plot, "mean_specificity", "anchor_lang",
             "Anchor-language Specificity (English Must Beat Pseudo-anchors)", "Specificity",
             path, baseline=0.0)
    generated.append(path)

    neighbors = pd.read_csv(metrics / "neighbor_direction_matrix.csv")
    en_neighbors = neighbors[(neighbors.representation == primary) & (neighbors.neighbor_lang == english)]
    path = figures / "english_hub_attraction_by_layer.png"
    lineplot(en_neighbors, "neighbor_rate", "query_lang",
             "English Cross-lingual Hub Attraction", "Share of English neighbors",
             path,
             baseline=float(en_neighbors.uniform_baseline.iloc[0]))
    generated.append(path)

    purity = pd.read_csv(metrics / "language_neighborhood_purity.csv")
    purity_plot = purity[purity.representation == primary]
    path = figures / "language_neighborhood_purity.png"
    lineplot(purity_plot, "neighborhood_purity", "lang",
             "Language Neighborhood Purity and Re-separation", "Same-language neighbor share",
             path,
             baseline=float(purity_plot.uniform_baseline.iloc[0]))
    generated.append(path)

    centroid = pd.read_csv(metrics / "centroid_separation.csv")
    representation_count = centroid.representation.nunique()
    path = figures / "centroid_separation_by_layer.png"
    lineplot(centroid, "centroid_separation", "representation",
             "Language Centroid Separation (Pooling Robustness)", "Mean centroid cosine distance",
             path)
    generated.append(path)

    sanity = pd.read_csv(metrics / "sanity_checks.csv")
    sanity_long = sanity.melt(
        id_vars=["representation", "layer"],
        value_vars=["parallel_similarity", "shuffled_similarity"],
        var_name="comparison", value_name="cosine_similarity"
    )
    path = figures / "similarity_sanity_check.png"
    lineplot(sanity_long, "cosine_similarity", "comparison",
             "Similarity Sanity Check", "Mean cosine similarity",
             path, style="representation" if representation_count > 1 else None)
    generated.append(path)

    retrieval = pd.read_csv(metrics / "semantic_retrieval.csv")
    retrieval["pair"] = retrieval["query_lang"] + "->" + retrieval["target_lang"]
    path = figures / "semantic_retrieval_recall1.png"
    lineplot(retrieval[retrieval.representation == primary], "recall_at_1", "pair",
             "Cross-lingual Parallel-sentence Retrieval", "Recall@1",
             path)
    generated.append(path)

    print("\n=== Figure output ===")
    for path in generated:
        print(f"- {path.name}")
    print(f"Saved {len(generated)} figures to {figures}")


if __name__ == "__main__":
    main()
