import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ensure_dirs, load_config


def lineplot(data, y, hue, title, ylabel, output, baseline=None, style=None,
             ci_lower=None, ci_upper=None):
    plt.figure(figsize=(10, 6))
    hue_order = sorted(data[hue].unique())
    palette = sns.color_palette(n_colors=len(hue_order))
    sns.lineplot(data=data, x="layer", y=y, hue=hue, hue_order=hue_order,
                 palette=palette, style=style, errorbar=None)
    if ci_lower and ci_upper and ci_lower in data.columns and ci_upper in data.columns:
        for color, label in zip(palette, hue_order):
            group = data[data[hue] == label].sort_values("layer")
            plt.fill_between(group["layer"].to_numpy(), group[ci_lower].to_numpy(),
                             group[ci_upper].to_numpy(), color=color, alpha=0.15)
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
    parser.add_argument("--result-tag", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    metrics = Path(paths["metrics"])
    figures = Path(paths["figures"])
    if args.result_tag:
        metrics = metrics / args.result_tag
        figures = figures / args.result_tag
        figures.mkdir(parents=True, exist_ok=True)
    primary = cfg["metrics"].get("primary_representation", "last_token")
    english = cfg["metrics"].get("english_language", "en")
    sns.set_theme(style="whitegrid")

    alignment = pd.read_csv(metrics / "alignment_gain.csv")
    alignment["pair"] = alignment["lang_a"] + "-" + alignment["lang_b"]
    generated = []

    path = figures / "alignment_gain_by_layer.png"
    lineplot(alignment[alignment.representation == primary], "alignment_gain", "pair",
             "Cross-lingual Semantic Alignment", "Parallel cosine - shuffled cosine",
             path, ci_lower="ci_lower", ci_upper="ci_upper")
    generated.append(path)

    specificity_contrasts = pd.read_csv(metrics / "anchor_specificity_contrasts.csv")
    contrast_plot = specificity_contrasts[specificity_contrasts.representation == primary]
    path = figures / "english_specificity_contrasts.png"
    lineplot(contrast_plot, "english_minus_pseudo", "pseudo_anchor",
             "English Specificity Minus Each Pseudo-anchor", "English - pseudo specificity",
             path, baseline=0.0, ci_lower="ci_lower", ci_upper="ci_upper")
    generated.append(path)

    specificity = pd.read_csv(metrics / "anchor_specificity_summary.csv")
    spec_plot = specificity[specificity.representation == primary]
    path = figures / "anchor_specificity_by_layer.png"
    lineplot(spec_plot, "mean_specificity", "anchor_lang",
             "Anchor-language Specificity (English Must Beat Pseudo-anchors)", "Specificity",
             path, baseline=0.0, ci_lower="ci_lower", ci_upper="ci_upper")
    generated.append(path)

    neighbors = pd.read_csv(metrics / "neighbor_direction_matrix.csv")
    en_neighbors = neighbors[(neighbors.representation == primary) & (neighbors.neighbor_lang == english)]
    path = figures / "english_hub_attraction_by_layer.png"
    lineplot(en_neighbors, "neighbor_rate", "query_lang",
             "English Cross-lingual Neighbor Attraction", "Share of English neighbors",
             path,
             baseline=float(en_neighbors.uniform_baseline.iloc[0]),
             ci_lower="ci_lower", ci_upper="ci_upper")
    generated.append(path)

    occurrence = pd.read_csv(metrics / "hubness_occurrence.csv")
    occurrence_plot = occurrence[occurrence.representation == primary]
    path = figures / "hubness_occurrence_by_layer.png"
    lineplot(occurrence_plot, "mean_k_occurrence", "candidate_lang",
             "Classical Cross-lingual k-occurrence Hubness", "Mean k-occurrence",
             path, ci_lower="ci_lower", ci_upper="ci_upper")
    generated.append(path)

    purity = pd.read_csv(metrics / "language_neighborhood_purity.csv")
    purity_plot = purity[purity.representation == primary]
    path = figures / "language_neighborhood_purity.png"
    lineplot(purity_plot, "neighborhood_purity", "lang",
             "Language Neighborhood Purity and Re-separation", "Same-language neighbor share",
             path,
             baseline=float(purity_plot.uniform_baseline.iloc[0]),
             ci_lower="ci_lower", ci_upper="ci_upper")
    generated.append(path)

    reseparation = pd.read_csv(metrics / "re_separation_summary.csv")
    reseparation_plot = reseparation[reseparation.representation == primary].sort_values("lang")
    path = figures / "re_separation_strength.png"
    plt.figure(figsize=(8, 5))
    yerr = [
        reseparation_plot["re_separation_strength"] - reseparation_plot["ci_lower"],
        reseparation_plot["ci_upper"] - reseparation_plot["re_separation_strength"],
    ]
    plt.bar(reseparation_plot["lang"], reseparation_plot["re_separation_strength"],
            yerr=yerr, capsize=4)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Late Language Re-separation Strength")
    plt.xlabel("Language")
    plt.ylabel("Late-window purity - mid-window purity")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
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
             path, ci_lower="recall1_ci_lower", ci_upper="recall1_ci_upper")
    generated.append(path)

    pooling = pd.read_csv(metrics / "pooling_robustness_summary.csv")
    path = figures / "pooling_robustness_summary.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("alignment_peak", "Alignment peak"),
        ("retrieval_peak_recall1", "Peak Recall@1"),
        ("english_specificity_peak", "English specificity peak"),
        ("mean_re_separation_strength", "Mean re-separation strength"),
    ]
    for ax, (column, title) in zip(axes.flat, panels):
        sns.barplot(data=pooling, x="representation", y=column, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("Pooling Robustness Summary")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path)

    print("\n=== Figure output ===")
    for path in generated:
        print(f"- {path.name}")
    print(f"Saved {len(generated)} figures to {figures}")


if __name__ == "__main__":
    main()
