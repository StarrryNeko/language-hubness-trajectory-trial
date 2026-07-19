import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ensure_dirs, load_config


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
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
        metrics /= args.result_tag
        figures /= args.result_tag
        figures.mkdir(parents=True, exist_ok=True)
    primary = cfg["metrics"].get("primary_representation", "mean_pool")
    english = cfg["metrics"].get("english_language", "en")
    sns.set_theme(style="whitegrid")
    generated = []

    evidence = pd.read_csv(metrics / "english_hubness_evidence.csv")
    evidence = evidence[
        (evidence.representation == primary) & (evidence.similarity_method == "cosine")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric, group) in zip(axes.flat, evidence.groupby("metric", sort=True)):
        group = group.sort_values("layer")
        ax.plot(group.layer, group["mean"], marker="o", markersize=2)
        ax.fill_between(group.layer, group.ci_lower, group.ci_upper, alpha=0.2)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("English minus balanced alternative")
    fig.suptitle("English Hubness: Complementary Same-semantics Evidence")
    generated.append(figures / "english_hubness_evidence.png")
    save(generated[-1])

    by_language = pd.read_csv(metrics / "hubness_by_language.csv")
    by_language = by_language[
        (by_language.representation == primary)
        & (by_language.similarity_method == "cosine")
        & (by_language.metric == "k_occurrence")
    ]
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=by_language, x="layer", y="mean", hue="candidate_lang", errorbar=None)
    plt.axhline(int(cfg["metrics"].get("nearest_neighbors_k", 5)), color="black", linestyle="--")
    plt.title("Within-semantics Reverse-kNN Occurrence by Language")
    plt.ylabel("Mean number of source languages selecting candidate")
    plt.legend(ncol=3, fontsize=8)
    generated.append(figures / "hubness_occurrence_by_language.png")
    save(generated[-1])

    controls = by_language = pd.read_csv(metrics / "english_hubness_evidence.csv")
    controls = controls[
        (controls.representation.isin([primary, cfg["metrics"].get("validation_representation", "sentinel_eos")]))
        & (controls.metric == "k_occurrence_excess")
    ]
    controls["control"] = controls.representation + " / " + controls.similarity_method
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=controls, x="layer", y="mean", hue="control", marker="o", errorbar=None)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("English Hubness under EOS and Local-density Controls")
    plt.ylabel("English k-occurrence minus balanced expectation")
    generated.append(figures / "english_hubness_controls.png")
    save(generated[-1])

    source = pd.read_csv(metrics / "english_source_group_attraction.csv")
    source = source[
        (source.representation == primary) & (source.similarity_method == "cosine")
    ]
    script = source.groupby(["layer", "source_script"], as_index=False).agg(
        mean=("mean", "mean"), baseline=("balanced_selection_baseline", "mean")
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=script, x="layer", y="mean", hue="source_script", errorbar=None)
    plt.axhline(float(script.baseline.iloc[0]), color="black", linestyle="--")
    plt.title("Breadth of English Attraction across Source Scripts")
    plt.ylabel("P(English is in same-semantics top-k)")
    plt.legend(ncol=2, fontsize=8)
    generated.append(figures / "english_attraction_by_source_script.png")
    save(generated[-1])

    agreement = pd.read_csv(metrics / "representation_agreement.csv")
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=agreement, x="layer", y="pairwise_similarity_pearson", marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylim(-1.05, 1.05)
    plt.title("Mean-pool vs Sentinel-EOS Geometry Agreement")
    plt.ylabel("Pearson r over within-semantics language pairs")
    generated.append(figures / "representation_agreement.png")
    save(generated[-1])

    pair = pd.read_csv(metrics / "within_semantic_pair_similarity.csv")
    pair = pair[pair.representation == primary]
    aggregate = pair.groupby("layer", as_index=False).agg(
        mean_similarity=("mean", "mean"), pair_std=("mean", "std")
    )
    plt.figure(figsize=(9, 5))
    plt.plot(aggregate.layer, aggregate.mean_similarity, marker="o", markersize=3)
    plt.fill_between(
        aggregate.layer,
        aggregate.mean_similarity - aggregate.pair_std,
        aggregate.mean_similarity + aggregate.pair_std,
        alpha=0.2,
    )
    plt.title("Within-semantics Cross-language Cohesion")
    plt.ylabel("Mean cosine across language pairs")
    generated.append(figures / "within_semantic_cohesion.png")
    save(generated[-1])

    print(f"Saved {len(generated)} figures to {figures}")
    for path in generated:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
