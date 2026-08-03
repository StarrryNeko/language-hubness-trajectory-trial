"""Create compact paper_v1 diagnostic figures from saved metric tables."""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config
from paper_common import ensure_paper_dirs


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot paper_v1 trajectories")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_paper_dirs(cfg)

    alignment = pd.read_csv(paths.metrics / "alignment" / "alignment_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group_name, group in alignment.groupby("pair_group"):
        group = group.sort_values("layer")
        ax.plot(group.layer, group["mean"], label=group_name)
        ax.fill_between(group.layer, group.ci_lower, group.ci_upper, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Layer (0 = embedding)", ylabel="AlignmentGain", title="Semantic alignment by layer")
    ax.legend(frameon=False)
    save(fig, paths.figures / "alignment_overall_by_layer.png")

    evidence = pd.read_csv(paths.metrics / "hubness" / "target_rotation_evidence.csv")
    english = cfg["metrics"].get("english_language", "en")
    selected = evidence[
        (evidence.candidate_lang == english)
        & evidence.similarity_method.isin(["cosine", "local_scaled_cosine"])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (metric, group) in zip(axes.flat, selected.groupby("metric", sort=True)):
        for method, curve in group.groupby("similarity_method"):
            curve = curve.sort_values("layer")
            ax.plot(curve.layer, curve["mean"], label=method)
            ax.fill_between(curve.layer, curve.ci_lower, curve.ci_upper, alpha=0.15)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(metric)
    axes[-1, 0].set_xlabel("Layer")
    axes[-1, 1].set_xlabel("Layer")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("English hubness: raw vs local-density control")
    save(fig, paths.figures / "english_hubness_raw_vs_density.png")

    specificity = pd.read_csv(paths.metrics / "hubness" / "target_rotation_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, curve in specificity[
        specificity.similarity_method.isin(["cosine", "local_scaled_cosine"])
    ].groupby("similarity_method"):
        ax.plot(curve.layer, curve.english_rank, marker="o", markersize=2.5, label=method)
    ax.invert_yaxis()
    ax.set(xlabel="Layer", ylabel="English rank (1 is highest)", title="English among all rotated hub candidates")
    ax.legend(frameon=False)
    save(fig, paths.figures / "english_target_rank_by_layer.png")

    competition = pd.read_csv(
        paths.metrics / "similarity_competition" / "english_competition_by_layer.csv"
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    competition_specs = [
        ("hard_margin", 0.0, "English − strongest non-English"),
        ("pairwise_win_rate", 0.5, "English pairwise win rate"),
        ("english_rank", 1.0, "English candidate rank"),
    ]
    for ax, (metric, baseline, title) in zip(axes, competition_specs):
        subset = competition[competition.metric == metric]
        for method, curve in subset.groupby("similarity_method"):
            curve = curve.sort_values("layer")
            ax.plot(curve.layer, curve["mean"], label=method)
            ax.fill_between(curve.layer, curve.ci_lower, curve.ci_upper, alpha=0.15)
        ax.axhline(baseline, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Layer")
    axes[2].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, paths.figures / "english_competition_by_layer.png")

    geometry = pd.read_csv(
        paths.metrics / "norm_trajectory" / "language_geometry_by_layer.csv"
    )
    geometry_features = [
        ("norm_rank", "Norm rank (1 = largest)"),
        ("distance_to_loo_semantic_centroid", "Distance to LOO semantic centroid"),
        ("distance_to_global_centroid", "Distance to calibration centroid"),
        ("local_density", "Local cosine density"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (feature, title) in zip(axes.flat, geometry_features):
        subset = geometry[geometry.feature == feature]
        other = subset[subset.language != english].groupby("layer", as_index=False)["mean"].agg(
            other_mean="mean", other_min="min", other_max="max"
        )
        english_curve = subset[subset.language == english].sort_values("layer")
        ax.fill_between(
            other.layer, other.other_min, other.other_max, color="gray", alpha=0.18,
            label="non-English range",
        )
        ax.plot(other.layer, other.other_mean, color="gray", linewidth=1.2, label="non-English mean")
        ax.plot(
            english_curve.layer, english_curve["mean"], color="#d62728", linewidth=2,
            label="English",
        )
        ax.set_title(title)
        if feature == "norm_rank":
            ax.invert_yaxis()
    axes[-1, 0].set_xlabel("Layer")
    axes[-1, 1].set_xlabel("Layer")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, paths.figures / "english_geometry_mechanisms.png")

    attraction = pd.read_csv(
        paths.metrics / "hubness" / "target_rotation_source_attraction.csv"
    )
    raw_competition = competition[
        (competition.similarity_method == "cosine") & (competition.metric == "hard_margin")
    ]
    peak_layer = int(raw_competition.loc[raw_competition["mean"].idxmax(), "layer"])
    peak = attraction[
        (attraction.similarity_method == "cosine") & (attraction.layer == peak_layer)
    ]
    matrix = peak.pivot(index="source_lang", columns="candidate_lang", values="mean").reindex(
        index=cfg["dataset"]["languages"], columns=cfg["dataset"]["languages"]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=7)
    ax.set(xlabel="Candidate language", ylabel="Source language",
           title=f"Directed top-k attraction at competitive peak layer {peak_layer}")
    fig.colorbar(image, ax=ax, label="Top-k selection rate")
    save(fig, paths.figures / "source_candidate_attraction_peak_layer.png")

    purity = pd.read_csv(paths.metrics / "language_structure" / "neighborhood_purity.csv")
    probe = pd.read_csv(paths.metrics / "language_probe" / "probe_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    purity_mean = purity.groupby("layer", as_index=False).neighborhood_purity.mean()
    axes[0].plot(purity_mean.layer, purity_mean.neighborhood_purity)
    axes[0].axhline(purity.uniform_language_baseline.iloc[0], color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="Layer", ylabel="Mean purity", title="Cross-semantic language-neighborhood purity")
    axes[1].plot(probe.layer, probe.macro_f1, label="test macro-F1")
    axes[1].plot(probe.layer, probe.permutation_macro_f1_mean, label="label-permutation mean")
    axes[1].set(xlabel="Layer", ylabel="Macro-F1", title="Semantic-ID-split language probe")
    axes[1].legend(frameon=False)
    save(fig, paths.figures / "language_identity_structure.png")
    print(f"Saved paper_v1 figures to {paths.figures}")


if __name__ == "__main__":
    main()
