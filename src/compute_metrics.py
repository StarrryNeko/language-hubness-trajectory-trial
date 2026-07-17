import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import ensure_dirs, l2_normalize, load_config, representation_file_map


def safe_topk(similarities, candidate_mask, k):
    candidates = np.flatnonzero(candidate_mask)
    if len(candidates) == 0:
        return np.array([], dtype=int)
    take = min(k, len(candidates))
    scores = similarities[candidates]
    chosen = np.argpartition(-scores, take - 1)[:take]
    return candidates[chosen]


def bootstrap_mean_ci(values, rng, n_boot=500, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1 or n_boot <= 0:
        return mean, mean, mean
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return mean, float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1.0 - alpha))


def derangement(rng, size):
    """Generate a permutation with no fixed points for a shuffled semantic baseline."""
    base = np.arange(size)
    for _ in range(100):
        candidate = rng.permutation(size)
        if np.all(candidate != base):
            return candidate
    return np.roll(base, 1)


def max_consecutive(values):
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def layer_window(configured, default_start, default_end, n_layers, excluded=None):
    excluded = set(excluded or [])
    start, end = configured if configured is not None else (default_start, default_end)
    start = max(0, int(start))
    end = min(n_layers - 1, int(end))
    return [layer for layer in range(start, end + 1) if layer not in excluded]


def validate_metadata(meta, languages):
    issues = []
    truncated = meta["was_truncated"]
    if truncated.dtype == object:
        truncated = truncated.astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        truncated = truncated.astype(bool)
    if truncated.any():
        issues.append(f"{int(truncated.sum())} inputs were truncated")
    counts = meta.groupby(["id", "lang"]).size()
    if (counts != 1).any():
        issues.append("some (semantic id, language) pairs are duplicated")
    group_sizes = meta.groupby("id")["lang"].nunique()
    incomplete = int((group_sizes != len(languages)).sum())
    if incomplete:
        issues.append(f"{incomplete} semantic IDs do not contain every language")
    duplicate_texts = int(meta.duplicated(["lang", "text"]).sum())
    if duplicate_texts:
        issues.append(f"{duplicate_texts} within-language texts are duplicated")
    return issues


def load_representation(paths, name):
    files = representation_file_map()
    if name not in files:
        raise ValueError(f"Unknown representation: {name}")
    return np.load(Path(paths["hidden"]) / files[name])


def select_peak(frame, value_col):
    return frame.loc[frame[value_col].idxmax()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=None, help="Override nearest-neighbor k without re-extracting hidden states")
    parser.add_argument("--result-tag", default=None, help="Write metrics into metrics/<tag> for robustness runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    metrics_dir = Path(paths["metrics"])
    if args.result_tag:
        metrics_dir = metrics_dir / args.result_tag
        metrics_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(Path(paths["hidden"]) / "metadata.csv")
    metrics_cfg = cfg["metrics"]
    english = metrics_cfg.get("english_language", "en")
    k = int(args.k if args.k is not None else metrics_cfg.get("nearest_neighbors_k", 10))
    representations = metrics_cfg.get("representations", ["last_token", "mean_pool"])
    n_boot = int(metrics_cfg.get("bootstrap_samples", 500))
    confidence = float(metrics_cfg.get("confidence_level", 0.95))
    n_shuffles = int(metrics_cfg.get("shuffled_permutations", 20))
    min_consecutive_layers = int(metrics_cfg.get("min_consecutive_layers", 3))
    if k < 1:
        raise ValueError("nearest-neighbor k must be positive")
    if n_boot < 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap_samples must be non-negative and confidence_level must be between 0 and 1")
    if n_shuffles < 1:
        raise ValueError("shuffled_permutations must be positive")
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + k)

    languages = sorted(meta["lang"].astype(str).unique())
    if len(languages) < 3:
        raise ValueError("Anchor specificity requires at least three languages")
    if english not in languages:
        raise ValueError(f"English language '{english}' is missing from metadata")
    issues = validate_metadata(meta, languages)
    ids = meta["id"].astype(str).to_numpy()
    langs = meta["lang"].astype(str).to_numpy()
    id_to_indices = {
        semantic_id: {str(meta.loc[i, "lang"]): i for i in group.index}
        for semantic_id, group in meta.groupby(meta["id"].astype(str), sort=True)
    }
    complete_ids = [i for i, mapping in id_to_indices.items() if set(mapping) == set(languages)]
    if not complete_ids:
        raise ValueError("No complete parallel semantic groups found")

    shuffle_indices = {
        pair: [derangement(rng, len(complete_ids)) for _ in range(n_shuffles)]
        for pair in combinations(languages, 2)
    }

    sanity_records, alignment_records, specificity_records = [], [], []
    specificity_sample_records, neighbor_records, neighbor_sample_records = [], [], []
    purity_records, purity_sample_records = [], []
    centroid_records, retrieval_records, occurrence_records = [], [], []

    for representation in representations:
        vectors = load_representation(paths, representation)
        n_samples, n_layers, _ = vectors.shape
        if n_samples != len(meta):
            raise ValueError(f"Metadata/vector row mismatch for {representation}")

        for layer in tqdm(range(n_layers), desc=f"Metrics ({representation}, k={k})"):
            x = l2_normalize(vectors[:, layer, :], axis=1)
            sim = x @ x.T
            parallel_values, shuffled_values = [], []

            for lang_a, lang_b in combinations(languages, 2):
                idx_a = np.array([id_to_indices[i][lang_a] for i in complete_ids])
                idx_b = np.array([id_to_indices[i][lang_b] for i in complete_ids])
                paired = sim[idx_a, idx_b]
                shuffled_matrix = np.stack(
                    [sim[idx_a, idx_b[permutation]] for permutation in shuffle_indices[(lang_a, lang_b)]],
                    axis=0,
                )
                shuffled = shuffled_matrix.mean(axis=0)
                differences = paired - shuffled
                mean, low, high = bootstrap_mean_ci(differences, rng, n_boot, confidence)
                parallel_values.extend(paired.tolist())
                shuffled_values.extend(shuffled.tolist())
                alignment_records.append({
                    "representation": representation, "layer": layer,
                    "lang_a": lang_a, "lang_b": lang_b,
                    "parallel_similarity": float(paired.mean()),
                    "shuffled_similarity": float(shuffled.mean()),
                    "alignment_gain": mean, "ci_lower": low, "ci_upper": high,
                    "n": len(differences), "n_shuffles": n_shuffles,
                })

                for query_lang, target_lang, query_idx, target_idx in [
                    (lang_a, lang_b, idx_a, idx_b), (lang_b, lang_a, idx_b, idx_a)
                ]:
                    ranks = np.argsort(-sim[np.ix_(query_idx, target_idx)], axis=1)
                    correct1 = ranks[:, 0] == np.arange(len(query_idx))
                    correct5 = np.array([
                        row_i in ranks[row_i, : min(5, len(target_idx))]
                        for row_i in range(len(query_idx))
                    ])
                    r1, r1_low, r1_high = bootstrap_mean_ci(correct1, rng, n_boot, confidence)
                    r5, r5_low, r5_high = bootstrap_mean_ci(correct5, rng, n_boot, confidence)
                    retrieval_records.append({
                        "representation": representation, "layer": layer,
                        "query_lang": query_lang, "target_lang": target_lang,
                        "recall_at_1": r1, "recall1_ci_lower": r1_low, "recall1_ci_upper": r1_high,
                        "recall_at_5": r5, "recall5_ci_lower": r5_low, "recall5_ci_upper": r5_high,
                        "random_recall_at_1": 1.0 / len(target_idx), "n": len(query_idx),
                    })

            sanity_records.append({
                "representation": representation, "layer": layer,
                "self_similarity": float(np.diag(sim).mean()),
                "parallel_similarity": float(np.mean(parallel_values)),
                "shuffled_similarity": float(np.mean(shuffled_values)),
                "parallel_minus_shuffled": float(np.mean(parallel_values) - np.mean(shuffled_values)),
                "off_diagonal_std": float(sim[~np.eye(n_samples, dtype=bool)].std()),
            })

            for source_lang in languages:
                for anchor_lang in languages:
                    if anchor_lang == source_lang:
                        continue
                    other_langs = [g for g in languages if g not in (source_lang, anchor_lang)]
                    values = []
                    for semantic_id in complete_ids:
                        source_idx = id_to_indices[semantic_id][source_lang]
                        anchor_idx = id_to_indices[semantic_id][anchor_lang]
                        value = float(sim[source_idx, anchor_idx] - np.mean([
                            sim[source_idx, id_to_indices[semantic_id][g]] for g in other_langs
                        ]))
                        values.append(value)
                        specificity_sample_records.append({
                            "representation": representation, "layer": layer,
                            "semantic_id": semantic_id, "source_lang": source_lang,
                            "anchor_lang": anchor_lang, "specificity": value,
                        })
                    mean, low, high = bootstrap_mean_ci(values, rng, n_boot, confidence)
                    specificity_records.append({
                        "representation": representation, "layer": layer,
                        "source_lang": source_lang, "anchor_lang": anchor_lang,
                        "mean_specificity": mean, "ci_lower": low, "ci_upper": high,
                        "n": len(values),
                    })

            # Cross-language hubness. Each query contributes a language share, enabling bootstrap CIs.
            layer_occurrences = np.zeros(n_samples, dtype=int)
            for query_lang in languages:
                per_neighbor_language = {g: [] for g in languages if g != query_lang}
                for query_idx in np.flatnonzero(langs == query_lang):
                    mask = (langs != query_lang) & (ids != ids[query_idx])
                    neighbors = safe_topk(sim[query_idx], mask, k)
                    layer_occurrences[neighbors] += 1
                    for neighbor_lang in per_neighbor_language:
                        share = float(np.mean(langs[neighbors] == neighbor_lang))
                        per_neighbor_language[neighbor_lang].append(share)
                        neighbor_sample_records.append({
                            "representation": representation, "layer": layer,
                            "semantic_id": ids[query_idx], "query_lang": query_lang,
                            "neighbor_lang": neighbor_lang, "neighbor_rate": share,
                        })
                baseline = 1.0 / (len(languages) - 1)
                for neighbor_lang, values in per_neighbor_language.items():
                    rate, low, high = bootstrap_mean_ci(values, rng, n_boot, confidence)
                    neighbor_records.append({
                        "representation": representation, "layer": layer,
                        "query_lang": query_lang, "neighbor_lang": neighbor_lang,
                        "neighbor_rate": rate, "ci_lower": low, "ci_upper": high,
                        "uniform_baseline": baseline, "excess_neighbor_rate": rate - baseline,
                        "n": len(values),
                    })

            # Classical k-occurrence hubness: how often each point appears in other points' kNN lists.
            for candidate_lang in languages:
                values = layer_occurrences[langs == candidate_lang].astype(float)
                std = float(values.std())
                skewness = float(np.mean(((values - values.mean()) / std) ** 3)) if std > 0 else 0.0
                occurrence_mean, occurrence_low, occurrence_high = bootstrap_mean_ci(
                    values, rng, n_boot, confidence
                )
                occurrence_records.append({
                    "representation": representation, "layer": layer,
                    "candidate_lang": candidate_lang,
                    "mean_k_occurrence": occurrence_mean,
                    "ci_lower": occurrence_low, "ci_upper": occurrence_high,
                    "std_k_occurrence": std,
                    "skewness_k_occurrence": skewness,
                    "p95_k_occurrence": float(np.quantile(values, 0.95)),
                    "max_k_occurrence": float(values.max()),
                    "n": len(values), "k": k,
                })

            # Re-separation: allow same-language neighbors, excluding the complete parallel group.
            language_purities = []
            for query_lang in languages:
                values = []
                for query_idx in np.flatnonzero(langs == query_lang):
                    neighbors = safe_topk(sim[query_idx], ids != ids[query_idx], k)
                    value = float(np.mean(langs[neighbors] == query_lang))
                    values.append(value)
                    purity_sample_records.append({
                        "representation": representation, "layer": layer,
                        "row_idx": int(query_idx), "lang": query_lang,
                        "neighborhood_purity": value,
                    })
                mean, low, high = bootstrap_mean_ci(values, rng, n_boot, confidence)
                language_purities.append(mean)
                purity_records.append({
                    "representation": representation, "layer": layer, "lang": query_lang,
                    "neighborhood_purity": mean, "ci_lower": low, "ci_upper": high,
                    "uniform_baseline": 1.0 / len(languages), "n": len(values),
                })

            centroids = np.stack([
                l2_normalize(x[langs == g].mean(axis=0, keepdims=True))[0] for g in languages
            ])
            centroid_sim = centroids @ centroids.T
            distances = [1.0 - centroid_sim[i, j] for i, j in combinations(range(len(languages)), 2)]
            centroid_records.append({
                "representation": representation, "layer": layer,
                "centroid_separation": float(np.mean(distances)),
                "mean_neighborhood_purity": float(np.mean(language_purities)),
            })

    alignment_df = pd.DataFrame(alignment_records)
    specificity_df = pd.DataFrame(specificity_records)
    specificity_sample_df = pd.DataFrame(specificity_sample_records)
    neighbor_df = pd.DataFrame(neighbor_records)
    occurrence_df = pd.DataFrame(occurrence_records)
    purity_df = pd.DataFrame(purity_records)
    purity_sample_df = pd.DataFrame(purity_sample_records)
    retrieval_df = pd.DataFrame(retrieval_records)

    # Aggregate rotated-anchor evidence by semantic ID first, so parallel translations
    # from the same semantic group are not treated as independent bootstrap samples.
    anchor_summary_records = []
    for (representation, layer, anchor), group in specificity_sample_df.groupby(
        ["representation", "layer", "anchor_lang"]
    ):
        semantic_values = group.groupby("semantic_id").specificity.mean()
        mean, low, high = bootstrap_mean_ci(semantic_values, rng, n_boot, confidence)
        anchor_summary_records.append({
            "representation": representation, "layer": layer, "anchor_lang": anchor,
            "mean_specificity": mean, "ci_lower": low, "ci_upper": high,
            "n_semantic_ids": len(semantic_values),
        })
    anchor_summary_df = pd.DataFrame(anchor_summary_records)

    specificity_contrast_records = []
    for (representation, layer), group in specificity_sample_df.groupby(["representation", "layer"]):
        for pseudo_anchor in [language for language in languages if language != english]:
            common_sources = [
                language for language in languages if language not in {english, pseudo_anchor}
            ]
            english_rows = group[
                (group.anchor_lang == english) & group.source_lang.isin(common_sources)
            ][["semantic_id", "source_lang", "specificity"]].rename(
                columns={"specificity": "english_specificity"}
            )
            pseudo_rows = group[
                (group.anchor_lang == pseudo_anchor) & group.source_lang.isin(common_sources)
            ][["semantic_id", "source_lang", "specificity"]].rename(
                columns={"specificity": "pseudo_specificity"}
            )
            paired = english_rows.merge(
                pseudo_rows, on=["semantic_id", "source_lang"], how="inner"
            )
            semantic_differences = (
                paired.assign(difference=paired.english_specificity - paired.pseudo_specificity)
                .groupby("semantic_id").difference.mean()
            )
            values = semantic_differences.to_numpy()
            mean, low, high = bootstrap_mean_ci(values, rng, n_boot, confidence)
            specificity_contrast_records.append({
                "representation": representation, "layer": layer,
                "english_anchor": english, "pseudo_anchor": pseudo_anchor,
                "english_minus_pseudo": mean, "ci_lower": low, "ci_upper": high,
                "common_source_languages": ",".join(common_sources),
                "n_semantic_ids": len(values),
            })
    specificity_contrast_df = pd.DataFrame(specificity_contrast_records)

    asymmetry_records = []
    neighbor_sample_df = pd.DataFrame(neighbor_sample_records)
    for representation in representations:
        rep_samples = neighbor_sample_df[neighbor_sample_df.representation == representation]
        for layer in sorted(rep_samples.layer.unique()):
            layer_samples = rep_samples[rep_samples.layer == layer]
            for language in [item for item in languages if item != english]:
                toward = layer_samples[
                    (layer_samples.query_lang == language) & (layer_samples.neighbor_lang == english)
                ][["semantic_id", "neighbor_rate"]].rename(columns={"neighbor_rate": "toward"})
                reverse = layer_samples[
                    (layer_samples.query_lang == english) & (layer_samples.neighbor_lang == language)
                ][["semantic_id", "neighbor_rate"]].rename(columns={"neighbor_rate": "reverse"})
                paired = toward.merge(reverse, on="semantic_id", how="inner")
                if paired.empty:
                    continue
                differences = (paired.toward - paired.reverse).to_numpy()
                mean, low, high = bootstrap_mean_ci(differences, rng, n_boot, confidence)
                asymmetry_records.append({
                    "representation": representation, "layer": layer, "language": language,
                    "p_english_given_language": float(paired.toward.mean()),
                    "p_language_given_english": float(paired.reverse.mean()),
                    "english_asymmetry": mean, "ci_lower": low, "ci_upper": high,
                    "n_semantic_ids": len(paired),
                })
    asymmetry_df = pd.DataFrame(asymmetry_records)

    # Re-separation is a pre-specified late-window minus mid-window contrast.
    # Layer 0 is excluded by default because a final-token embedding can mostly encode punctuation.
    window_cfg = metrics_cfg.get("reseparation_windows", {})
    max_layer_number = int(purity_df.layer.max())
    total_layers = max_layer_number + 1
    excluded_layers = window_cfg.get("exclude_layers", [0])
    early_layers = layer_window(window_cfg.get("early"), 1, min(8, max_layer_number), total_layers, excluded_layers)
    mid_layers = layer_window(window_cfg.get("mid"), min(9, max_layer_number), min(17, max_layer_number), total_layers, excluded_layers)
    late_layers = layer_window(window_cfg.get("late"), max(1, max_layer_number - 6), max_layer_number, total_layers, excluded_layers)
    if not mid_layers or not late_layers:
        raise ValueError("Re-separation windows must contain at least one valid mid and late layer")

    reseparation_records = []
    for (representation, language), curve in purity_df.groupby(["representation", "lang"]):
        curve = curve.sort_values("layer")
        final_layer = int(curve.layer.max())
        samples = purity_sample_df[
            (purity_sample_df.representation == representation) & (purity_sample_df.lang == language)
        ]
        pivot = samples.pivot(index="row_idx", columns="layer", values="neighborhood_purity")
        available_early = [layer for layer in early_layers if layer in pivot.columns]
        available_mid = [layer for layer in mid_layers if layer in pivot.columns]
        available_late = [layer for layer in late_layers if layer in pivot.columns]
        early_values = pivot[available_early].mean(axis=1) if available_early else None
        mid_values = pivot[available_mid].mean(axis=1)
        late_values = pivot[available_late].mean(axis=1)
        strengths = (late_values - mid_values).dropna().to_numpy()
        mean, low, high = bootstrap_mean_ci(strengths, rng, n_boot, confidence)
        early_minus_mid = np.nan
        early_mid_low = np.nan
        early_mid_high = np.nan
        if early_values is not None:
            early_mid = (early_values - mid_values).dropna().to_numpy()
            early_minus_mid, early_mid_low, early_mid_high = bootstrap_mean_ci(
                early_mid, rng, n_boot, confidence
            )
        reseparation_records.append({
            "representation": representation, "lang": language,
            "early_layers": ",".join(map(str, available_early)),
            "mid_layers": ",".join(map(str, available_mid)),
            "late_layers": ",".join(map(str, available_late)),
            "early_purity": float(curve[curve.layer.isin(available_early)].neighborhood_purity.mean()) if available_early else np.nan,
            "mid_purity": float(curve[curve.layer.isin(available_mid)].neighborhood_purity.mean()),
            "late_purity": float(curve[curve.layer.isin(available_late)].neighborhood_purity.mean()),
            "final_purity": float(curve[curve.layer == final_layer].neighborhood_purity.iloc[0]),
            "re_separation_strength": mean, "ci_lower": low, "ci_upper": high, "n": len(strengths),
            "early_minus_mid": early_minus_mid,
            "early_minus_mid_ci_lower": early_mid_low,
            "early_minus_mid_ci_upper": early_mid_high,
        })
    reseparation_df = pd.DataFrame(reseparation_records)

    sanity_df = pd.DataFrame(sanity_records)

    # Representation-level comparison uses the same substantive checks as the research claims.
    pooling_records = []
    for representation in representations:
        align_curve = alignment_df[alignment_df.representation == representation].groupby("layer", as_index=False).alignment_gain.mean()
        retrieval_curve = retrieval_df[retrieval_df.representation == representation].groupby("layer", as_index=False).recall_at_1.mean()
        en_curve = anchor_summary_df[
            (anchor_summary_df.representation == representation) & (anchor_summary_df.anchor_lang == english)
        ]
        hub_curve = neighbor_df[
            (neighbor_df.representation == representation) & (neighbor_df.neighbor_lang == english)
        ].groupby("layer", as_index=False).agg(
            english_hub_rate=("neighbor_rate", "mean"), baseline=("uniform_baseline", "mean")
        )
        rep_resep = reseparation_df[reseparation_df.representation == representation]
        rep_sanity = sanity_df[sanity_df.representation == representation]
        geometry_warning_layers = int((rep_sanity.off_diagonal_std < 0.01).sum())
        rep_contrasts = specificity_contrast_df[specificity_contrast_df.representation == representation]
        specificity_by_layer = rep_contrasts.groupby("layer").ci_lower.min().sort_index() > 0
        specificity_supported_layers = int(specificity_by_layer.sum())
        specificity_longest_run = max_consecutive(specificity_by_layer.tolist())

        hub_supported_languages = 0
        hub_language_names = []
        for language in [item for item in languages if item != english]:
            attraction = neighbor_df[
                (neighbor_df.representation == representation) &
                (neighbor_df.query_lang == language) &
                (neighbor_df.neighbor_lang == english)
            ][["layer", "ci_lower", "uniform_baseline"]]
            asymmetry = asymmetry_df[
                (asymmetry_df.representation == representation) &
                (asymmetry_df.language == language)
            ][["layer", "ci_lower"]].rename(columns={"ci_lower": "asymmetry_ci_lower"})
            combined = attraction.merge(asymmetry, on="layer", how="inner").sort_values("layer")
            supported = (
                (combined.ci_lower > combined.uniform_baseline) &
                (combined.asymmetry_ci_lower > 0)
            )
            if max_consecutive(supported.tolist()) >= min_consecutive_layers:
                hub_supported_languages += 1
                hub_language_names.append(language)

        retrieval_direction_support = retrieval_df[
            retrieval_df.representation == representation
        ].groupby(["query_lang", "target_lang"]).apply(
            lambda group: bool((group.recall1_ci_lower > group.random_recall_at_1).any()),
            include_groups=False,
        )
        align_peak, retrieval_peak = select_peak(align_curve, "alignment_gain"), select_peak(retrieval_curve, "recall_at_1")
        en_peak, hub_peak = select_peak(en_curve, "mean_specificity"), select_peak(hub_curve, "english_hub_rate")
        pooling_records.append({
            "representation": representation,
            "alignment_peak": float(align_peak.alignment_gain), "alignment_peak_layer": int(align_peak.layer),
            "retrieval_peak_recall1": float(retrieval_peak.recall_at_1), "retrieval_peak_layer": int(retrieval_peak.layer),
            "english_specificity_peak": float(en_peak.mean_specificity), "english_specificity_peak_layer": int(en_peak.layer),
            "english_hub_peak": float(hub_peak.english_hub_rate), "english_hub_baseline": float(hub_peak.baseline),
            "english_hub_peak_layer": int(hub_peak.layer),
            "mean_re_separation_strength": float(rep_resep.re_separation_strength.mean()),
            "languages_with_positive_re_separation_ci": int((rep_resep.ci_lower > 0).sum()),
            "retrieval_directions_above_random_ci": int(retrieval_direction_support.sum()),
            "retrieval_direction_count": int(len(retrieval_direction_support)),
            "english_specificity_supported_layers": specificity_supported_layers,
            "english_specificity_longest_run": specificity_longest_run,
            "english_hub_supported_language_count": hub_supported_languages,
            "english_hub_supported_languages": ",".join(hub_language_names),
            "geometry_warning_layers": geometry_warning_layers,
        })
    pooling_df = pd.DataFrame(pooling_records)

    outputs = {
        "sanity_checks.csv": sanity_df,
        "alignment_gain.csv": alignment_df,
        "anchor_specificity.csv": specificity_df,
        "anchor_specificity_samples.csv": specificity_sample_df,
        "anchor_specificity_summary.csv": anchor_summary_df,
        "anchor_specificity_contrasts.csv": specificity_contrast_df,
        "neighbor_direction_matrix.csv": neighbor_df,
        "neighbor_rate_samples.csv": neighbor_sample_df,
        "english_directional_asymmetry.csv": asymmetry_df,
        "hubness_occurrence.csv": occurrence_df,
        "language_neighborhood_purity.csv": purity_df,
        "language_neighborhood_purity_samples.csv": purity_sample_df,
        "centroid_separation.csv": pd.DataFrame(centroid_records),
        "semantic_retrieval.csv": retrieval_df,
        "re_separation_summary.csv": reseparation_df,
        "pooling_robustness_summary.csv": pooling_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(metrics_dir / filename, index=False, encoding="utf-8")

    primary = metrics_cfg.get("primary_representation", "last_token")
    primary_pool = pooling_df[pooling_df.representation == primary].iloc[0]
    primary_align = alignment_df[alignment_df.representation == primary]
    alignment_pair_support = primary_align.groupby(["lang_a", "lang_b"]).apply(
        lambda group: max_consecutive((group.sort_values("layer").ci_lower > 0).tolist()) >= min_consecutive_layers,
        include_groups=False,
    )
    semantic_supported = bool(
        alignment_pair_support.all() and
        primary_pool.retrieval_directions_above_random_ci == primary_pool.retrieval_direction_count
    )
    en_anchor = anchor_summary_df[
        (anchor_summary_df.representation == primary) & (anchor_summary_df.anchor_lang == english)
    ]
    en_peak = select_peak(en_anchor, "mean_specificity")
    pseudo_at_peak = anchor_summary_df[
        (anchor_summary_df.representation == primary) &
        (anchor_summary_df.layer == en_peak.layer) &
        (anchor_summary_df.anchor_lang != english)
    ].mean_specificity.max()
    english_specific = bool(primary_pool.english_specificity_longest_run >= min_consecutive_layers)
    supported_hub_languages = int(primary_pool.english_hub_supported_language_count)
    hub_status = "SUPPORTED" if supported_hub_languages >= 2 else (
        "PARTIAL" if supported_hub_languages == 1 else "NOT SUPPORTED"
    )
    primary_resep = reseparation_df[reseparation_df.representation == primary]
    positive_resep = int((primary_resep.ci_lower > 0).sum())
    reseparation_status = "SUPPORTED" if positive_resep == len(languages) else ("PARTIAL" if positive_resep else "NOT SUPPORTED")
    pooling_status = "NOT TESTED"
    if len(pooling_df) >= 2:
        reference = pooling_df[pooling_df.representation == primary].iloc[0]
        comparisons = pooling_df[pooling_df.representation != primary]
        same_claims = all(
            row.retrieval_directions_above_random_ci == reference.retrieval_directions_above_random_ci and
            (row.english_specificity_longest_run >= min_consecutive_layers) ==
            (reference.english_specificity_longest_run >= min_consecutive_layers) and
            ("" if pd.isna(row.english_hub_supported_languages) else str(row.english_hub_supported_languages)) ==
            ("" if pd.isna(reference.english_hub_supported_languages) else str(reference.english_hub_supported_languages)) and
            row.languages_with_positive_re_separation_ci == reference.languages_with_positive_re_separation_ci
            for _, row in comparisons.iterrows()
        )
        no_geometry_warning = bool((pooling_df.geometry_warning_layers == 0).all())
        max_layer = max(1, int(alignment_df.layer.max()))
        similar_peak_location = all(
            abs(int(row.alignment_peak_layer) - int(reference.alignment_peak_layer)) / max_layer <= 0.25 and
            abs(int(row.retrieval_peak_layer) - int(reference.retrieval_peak_layer)) / max_layer <= 0.25
            for _, row in comparisons.iterrows()
        )
        pooling_status = "CONSISTENT" if same_claims and no_geometry_warning and similar_peak_location else "SENSITIVE"

    pooling_verdict_df = pd.DataFrame([{
        "primary_representation": primary,
        "representations_compared": ",".join(pooling_df.representation.astype(str)),
        "status": pooling_status,
        "min_consecutive_layers": min_consecutive_layers,
        "reason": (
            "All claim-level directions, peak locations, and geometry checks agree."
            if pooling_status == "CONSISTENT" else
            "At least one representation differs in claim support, peak location, or geometry validity."
        ),
    }])
    pooling_verdict_df.to_csv(metrics_dir / "pooling_robustness_verdict.csv", index=False, encoding="utf-8")

    def verdict(value):
        return "SUPPORTED" if value else "NOT SUPPORTED"

    report = [
        "=== PILOT VALIDATION REPORT ===",
        f"Result tag: {args.result_tag or 'main'} | k={k} | bootstrap={n_boot} ({confidence:.0%} CI) "
        f"| shuffled derangements={n_shuffles}",
        f"Rows={len(meta)}, complete semantic groups={len(complete_ids)}, languages={languages}",
        f"Metadata checks: {'PASS' if not issues else 'WARN - ' + '; '.join(issues)}",
        "Input isolation: one sentence per forward pass; model.eval(); use_cache=False; no chat template",
        f"Semantic alignment: {verdict(semantic_supported)}",
        f"English specificity over rotated pseudo-anchors: {verdict(english_specific)}",
        f"English cross-lingual hub advantage: {hub_status} "
        f"({supported_hub_languages}/{len(languages) - 1} non-English languages)",
        f"Language re-separation: {reseparation_status} ({positive_resep}/{len(languages)} languages with CI > 0)",
        f"Pooling robustness: {pooling_status}",
        f"Primary alignment peak: {primary_pool.alignment_peak:.4f} at layer {int(primary_pool.alignment_peak_layer)}",
        f"Primary retrieval peak Recall@1: {primary_pool.retrieval_peak_recall1:.3f}",
        f"Primary English specificity peak: {primary_pool.english_specificity_peak:.4f}",
        f"English specificity longest all-pseudo-anchor CI run: "
        f"{int(primary_pool.english_specificity_longest_run)} layers",
        f"Primary English hub peak/baseline: {primary_pool.english_hub_peak:.3f}/{primary_pool.english_hub_baseline:.3f}",
        f"Re-separation contrast: late layers {late_layers} minus mid layers {mid_layers}; excluded={excluded_layers}",
        "Interpretation: neighbor-language attraction and classical k-occurrence hubness are both saved; "
        "proximity, attraction, hubness, and re-separation remain separate claims.",
    ]
    report_text = "\n".join(report)
    (metrics_dir / "validation_report.txt").write_text(report_text, encoding="utf-8")
    (metrics_dir / "research_summary.txt").write_text("\n".join(report[5:]), encoding="utf-8")
    print("\n" + report_text)
    print(f"\nSaved metric tables to {metrics_dir}")


if __name__ == "__main__":
    main()
