import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import ensure_dirs, l2_normalize, load_config


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
    files = {
        "last_token": "sentence_layer_last_token.npy",
        "mean_pool": "sentence_layer_mean_pool.npy",
    }
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
    if k < 1:
        raise ValueError("nearest-neighbor k must be positive")
    if n_boot < 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap_samples must be non-negative and confidence_level must be between 0 and 1")
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

    sanity_records, alignment_records, specificity_records = [], [], []
    specificity_sample_records, neighbor_records = [], []
    purity_records, purity_sample_records = [], []
    centroid_records, retrieval_records = [], []

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
                shuffled = sim[idx_a, np.roll(idx_b, 1)]
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
                    "n": len(differences),
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
            for query_lang in languages:
                per_neighbor_language = {g: [] for g in languages if g != query_lang}
                for query_idx in np.flatnonzero(langs == query_lang):
                    mask = (langs != query_lang) & (ids != ids[query_idx])
                    neighbors = safe_topk(sim[query_idx], mask, k)
                    for neighbor_lang in per_neighbor_language:
                        per_neighbor_language[neighbor_lang].append(float(np.mean(langs[neighbors] == neighbor_lang)))
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
    purity_df = pd.DataFrame(purity_records)
    purity_sample_df = pd.DataFrame(purity_sample_records)
    retrieval_df = pd.DataFrame(retrieval_records)

    # Aggregate rotated-anchor evidence across source languages and semantic IDs.
    anchor_summary_records = []
    for (representation, layer, anchor), group in specificity_sample_df.groupby(
        ["representation", "layer", "anchor_lang"]
    ):
        mean, low, high = bootstrap_mean_ci(group.specificity, rng, n_boot, confidence)
        anchor_summary_records.append({
            "representation": representation, "layer": layer, "anchor_lang": anchor,
            "mean_specificity": mean, "ci_lower": low, "ci_upper": high, "n": len(group),
        })
    anchor_summary_df = pd.DataFrame(anchor_summary_records)

    asymmetry_records = []
    for representation in representations:
        rep_df = neighbor_df[neighbor_df.representation == representation]
        for layer in sorted(rep_df.layer.unique()):
            layer_df = rep_df[rep_df.layer == layer]
            for language in languages:
                if language == english:
                    continue
                toward = layer_df[(layer_df.query_lang == language) & (layer_df.neighbor_lang == english)]
                reverse = layer_df[(layer_df.query_lang == english) & (layer_df.neighbor_lang == language)]
                if len(toward) and len(reverse):
                    asymmetry_records.append({
                        "representation": representation, "layer": layer, "language": language,
                        "p_english_given_language": float(toward.neighbor_rate.iloc[0]),
                        "p_language_given_english": float(reverse.neighbor_rate.iloc[0]),
                        "english_asymmetry": float(toward.neighbor_rate.iloc[0] - reverse.neighbor_rate.iloc[0]),
                    })
    asymmetry_df = pd.DataFrame(asymmetry_records)

    # Re-separation strength is final-layer purity minus the language-specific minimum layer.
    reseparation_records = []
    for (representation, language), curve in purity_df.groupby(["representation", "lang"]):
        curve = curve.sort_values("layer")
        min_layer = int(curve.loc[curve.neighborhood_purity.idxmin(), "layer"])
        final_layer = int(curve.layer.max())
        samples = purity_sample_df[
            (purity_sample_df.representation == representation) & (purity_sample_df.lang == language)
        ]
        pivot = samples.pivot(index="row_idx", columns="layer", values="neighborhood_purity")
        strengths = (pivot[final_layer] - pivot[min_layer]).dropna().to_numpy()
        mean, low, high = bootstrap_mean_ci(strengths, rng, n_boot, confidence)
        reseparation_records.append({
            "representation": representation, "lang": language,
            "minimum_layer": min_layer, "final_layer": final_layer,
            "minimum_purity": float(curve[curve.layer == min_layer].neighborhood_purity.iloc[0]),
            "final_purity": float(curve[curve.layer == final_layer].neighborhood_purity.iloc[0]),
            "re_separation_strength": mean, "ci_lower": low, "ci_upper": high, "n": len(strengths),
        })
    reseparation_df = pd.DataFrame(reseparation_records)

    # Compact representation-level comparison for the weekly robustness summary.
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
        })
    pooling_df = pd.DataFrame(pooling_records)

    outputs = {
        "sanity_checks.csv": pd.DataFrame(sanity_records),
        "alignment_gain.csv": alignment_df,
        "anchor_specificity.csv": specificity_df,
        "anchor_specificity_summary.csv": anchor_summary_df,
        "neighbor_direction_matrix.csv": neighbor_df,
        "english_directional_asymmetry.csv": asymmetry_df,
        "language_neighborhood_purity.csv": purity_df,
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
    semantic_supported = bool(primary_align.ci_lower.max() > 0)
    en_anchor = anchor_summary_df[
        (anchor_summary_df.representation == primary) & (anchor_summary_df.anchor_lang == english)
    ]
    en_peak = select_peak(en_anchor, "mean_specificity")
    pseudo_at_peak = anchor_summary_df[
        (anchor_summary_df.representation == primary) &
        (anchor_summary_df.layer == en_peak.layer) &
        (anchor_summary_df.anchor_lang != english)
    ].mean_specificity.max()
    english_specific = bool(en_peak.ci_lower > 0 and en_peak.mean_specificity > pseudo_at_peak)
    primary_hub = neighbor_df[
        (neighbor_df.representation == primary) & (neighbor_df.neighbor_lang == english)
    ]
    hub_supported = bool((primary_hub.ci_lower > primary_hub.uniform_baseline).any())
    primary_resep = reseparation_df[reseparation_df.representation == primary]
    positive_resep = int((primary_resep.ci_lower > 0).sum())
    reseparation_status = "SUPPORTED" if positive_resep == len(languages) else ("PARTIAL" if positive_resep else "NOT SUPPORTED")
    pooling_status = "NOT TESTED"
    if len(pooling_df) >= 2:
        reference = pooling_df.iloc[0]
        comparisons = pooling_df.iloc[1:]
        same_directions = all(
            np.sign(row.alignment_peak) == np.sign(reference.alignment_peak) and
            np.sign(row.english_specificity_peak) == np.sign(reference.english_specificity_peak) and
            np.sign(row.mean_re_separation_strength) == np.sign(reference.mean_re_separation_strength)
            for _, row in comparisons.iterrows()
        )
        max_layer = max(1, int(alignment_df.layer.max()))
        similar_peak_location = all(
            abs(int(row.alignment_peak_layer) - int(reference.alignment_peak_layer)) / max_layer <= 0.25
            for _, row in comparisons.iterrows()
        )
        pooling_status = "CONSISTENT" if same_directions and similar_peak_location else "SENSITIVE"

    def verdict(value):
        return "SUPPORTED" if value else "NOT SUPPORTED"

    report = [
        "=== PILOT VALIDATION REPORT ===",
        f"Result tag: {args.result_tag or 'main'} | k={k} | bootstrap={n_boot} ({confidence:.0%} CI)",
        f"Rows={len(meta)}, complete semantic groups={len(complete_ids)}, languages={languages}",
        f"Metadata checks: {'PASS' if not issues else 'WARN - ' + '; '.join(issues)}",
        "Input isolation: one sentence per forward pass; model.eval(); use_cache=False; no chat template",
        f"Semantic alignment: {verdict(semantic_supported)}",
        f"English specificity over rotated pseudo-anchors: {verdict(english_specific)}",
        f"English cross-lingual hub advantage: {verdict(hub_supported)}",
        f"Language re-separation: {reseparation_status} ({positive_resep}/{len(languages)} languages with CI > 0)",
        f"Pooling robustness: {pooling_status}",
        f"Primary alignment peak: {primary_pool.alignment_peak:.4f} at layer {int(primary_pool.alignment_peak_layer)}",
        f"Primary retrieval peak Recall@1: {primary_pool.retrieval_peak_recall1:.3f}",
        f"Primary English specificity peak: {primary_pool.english_specificity_peak:.4f}",
        f"Primary English hub peak/baseline: {primary_pool.english_hub_peak:.3f}/{primary_pool.english_hub_baseline:.3f}",
        "Interpretation: proximity, hubness, and re-separation are separate claims and require separate evidence.",
    ]
    report_text = "\n".join(report)
    (metrics_dir / "validation_report.txt").write_text(report_text, encoding="utf-8")
    (metrics_dir / "research_summary.txt").write_text("\n".join(report[5:]), encoding="utf-8")
    print("\n" + report_text)
    print(f"\nSaved metric tables to {metrics_dir}")


if __name__ == "__main__":
    main()
