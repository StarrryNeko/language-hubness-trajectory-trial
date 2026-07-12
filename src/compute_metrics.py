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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    meta = pd.read_csv(Path(paths["hidden"]) / "metadata.csv")
    metrics_cfg = cfg["metrics"]
    english = metrics_cfg.get("english_language", "en")
    k = int(metrics_cfg.get("nearest_neighbors_k", 10))
    representations = metrics_cfg.get("representations", ["last_token", "mean_pool"])
    languages = sorted(meta["lang"].unique())
    if english not in languages:
        raise ValueError(f"English language '{english}' is missing from metadata")
    issues = validate_metadata(meta, languages)
    ids = meta["id"].astype(str).to_numpy()
    langs = meta["lang"].astype(str).to_numpy()
    id_to_indices = {
        semantic_id: {meta.loc[i, "lang"]: i for i in group.index}
        for semantic_id, group in meta.groupby(meta["id"].astype(str), sort=True)
    }
    complete_ids = [i for i, mapping in id_to_indices.items() if set(mapping) == set(languages)]
    if not complete_ids:
        raise ValueError("No complete parallel semantic groups found")

    sanity_records = []
    alignment_records = []
    specificity_records = []
    neighbor_records = []
    purity_records = []
    centroid_records = []
    retrieval_records = []

    for representation in representations:
        vectors = load_representation(paths, representation)
        n_samples, n_layers, _ = vectors.shape
        if n_samples != len(meta):
            raise ValueError(f"Metadata/vector row mismatch for {representation}")

        for layer in tqdm(range(n_layers), desc=f"Metrics ({representation})"):
            x = l2_normalize(vectors[:, layer, :], axis=1)
            sim = x @ x.T

            parallel_values = []
            shuffled_values = []
            for lang_a, lang_b in combinations(languages, 2):
                idx_a = np.array([id_to_indices[i][lang_a] for i in complete_ids])
                idx_b = np.array([id_to_indices[i][lang_b] for i in complete_ids])
                paired = sim[idx_a, idx_b]
                # A fixed cyclic shift makes the non-parallel baseline identical across layers.
                shuffled_b = np.roll(idx_b, 1)
                shuffled = sim[idx_a, shuffled_b]
                differences = paired - shuffled
                parallel_values.extend(paired.tolist())
                shuffled_values.extend(shuffled.tolist())
                alignment_records.append({
                    "representation": representation,
                    "layer": layer,
                    "lang_a": lang_a,
                    "lang_b": lang_b,
                    "parallel_similarity": float(paired.mean()),
                    "shuffled_similarity": float(shuffled.mean()),
                    "alignment_gain": float(paired.mean() - shuffled.mean()),
                    "alignment_gain_std": float(differences.std(ddof=1)),
                    "n": len(differences),
                })

                for query_lang, target_lang, query_idx, target_idx in [
                    (lang_a, lang_b, idx_a, idx_b),
                    (lang_b, lang_a, idx_b, idx_a),
                ]:
                    ranks = np.argsort(-sim[np.ix_(query_idx, target_idx)], axis=1)
                    recall1 = float(np.mean(ranks[:, 0] == np.arange(len(query_idx))))
                    recall5 = float(np.mean([
                        row_i in ranks[row_i, : min(5, len(target_idx))]
                        for row_i in range(len(query_idx))
                    ]))
                    retrieval_records.append({
                        "representation": representation, "layer": layer,
                        "query_lang": query_lang, "target_lang": target_lang,
                        "recall_at_1": recall1, "recall_at_5": recall5,
                    })

            sanity_records.append({
                "representation": representation,
                "layer": layer,
                "self_similarity": float(np.diag(sim).mean()),
                "parallel_similarity": float(np.mean(parallel_values)),
                "shuffled_similarity": float(np.mean(shuffled_values)),
                "parallel_minus_shuffled": float(np.mean(parallel_values) - np.mean(shuffled_values)),
                "off_diagonal_std": float(sim[~np.eye(n_samples, dtype=bool)].std()),
            })

            # Anchor specificity: rotate every language through the anchor role.
            for source_lang in languages:
                for anchor_lang in languages:
                    if anchor_lang == source_lang:
                        continue
                    other_langs = [g for g in languages if g not in (source_lang, anchor_lang)]
                    values = []
                    for semantic_id in complete_ids:
                        source_idx = id_to_indices[semantic_id][source_lang]
                        anchor_idx = id_to_indices[semantic_id][anchor_lang]
                        anchor_sim = sim[source_idx, anchor_idx]
                        baseline = np.mean([
                            sim[source_idx, id_to_indices[semantic_id][g]] for g in other_langs
                        ])
                        values.append(anchor_sim - baseline)
                    specificity_records.append({
                        "representation": representation, "layer": layer,
                        "source_lang": source_lang, "anchor_lang": anchor_lang,
                        "mean_specificity": float(np.mean(values)),
                        "std_specificity": float(np.std(values, ddof=1)),
                        "n": len(values),
                    })

            # Cross-language hubness pool: exclude same language and all translations of same semantic ID.
            for query_lang in languages:
                query_indices = np.flatnonzero(langs == query_lang)
                counts = {g: 0 for g in languages if g != query_lang}
                slots = 0
                for query_idx in query_indices:
                    mask = (langs != query_lang) & (ids != ids[query_idx])
                    neighbors = safe_topk(sim[query_idx], mask, k)
                    for neighbor_lang in langs[neighbors]:
                        counts[neighbor_lang] += 1
                    slots += len(neighbors)
                for neighbor_lang, count in counts.items():
                    candidate_languages = len(languages) - 1
                    baseline = 1.0 / candidate_languages
                    rate = count / max(slots, 1)
                    neighbor_records.append({
                        "representation": representation, "layer": layer,
                        "query_lang": query_lang, "neighbor_lang": neighbor_lang,
                        "neighbor_rate": rate, "uniform_baseline": baseline,
                        "excess_neighbor_rate": rate - baseline,
                    })

            # Re-separation pool: allow same-language neighbors, but exclude the whole parallel group.
            language_purities = []
            for query_lang in languages:
                values = []
                for query_idx in np.flatnonzero(langs == query_lang):
                    mask = ids != ids[query_idx]
                    neighbors = safe_topk(sim[query_idx], mask, k)
                    values.append(float(np.mean(langs[neighbors] == query_lang)))
                mean_purity = float(np.mean(values))
                language_purities.append(mean_purity)
                purity_records.append({
                    "representation": representation, "layer": layer,
                    "lang": query_lang, "neighborhood_purity": mean_purity,
                    "purity_std": float(np.std(values, ddof=1)),
                    "n": len(values),
                    "uniform_baseline": 1.0 / len(languages),
                })

            centroids = np.stack([l2_normalize(x[langs == g].mean(axis=0, keepdims=True))[0] for g in languages])
            centroid_sim = centroids @ centroids.T
            distances = [1.0 - centroid_sim[i, j] for i, j in combinations(range(len(languages)), 2)]
            centroid_records.append({
                "representation": representation, "layer": layer,
                "centroid_separation": float(np.mean(distances)),
                "mean_neighborhood_purity": float(np.mean(language_purities)),
            })

    neighbor_df = pd.DataFrame(neighbor_records)
    asymmetry_records = []
    for representation in representations:
        rep_df = neighbor_df[neighbor_df.representation == representation]
        for layer in sorted(rep_df.layer.unique()):
            layer_df = rep_df[rep_df.layer == layer]
            for language in languages:
                if language == english:
                    continue
                toward_en = layer_df[(layer_df.query_lang == language) & (layer_df.neighbor_lang == english)]
                from_en = layer_df[(layer_df.query_lang == english) & (layer_df.neighbor_lang == language)]
                if len(toward_en) and len(from_en):
                    asymmetry_records.append({
                        "representation": representation,
                        "layer": layer,
                        "language": language,
                        "p_english_given_language": float(toward_en.neighbor_rate.iloc[0]),
                        "p_language_given_english": float(from_en.neighbor_rate.iloc[0]),
                        "english_asymmetry": float(toward_en.neighbor_rate.iloc[0] - from_en.neighbor_rate.iloc[0]),
                    })

    outputs = {
        "sanity_checks.csv": pd.DataFrame(sanity_records),
        "alignment_gain.csv": pd.DataFrame(alignment_records),
        "anchor_specificity.csv": pd.DataFrame(specificity_records),
        "neighbor_direction_matrix.csv": neighbor_df,
        "english_directional_asymmetry.csv": pd.DataFrame(asymmetry_records),
        "language_neighborhood_purity.csv": pd.DataFrame(purity_records),
        "centroid_separation.csv": pd.DataFrame(centroid_records),
        "semantic_retrieval.csv": pd.DataFrame(retrieval_records),
    }
    for filename, frame in outputs.items():
        frame.to_csv(Path(paths["metrics"]) / filename, index=False, encoding="utf-8")

    primary = metrics_cfg.get("primary_representation", "last_token")
    sanity = outputs["sanity_checks.csv"]
    align = outputs["alignment_gain.csv"]
    spec = outputs["anchor_specificity.csv"]
    neighbors = outputs["neighbor_direction_matrix.csv"]
    purity = outputs["language_neighborhood_purity.csv"]
    retrieval = outputs["semantic_retrieval.csv"]
    s = sanity[sanity.representation == primary]
    a = align[align.representation == primary].groupby("layer")["alignment_gain"].mean()
    en_spec = spec[(spec.representation == primary) & (spec.anchor_lang == english)].groupby("layer")["mean_specificity"].mean()
    en_hub = neighbors[(neighbors.representation == primary) & (neighbors.neighbor_lang == english)].groupby("layer")["excess_neighbor_rate"].mean()
    p = purity[purity.representation == primary].groupby("layer")["neighborhood_purity"].mean()
    r = retrieval[retrieval.representation == primary].groupby("layer")["recall_at_1"].mean()

    report = [
        "=== PILOT VALIDATION REPORT ===",
        f"Representations: {', '.join(representations)} (primary={primary})",
        f"Rows={len(meta)}, complete semantic groups={len(complete_ids)}, languages={languages}",
        f"Metadata checks: {'PASS' if not issues else 'WARN - ' + '; '.join(issues)}",
        "Input isolation: one sentence per forward pass; model.eval(); use_cache=False; no chat template added by this script",
        f"Cosine self-similarity max error: {float(np.max(np.abs(s.self_similarity - 1.0))):.3e}",
        f"Best mean AlignmentGain: {a.max():.4f} at layer {int(a.idxmax())}",
        f"Best mean cross-lingual Recall@1: {r.max():.3f} at layer {int(r.idxmax())}",
        f"Peak English specificity: {en_spec.max():.4f} at layer {int(en_spec.idxmax())}",
        f"Peak excess English hub attraction: {en_hub.max():.4f} at layer {int(en_hub.idxmax())}",
        f"Neighborhood purity minimum/final: {p.min():.3f} / {p.iloc[-1]:.3f}",
        f"Re-separation strength (final - minimum): {p.iloc[-1] - p.min():.3f}",
        "Interpretation guardrail: English proximity, hubness, and re-separation require separate evidence.",
    ]
    report_text = "\n".join(report)
    (Path(paths["metrics"]) / "validation_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nSaved metric tables to {paths['metrics']}")


if __name__ == "__main__":
    main()
