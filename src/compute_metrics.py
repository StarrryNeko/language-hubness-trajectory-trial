import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except (ImportError, PermissionError):  # Metrics remain runnable in minimal audit environments.
    def tqdm(iterable, **_kwargs):
        return iterable

from common import (
    configured_representations,
    ensure_dirs,
    l2_normalize,
    load_config,
    representation_file_map,
    validate_language_inventory,
)
from numerical_validation import (
    require_finite,
    require_nonzero_row_norms,
    validate_representation_array,
    validate_similarity_matrix,
)
from evidence_rules import joint_positive_layers, max_consecutive_layers


def bootstrap_mean_ci(values, rng, n_boot=500, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("bootstrap_mean_ci: values must not be empty")
    require_finite(values, "bootstrap_mean_ci input")
    mean = float(values.mean())
    if len(values) == 1 or n_boot <= 0:
        return mean, mean, mean
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return mean, float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def rank_percentiles(values):
    """Return 1 for the largest value and 0 for the smallest, with average tie ranks."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    order = pd.Series(-values).rank(method="average").to_numpy()
    return (n - order) / max(1, n - 1)


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or np.allclose(values, 0):
        return 0.0
    differences = np.abs(values[:, None] - values[None, :]).sum()
    return float(differences / (2 * len(values) * values.sum()))


def skewness(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    return float(np.mean(((values - values.mean()) / std) ** 3)) if std > 0 else 0.0


def locally_scaled_similarity(cosine, density_k):
    """Multiway CSLS-style control computed only inside one semantic group."""
    cosine = validate_similarity_matrix(cosine, "local-scaled cosine input")
    n = cosine.shape[0]
    if not 1 <= int(density_k) < n:
        raise ValueError(f"density_k must be in 1..{n - 1}, got {density_k}")
    masked = cosine.copy()
    np.fill_diagonal(masked, -np.inf)
    take = min(max(1, density_k), n - 1)
    local_density = np.partition(masked, n - take, axis=1)[:, -take:].mean(axis=1)
    adjusted = 2 * cosine - local_density[:, None] - local_density[None, :]
    np.fill_diagonal(adjusted, 1.0)
    return validate_similarity_matrix(adjusted, "local-scaled cosine output")


def group_statistics(similarity, k):
    """Compute graph statistics for one parallel semantic group (languages x languages)."""
    similarity = validate_similarity_matrix(similarity, "group_statistics similarity")
    n = similarity.shape[0]
    if not 1 <= int(k) < n:
        raise ValueError(f"k must be in 1..{n - 1}, got {k}")
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    take = int(k)
    selected = np.zeros((n, n), dtype=np.float64)
    # Fractionally split the final slots across boundary ties. This preserves
    # exactly k mass per query and prevents array/language order from creating
    # artificial hubs when a layer's geometry is collapsed or quantized.
    for row in range(n):
        scores = masked[row]
        threshold = np.partition(scores, n - take)[n - take]
        tied = np.isclose(scores, threshold, rtol=1e-7, atol=1e-8)
        above = (scores > threshold) & ~tied
        selected[row, above] = 1.0
        remaining = take - int(above.sum())
        tie_count = int(tied.sum())
        if tie_count:
            selected[row, tied] = remaining / tie_count
    occurrence = selected.sum(axis=0)
    centrality = np.nanmean(np.where(np.eye(n, dtype=bool), np.nan, similarity), axis=1)
    percentile = rank_percentiles(centrality)
    max_centrality = centrality.max()
    medoid = np.isclose(centrality, max_centrality).astype(np.float64)
    medoid /= medoid.sum()
    row_mass = selected.sum(axis=1)
    if not np.allclose(row_mass, take, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"kNN mass conservation failed per query: expected={take}, "
            f"range={row_mass.min()}..{row_mass.max()}"
        )
    total_mass = float(occurrence.sum())
    expected_mass = float(n * take)
    if not np.isclose(total_mass, expected_mass, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"kNN occurrence mass conservation failed: expected={expected_mass}, got={total_mass}"
        )
    for name, values in {
        "selected": selected,
        "occurrence": occurrence,
        "centrality": centrality,
        "percentile": percentile,
        "medoid": medoid,
    }.items():
        require_finite(values, f"group_statistics {name}")
    if not np.isclose(medoid.sum(), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"medoid mass conservation failed: expected=1, got={medoid.sum()}")
    return selected, occurrence, centrality, percentile, medoid


def metric_record(prefix, values, rng, n_boot, confidence):
    mean, low, high = bootstrap_mean_ci(values, rng, n_boot, confidence)
    return {**prefix, "mean": mean, "ci_lower": low, "ci_upper": high, "n_semantic_ids": len(values)}


def validate_metadata(meta, languages):
    required = {"row_idx", "id", "lang", "was_truncated"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")
    truncated = meta.was_truncated.astype(str).str.lower().isin({"true", "1", "yes"})
    if truncated.any():
        raise ValueError(f"{int(truncated.sum())} inputs were truncated; rerun with a larger max_length")
    if meta.duplicated(["id", "lang"]).any():
        raise ValueError("metadata contains duplicate (semantic ID, language) rows")
    if sorted(meta.row_idx.astype(int).tolist()) != list(range(len(meta))):
        raise ValueError("metadata.row_idx must be a complete 0..N-1 mapping to vector rows")
    expected = set(languages)
    invalid = [
        str(semantic_id)
        for semantic_id, group in meta.groupby(meta.id.astype(str))
        if set(group.lang.astype(str)) != expected
    ]
    if invalid:
        raise ValueError(f"{len(invalid)} semantic groups are not complete; first IDs: {invalid[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Same-semantics-only multilingual hubness metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--result-tag", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    configured_languages = validate_language_inventory(cfg)
    representations = configured_representations(cfg)
    paths = ensure_dirs(cfg)
    metrics_dir = Path(paths["metrics"])
    if args.result_tag:
        metrics_dir = metrics_dir / args.result_tag
        metrics_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(Path(paths["hidden"]) / "metadata.csv")
    languages = list(configured_languages)
    validate_metadata(meta, languages)
    english = cfg["metrics"].get("english_language", "en")
    if english not in languages:
        raise ValueError(f"English language {english!r} is not configured")
    k = int(args.k if args.k is not None else cfg["metrics"].get("nearest_neighbors_k", 5))
    if not 1 <= k < len(languages):
        raise ValueError(f"k must be in 1..{len(languages) - 1}")
    n_boot = int(cfg["metrics"].get("bootstrap_samples", 500))
    confidence = float(cfg["metrics"].get("confidence_level", 0.95))
    density_k = int(cfg.get("similarity_controls", {}).get("local_scaling_k", k))
    methods = list(cfg.get("similarity_controls", {}).get(
        "methods", ["cosine", "local_scaled_cosine"]
    ))
    if any(method not in {"cosine", "local_scaled_cosine"} for method in methods):
        raise ValueError("similarity methods must be cosine and/or local_scaled_cosine")
    rng = np.random.default_rng(int(cfg.get("seed", 42)) + k)

    semantic_ids = sorted(meta.id.astype(str).unique())
    id_indices = {
        semantic_id: np.array([
            int(meta[(meta.id.astype(str) == semantic_id) & (meta.lang.astype(str) == lang)].row_idx.iloc[0])
            for lang in languages
        ])
        for semantic_id in semantic_ids
    }
    lang_index = {lang: index for index, lang in enumerate(languages)}
    en_index = lang_index[english]
    language_metadata = cfg["dataset"].get("language_metadata", {})

    pair_records = []
    knn_records = []
    hub_records = []
    global_records = []
    english_evidence_records = []
    english_source_records = []
    pair_samples = {}

    for representation in representations:
        vector_path = Path(paths["hidden"]) / representation_file_map()[representation]
        vectors = np.load(vector_path, mmap_mode="r")
        validate_representation_array(vectors, len(meta), f"representation={representation}")
        n_layers = vectors.shape[1]

        for layer in tqdm(range(n_layers), desc=f"Same-semantics metrics ({representation}, k={k})"):
            raw_by_id = {}
            for semantic_id in semantic_ids:
                group = np.asarray(vectors[id_indices[semantic_id], layer, :], dtype=np.float32)
                context = f"representation={representation} semantic_id={semantic_id} layer={layer}"
                require_nonzero_row_norms(group, context)
                normalized = l2_normalize(group, axis=1)
                cosine = normalized @ normalized.T
                raw_by_id[semantic_id] = validate_similarity_matrix(cosine, f"{context} cosine")

            upper = np.triu_indices(len(languages), 1)
            pair_samples[(representation, layer)] = np.concatenate([
                raw_by_id[semantic_id][upper] for semantic_id in semantic_ids
            ])
            for i, j in combinations(range(len(languages)), 2):
                values = [raw_by_id[semantic_id][i, j] for semantic_id in semantic_ids]
                pair_records.append(metric_record({
                    "representation": representation,
                    "layer": layer,
                    "lang_a": languages[i],
                    "lang_b": languages[j],
                    "similarity_method": "cosine",
                }, values, rng, n_boot, confidence))

            method_matrices = {
                "cosine": raw_by_id,
                "local_scaled_cosine": {
                    semantic_id: locally_scaled_similarity(raw_by_id[semantic_id], density_k)
                    for semantic_id in semantic_ids
                },
            }
            for method in methods:
                selected_samples = []
                occurrence_samples = []
                centrality_samples = []
                percentile_samples = []
                medoid_samples = []
                for semantic_id in semantic_ids:
                    stats = group_statistics(method_matrices[method][semantic_id], k)
                    selected, occurrence, centrality, percentile, medoid = stats
                    selected_samples.append(selected)
                    occurrence_samples.append(occurrence)
                    centrality_samples.append(centrality)
                    percentile_samples.append(percentile)
                    medoid_samples.append(medoid)
                selected_samples = np.stack(selected_samples)
                occurrence_samples = np.stack(occurrence_samples)
                centrality_samples = np.stack(centrality_samples)
                percentile_samples = np.stack(percentile_samples)
                medoid_samples = np.stack(medoid_samples)

                for source_idx, source_lang in enumerate(languages):
                    for candidate_idx, candidate_lang in enumerate(languages):
                        if source_idx == candidate_idx:
                            continue
                        values = selected_samples[:, source_idx, candidate_idx]
                        knn_records.append(metric_record({
                            "representation": representation,
                            "similarity_method": method,
                            "layer": layer,
                            "source_lang": source_lang,
                            "candidate_lang": candidate_lang,
                            "balanced_selection_baseline": k / (len(languages) - 1),
                            "k": k,
                        }, values, rng, n_boot, confidence))

                for candidate_idx, candidate_lang in enumerate(languages):
                    prefix = {
                        "representation": representation,
                        "similarity_method": method,
                        "layer": layer,
                        "candidate_lang": candidate_lang,
                        "candidate_family": language_metadata.get(candidate_lang, {}).get("family", "unknown"),
                        "candidate_script": language_metadata.get(candidate_lang, {}).get("script", "unknown"),
                        "k": k,
                    }
                    measures = {
                        "k_occurrence": occurrence_samples[:, candidate_idx],
                        "centrality": centrality_samples[:, candidate_idx],
                        "centrality_rank_percentile": percentile_samples[:, candidate_idx],
                        "medoid_rate": medoid_samples[:, candidate_idx],
                    }
                    for metric_name, values in measures.items():
                        hub_records.append(metric_record(
                            {**prefix, "metric": metric_name}, values, rng, n_boot, confidence
                        ))

                per_group_skew = np.array([skewness(row) for row in occurrence_samples])
                per_group_gini = np.array([gini(row) for row in occurrence_samples])
                per_group_max_share = occurrence_samples.max(axis=1) / (len(languages) * k)
                per_group_tie_rate = np.mean(
                    np.any((selected_samples > 0) & (selected_samples < 1), axis=2), axis=1
                )
                for metric_name, values in {
                    "k_occurrence_skewness": per_group_skew,
                    "k_occurrence_gini": per_group_gini,
                    "largest_hub_share": per_group_max_share,
                    "topk_boundary_tie_rate": per_group_tie_rate,
                }.items():
                    global_records.append(metric_record({
                        "representation": representation,
                        "similarity_method": method,
                        "layer": layer,
                        "metric": metric_name,
                        "k": k,
                    }, values, rng, n_boot, confidence))

                english_measures = {
                    "k_occurrence_excess": occurrence_samples[:, en_index] - k,
                    "centrality_advantage": centrality_samples[:, en_index] - np.delete(
                        centrality_samples, en_index, axis=1
                    ).mean(axis=1),
                    "rank_percentile_advantage": percentile_samples[:, en_index] - np.delete(
                        percentile_samples, en_index, axis=1
                    ).mean(axis=1),
                    "medoid_rate_excess": medoid_samples[:, en_index] - 1 / len(languages),
                }
                for metric_name, values in english_measures.items():
                    english_evidence_records.append(metric_record({
                        "representation": representation,
                        "similarity_method": method,
                        "layer": layer,
                        "metric": metric_name,
                        "english_language": english,
                        "null_value": 0.0,
                        "k": k,
                    }, values, rng, n_boot, confidence))

                for source_idx, source_lang in enumerate(languages):
                    if source_lang == english:
                        continue
                    selected_values = selected_samples[:, source_idx, en_index]
                    english_source_records.append(metric_record({
                        "representation": representation,
                        "similarity_method": method,
                        "layer": layer,
                        "source_lang": source_lang,
                        "source_family": language_metadata.get(source_lang, {}).get("family", "unknown"),
                        "source_script": language_metadata.get(source_lang, {}).get("script", "unknown"),
                        "metric": "english_topk_selection_rate",
                        "balanced_selection_baseline": k / (len(languages) - 1),
                        "k": k,
                    }, selected_values, rng, n_boot, confidence))

    agreement_records = []
    primary = cfg["metrics"].get("primary_representation", "mean_pool")
    validation = cfg["metrics"].get("validation_representation", "sentinel_eos")
    if primary in representations and validation in representations:
        layers = sorted({layer for rep, layer in pair_samples if rep == primary})
        for layer in layers:
            a = pair_samples[(primary, layer)]
            b = pair_samples[(validation, layer)]
            correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
            primary_evidence = [
                row for row in english_evidence_records
                if row["representation"] == primary and row["similarity_method"] == "cosine" and row["layer"] == layer
            ]
            validation_evidence = [
                row for row in english_evidence_records
                if row["representation"] == validation and row["similarity_method"] == "cosine" and row["layer"] == layer
            ]
            a_values = {row["metric"]: row["mean"] for row in primary_evidence}
            b_values = {row["metric"]: row["mean"] for row in validation_evidence}
            agreement_records.append({
                "layer": layer,
                "primary_representation": primary,
                "validation_representation": validation,
                "pairwise_similarity_pearson": correlation,
                "english_evidence_sign_agreement": all(
                    np.sign(a_values[key]) == np.sign(b_values[key]) for key in a_values.keys() & b_values.keys()
                ),
            })

    source_frame = pd.DataFrame(english_source_records)
    breadth_records = []
    for (representation, method, layer), group in source_frame.groupby(
        ["representation", "similarity_method", "layer"]
    ):
        supported = group[group.ci_lower > group.balanced_selection_baseline]
        breadth_records.append({
            "representation": representation,
            "similarity_method": method,
            "layer": layer,
            "supported_source_languages": int(supported.source_lang.nunique()),
            "total_source_languages": int(group.source_lang.nunique()),
            "supported_source_families": int(supported.source_family.nunique()),
            "supported_source_scripts": int(supported.source_script.nunique()),
            "supported_non_latin_languages": int(
                supported[supported.source_script != "Latin"].source_lang.nunique()
            ),
        })

    outputs = {
        "within_semantic_pair_similarity.csv": pd.DataFrame(pair_records),
        "within_semantic_knn.csv": pd.DataFrame(knn_records),
        "hubness_by_language.csv": pd.DataFrame(hub_records),
        "hubness_global.csv": pd.DataFrame(global_records),
        "english_hubness_evidence.csv": pd.DataFrame(english_evidence_records),
        "english_source_group_attraction.csv": pd.DataFrame(english_source_records),
        "english_hubness_breadth.csv": pd.DataFrame(breadth_records),
        "representation_agreement.csv": pd.DataFrame(agreement_records),
    }
    for filename, frame in outputs.items():
        frame.to_csv(metrics_dir / filename, index=False, encoding="utf-8")

    evidence = outputs["english_hubness_evidence.csv"]
    main_evidence = evidence[
        (evidence.representation == primary) & (evidence.similarity_method == "cosine")
    ]
    joint_layers = joint_positive_layers(main_evidence)
    joint_run = max_consecutive_layers(joint_layers)
    supported = main_evidence.assign(supported=main_evidence.ci_lower > 0).groupby("metric").supported.sum()
    max_layers = main_evidence.groupby("metric").apply(
        lambda group: int(group.loc[group["mean"].idxmax(), "layer"]), include_groups=False
    )
    report = [
        "=== SAME-SEMANTICS MULTILINGUAL HUBNESS REPORT ===",
        f"Rows={len(meta)}; semantic_groups={len(semantic_ids)}; languages={len(languages)}; k={k}",
        "Candidate scope: same semantic ID only (cross-semantic candidates are never compared)",
        f"Primary={primary}; EOS validation={validation}; similarity controls={methods}",
        "English positive-CI layer counts: " + ", ".join(
            f"{name}={int(count)}" for name, count in supported.items()
        ),
        f"English four-metric joint-positive layers: {joint_layers}; longest run={joint_run}",
        "English evidence peak layers: " + ", ".join(
            f"{name}={int(layer)}" for name, layer in max_layers.items()
        ),
        "Interpretation rule: call English a hub only when k-occurrence, centrality rank/medoid evidence, "
        "source-language breadth, and EOS/local-scaling controls agree over a stable layer interval.",
    ]
    (metrics_dir / "research_summary.txt").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "candidate_scope": "same_semantic_id_only",
        "cross_semantic_similarity_computed": False,
        "semantic_groups": len(semantic_ids),
        "languages": languages,
        "k": k,
        "representations": representations,
        "similarity_methods": methods,
        "bootstrap_unit": "semantic_id",
        "joint_evidence_rule": "all_four_ci_lower_gt_zero_on_same_layer",
        "output_files": list(outputs),
    }
    (metrics_dir / "metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "\n".join(report))
    print(f"Saved {len(outputs)} metric tables and metrics_manifest.json to {metrics_dir}")


if __name__ == "__main__":
    main()
