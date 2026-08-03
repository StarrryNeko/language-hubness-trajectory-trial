"""Competitive cosine analysis that cannot hide a stronger non-English candidate."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import load_config
from compute_metrics import locally_scaled_similarity
from evidence_rules import max_consecutive_layers
from paper_common import (
    ensure_paper_dirs,
    group_vectors,
    load_hidden_dataset,
    metric_record,
    paper_settings,
    semantic_id_hash,
    write_manifest,
)


def candidate_competition(similarities, english_index):
    """Return English-vs-competitor values for every non-English source.

    ``similarities`` has shape semantic x source-language x candidate-language.
    Self-candidates are excluded. Ties receive half credit and average ranks.
    """
    similarities = np.asarray(similarities, dtype=np.float64)
    if similarities.ndim != 3 or similarities.shape[1] != similarities.shape[2]:
        raise ValueError("similarities must have shape (semantic, language, language)")
    n_semantics, n_languages, _ = similarities.shape
    if not 0 <= int(english_index) < n_languages:
        raise ValueError("english_index is out of range")
    sources = np.array([index for index in range(n_languages) if index != english_index])
    english_scores = similarities[:, sources, english_index]
    best_values = np.empty((n_semantics, len(sources)), dtype=np.float64)
    best_indices = np.empty((n_semantics, len(sources)), dtype=np.int16)
    ranks = np.empty_like(best_values)
    win_rates = np.empty_like(best_values)
    for column, source in enumerate(sources):
        competitors = np.array([
            index for index in range(n_languages)
            if index not in {int(source), int(english_index)}
        ])
        values = similarities[:, source, :][:, competitors]
        best_local = np.argmax(values, axis=1)
        best_values[:, column] = values[np.arange(n_semantics), best_local]
        best_indices[:, column] = competitors[best_local]
        english = english_scores[:, column, None]
        greater = (english > values).sum(axis=1)
        ties = np.isclose(english, values, rtol=1e-7, atol=1e-8).sum(axis=1)
        win_rates[:, column] = (greater + 0.5 * ties) / len(competitors)
        ranks[:, column] = 1 + (values > english).sum(axis=1) + 0.5 * ties
    return {
        "source_indices": sources,
        "english_similarity": english_scores,
        "best_non_english_similarity": best_values,
        "hard_margin": english_scores - best_values,
        "pairwise_win_rate": win_rates,
        "english_rank": ranks,
        "best_non_english_index": best_indices,
    }


def supported_layers(frame, method, min_run):
    selected = frame[frame.similarity_method == method]
    hard = set(selected.loc[
        (selected.metric == "hard_margin") & (selected.ci_lower > 0), "layer"
    ].astype(int))
    wins = set(selected.loc[
        (selected.metric == "pairwise_win_rate") & (selected.ci_lower > 0.5), "layer"
    ].astype(int))
    layers = sorted(hard & wins)
    return layers, max_consecutive_layers(layers) >= int(min_run)


def descriptive_record(prefix, values):
    """Compact cell/source description; confirmatory CIs stay at semantic-ID aggregate level."""
    values = np.asarray(values, dtype=np.float64)
    return {
        **prefix,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "n_semantic_ids": int(len(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Competitive cosine similarity for paper_v1")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = paper_settings(cfg)
    paths = ensure_paper_dirs(cfg)
    output = paths.metrics / "similarity_competition"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_hidden_dataset(cfg)
    languages = dataset.languages
    english = cfg["metrics"].get("english_language", "en")
    english_index = languages.index(english)
    density_k = int(cfg.get("similarity_controls", {}).get("local_scaling_k", 5))
    methods = ("cosine", "local_scaled_cosine")
    n_semantics = len(dataset.semantic_ids)
    n_layers = int(dataset.vectors.shape[1])
    source_indices = np.array([index for index in range(len(languages)) if index != english_index])
    arrays = {
        f"{method}__{metric}": np.empty(
            (n_semantics, n_layers, len(source_indices)), dtype=np.float32
        )
        for method in methods
        for metric in (
            "english_similarity", "best_non_english_similarity", "hard_margin",
            "pairwise_win_rate", "english_rank",
        )
    }
    best_index_arrays = {
        f"{method}__best_non_english_index": np.empty(
            (n_semantics, n_layers, len(source_indices)), dtype=np.int16
        )
        for method in methods
    }
    rng = np.random.default_rng(settings["seed"] + 701)
    aggregate_records = []
    source_records = []
    matrix_records = []
    candidate_records = []

    for layer in range(n_layers):
        groups = group_vectors(dataset, dataset.semantic_ids, layer, normalize=True)
        raw = groups @ np.swapaxes(groups, 1, 2)
        matrices = {
            "cosine": raw,
            "local_scaled_cosine": np.stack([
                locally_scaled_similarity(matrix, density_k) for matrix in raw
            ]),
        }
        for method, matrix in matrices.items():
            result = candidate_competition(matrix, english_index)
            for metric in (
                "english_similarity", "best_non_english_similarity", "hard_margin",
                "pairwise_win_rate", "english_rank",
            ):
                arrays[f"{method}__{metric}"][:, layer, :] = result[metric].astype(np.float32)
            best_index_arrays[f"{method}__best_non_english_index"][:, layer, :] = result[
                "best_non_english_index"
            ]
            for metric in (
                "english_similarity", "best_non_english_similarity", "hard_margin",
                "pairwise_win_rate", "english_rank",
            ):
                per_semantic = result[metric].mean(axis=1)
                aggregate_records.append(metric_record({
                    "representation": "mean_pool",
                    "similarity_method": method,
                    "layer": layer,
                    "metric": metric,
                    "aggregation": "mean_across_non_english_sources_per_semantic_id",
                }, per_semantic, rng, settings["bootstrap_samples"], settings["confidence_level"]))
                for source_column, source_index in enumerate(source_indices):
                    source_records.append(descriptive_record({
                        "representation": "mean_pool",
                        "similarity_method": method,
                        "layer": layer,
                        "source_lang": languages[int(source_index)],
                        "metric": metric,
                    }, result[metric][:, source_column]))

            for source_index, source_lang in enumerate(languages):
                candidate_means = []
                for candidate_index, candidate_lang in enumerate(languages):
                    if source_index == candidate_index:
                        continue
                    values = matrix[:, source_index, candidate_index]
                    candidate_means.append((candidate_index, float(values.mean())))
                    matrix_records.append(descriptive_record({
                        "representation": "mean_pool",
                        "similarity_method": method,
                        "layer": layer,
                        "source_lang": source_lang,
                        "candidate_lang": candidate_lang,
                    }, values))
                mean_map = dict(candidate_means)
                ordered = sorted(candidate_means, key=lambda item: (-item[1], item[0]))
                rank_map = {candidate: rank + 1 for rank, (candidate, _) in enumerate(ordered)}
                for candidate_index, mean_value in candidate_means:
                    candidate_records.append({
                        "representation": "mean_pool",
                        "similarity_method": method,
                        "layer": layer,
                        "source_lang": source_lang,
                        "candidate_lang": languages[candidate_index],
                        "mean_similarity": mean_value,
                        "candidate_rank": rank_map[candidate_index],
                        "is_english": candidate_index == english_index,
                        "english_minus_candidate": (
                            np.nan if candidate_index == english_index or source_index == english_index
                            else mean_map.get(english_index, np.nan) - mean_value
                        ),
                    })

    aggregate = pd.DataFrame(aggregate_records)
    source_summary = pd.DataFrame(source_records)
    matrix_summary = pd.DataFrame(matrix_records)
    candidates = pd.DataFrame(candidate_records)
    aggregate.to_csv(output / "english_competition_by_layer.csv", index=False, encoding="utf-8")
    source_summary.to_csv(
        output / "english_competition_by_layer_source.csv", index=False, encoding="utf-8"
    )
    matrix_summary.to_csv(
        output / "source_candidate_similarity.csv", index=False, encoding="utf-8"
    )
    candidates.to_csv(output / "candidate_ranks_by_source.csv", index=False, encoding="utf-8")
    np.savez_compressed(output / "english_competition_samples.npz", **{
        "semantic_ids": np.asarray(dataset.semantic_ids, dtype="U"),
        "languages": np.asarray(languages, dtype="U"),
        "source_indices": source_indices.astype(np.int16),
        "layers": np.arange(n_layers, dtype=np.int16),
        **arrays,
        **best_index_arrays,
    })
    min_run = int(cfg["metrics"].get("min_consecutive_layers", 3))
    raw_layers, raw_supported = supported_layers(aggregate, "cosine", min_run)
    density_layers, density_supported = supported_layers(
        aggregate, "local_scaled_cosine", min_run
    )
    status = {
        "status": (
            "ROBUST" if raw_supported and density_supported
            else "DENSITY_SENSITIVE" if raw_supported
            else "NOT_SUPPORTED"
        ),
        "raw_supported_layers": raw_layers,
        "local_scaled_supported_layers": density_layers,
        "required_consecutive_layers": min_run,
    }
    (output / "similarity_competition_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(output / "similarity_competition_manifest.json", {
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "bootstrap_unit": "semantic_id",
        "candidate_scope": "same_semantic_id_only",
        "english_language": english,
        "methods": list(methods),
        "hard_margin_reference": "strongest non-English candidate excluding the source",
        "tie_policy": "half-credit pairwise wins and average candidate rank",
        "status": status,
        "output_files": [
            "english_competition_by_layer.csv",
            "english_competition_by_layer_source.csv",
            "source_candidate_similarity.csv",
            "candidate_ranks_by_source.csv",
            "english_competition_samples.npz",
            "similarity_competition_status.json",
        ],
    })
    (paths.validation / "15_similarity_competition.json").write_text(
        json.dumps({
            "module": "15_similarity_competition",
            "status": "PASS",
            "checks": {
                "all_sources_present": source_summary.source_lang.nunique() == len(languages) - 1,
                "all_candidates_present": candidates.candidate_lang.nunique() == len(languages),
                "hard_margin_present": "hard_margin" in set(aggregate.metric),
                "pairwise_win_rate_present": "pairwise_win_rate" in set(aggregate.metric),
                "sample_arrays_finite": all(np.isfinite(value).all() for value in arrays.values()),
            },
            "scientific_status": status,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved competitive similarity outputs to {output}; status={status['status']}")


if __name__ == "__main__":
    main()
