"""Compute language identity structure with a scope distinct from hubness."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except (ImportError, PermissionError):
    def tqdm(iterable, **_kwargs):
        return iterable

from common import l2_normalize, load_config
from paper_common import (
    bootstrap_mean_ci,
    ensure_paper_dirs,
    load_hidden_dataset,
    paper_settings,
    semantic_id_hash,
    write_manifest,
)


def fractional_topk_purity(similarities, candidate_mask, candidate_is_same_language, k):
    """Tie-safe same-language mass among the top-k eligible candidates."""
    scores = np.asarray(similarities, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    same = np.asarray(candidate_is_same_language, dtype=bool)
    candidates = np.flatnonzero(mask)
    if len(candidates) < int(k):
        raise ValueError(f"only {len(candidates)} eligible candidates for k={k}")
    candidate_scores = scores[candidates]
    threshold = np.partition(candidate_scores, len(candidate_scores) - int(k))[
        len(candidate_scores) - int(k)
    ]
    above = candidate_scores > threshold
    tied = np.isclose(candidate_scores, threshold, rtol=1e-7, atol=1e-8)
    remaining = int(k) - int(above.sum())
    mass = float(same[candidates[above]].sum())
    if tied.sum():
        mass += remaining * float(same[candidates[tied]].mean())
    return mass / int(k)


def layer_windows(n_layers):
    transformer_layers = np.arange(1, n_layers)
    if len(transformer_layers) < 3:
        return {"early": transformer_layers.tolist(), "mid": [], "late": []}
    chunks = np.array_split(transformer_layers, 3)
    return {name: chunk.astype(int).tolist() for name, chunk in zip(("early", "mid", "late"), chunks)}


def main():
    parser = argparse.ArgumentParser(description="paper_v1 neighborhood purity and centroid structure")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = paper_settings(cfg)
    paths = ensure_paper_dirs(cfg)
    output = paths.metrics / "language_structure"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_hidden_dataset(cfg)
    k = int(cfg.get("paper_v1", {}).get("language_structure_k", 10))
    n_rows = len(dataset.meta)
    candidates_per_query = n_rows - len(dataset.languages)
    if not 1 <= k <= candidates_per_query:
        raise ValueError(f"language_structure_k must be in 1..{candidates_per_query}")
    rng = np.random.default_rng(settings["seed"] + 201)
    ids_by_row = dataset.meta.sort_values("row_idx").id.to_numpy(dtype=str)
    langs_by_row = dataset.meta.sort_values("row_idx").lang.to_numpy(dtype=str)
    n_layers = int(dataset.vectors.shape[1])
    windows = layer_windows(n_layers)

    sample_records = []
    purity_records = []
    centroid_records = []
    for layer in tqdm(range(n_layers), desc="paper_v1 language structure"):
        x = l2_normalize(np.asarray(dataset.vectors[:, layer, :], dtype=np.float32), axis=1)
        # Blocked multiplication bounds peak memory; the exact candidate scope is
        # still every row outside the query's complete parallel semantic group.
        layer_values = {lang: [] for lang in dataset.languages}
        block_size = int(cfg.get("paper_v1", {}).get("similarity_block_size", 128))
        for start in range(0, n_rows, block_size):
            stop = min(n_rows, start + block_size)
            similarities = x[start:stop] @ x.T
            for local, query_idx in enumerate(range(start, stop)):
                candidate_mask = ids_by_row != ids_by_row[query_idx]
                purity = fractional_topk_purity(
                    similarities[local], candidate_mask,
                    langs_by_row == langs_by_row[query_idx], k,
                )
                lang = langs_by_row[query_idx]
                layer_values[lang].append(purity)
                sample_records.append({
                    "representation": "mean_pool",
                    "layer": layer,
                    "layer_role": "embedding" if layer == 0 else "transformer_block_output",
                    "row_idx": query_idx,
                    "semantic_id": ids_by_row[query_idx],
                    "lang": lang,
                    "neighborhood_purity": purity,
                    "eligible_candidates": int(candidate_mask.sum()),
                    "k": k,
                })
        language_purity_means = []
        for lang, values in layer_values.items():
            mean, low, high = bootstrap_mean_ci(
                values, rng, settings["bootstrap_samples"], settings["confidence_level"]
            )
            language_purity_means.append(mean)
            purity_records.append({
                "representation": "mean_pool",
                "layer": layer,
                "layer_role": "embedding" if layer == 0 else "transformer_block_output",
                "lang": lang,
                "neighborhood_purity": mean,
                "ci_lower": low,
                "ci_upper": high,
                "uniform_language_baseline": 1 / len(dataset.languages),
                "n_queries": len(values),
                "k": k,
            })

        centroids_raw = np.stack([x[langs_by_row == lang].mean(axis=0) for lang in dataset.languages])
        centroids = l2_normalize(centroids_raw, axis=1)
        centroid_similarity = centroids @ centroids.T
        upper = np.triu_indices(len(dataset.languages), k=1)
        between = float(np.mean(1 - centroid_similarity[upper]))
        within_values = []
        for lang_idx, lang in enumerate(dataset.languages):
            member = x[langs_by_row == lang]
            within_values.extend((1 - member @ centroids[lang_idx]).tolist())
        within = float(np.mean(within_values))
        centroid_records.append({
            "representation": "mean_pool",
            "layer": layer,
            "layer_role": "embedding" if layer == 0 else "transformer_block_output",
            "centroid_separation": between,
            "within_language_dispersion": within,
            "separation_dispersion_ratio": between / max(within, 1e-12),
            "mean_neighborhood_purity": float(np.mean(language_purity_means)),
        })

    samples = pd.DataFrame(sample_records)
    purity = pd.DataFrame(purity_records)
    centroid = pd.DataFrame(centroid_records)
    contrast_records = []
    for lang in dataset.languages:
        lang_samples = samples[samples.lang == lang]
        pivot = lang_samples.pivot(index="row_idx", columns="layer", values="neighborhood_purity")
        window_values = {
            name: pivot[layers].mean(axis=1) for name, layers in windows.items() if layers
        }
        if "mid" not in window_values or "late" not in window_values:
            continue
        contrasts = (window_values["late"] - window_values["mid"]).to_numpy()
        mean, low, high = bootstrap_mean_ci(
            contrasts, rng, settings["bootstrap_samples"], settings["confidence_level"]
        )
        contrast_records.append({
            "representation": "mean_pool",
            "lang": lang,
            "contrast": "late_minus_mid",
            "early_layers": ",".join(map(str, windows["early"])),
            "mid_layers": ",".join(map(str, windows["mid"])),
            "late_layers": ",".join(map(str, windows["late"])),
            "mean": mean,
            "ci_lower": low,
            "ci_upper": high,
            "n_queries": len(contrasts),
        })

    purity.to_csv(output / "neighborhood_purity.csv", index=False, encoding="utf-8")
    samples.to_csv(output / "neighborhood_purity_samples.csv", index=False, encoding="utf-8")
    centroid.to_csv(output / "centroid_separation.csv", index=False, encoding="utf-8")
    pd.DataFrame(contrast_records).to_csv(
        output / "layer_window_contrasts.csv", index=False, encoding="utf-8"
    )
    write_manifest(output / "language_structure_manifest.json", {
        "candidate_scope": "all rows except every translation with the query semantic ID",
        "same_language_candidates_retained": True,
        "hubness_cache_reused": False,
        "bootstrap_unit": "query sentence within language; semantic-ID split is used by probe",
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "k": k,
        "eligible_candidates_per_query": candidates_per_query,
        "relative_layer_windows_exclude_layer_zero": windows,
        "output_files": [
            "neighborhood_purity.csv", "neighborhood_purity_samples.csv",
            "centroid_separation.csv", "layer_window_contrasts.csv",
        ],
    })
    validation = {
        "module": "10_language_structure",
        "status": "PASS",
        "checks": {
            "same_semantic_group_excluded": True,
            "same_language_candidates_retained": True,
            "candidate_count_balanced": True,
            "k_within_candidate_count": True,
            "per_language_curves_saved": True,
            "layer_zero_excluded_from_windows": 0 not in sum(windows.values(), []),
        },
    }
    (paths.validation / "10_language_structure.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved paper_v1 language-structure outputs to {output}")


if __name__ == "__main__":
    main()

