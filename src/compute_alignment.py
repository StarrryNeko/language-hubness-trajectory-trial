"""Compute paper_v1 semantic alignment from existing sentence hidden states."""

from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except (ImportError, PermissionError):
    def tqdm(iterable, **_kwargs):
        return iterable

from common import load_config
from paper_common import (
    bootstrap_mean_ci,
    derangement,
    ensure_paper_dirs,
    group_vectors,
    load_hidden_dataset,
    metric_record,
    paper_settings,
    semantic_id_hash,
    write_manifest,
)


def retrieval_hits(query, target, top_values=(1, 5)):
    """Return correct-pair retrieval hits with target-language-only candidates."""
    query = np.asarray(query, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if query.shape != target.shape or query.ndim != 2:
        raise ValueError("query and target must be equally shaped two-dimensional arrays")
    similarity = query @ target.T
    order = np.argsort(-similarity, axis=1, kind="stable")
    correct = np.arange(len(query))
    return {
        int(k): np.any(order[:, : min(int(k), len(target))] == correct[:, None], axis=1)
        for k in top_values
    }


def summarize_alignment_samples(samples, english, rng, n_boot, confidence):
    records = []
    for (representation, layer), layer_frame in samples.groupby(["representation", "layer"]):
        categories = {
            "all_pairs": np.ones(len(layer_frame), dtype=bool),
            "english_pairs": (layer_frame.lang_a == english) | (layer_frame.lang_b == english),
            "non_english_pairs": (layer_frame.lang_a != english) & (layer_frame.lang_b != english),
        }
        for category, mask in categories.items():
            selected = layer_frame.loc[mask]
            semantic_values = selected.groupby("semantic_id").alignment_gain.mean().to_numpy()
            records.append(metric_record({
                "representation": representation,
                "layer": int(layer),
                "pair_group": category,
                "language_pairs": int(selected[["lang_a", "lang_b"]].drop_duplicates().shape[0]),
            }, semantic_values, rng, n_boot, confidence))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="paper_v1 AlignmentGain and semantic retrieval")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = paper_settings(cfg)
    paths = ensure_paper_dirs(cfg)
    output = paths.metrics / "alignment"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_hidden_dataset(cfg)
    english = cfg["metrics"].get("english_language", "en")
    if english not in dataset.languages:
        raise ValueError(f"English language {english!r} is missing")
    rng = np.random.default_rng(settings["seed"] + 101)
    shuffle_rng = np.random.default_rng(settings["seed"] + 102)
    permutations = np.stack([
        derangement(shuffle_rng, len(dataset.semantic_ids))
        for _ in range(settings["shuffled_permutations"])
    ])
    if np.any(permutations == np.arange(len(dataset.semantic_ids))[None, :]):
        raise AssertionError("shuffled semantic baseline contains a fixed point")

    sample_records = []
    alignment_records = []
    retrieval_records = []
    n_layers = int(dataset.vectors.shape[1])
    for layer in tqdm(range(n_layers), desc="paper_v1 alignment"):
        groups = group_vectors(dataset, dataset.semantic_ids, layer, normalize=True)
        for lang_a_idx, lang_b_idx in combinations(range(len(dataset.languages)), 2):
            lang_a = dataset.languages[lang_a_idx]
            lang_b = dataset.languages[lang_b_idx]
            a = groups[:, lang_a_idx, :]
            b = groups[:, lang_b_idx, :]
            paired = np.einsum("sd,sd->s", a, b)
            shuffled = np.stack([
                np.einsum("sd,sd->s", a, b[permutation])
                for permutation in permutations
            ]).mean(axis=0)
            differences = paired - shuffled
            mean, low, high = bootstrap_mean_ci(
                differences, rng, settings["bootstrap_samples"], settings["confidence_level"]
            )
            alignment_records.append({
                "representation": "mean_pool",
                "layer": layer,
                "layer_role": "embedding" if layer == 0 else "transformer_block_output",
                "lang_a": lang_a,
                "lang_b": lang_b,
                "parallel_similarity": float(paired.mean()),
                "shuffled_similarity": float(shuffled.mean()),
                "alignment_gain": mean,
                "ci_lower": low,
                "ci_upper": high,
                "n_semantic_ids": len(differences),
                "n_shuffles": len(permutations),
            })
            sample_records.extend({
                "representation": "mean_pool",
                "layer": layer,
                "semantic_id": semantic_id,
                "lang_a": lang_a,
                "lang_b": lang_b,
                "parallel_similarity": float(paired[index]),
                "shuffled_similarity": float(shuffled[index]),
                "alignment_gain": float(differences[index]),
            } for index, semantic_id in enumerate(dataset.semantic_ids))

            for query_lang, target_lang, query, target in (
                (lang_a, lang_b, a, b),
                (lang_b, lang_a, b, a),
            ):
                hits = retrieval_hits(query, target)
                r1, r1_low, r1_high = bootstrap_mean_ci(
                    hits[1], rng, settings["bootstrap_samples"], settings["confidence_level"]
                )
                r5, r5_low, r5_high = bootstrap_mean_ci(
                    hits[5], rng, settings["bootstrap_samples"], settings["confidence_level"]
                )
                retrieval_records.append({
                    "representation": "mean_pool",
                    "layer": layer,
                    "layer_role": "embedding" if layer == 0 else "transformer_block_output",
                    "query_lang": query_lang,
                    "target_lang": target_lang,
                    "recall_at_1": r1,
                    "recall1_ci_lower": r1_low,
                    "recall1_ci_upper": r1_high,
                    "recall_at_5": r5,
                    "recall5_ci_lower": r5_low,
                    "recall5_ci_upper": r5_high,
                    "random_recall_at_1": 1 / len(dataset.semantic_ids),
                    "random_recall_at_5": min(5, len(dataset.semantic_ids)) / len(dataset.semantic_ids),
                    "n_semantic_ids": len(dataset.semantic_ids),
                    "candidate_scope": "target_language_only",
                })

    alignment = pd.DataFrame(alignment_records)
    samples = pd.DataFrame(sample_records)
    retrieval = pd.DataFrame(retrieval_records)
    summary = summarize_alignment_samples(
        samples, english, rng, settings["bootstrap_samples"], settings["confidence_level"]
    )
    alignment.to_csv(output / "alignment_gain.csv", index=False, encoding="utf-8")
    samples.to_csv(output / "alignment_gain_samples.csv", index=False, encoding="utf-8")
    retrieval.to_csv(output / "semantic_retrieval.csv", index=False, encoding="utf-8")
    summary.to_csv(output / "alignment_summary.csv", index=False, encoding="utf-8")
    write_manifest(output / "alignment_manifest.json", {
        "candidate_scope": "paired language pair; shuffled baseline changes semantic ID only",
        "retrieval_candidate_scope": "target_language_only",
        "bootstrap_unit": "semantic_id",
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "languages": dataset.languages,
        "language_pairs": len(dataset.languages) * (len(dataset.languages) - 1) // 2,
        "shuffled_permutations": settings["shuffled_permutations"],
        "shuffles_reused_across_language_pairs_and_models": True,
        "layer_zero_reported_separately": True,
        "output_files": [
            "alignment_gain.csv", "alignment_gain_samples.csv",
            "semantic_retrieval.csv", "alignment_summary.csv",
        ],
    })
    validation = {
        "module": "04_alignment",
        "status": "PASS",
        "checks": {
            "derangements_have_no_fixed_points": True,
            "bootstrap_unit_is_semantic_id": True,
            "equal_pair_sample_counts": bool(alignment.n_semantic_ids.nunique() == 1),
            "retrieval_target_language_only": True,
            "layer_zero_labeled": True,
        },
        "claim_boundary": "Alignment supports shared semantic structure, not functional causality.",
    }
    (paths.validation / "04_alignment.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved paper_v1 alignment outputs to {output}")


if __name__ == "__main__":
    main()

