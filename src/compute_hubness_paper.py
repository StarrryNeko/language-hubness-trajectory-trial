"""Selection-corrected 24-language hubness analysis for paper_v1."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except (ImportError, PermissionError):
    def tqdm(iterable, **_kwargs):
        return iterable

from common import load_config
from compute_geometry_controls import crossfit_geometry_groups
from compute_metrics import group_statistics, locally_scaled_similarity
from evidence_rules import classify_model_status, max_consecutive_layers
from paper_common import (
    ensure_paper_dirs,
    group_vectors,
    load_hidden_dataset,
    metric_record,
    paper_settings,
    semantic_id_hash,
    write_manifest,
)


HUB_METRICS = (
    "k_occurrence_excess",
    "centrality_advantage",
    "rank_percentile_advantage",
    "medoid_rate_excess",
)


def candidate_measure_samples(occurrence, centrality, percentile, medoid, k):
    n_languages = occurrence.shape[1]
    centrality_other = (centrality.sum(axis=1, keepdims=True) - centrality) / (n_languages - 1)
    percentile_other = (percentile.sum(axis=1, keepdims=True) - percentile) / (n_languages - 1)
    return {
        "k_occurrence_excess": occurrence - int(k),
        "centrality_advantage": centrality - centrality_other,
        "rank_percentile_advantage": percentile - percentile_other,
        "medoid_rate_excess": medoid - 1 / n_languages,
    }


def make_label_permutations(rng, n_permutations, n_semantics, n_languages):
    permutations = np.empty((n_permutations, n_semantics, n_languages), dtype=np.int16)
    for permutation in range(n_permutations):
        for semantic in range(n_semantics):
            permutations[permutation, semantic] = rng.permutation(n_languages)
    return permutations


def max_target_null(samples, permutations):
    samples = np.asarray(samples, dtype=np.float64)
    relabeled = np.take_along_axis(samples[None, :, :], permutations, axis=2)
    candidate_means = relabeled.mean(axis=1)
    return candidate_means.max(axis=1)


def breadth_supported(frame):
    return (
        (frame.supported_source_languages >= np.ceil(frame.total_source_languages / 2))
        & (frame.supported_source_scripts >= 4)
        & (frame.supported_non_latin_languages >= 3)
    )


def joint_layers(evidence, language, method):
    selected = evidence[
        (evidence.candidate_lang == language) & (evidence.similarity_method == method)
    ]
    layers = []
    for layer, group in selected.groupby("layer"):
        if set(group.metric) == set(HUB_METRICS) and bool((group.ci_lower > 0).all()):
            layers.append(int(layer))
    return sorted(layers)


def main():
    parser = argparse.ArgumentParser(description="paper_v1 rotated-target hubness")
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--result-tag", default=None)
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip large per-semantic artifacts (used by non-primary k sweep runs).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = paper_settings(cfg)
    paths = ensure_paper_dirs(cfg)
    output = paths.metrics / "hubness"
    if args.result_tag:
        output = output / args.result_tag
    output.mkdir(parents=True, exist_ok=True)
    sample_array_dir = output / "sample_arrays"
    sample_array_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_hidden_dataset(cfg)
    languages = dataset.languages
    language_metadata = cfg["dataset"].get("language_metadata", {})
    english = cfg["metrics"].get("english_language", "en")
    english_index = languages.index(english)
    k = int(args.k if args.k is not None else cfg["metrics"].get("nearest_neighbors_k", 5))
    if not 1 <= k < len(languages):
        raise ValueError(f"k must be in 1..{len(languages) - 1}")
    density_k = int(cfg.get("similarity_controls", {}).get("local_scaling_k", k))
    remove_counts = sorted(set(map(int, cfg.get("paper_v1", {}).get("geometry_remove_pcs", [1, 3, 5]))))
    methods = ["cosine", "local_scaled_cosine", "centered_cosine", *[
        f"remove_pc_{value}" for value in remove_counts
    ]]
    cache_signature = hashlib.sha256(json.dumps({
        "protocol": "paper_v1",
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "methods": methods,
        "density_k": density_k,
        "remove_counts": remove_counts,
        "geometry_seed": settings["seed"] + 503,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    similarity_cache = paths.root / "cache" / f"hubness_similarity_{cache_signature}"
    similarity_cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(settings["seed"] + 501 + k)
    permutation_rng = np.random.default_rng(settings["seed"] + 502)
    label_permutations = make_label_permutations(
        permutation_rng, settings["label_permutations"],
        len(dataset.semantic_ids), len(languages),
    )

    sample_records = []
    evidence_records = []
    source_records = []
    breadth_records = []
    specificity_records = []
    permutation_records = []
    null_by_method = {"cosine": [], "local_scaled_cosine": []}
    observed_english_by_method = {"cosine": [], "local_scaled_cosine": []}
    n_layers = int(dataset.vectors.shape[1])
    for layer in tqdm(range(n_layers), desc=f"paper_v1 rotated hubness (k={k})"):
        cache_path = similarity_cache / f"layer_{layer:03d}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            matrices = {method: cached[method] for method in methods}
        else:
            raw_groups = group_vectors(dataset, dataset.semantic_ids, layer, normalize=True)
            transformed = crossfit_geometry_groups(
                dataset, layer, dataset.semantic_ids, remove_counts, settings["seed"] + 503
            )
            cosine_matrices = raw_groups @ np.swapaxes(raw_groups, 1, 2)
            matrices = {
                "cosine": cosine_matrices,
                "local_scaled_cosine": np.stack([
                    locally_scaled_similarity(matrix, density_k) for matrix in cosine_matrices
                ]),
            }
            matrices.update({
                method: values @ np.swapaxes(values, 1, 2)
                for method, values in transformed.items()
            })
            np.savez_compressed(
                cache_path, **{name: values.astype(np.float32) for name, values in matrices.items()}
            )
        layer_arrays = None
        if not args.summary_only:
            layer_arrays = {
                "semantic_ids": np.asarray(dataset.semantic_ids, dtype="U"),
                "languages": np.asarray(languages, dtype="U"),
                "methods": np.asarray(methods, dtype="U"),
            }
        for method in methods:
            statistics = [group_statistics(matrix, k) for matrix in matrices[method]]
            selected = np.stack([item[0] for item in statistics])
            occurrence = np.stack([item[1] for item in statistics])
            centrality = np.stack([item[2] for item in statistics])
            percentile = np.stack([item[3] for item in statistics])
            medoid = np.stack([item[4] for item in statistics])
            measures = candidate_measure_samples(occurrence, centrality, percentile, medoid, k)
            safe_method = method.replace("-", "_")
            if layer_arrays is not None:
                layer_arrays[f"{safe_method}__selected"] = selected.astype(np.float16)
                for metric, values in measures.items():
                    layer_arrays[f"{safe_method}__{metric}"] = values.astype(np.float32)
            for metric, values in measures.items():
                for candidate_index, candidate_lang in enumerate(languages):
                    candidate_values = values[:, candidate_index]
                    if not args.summary_only:
                        sample_records.extend({
                            "representation": "mean_pool",
                            "similarity_method": method,
                            "layer": layer,
                            "semantic_id": semantic_id,
                            "candidate_lang": candidate_lang,
                            "metric": metric,
                            "value": float(candidate_values[index]),
                            "k": k,
                        } for index, semantic_id in enumerate(dataset.semantic_ids))
                    evidence_records.append(metric_record({
                        "representation": "mean_pool",
                        "similarity_method": method,
                        "layer": layer,
                        "layer_role": "embedding" if layer == 0 else "transformer_block_output",
                        "candidate_lang": candidate_lang,
                        "metric": metric,
                        "null_value": 0.0,
                        "k": k,
                    }, candidate_values, rng, settings["bootstrap_samples"], settings["confidence_level"]))

            primary_values = measures[settings["primary_permutation_metric"]]
            candidate_means = primary_values.mean(axis=0)
            order = pd.Series(-candidate_means).rank(method="min").to_numpy(dtype=int)
            best_non_english_index = int(np.argmax(np.where(
                np.arange(len(languages)) == english_index, -np.inf, candidate_means
            )))
            specificity_records.append({
                "representation": "mean_pool",
                "similarity_method": method,
                "layer": layer,
                "primary_metric": settings["primary_permutation_metric"],
                "english_mean": float(candidate_means[english_index]),
                "english_rank": int(order[english_index]),
                "candidate_count": len(languages),
                "best_non_english_lang": languages[best_non_english_index],
                "best_non_english_mean": float(candidate_means[best_non_english_index]),
                "english_minus_best_non_english": float(
                    candidate_means[english_index] - candidate_means[best_non_english_index]
                ),
            })

            for source_index, source_lang in enumerate(languages):
                for candidate_index, candidate_lang in enumerate(languages):
                    if source_index == candidate_index:
                        continue
                    values = selected[:, source_index, candidate_index]
                    source_records.append(metric_record({
                        "representation": "mean_pool",
                        "similarity_method": method,
                        "layer": layer,
                        "source_lang": source_lang,
                        "source_family": language_metadata.get(source_lang, {}).get("family", "unknown"),
                        "source_script": language_metadata.get(source_lang, {}).get("script", "unknown"),
                        "candidate_lang": candidate_lang,
                        "metric": "topk_selection_rate",
                        "balanced_selection_baseline": k / (len(languages) - 1),
                        "k": k,
                    }, values, rng, settings["bootstrap_samples"], settings["confidence_level"]))

            if method in null_by_method:
                null_maxima = max_target_null(primary_values, label_permutations)
                observed = float(candidate_means[english_index])
                p_layer = (1 + int(np.sum(null_maxima >= observed))) / (1 + len(null_maxima))
                null_by_method[method].append(null_maxima)
                observed_english_by_method[method].append(observed)
                permutation_records.append({
                    "representation": "mean_pool",
                    "similarity_method": method,
                    "layer": layer,
                    "primary_metric": settings["primary_permutation_metric"],
                    "observed_english": observed,
                    "observed_max_candidate": float(candidate_means.max()),
                    "layerwise_max_target_p_value": p_layer,
                    "n_permutations": settings["label_permutations"],
                })
        if layer_arrays is not None:
            np.savez_compressed(sample_array_dir / f"layer_{layer:03d}.npz", **layer_arrays)

    source_frame = pd.DataFrame(source_records)
    for (method, layer, candidate), group in source_frame.groupby(
        ["similarity_method", "layer", "candidate_lang"]
    ):
        supported = group[group.ci_lower > group.balanced_selection_baseline]
        breadth_records.append({
            "representation": "mean_pool",
            "similarity_method": method,
            "layer": int(layer),
            "candidate_lang": candidate,
            "supported_source_languages": int(supported.source_lang.nunique()),
            "total_source_languages": int(group.source_lang.nunique()),
            "supported_source_families": int(supported.source_family.nunique()),
            "supported_source_scripts": int(supported.source_script.nunique()),
            "supported_non_latin_languages": int(
                supported[supported.source_script != "Latin"].source_lang.nunique()
            ),
            "k": k,
        })

    for method in ("cosine", "local_scaled_cosine"):
        null_global = np.max(np.stack(null_by_method[method], axis=1), axis=1)
        observed_global = max(observed_english_by_method[method])
        global_p = (1 + int(np.sum(null_global >= observed_global))) / (1 + len(null_global))
        for record in permutation_records:
            if record["similarity_method"] == method:
                record["global_max_target_layer_p_value"] = global_p
                record["observed_english_max_across_layers"] = observed_global

    evidence = pd.DataFrame(evidence_records)
    breadth = pd.DataFrame(breadth_records)
    specificity = pd.DataFrame(specificity_records)
    english_breadth = breadth[breadth.candidate_lang == english].copy()
    raw_joint = joint_layers(evidence, english, "cosine")
    density_joint = joint_layers(evidence, english, "local_scaled_cosine")
    raw_broad = english_breadth[
        (english_breadth.similarity_method == "cosine") & breadth_supported(
            english_breadth[english_breadth.similarity_method == "cosine"]
        )
    ].layer.astype(int).tolist()
    density_subset = english_breadth[english_breadth.similarity_method == "local_scaled_cosine"]
    density_broad = density_subset[breadth_supported(density_subset)].layer.astype(int).tolist()
    primary_layers = sorted(set(raw_joint) & set(raw_broad))
    density_complete = sorted(set(density_joint) & set(density_broad))
    status = classify_model_status(
        primary_layers, primary_layers, density_complete,
        int(cfg["metrics"].get("min_consecutive_layers", 3)),
    )
    status.update({
        "raw_four_metric_layers": raw_joint,
        "raw_breadth_layers": sorted(raw_broad),
        "local_scaled_four_metric_layers": density_joint,
        "local_scaled_breadth_layers": sorted(density_broad),
        "density_rule": "raw four metrics + raw breadth + local-scaled four metrics + local-scaled breadth",
    })

    if not args.summary_only:
        pd.DataFrame(sample_records).to_csv(
            output / "target_rotation_samples.csv", index=False, encoding="utf-8"
        )
    evidence.to_csv(output / "target_rotation_evidence.csv", index=False, encoding="utf-8")
    source_frame.to_csv(output / "target_rotation_source_attraction.csv", index=False, encoding="utf-8")
    breadth.to_csv(output / "target_rotation_breadth.csv", index=False, encoding="utf-8")
    specificity.to_csv(output / "target_rotation_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(permutation_records).to_csv(
        output / "target_rotation_permutation.csv", index=False, encoding="utf-8"
    )
    (output / "paper_model_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(output / "hubness_manifest.json", {
        "candidate_scope": "same_semantic_id_only",
        "cross_semantic_similarity_computed": False,
        "bootstrap_unit": "semantic_id",
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "languages": languages,
        "target_rotation": "all configured languages",
        "methods": methods,
        "geometry_fit": "two-fold semantic-ID cross-fitting",
        "similarity_cache_signature": cache_signature,
        "similarity_cache_directory": str(similarity_cache),
        "primary_permutation_metric": settings["primary_permutation_metric"],
        "label_permutations": settings["label_permutations"],
        "permutation_family": "maximum across candidate targets and layers",
        "density_breadth_required": True,
        "k": k,
        "summary_only": bool(args.summary_only),
        "output_files": [
            *([] if args.summary_only else [
                "target_rotation_samples.csv", "sample_arrays/layer_*.npz"
            ]),
            "target_rotation_evidence.csv",
            "target_rotation_source_attraction.csv", "target_rotation_breadth.csv",
            "target_rotation_summary.csv", "target_rotation_permutation.csv",
            "paper_model_status.json",
        ],
    })
    validation = {
        "module": "05_06_target_rotation_hubness",
        "status": "PASS",
        "checks": {
            "all_languages_are_targets": evidence.candidate_lang.nunique() == len(languages),
            "all_four_metrics_present": set(evidence.metric) == set(HUB_METRICS),
            "raw_and_density_breadth_saved": {"cosine", "local_scaled_cosine"}.issubset(
                set(breadth.similarity_method)
            ),
            "max_target_max_layer_permutation_saved": True,
            "crossfit_geometry_saved": True,
        },
        "paper_model_status": status,
    }
    validation_name = (
        f"05_06_target_rotation_hubness_{args.result_tag}.json"
        if args.result_tag else "05_06_target_rotation_hubness.json"
    )
    (paths.validation / validation_name).write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved paper_v1 target-rotation hubness outputs to {output}; status={status['status']}")


if __name__ == "__main__":
    main()
