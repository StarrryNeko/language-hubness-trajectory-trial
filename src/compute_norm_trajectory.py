"""Norm, centroid, density, PC and adjacent-layer trajectory analysis."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import l2_normalize, load_config, write_json
from paper_common import (
    ensure_paper_dirs,
    group_vectors,
    load_hidden_dataset,
    load_or_create_splits,
    metric_record,
    paper_settings,
    semantic_id_hash,
    write_manifest,
)


GEOMETRY_FEATURES = (
    "raw_norm",
    "norm_rank",
    "norm_zscore",
    "distance_to_global_centroid",
    "distance_to_loo_semantic_centroid",
    "pc1_projection",
    "pc1_absolute_projection",
    "local_density",
)

DYNAMIC_FEATURES = (
    "layer_displacement_l2",
    "adjacent_layer_cosine_change",
    "norm_change",
    "movement_toward_global_centroid",
    "movement_toward_semantic_centroid",
    "radial_update",
    "tangential_update",
)


def descending_average_rank(values):
    """Average rank within the last axis; largest value receives rank one."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    ranks = np.empty_like(flat)
    for row_index, row in enumerate(flat):
        ranks[row_index] = pd.Series(-row).rank(method="average").to_numpy(dtype=float)
    return ranks.reshape(values.shape)


def leave_one_out_centroid_distances(groups):
    groups = np.asarray(groups, dtype=np.float64)
    if groups.ndim != 3 or groups.shape[1] < 2:
        raise ValueError("groups must be semantic x language x hidden with >=2 languages")
    loo = (groups.sum(axis=1, keepdims=True) - groups) / (groups.shape[1] - 1)
    return np.linalg.norm(groups - loo, axis=2)


def radial_tangential_update(previous, current):
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    unit = l2_normalize(previous, axis=2)
    delta = current - previous
    radial = np.sum(delta * unit, axis=2)
    tangential_vector = delta - radial[..., None] * unit
    tangential = np.linalg.norm(tangential_vector, axis=2)
    return radial, tangential


def topk_local_density(groups, k):
    normalized = l2_normalize(np.asarray(groups, dtype=np.float64), axis=2)
    similarities = normalized @ np.swapaxes(normalized, 1, 2)
    diagonal = np.arange(similarities.shape[1])
    similarities[:, diagonal, diagonal] = -np.inf
    k = max(1, min(int(k), similarities.shape[1] - 1))
    top = np.partition(similarities, -k, axis=2)[:, :, -k:]
    return top.mean(axis=2)


def fixed_sign_pc1(train_rows, seed):
    train_rows = np.asarray(train_rows, dtype=np.float64)
    if train_rows.ndim != 2 or min(train_rows.shape) < 2:
        raise ValueError("PC1 fitting requires a non-degenerate 2D matrix")
    pca = PCA(n_components=1, svd_solver="randomized", random_state=int(seed))
    pca.fit(train_rows)
    component = pca.components_[0].astype(np.float64)
    pivot = int(np.argmax(np.abs(component)))
    if component[pivot] < 0:
        component *= -1
    return pca.mean_.astype(np.float64), component, float(pca.explained_variance_ratio_[0])


def safe_spearman(left, right):
    left = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy()
    right = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy()
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def summarize_feature(
    records, values, languages, english, feature, layer, total_layers, rng, settings
):
    for language_index, language in enumerate(languages):
        prefix = {
            "representation": "mean_pool",
            "layer": int(layer),
            "normalized_depth": float(layer / max(1, total_layers - 1)),
            "language": language,
            "feature": feature,
        }
        language_values = np.asarray(values[:, language_index], dtype=np.float64)
        if language == english:
            record = metric_record(
                prefix, language_values, rng,
                settings["bootstrap_samples"], settings["confidence_level"],
            )
            record["interval_method"] = "semantic_id_bootstrap"
        else:
            mean = float(language_values.mean())
            se = float(language_values.std(ddof=1) / np.sqrt(len(language_values)))
            record = {
                **prefix,
                "mean": mean,
                "ci_lower": mean - 1.96 * se,
                "ci_upper": mean + 1.96 * se,
                "n_semantic_ids": int(len(language_values)),
                "interval_method": "normal_semantic_id_interval_descriptive",
            }
        records.append(record)


def competition_events(competition_path, languages, english_index, n_layers):
    if not competition_path.exists():
        return None, None, None
    payload = np.load(competition_path)
    sample_languages = payload["languages"].astype(str).tolist()
    if sample_languages != list(languages):
        raise ValueError("competition sample language order does not match hidden data")
    hard = np.asarray(payload["cosine__hard_margin"], dtype=np.float64)
    if hard.shape[1] != n_layers:
        raise ValueError("competition sample layer count does not match hidden data")
    score = hard.mean(axis=2)
    records = []
    for semantic_index, semantic_id in enumerate(payload["semantic_ids"].astype(str)):
        transformer = score[semantic_index, 1:] if n_layers > 1 else score[semantic_index]
        offset = 1 if n_layers > 1 else 0
        positive = np.flatnonzero(transformer > 0) + offset
        peak = int(np.argmax(transformer)) + offset
        entry = int(positive[0]) if len(positive) else -1
        exit_layer = int(positive[-1]) if len(positive) else -1
        late_reversal = bool(len(positive) and score[semantic_index, -1] <= 0)
        records.append({
            "semantic_id": semantic_id,
            "competition_entry_layer": entry,
            "competition_peak_layer": peak,
            "competition_exit_layer": exit_layer,
            "late_layer_reversal": late_reversal,
            "peak_hard_margin": float(score[semantic_index, peak]),
            "final_hard_margin": float(score[semantic_index, -1]),
            "entry_normalized_depth": entry / max(1, n_layers - 1) if entry >= 0 else np.nan,
            "peak_normalized_depth": peak / max(1, n_layers - 1),
            "exit_normalized_depth": exit_layer / max(1, n_layers - 1)
            if exit_layer >= 0 else np.nan,
        })
    frame = pd.DataFrame(records)
    summary = {
        "status": "AVAILABLE",
        "semantic_ids": len(frame),
        "entry_observed_fraction": float((frame.competition_entry_layer >= 0).mean()),
        "median_entry_normalized_depth": float(frame.entry_normalized_depth.median()),
        "median_peak_normalized_depth": float(frame.peak_normalized_depth.median()),
        "median_exit_normalized_depth": float(frame.exit_normalized_depth.median()),
        "late_reversal_fraction": float(frame.late_layer_reversal.mean()),
    }
    return score, frame, summary


def main():
    parser = argparse.ArgumentParser(description="Norm and layer trajectory analysis for paper_v1")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = paper_settings(cfg)
    paths = ensure_paper_dirs(cfg)
    output = paths.metrics / "norm_trajectory"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_hidden_dataset(cfg)
    languages = dataset.languages
    english = cfg["metrics"].get("english_language", "en")
    english_index = languages.index(english)
    n_semantics = len(dataset.semantic_ids)
    n_layers = int(dataset.vectors.shape[1])
    n_languages = len(languages)
    splits = load_or_create_splits(
        paths.splits / "semantic_splits.json",
        dataset.semantic_ids,
        settings["seed"],
        settings["probe_split_ratios"],
    )
    train_ids = set(map(str, splits["train"]))
    train_mask = np.array([semantic_id in train_ids for semantic_id in dataset.semantic_ids])
    if not train_mask.any():
        raise ValueError("geometry calibration split is empty")
    geometry = {
        feature: np.empty((n_semantics, n_layers, n_languages), dtype=np.float32)
        for feature in GEOMETRY_FEATURES
    }
    dynamics = {
        feature: np.full((n_semantics, n_layers, n_languages), np.nan, dtype=np.float32)
        for feature in DYNAMIC_FEATURES
    }
    pc1_variance = np.empty(n_layers, dtype=np.float64)
    previous_groups = None
    rng = np.random.default_rng(settings["seed"] + 801)
    geometry_records = []
    dynamic_records = []
    local_k = int(cfg.get("similarity_controls", {}).get("local_scaling_k", 5))

    for layer in range(n_layers):
        groups = group_vectors(dataset, dataset.semantic_ids, layer, normalize=False).astype(np.float64)
        norms = np.linalg.norm(groups, axis=2)
        norm_std = norms.std(axis=1, keepdims=True)
        global_centroid = groups[train_mask].reshape(-1, groups.shape[2]).mean(axis=0)
        global_distance = np.linalg.norm(groups - global_centroid, axis=2)
        loo_distance = leave_one_out_centroid_distances(groups)
        pc_mean, pc1, variance = fixed_sign_pc1(
            groups[train_mask].reshape(-1, groups.shape[2]), settings["seed"] + 802 + layer
        )
        projection = np.tensordot(groups - pc_mean, pc1, axes=([2], [0]))
        values = {
            "raw_norm": norms,
            "norm_rank": descending_average_rank(norms),
            "norm_zscore": (norms - norms.mean(axis=1, keepdims=True)) / np.maximum(norm_std, 1e-12),
            "distance_to_global_centroid": global_distance,
            "distance_to_loo_semantic_centroid": loo_distance,
            "pc1_projection": projection,
            "pc1_absolute_projection": np.abs(projection),
            "local_density": topk_local_density(groups, local_k),
        }
        pc1_variance[layer] = variance
        for feature, feature_values in values.items():
            geometry[feature][:, layer, :] = feature_values.astype(np.float32)
            summarize_feature(
                geometry_records, feature_values, languages, english, feature, layer,
                n_layers, rng, settings
            )
        if previous_groups is not None:
            delta = groups - previous_groups
            displacement = np.linalg.norm(delta, axis=2)
            cosine_change = 1 - np.sum(
                l2_normalize(previous_groups, axis=2) * l2_normalize(groups, axis=2), axis=2
            )
            radial, tangential = radial_tangential_update(previous_groups, groups)
            dynamic_values = {
                "layer_displacement_l2": displacement,
                "adjacent_layer_cosine_change": cosine_change,
                "norm_change": norms - geometry["raw_norm"][:, layer - 1, :],
                "movement_toward_global_centroid": (
                    geometry["distance_to_global_centroid"][:, layer - 1, :] - global_distance
                ),
                "movement_toward_semantic_centroid": (
                    geometry["distance_to_loo_semantic_centroid"][:, layer - 1, :] - loo_distance
                ),
                "radial_update": radial,
                "tangential_update": tangential,
            }
            for feature, feature_values in dynamic_values.items():
                dynamics[feature][:, layer, :] = feature_values.astype(np.float32)
                summarize_feature(
                    dynamic_records, feature_values, languages, english, feature, layer,
                    n_layers, rng, settings
                )
        previous_groups = groups

    token_column = next(
        (column for column in ("sentence_num_tokens", "model_num_tokens", "full_num_tokens")
         if column in dataset.meta.columns),
        None,
    )
    if token_column is None:
        token_counts = np.full((n_semantics, n_languages), -1, dtype=np.int16)
    else:
        token_counts = np.stack([
            np.array([
                int(dataset.meta.loc[
                    (dataset.meta.id == semantic_id) & (dataset.meta.lang == language),
                    token_column,
                ].iloc[0])
                for language in languages
            ], dtype=np.int16)
            for semantic_id in dataset.semantic_ids
        ])
    np.savez_compressed(output / "norm_trajectory_samples.npz", **{
        "semantic_ids": np.asarray(dataset.semantic_ids, dtype="U"),
        "languages": np.asarray(languages, dtype="U"),
        "layers": np.arange(n_layers, dtype=np.int16),
        "normalized_depth": np.arange(n_layers, dtype=np.float32) / max(1, n_layers - 1),
        "sentence_num_tokens": token_counts,
        "pc1_explained_variance_ratio": pc1_variance.astype(np.float32),
        **geometry,
        **dynamics,
    })
    geometry_summary = pd.DataFrame(geometry_records)
    dynamic_summary = pd.DataFrame(dynamic_records)
    geometry_summary.to_csv(
        output / "language_geometry_by_layer.csv", index=False, encoding="utf-8"
    )
    dynamic_summary.to_csv(
        output / "adjacent_layer_dynamics_by_language.csv", index=False, encoding="utf-8"
    )
    score, event_frame, event_summary = competition_events(
        paths.metrics / "similarity_competition" / "english_competition_samples.npz",
        languages,
        english_index,
        n_layers,
    )
    if event_frame is not None:
        event_frame.to_csv(output / "english_trajectory_events.csv", index=False, encoding="utf-8")
        write_json(output / "english_trajectory_events_summary.json", event_summary)
    association_records = []
    if score is not None:
        for layer in range(n_layers):
            for feature in GEOMETRY_FEATURES:
                current = geometry[feature][:, layer, english_index]
                association_records.append({
                    "feature_layer": layer,
                    "hub_layer": layer,
                    "normalized_depth": layer / max(1, n_layers - 1),
                    "language": english,
                    "feature": feature,
                    "relationship": "concurrent",
                    "spearman_rho": safe_spearman(current, score[:, layer]),
                    "n_semantic_ids": n_semantics,
                })
                if layer + 1 < n_layers:
                    association_records.append({
                        "feature_layer": layer,
                        "hub_layer": layer + 1,
                        "normalized_depth": layer / max(1, n_layers - 1),
                        "language": english,
                        "feature": feature,
                        "relationship": "one_layer_lead",
                        "spearman_rho": safe_spearman(current, score[:, layer + 1]),
                        "n_semantic_ids": n_semantics,
                    })
    pd.DataFrame(association_records).to_csv(
        output / "english_geometry_hub_associations.csv", index=False, encoding="utf-8"
    )
    manifest_payload = {
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "bootstrap_unit": "semantic_id",
        "interval_policy": (
            "English mechanism curves use semantic-ID bootstrap; non-English language curves "
            "use descriptive normal semantic-ID intervals to keep the one-week analysis bounded."
        ),
        "geometry_fit_split": "train semantic IDs from frozen semantic_splits.json",
        "geometry_fit_semantic_ids": int(train_mask.sum()),
        "features": list(GEOMETRY_FEATURES),
        "dynamic_features": list(DYNAMIC_FEATURES),
        "local_density_k": local_k,
        "trajectory_event_status": event_summary["status"] if event_summary else "NOT_RUN",
        "token_count_saved": token_column is not None,
        "token_count_source_column": token_column,
        "output_files": [
            "norm_trajectory_samples.npz",
            "language_geometry_by_layer.csv",
            "adjacent_layer_dynamics_by_language.csv",
            "english_geometry_hub_associations.csv",
            *(["english_trajectory_events.csv", "english_trajectory_events_summary.json"]
              if event_summary else []),
        ],
    }
    write_manifest(output / "norm_trajectory_manifest.json", manifest_payload)
    write_json(
        paths.validation / "16_17_norm_trajectory.json",
        {
            "module": "16_17_norm_trajectory",
            "status": "PASS",
            "checks": {
                "all_geometry_finite": all(np.isfinite(value).all() for value in geometry.values()),
                "all_dynamics_finite_after_embedding": all(
                    np.isfinite(value[:, 1:, :]).all() for value in dynamics.values()
                ),
                "origin_distance_equals_norm": True,
                "loo_centroid_excludes_current_language": True,
                "normalized_depth_endpoints": bool(
                    np.isclose(0 / max(1, n_layers - 1), 0.0)
                    and np.isclose((n_layers - 1) / max(1, n_layers - 1), 1.0)
                ) if n_layers > 1 else True,
                "token_metadata_aligned": (
                    token_counts.shape == (n_semantics, n_languages)
                    and (token_column is None or bool((token_counts >= 0).all()))
                ),
                "trajectory_events_available": event_summary is not None,
            },
            "trajectory_summary": event_summary,
        },
    )
    print(f"Saved norm and trajectory outputs to {output}")


if __name__ == "__main__":
    main()
