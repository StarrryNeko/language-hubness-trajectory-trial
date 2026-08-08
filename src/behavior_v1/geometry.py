"""Export task-level geometry predictors for behavior_v1 association tests."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from behavior_v1.common import (
    behavior_settings, ensure_behavior_dirs, load_tasks, read_checkpoint_identity,
    sha256_file, write_manifest,
)
from common import l2_normalize, load_config
from compute_metrics import locally_scaled_similarity
from paper_common import group_vectors, load_hidden_dataset, semantic_id_hash


def descending_rank(values, index):
    order = np.argsort(-np.asarray(values), kind="stable")
    return int(np.where(order == int(index))[0][0]) + 1


def canonical_semantic_id(value):
    """Match JSONL IDs such as 00040 to CSV-inferred IDs such as 40."""
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def align_semantic_ids(task_ids, hidden_ids):
    task_ids = sorted({str(value) for value in task_ids}, key=lambda value: int(value))
    hidden_by_canonical = {}
    for hidden_id in hidden_ids:
        canonical = canonical_semantic_id(hidden_id)
        if canonical in hidden_by_canonical:
            raise ValueError(f"hidden data has colliding semantic IDs after normalization: {canonical}")
        hidden_by_canonical[canonical] = str(hidden_id)
    missing = [
        task_id for task_id in task_ids
        if canonical_semantic_id(task_id) not in hidden_by_canonical
    ]
    if missing:
        raise ValueError(f"behavior tasks are absent from hidden data: {missing[:5]}")
    aligned_hidden_ids = [
        hidden_by_canonical[canonical_semantic_id(task_id)] for task_id in task_ids
    ]
    return task_ids, aligned_hidden_ids


def fit_pc1(vectors, seed):
    values = np.asarray(vectors, dtype=np.float64)
    estimator = PCA(n_components=1, svd_solver="randomized", random_state=int(seed))
    estimator.fit(values)
    return estimator.mean_, estimator.components_[0]


def similarity_matrices(raw_groups, local_scaling_k, resources):
    """Compute cosine and local scaling on the configured accelerator."""
    device_name = resources["geometry_device"]
    if resources["geometry_dtype"] != "float32":
        raise ValueError("behavior geometry currently requires geometry_dtype=float32")
    if device_name.startswith("cuda"):
        try:
            import torch
            import torch.nn.functional as torch_functional
        except ImportError as exc:
            raise ImportError("CUDA geometry requires a PyTorch-enabled runtime") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("behavior geometry_device requests CUDA but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = resources["allow_tf32"]
        with torch.inference_mode():
            tensor = torch.as_tensor(
                np.asarray(raw_groups), dtype=torch.float32, device=torch.device(device_name)
            )
            normalized = torch_functional.normalize(tensor, p=2, dim=2)
            cosine = torch.bmm(normalized, normalized.transpose(1, 2))
            masked = cosine.clone()
            masked.diagonal(dim1=1, dim2=2).fill_(float("-inf"))
            density = torch.topk(masked, k=int(local_scaling_k), dim=2).values.mean(dim=2)
            scaled = 2 * cosine - density[:, :, None] - density[:, None, :]
            scaled.diagonal(dim1=1, dim2=2).fill_(1.0)
            return cosine.cpu().numpy(), scaled.cpu().numpy()
    if device_name != "cpu":
        raise ValueError("behavior geometry_device must be cpu, cuda, or cuda:<index>")
    normalized = l2_normalize(raw_groups, axis=2)
    cosine = normalized @ np.swapaxes(normalized, 1, 2)
    scaled = np.stack([
        locally_scaled_similarity(matrix, local_scaling_k) for matrix in cosine
    ])
    return cosine, scaled


def main():
    parser = argparse.ArgumentParser(description="Export behavior_v1 geometry predictors")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = behavior_settings(cfg)
    paths = ensure_behavior_dirs(cfg)
    tasks = load_tasks(cfg)
    dataset = load_hidden_dataset(cfg)
    languages = dataset.languages
    language_index = {language: index for index, language in enumerate(languages)}
    english_index = language_index["en"]
    layers = sorted(set(int(value) for value in settings["analysis_layers"]))
    if not layers:
        raise ValueError("behavior_v1.analysis_layers must be frozen before predictor export")
    maximum_layer = int(dataset.vectors.shape[1]) - 1
    if min(layers) < 0 or max(layers) > maximum_layer:
        raise ValueError(f"behavior analysis layers must be within 0..{maximum_layer}: {layers}")
    semantic_ids, hidden_semantic_ids = align_semantic_ids(
        (task["semantic_id"] for task in tasks), dataset.semantic_ids
    )
    task_by_semantic = {}
    for task in tasks:
        task_by_semantic.setdefault(str(task["semantic_id"]), []).append(task)
    records = []
    token_counts = {
        (canonical_semantic_id(row["id"]), str(row["lang"])): row.get("sentence_num_tokens")
        for row in dataset.meta.to_dict("records")
    }
    for layer in layers:
        raw_groups = group_vectors(dataset, hidden_semantic_ids, layer, normalize=False)
        cosine, scaled = similarity_matrices(
            raw_groups, settings["local_scaling_k"], settings["resources"]
        )
        flat = raw_groups.reshape(-1, raw_groups.shape[-1])
        global_centroid, pc1 = fit_pc1(flat, settings["seed"] + layer)
        english_centered = raw_groups[:, english_index, :] - global_centroid
        if float((english_centered @ pc1).mean()) < 0:
            pc1 = -pc1
        for semantic_position, semantic_id in enumerate(semantic_ids):
            raw_vectors = raw_groups[semantic_position]
            raw_matrix = cosine[semantic_position]
            scaled_matrix = scaled[semantic_position]
            norms = np.linalg.norm(raw_vectors, axis=1)
            english_norm = float(norms[english_index])
            english_norm_rank = descending_rank(-norms, english_index)
            semantic_loo = np.delete(raw_vectors, english_index, axis=0).mean(axis=0)
            english_global_distance = float(np.linalg.norm(
                raw_vectors[english_index] - global_centroid
            ))
            english_loo_distance = float(np.linalg.norm(
                raw_vectors[english_index] - semantic_loo
            ))
            english_pc1 = float((raw_vectors[english_index] - global_centroid) @ pc1)
            english_neighbors = np.delete(raw_matrix[english_index], english_index)
            local_k = min(settings["local_scaling_k"], len(english_neighbors))
            english_local_density = float(np.sort(english_neighbors)[-local_k:].mean())
            raw_topk = np.argsort(-raw_matrix, axis=1)[:, : settings["local_scaling_k"] + 1]
            scaled_topk = np.argsort(-scaled_matrix, axis=1)[:, : settings["local_scaling_k"] + 1]
            raw_occurrence = sum(
                english_index in [value for value in row if value != source][: settings["local_scaling_k"]]
                for source, row in enumerate(raw_topk)
                if source != english_index
            )
            scaled_occurrence = sum(
                english_index in [value for value in row if value != source][: settings["local_scaling_k"]]
                for source, row in enumerate(scaled_topk)
                if source != english_index
            )
            for task in task_by_semantic[semantic_id]:
                source = language_index[str(task["source_lang"])]
                target = language_index[str(task["target_lang"])]
                non_english_candidates = [
                    index for index in range(len(languages))
                    if index not in {source, english_index}
                ]
                raw_candidates = [index for index in range(len(languages)) if index != source]
                raw_values = raw_matrix[source, raw_candidates]
                scaled_values = scaled_matrix[source, raw_candidates]
                source_is_english = source == english_index
                english_cosine = np.nan if source_is_english else float(raw_matrix[source, english_index])
                english_scaled = np.nan if source_is_english else float(scaled_matrix[source, english_index])
                records.append({
                    "task_id": task["task_id"],
                    "semantic_id": semantic_id,
                    "condition": task["condition"],
                    "source_lang": task["source_lang"],
                    "target_lang": task["target_lang"],
                    "source_sentence_token_count": token_counts.get((
                        canonical_semantic_id(semantic_id), str(task["source_lang"])
                    )),
                    "target_sentence_token_count": token_counts.get((
                        canonical_semantic_id(semantic_id), str(task["target_lang"])
                    )),
                    "layer": layer,
                    "normalized_depth": layer / maximum_layer,
                    "source_english_cosine": english_cosine,
                    "source_target_cosine": float(raw_matrix[source, target]),
                    "english_minus_target_cosine": (
                        np.nan if source_is_english else
                        float(raw_matrix[source, english_index] - raw_matrix[source, target])
                    ),
                    "english_hard_margin": np.nan if source_is_english else float(
                        raw_matrix[source, english_index]
                        - max(raw_matrix[source, index] for index in non_english_candidates)
                    ),
                    "english_source_rank": (
                        np.nan if source_is_english else
                        descending_rank(raw_values, raw_candidates.index(english_index))
                    ),
                    "source_english_local_scaled": english_scaled,
                    "source_target_local_scaled": float(scaled_matrix[source, target]),
                    "english_minus_target_local_scaled": (
                        np.nan if source_is_english else
                        float(scaled_matrix[source, english_index] - scaled_matrix[source, target])
                    ),
                    "english_local_scaled_rank": (
                        np.nan if source_is_english else
                        descending_rank(scaled_values, raw_candidates.index(english_index))
                    ),
                    "english_k_occurrence": int(raw_occurrence),
                    "english_local_scaled_k_occurrence": int(scaled_occurrence),
                    "english_norm": english_norm,
                    "english_norm_rank": english_norm_rank,
                    "english_distance_to_global_centroid": english_global_distance,
                    "english_distance_to_loo_semantic_centroid": english_loo_distance,
                    "english_pc1_projection": english_pc1,
                    "english_local_density": english_local_density,
                })
    frame = pd.DataFrame.from_records(records)
    output = paths.metrics / "behavior_geometry_predictors.csv"
    frame.to_csv(output, index=False, encoding="utf-8")
    checkpoint = read_checkpoint_identity(cfg)
    write_manifest(paths.metrics / "behavior_geometry_manifest.json", {
        "rows": len(frame),
        "tasks": frame.task_id.nunique(),
        "semantic_ids": frame.semantic_id.nunique(),
        "layers": layers,
        "primary_layer": settings["primary_layer"],
        "local_scaling_k": settings["local_scaling_k"],
        "resources": settings["resources"],
        "gpu_accelerated_similarity": settings["resources"]["geometry_device"].startswith("cuda"),
        "hidden_semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "task_file_sha256": sha256_file(paths.data / "behavior_tasks.jsonl"),
        "predictor_file_sha256": sha256_file(output),
        "pc1_fit_scope": "all held-out evaluation vectors; predictor-only, no behavior outcomes",
    })
    print(f"Saved {len(frame)} behavior predictors to {output}")


if __name__ == "__main__":
    main()
