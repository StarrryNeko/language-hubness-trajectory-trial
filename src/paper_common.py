"""Shared, auditable inputs and outputs for the paper_v1 analyses.

The paper modules deliberately read the existing extraction artifacts and write to
``<output_dir>/paper_v1``.  This keeps confirmatory results separate without
requiring another model forward pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    configured_representations,
    l2_normalize,
    representation_file_map,
    validate_language_inventory,
)
from numerical_validation import validate_representation_array


PAPER_PROTOCOL_VERSION = "paper_v1"


@dataclass(frozen=True)
class PaperPaths:
    root: Path
    metrics: Path
    figures: Path
    validation: Path
    splits: Path


@dataclass
class HiddenDataset:
    meta: pd.DataFrame
    vectors: np.ndarray
    languages: list[str]
    semantic_ids: list[str]
    indices_by_id: dict[str, np.ndarray]
    row_language_indices: np.ndarray


def ensure_paper_dirs(cfg) -> PaperPaths:
    root = Path(cfg["output_dir"]) / str(
        cfg.get("paper_v1", {}).get("result_directory", PAPER_PROTOCOL_VERSION)
    )
    paths = PaperPaths(
        root=root,
        metrics=root / "metrics",
        figures=root / "figures",
        validation=root / "validation",
        splits=root / "splits",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def paper_settings(cfg) -> dict:
    paper = cfg.get("paper_v1", {})
    settings = {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "seed": int(paper.get("seed", cfg.get("seed", 42))),
        "bootstrap_samples": int(
            paper.get("bootstrap_samples", cfg.get("metrics", {}).get("bootstrap_samples", 500))
        ),
        "confidence_level": float(
            paper.get("confidence_level", cfg.get("metrics", {}).get("confidence_level", 0.95))
        ),
        "shuffled_permutations": int(paper.get("shuffled_permutations", 20)),
        "label_permutations": int(paper.get("label_permutations", 500)),
        "probe_split_ratios": paper.get("probe_split_ratios", [0.6, 0.2, 0.2]),
        "geometry_fit_fraction": float(paper.get("geometry_fit_fraction", 0.5)),
        "primary_permutation_metric": str(
            paper.get("primary_permutation_metric", "k_occurrence_excess")
        ),
    }
    if settings["bootstrap_samples"] < 0:
        raise ValueError("paper_v1.bootstrap_samples must be non-negative")
    if not 0 < settings["confidence_level"] < 1:
        raise ValueError("paper_v1.confidence_level must be between 0 and 1")
    if settings["shuffled_permutations"] < 1:
        raise ValueError("paper_v1.shuffled_permutations must be positive")
    if settings["label_permutations"] < 1:
        raise ValueError("paper_v1.label_permutations must be positive")
    if not 0 < settings["geometry_fit_fraction"] < 1:
        raise ValueError("paper_v1.geometry_fit_fraction must be between 0 and 1")
    return settings


def validate_hidden_metadata(meta: pd.DataFrame, languages: list[str]) -> None:
    required = {"row_idx", "id", "lang", "was_truncated"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")
    if meta.empty:
        raise ValueError("metadata.csv is empty")
    truncated = meta.was_truncated.astype(str).str.lower().isin({"true", "1", "yes"})
    if truncated.any():
        raise ValueError(f"{int(truncated.sum())} inputs were truncated")
    if meta.duplicated(["id", "lang"]).any():
        raise ValueError("metadata contains duplicate (semantic ID, language) rows")
    rows = sorted(meta.row_idx.astype(int).tolist())
    if rows != list(range(len(meta))):
        raise ValueError("metadata.row_idx must be a complete 0..N-1 mapping")
    expected = set(languages)
    invalid = []
    for semantic_id, group in meta.groupby(meta.id.astype(str), sort=True):
        if set(group.lang.astype(str)) != expected or len(group) != len(languages):
            invalid.append(str(semantic_id))
    if invalid:
        raise ValueError(
            f"{len(invalid)} semantic groups are incomplete; first IDs: {invalid[:5]}"
        )


def load_hidden_dataset(cfg, representation="mean_pool", mmap_mode="r") -> HiddenDataset:
    if representation not in configured_representations(cfg):
        raise ValueError(f"representation {representation!r} is not active")
    languages = list(validate_language_inventory(cfg))
    hidden = Path(cfg["output_dir"]) / "hidden"
    meta = pd.read_csv(hidden / "metadata.csv")
    validate_hidden_metadata(meta, languages)
    vectors = np.load(hidden / representation_file_map()[representation], mmap_mode=mmap_mode)
    validate_representation_array(vectors, len(meta), f"paper_v1 representation={representation}")
    meta = meta.copy()
    meta["id"] = meta.id.astype(str)
    meta["lang"] = meta.lang.astype(str)
    semantic_ids = sorted(meta.id.unique().tolist())
    indices_by_id = {
        semantic_id: np.array([
            int(meta.loc[(meta.id == semantic_id) & (meta.lang == lang), "row_idx"].iloc[0])
            for lang in languages
        ], dtype=np.int64)
        for semantic_id in semantic_ids
    }
    language_index = {lang: index for index, lang in enumerate(languages)}
    row_language_indices = meta.sort_values("row_idx").lang.map(language_index).to_numpy(dtype=int)
    return HiddenDataset(
        meta=meta,
        vectors=vectors,
        languages=languages,
        semantic_ids=semantic_ids,
        indices_by_id=indices_by_id,
        row_language_indices=row_language_indices,
    )


def semantic_id_hash(semantic_ids) -> str:
    payload = "\n".join(map(str, semantic_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derangement(rng: np.random.Generator, size: int) -> np.ndarray:
    if size < 2:
        raise ValueError("a derangement requires at least two semantic IDs")
    base = np.arange(size)
    for _ in range(100):
        candidate = rng.permutation(size)
        if np.all(candidate != base):
            return candidate
    return np.roll(base, 1)


def bootstrap_mean_ci(values, rng, n_boot=500, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap values must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap values contain non-finite observations")
    mean = float(values.mean())
    if len(values) == 1 or n_boot <= 0:
        return mean, mean, mean
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return mean, float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def metric_record(prefix, values, rng, n_boot, confidence):
    mean, low, high = bootstrap_mean_ci(values, rng, n_boot, confidence)
    return {
        **prefix,
        "mean": mean,
        "ci_lower": low,
        "ci_upper": high,
        "n_semantic_ids": int(len(values)),
    }


def build_semantic_splits(semantic_ids, seed, ratios=(0.6, 0.2, 0.2)) -> dict:
    semantic_ids = list(map(str, semantic_ids))
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("split ratios must be three positive values summing to 1")
    if len(semantic_ids) < 5:
        raise ValueError("at least five semantic IDs are required for train/validation/test")
    shuffled = np.asarray(semantic_ids, dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled)
    train_end = max(1, int(np.floor(len(shuffled) * ratios[0])))
    validation_end = max(train_end + 1, int(np.floor(len(shuffled) * ratios[:2].sum())))
    validation_end = min(validation_end, len(shuffled) - 1)
    result = {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "seed": int(seed),
        "ratios": ratios.tolist(),
        "semantic_id_sha256": semantic_id_hash(semantic_ids),
        "train": shuffled[:train_end].tolist(),
        "validation": shuffled[train_end:validation_end].tolist(),
        "test": shuffled[validation_end:].tolist(),
    }
    assert set(result["train"]).isdisjoint(result["validation"])
    assert set(result["train"]).isdisjoint(result["test"])
    assert set(result["validation"]).isdisjoint(result["test"])
    return result


def load_or_create_splits(path, semantic_ids, seed, ratios=(0.6, 0.2, 0.2)) -> dict:
    path = Path(path)
    expected = build_semantic_splits(semantic_ids, seed, ratios)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        for key in ("protocol_version", "seed", "ratios", "semantic_id_sha256"):
            if actual.get(key) != expected.get(key):
                raise ValueError(
                    f"frozen split {path} disagrees with current data/settings on {key}"
                )
        covered = actual.get("train", []) + actual.get("validation", []) + actual.get("test", [])
        if len(covered) != len(set(covered)) or set(covered) != set(map(str, semantic_ids)):
            raise ValueError(f"frozen split {path} does not partition the current semantic IDs")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")
    return expected


def group_vectors(dataset: HiddenDataset, semantic_ids, layer, normalize=True) -> np.ndarray:
    groups = np.stack([
        np.asarray(dataset.vectors[dataset.indices_by_id[str(semantic_id)], layer, :], dtype=np.float32)
        for semantic_id in semantic_ids
    ])
    if not np.isfinite(groups).all():
        raise ValueError(f"non-finite hidden vectors at layer {layer}")
    if normalize:
        groups = l2_normalize(groups, axis=2)
    return groups


def write_manifest(path, payload):
    body = {"protocol_version": PAPER_PROTOCOL_VERSION, **payload}
    Path(path).write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

