import json
import os
import random
from pathlib import Path

import numpy as np


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    parent = cfg.pop("extends", None)
    if not parent:
        return cfg

    base = load_config(path.parent / parent)

    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    return merge(base, cfg)


def ensure_dirs(cfg):
    output_dir = Path(cfg["output_dir"])
    paths = {
        "output": output_dir,
        "data": output_dir / "data",
        "hidden": output_dir / "hidden",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "validation": output_dir / "validation",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def representation_file_map():
    """Canonical filenames for supported sentence representations.

    ``shared_sentinel`` is retained as a read-only compatibility alias for old
    experiments. New experiments use the clearer ``sentinel_eos`` name.
    """
    return {
        "mean_pool": "sentence_layer_mean_pool.npy",
        "sentinel_eos": "sentence_layer_sentinel_eos.npy",
        "shared_sentinel": "sentence_layer_shared_sentinel.npy",
    }


def configured_representations(cfg):
    """Return and validate the two representations in the revised protocol."""
    metrics = cfg.get("metrics", {})
    names = list(metrics.get("representations", ["mean_pool", "sentinel_eos"]))
    allowed = {"mean_pool", "sentinel_eos"}
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(
            "The revised protocol only supports mean_pool and sentinel_eos; "
            f"remove: {unknown}"
        )
    if len(names) != len(set(names)):
        raise ValueError("metrics.representations contains duplicates")
    if "mean_pool" not in names:
        raise ValueError("mean_pool must be included as the primary sentence representation")
    primary = metrics.get("primary_representation", "mean_pool")
    if primary != "mean_pool":
        raise ValueError("metrics.primary_representation must be mean_pool in the revised protocol")
    validation = metrics.get("validation_representation", "sentinel_eos")
    if validation not in names:
        raise ValueError("metrics.validation_representation must be present in metrics.representations")
    return names


def validate_language_inventory(cfg):
    """Require a balanced, genuinely multilingual same-semantics candidate set."""
    dataset = cfg.get("dataset", {})
    languages = dataset.get("languages", {})
    minimum = int(dataset.get("minimum_languages_per_semantic_group", 20))
    if minimum < 20:
        raise ValueError("minimum_languages_per_semantic_group must be at least 20")
    if len(languages) < minimum:
        raise ValueError(
            f"Configured language count is {len(languages)}; at least {minimum} are required"
        )
    return list(languages)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)
