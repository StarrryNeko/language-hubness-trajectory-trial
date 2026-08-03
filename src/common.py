import json
import os
import random
from pathlib import Path

import numpy as np


MODEL_SIZE_CLASSES = ("S", "M", "L")


def portable_model_directory_name(model_id):
    """Return the stable folder name used by portable/offline model bundles."""
    text = str(model_id).strip().replace("\\", "/").strip("/")
    if not text:
        raise ValueError("model ID must be non-empty")
    return text.replace("/", "__")


def resolve_model_source(model_id, explicit_local_path=None, model_root=None):
    """Resolve a configured model ID to an uploaded local directory when present.

    The canonical Hugging Face ID remains the experiment identity.  A local path is
    only a transport/runtime source and therefore must not change manifests or
    extraction reuse checks.
    """
    canonical = str(model_id)
    candidates = []
    if explicit_local_path:
        candidates.append(Path(os.path.expandvars(os.path.expanduser(str(explicit_local_path)))))
    root = model_root or os.environ.get("LHT_MODEL_ROOT")
    if root:
        candidates.append(
            Path(os.path.expandvars(os.path.expanduser(str(root))))
            / portable_model_directory_name(canonical)
        )
    for candidate in candidates:
        if candidate.is_dir():
            return canonical, str(candidate.resolve()), True
    if explicit_local_path:
        raise FileNotFoundError(
            f"Configured local model directory does not exist: {explicit_local_path}"
        )
    return canonical, canonical, False


def select_semantic_indices(row_count, sample_size, strategy, seed):
    """Select one reproducible semantic-index set shared by every language."""
    row_count = int(row_count)
    sample_size = min(int(sample_size), row_count)
    if row_count < 1 or sample_size < 1:
        raise ValueError("dataset split and requested sample must be non-empty")
    if strategy == "first_n":
        return list(range(sample_size))
    if strategy == "random_without_replacement":
        selected = random.Random(int(seed)).sample(range(row_count), sample_size)
        return sorted(selected)
    raise ValueError(
        "dataset.sample_selection.strategy must be first_n or random_without_replacement"
    )


def classify_model_size(parameter_count_billions):
    """Classify dense models by total parameter count, not checkpoint bytes."""
    count = float(parameter_count_billions)
    if not np.isfinite(count) or count <= 0:
        raise ValueError("model.parameter_count_billions must be a positive finite number")
    if count < 7:
        return "S"
    if count < 12:
        return "M"
    if count < 20:
        return "L"
    raise ValueError(
        "Mainline model size must be below 20B parameters; "
        f"got {count:g}B"
    )


def model_metadata(cfg, require=False):
    """Return audited model metadata used for generation and scale comparisons."""
    model = cfg.get("model", {})
    required_fields = {
        "model_family": model.get("family"),
        "model_generation": model.get("generation"),
        "parameter_count_billions": model.get("parameter_count_billions"),
        "size_class": model.get("size_class"),
        "training_stage": model.get("training_stage"),
    }
    if not require and all(value is None for value in required_fields.values()):
        return {
            **required_fields,
            "architecture_type": None,
            "active_parameter_count_billions": None,
        }
    missing = [key for key, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(f"model metadata is missing fields: {missing}")
    expected = classify_model_size(required_fields["parameter_count_billions"])
    declared = str(required_fields["size_class"]).upper()
    if declared not in MODEL_SIZE_CLASSES:
        raise ValueError(f"model.size_class must be one of {MODEL_SIZE_CLASSES}")
    if declared != expected:
        raise ValueError(
            "model.size_class does not match parameter_count_billions: "
            f"declared={declared}, expected={expected}"
        )
    total = float(required_fields["parameter_count_billions"])
    architecture_type = str(model.get("architecture_type", "dense")).lower()
    if architecture_type not in {"dense", "moe"}:
        raise ValueError("model.architecture_type must be 'dense' or 'moe'")
    active = float(model.get("active_parameter_count_billions", total))
    if not np.isfinite(active) or active <= 0 or active > total:
        raise ValueError(
            "model.active_parameter_count_billions must be positive and no larger "
            "than parameter_count_billions"
        )
    return {
        **required_fields,
        "parameter_count_billions": total,
        "size_class": declared,
        "architecture_type": architecture_type,
        "active_parameter_count_billions": active,
    }


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    parent = cfg.pop("extends", None)
    if not parent:
        return cfg

    base = load_config(path.parent / parent)

    def merge(left, right):
        if isinstance(right, dict) and right.get("__replace__") is True:
            return {
                key: value for key, value in right.items()
                if key != "__replace__"
            }
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
    """Canonical filenames for the active sentence representation protocol."""
    return {
        "mean_pool": "sentence_layer_mean_pool.npy",
    }


def configured_representations(cfg):
    """Require mean pooling as the only active sentence representation."""
    metrics = cfg.get("metrics", {})
    names = list(metrics.get("representations", ["mean_pool"]))
    if names != ["mean_pool"]:
        raise ValueError(
            "The active protocol computes only mean_pool; "
            f"set metrics.representations to ['mean_pool'], got {names}"
        )
    primary = metrics.get("primary_representation", "mean_pool")
    if primary != "mean_pool":
        raise ValueError("metrics.primary_representation must be mean_pool")
    if "validation_representation" in metrics:
        raise ValueError("Remove metrics.validation_representation; EOS validation is disabled")
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
