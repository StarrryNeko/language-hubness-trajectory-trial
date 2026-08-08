"""Shared contracts for the observational behavior_v1 pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import json_dumps_strict, read_jsonl, write_json


BEHAVIOR_PROTOCOL_VERSION = "behavior_v1"


def resource_settings(cfg) -> dict:
    resources = cfg.get("behavior_v1", {}).get("resources", {})
    cpu_threads = int(resources.get("cpu_threads", 24))
    evaluation_workers = int(resources.get("evaluation_workers", cpu_threads))
    if cpu_threads < 1 or evaluation_workers < 1:
        raise ValueError("behavior_v1 resource thread counts must be positive")
    return {
        "cpu_threads": cpu_threads,
        "evaluation_workers": evaluation_workers,
        "geometry_device": str(resources.get("geometry_device", "cuda")),
        "geometry_dtype": str(resources.get("geometry_dtype", "float32")),
        "allow_tf32": bool(resources.get("allow_tf32", False)),
    }


def configure_cpu_environment(cpu_threads):
    """Freeze native CPU pools before launching numerical subprocesses."""
    value = str(int(cpu_threads))
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[name] = value
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass(frozen=True)
class BehaviorPaths:
    root: Path
    data: Path
    generations: Path
    metrics: Path
    validation: Path


def ensure_behavior_dirs(cfg) -> BehaviorPaths:
    behavior = cfg.get("behavior_v1", {})
    root = Path(cfg["output_dir"]) / str(behavior.get("result_directory", BEHAVIOR_PROTOCOL_VERSION))
    paths = BehaviorPaths(
        root=root,
        data=root / "data",
        generations=root / "generations",
        metrics=root / "metrics",
        validation=root / "validation",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload) -> str:
    return sha256_bytes(
        json_dumps_strict(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )


def behavior_settings(cfg) -> dict:
    behavior = cfg.get("behavior_v1", {})
    if not behavior.get("enabled", False):
        raise ValueError("behavior_v1.enabled must be true")
    languages = list(behavior.get("evaluation_languages", []))
    if len(languages) < 4 or "en" in languages:
        raise ValueError("behavior_v1.evaluation_languages must contain at least four non-English languages")
    available = set(cfg.get("dataset", {}).get("languages", {}))
    missing = sorted(set(languages + ["en"]) - available)
    if missing:
        raise ValueError(f"behavior languages are absent from dataset.languages: {missing}")
    demo_count = int(behavior.get("demonstration_semantic_ids", 2))
    evaluation_count = int(behavior.get("evaluation_semantic_ids", 200))
    if demo_count < 2 or evaluation_count < 5:
        raise ValueError("behavior_v1 requires at least two demonstrations and five evaluation IDs")
    decoding = behavior.get("decoding", {})
    if bool(decoding.get("do_sample", False)):
        raise ValueError("behavior_v1 main decoding must use do_sample=false")
    if int(decoding.get("num_beams", 1)) != 1:
        raise ValueError("behavior_v1 main decoding must use num_beams=1")
    if bool(decoding.get("prompt_add_special_tokens", False)):
        raise ValueError("behavior_v1 prompt encoding must not add tokenizer special tokens")
    output_extraction = str(decoding.get("output_extraction", "first_nonempty_line"))
    if output_extraction not in {"first_nonempty_line", "full_generation"}:
        raise ValueError("behavior_v1 decoding.output_extraction is unsupported")
    primary_layer = behavior.get("primary_layer")
    analysis_layers = list(behavior.get(
        "analysis_layers", [primary_layer] if primary_layer is not None else []
    ))
    if primary_layer is None or int(primary_layer) not in [int(value) for value in analysis_layers]:
        raise ValueError("behavior_v1 primary_layer must be included in analysis_layers")
    language_names = dict(behavior.get("language_names", {}))
    if not set(["en", *languages]).issubset(language_names):
        raise ValueError("behavior_v1.language_names must cover English and every evaluation language")
    generation_runtime = behavior.get("generation_runtime", {})
    maximum_batch_size = int(generation_runtime.get(
        "maximum_batch_size", decoding.get("batch_size", cfg.get("batch_size", 1))
    ))
    minimum_batch_size = int(generation_runtime.get("minimum_batch_size", 1))
    if minimum_batch_size < 1 or maximum_batch_size < minimum_batch_size:
        raise ValueError("behavior_v1 generation runtime batch sizes are invalid")
    return {
        "protocol_version": BEHAVIOR_PROTOCOL_VERSION,
        "seed": int(behavior.get("seed", cfg.get("seed", 42))),
        "evaluation_languages": languages,
        "language_names": language_names,
        "demonstration_semantic_ids": demo_count,
        "evaluation_semantic_ids": evaluation_count,
        "prompt_template": str(behavior.get(
            "prompt_template",
            "{demonstrations}\n\n{source_name}: {source_text}\n{target_name}:",
        )),
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(decoding.get("max_new_tokens", 256)),
            "use_cache": bool(decoding.get("use_cache", True)),
            "batch_size": int(decoding.get("batch_size", cfg.get("batch_size", 1))),
            "prompt_add_special_tokens": False,
            "max_prompt_tokens": int(decoding.get("max_prompt_tokens", 1024)),
            "output_extraction": output_extraction,
        },
        "primary_layer": int(primary_layer),
        "analysis_layers": [int(value) for value in analysis_layers],
        "local_scaling_k": int(behavior.get(
            "local_scaling_k", cfg.get("similarity_controls", {}).get("local_scaling_k", 5)
        )),
        "bootstrap_samples": int(behavior.get("bootstrap_samples", 1000)),
        "confidence_level": float(behavior.get("confidence_level", 0.95)),
        "language_id": dict(behavior.get("language_id", {})),
        "quality": dict(behavior.get("quality", {})),
        "resources": resource_settings(cfg),
        "generation_runtime": {
            "maximum_batch_size": maximum_batch_size,
            "minimum_batch_size": minimum_batch_size,
            "oom_backoff": bool(generation_runtime.get("oom_backoff", True)),
            "length_bucketed_batching": bool(
                generation_runtime.get("length_bucketed_batching", False)
            ),
        },
    }


def task_file(cfg) -> Path:
    return ensure_behavior_dirs(cfg).data / "behavior_tasks.jsonl"


def generation_file(cfg) -> Path:
    return ensure_behavior_dirs(cfg).generations / "generations.jsonl"


def validate_tasks(rows, settings):
    required = {
        "task_id", "semantic_id", "condition", "source_lang", "target_lang",
        "source_text", "reference_text", "prompt", "prompt_template_sha256",
    }
    if not rows:
        raise ValueError("behavior task file is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"behavior tasks are missing fields: {sorted(missing)}")
    ids = [str(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("behavior task IDs must be unique")
    expected_conditions = {
        "non_english_to_non_english", "english_to_non_english", "non_english_to_english"
    }
    if set(str(row["condition"]) for row in rows) != expected_conditions:
        raise ValueError("behavior task conditions are incomplete")
    evaluation_ids = {str(row["semantic_id"]) for row in rows}
    if len(evaluation_ids) != settings["evaluation_semantic_ids"]:
        raise ValueError(
            f"expected {settings['evaluation_semantic_ids']} evaluation IDs, got {len(evaluation_ids)}"
        )
    expected_languages = set(settings["evaluation_languages"])
    for semantic_id in evaluation_ids:
        group = [row for row in rows if str(row["semantic_id"]) == semantic_id]
        by_condition = {
            condition: [row for row in group if row["condition"] == condition]
            for condition in expected_conditions
        }
        if any(len(values) != len(expected_languages) for values in by_condition.values()):
            raise ValueError(f"behavior task condition is unbalanced for semantic ID {semantic_id}")
        nene = by_condition["non_english_to_non_english"]
        en_to_non = by_condition["english_to_non_english"]
        non_to_en = by_condition["non_english_to_english"]
        if (
            {row["source_lang"] for row in nene} != expected_languages
            or {row["target_lang"] for row in nene} != expected_languages
            or any(row["source_lang"] == row["target_lang"] for row in nene)
            or {row["source_lang"] for row in en_to_non} != {"en"}
            or {row["target_lang"] for row in en_to_non} != expected_languages
            or {row["source_lang"] for row in non_to_en} != expected_languages
            or {row["target_lang"] for row in non_to_en} != {"en"}
        ):
            raise ValueError(f"behavior task language mapping is invalid for semantic ID {semantic_id}")
    if len({str(row["prompt_template_sha256"]) for row in rows}) != 1:
        raise ValueError("behavior tasks must share one frozen prompt-template hash")
    return {
        "rows": len(rows),
        "semantic_ids": len(evaluation_ids),
        "conditions": sorted(expected_conditions),
        "task_sha256": stable_json_sha256(rows),
    }


def load_tasks(cfg):
    settings = behavior_settings(cfg)
    rows = read_jsonl(task_file(cfg))
    validate_tasks(rows, settings)
    return rows


def read_checkpoint_identity(cfg):
    path = Path(cfg["output_dir"]) / "checkpoint_identity.json"
    if not path.exists():
        raise FileNotFoundError(
            f"behavior_v1 requires checkpoint identity before generation: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.get("checkpoint_sha256")
    if not digest:
        raise ValueError(f"checkpoint identity has no checkpoint_sha256: {path}")
    return payload


def clustered_bootstrap_mean(values, clusters, seed, n_boot=1000, confidence=0.95):
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "cluster": list(map(str, clusters))})
    if frame.empty or not np.isfinite(frame.value).all():
        raise ValueError("clustered bootstrap requires finite observations")
    unique = frame.cluster.unique()
    mean = float(frame.value.mean())
    if n_boot <= 0 or len(unique) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    draws = []
    grouped = {key: group.value.to_numpy() for key, group in frame.groupby("cluster")}
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws.append(float(np.concatenate([grouped[key] for key in sampled]).mean()))
    alpha = (1 - float(confidence)) / 2
    return mean, float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def write_manifest(path, payload):
    write_json(path, {"protocol_version": BEHAVIOR_PROTOCOL_VERSION, **payload})
