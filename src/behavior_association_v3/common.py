"""Configuration, paths, hashes, and task contracts for behavior_association_v3."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from behavior_association_v3 import PROTOCOL_VERSION, REPRESENTATION_PROTOCOL
from behavior_v2.common import LEXICAL_DETECTOR_RULE, lexical_script_features
from common import json_dumps_strict, read_jsonl, write_json


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    generations: Path
    metrics: Path
    measurement: Path
    validation: Path


def paths(cfg):
    root = Path(cfg["output_dir"]) / PROTOCOL_VERSION
    result = Paths(
        root, root / "data", root / "generations", root / "metrics",
        root / "measurement", root / "validation",
    )
    for value in result.__dict__.values():
        value.mkdir(parents=True, exist_ok=True)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value):
    return hashlib.sha256(
        json_dumps_strict(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_manifest(path, values):
    payload = {"protocol_version": PROTOCOL_VERSION, **values}
    write_json(path, payload)
    return payload


def configure_cpu_threads(count):
    # One global pool is intentionally divided by callers; avoid 24 threads per library.
    value = str(max(1, int(count)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def trim_completion(text, stop_strings):
    """Return the first frozen answer span and whether a text boundary was seen."""
    text = str(text)
    hits = [(text.find(marker), marker) for marker in stop_strings if text.find(marker) >= 0]
    if hits:
        index, marker = min(hits, key=lambda item: item[0])
        return text[:index].strip(), True, marker
    return text.strip(), False, None


def classify_finish(eos_position, has_text_boundary, generated_count, maximum):
    """Classify the earliest recognized completion boundary.

    Text is inspected only before the first EOS, so a detected text boundary is
    necessarily earlier than EOS and must take precedence in the audit label.
    """
    if has_text_boundary:
        return "text_boundary"
    if eos_position is not None:
        return "native_eos"
    if int(generated_count) >= int(maximum):
        return "token_ceiling"
    raise ValueError("generation stopped without a recognized V3 boundary")


def settings(cfg):
    raw = cfg.get(PROTOCOL_VERSION, {})
    if not raw.get("enabled", False):
        raise ValueError(f"{PROTOCOL_VERSION}.enabled must be true")
    languages = list(cfg["dataset"]["languages"])
    metadata = cfg["dataset"]["language_metadata"]
    sources = list(raw["source_languages"])
    targets = list(raw["target_languages"])
    if "en" in sources or "en" in targets:
        raise ValueError("English cannot be a V3 source or target")
    if any(metadata[x]["script"] == "Latin" for x in targets):
        raise ValueError("V3 confirmatory targets must be non-Latin")
    if not set(sources + targets).issubset(languages):
        raise ValueError("V3 task languages are outside the dataset inventory")
    split = raw.get("semantic_split", {})
    counts = {
        "demonstration": int(split.get("demonstration", 4)),
        "calibration": int(split.get("calibration", 80)),
        "formal": int(split.get("formal", 720)),
    }
    if sum(counts.values()) != int(cfg["dataset"]["sample_size_per_language"]):
        raise ValueError("demonstration/calibration/formal counts must consume the frozen sample")
    decoding = raw.get("decoding", {})
    if decoding.get("do_sample", False) or int(decoding.get("num_beams", 1)) != 1:
        raise ValueError("V3 requires deterministic greedy decoding")
    stop_strings = list(decoding.get("stop_strings", ["\n"]))
    if not stop_strings:
        raise ValueError("V3 requires a frozen text-boundary stop string")
    runtime = raw.get("generation_runtime", {})
    minimum = int(runtime.get("minimum_batch_size", 1))
    initial = int(runtime.get("initial_batch_size", 8))
    maximum = int(runtime.get("maximum_batch_size", initial))
    if not 1 <= minimum <= initial <= maximum:
        raise ValueError("invalid V3 adaptive batch range")
    lexical = raw.get("lexical_leakage", {})
    return {
        "seed": int(raw.get("seed", 20260810)),
        "role_seed": int(raw.get("role_assignment_seed", 20260810)),
        "languages": languages,
        "metadata": metadata,
        "source_languages": sources,
        "target_languages": targets,
        "language_names": dict(raw["language_names"]),
        "counts": counts,
        "prompt_template": str(raw["prompt_template"]),
        "representation_protocol": REPRESENTATION_PROTOCOL,
        "primary_layer": int(raw["primary_layer"]),
        "analysis_layers": sorted(set(map(int, raw["analysis_layers"]))),
        "local_scaling_k": int(raw.get("local_scaling_k", 5)),
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(decoding.get("max_new_tokens", 192)),
            "max_prompt_tokens": int(decoding.get("max_prompt_tokens", 4096)),
            "use_cache": bool(decoding.get("use_cache", True)),
            "stop_strings": stop_strings,
            "prompt_add_special_tokens": False,
        },
        "runtime": {
            "minimum_batch_size": minimum,
            "initial_batch_size": initial,
            "maximum_batch_size": maximum,
            "adaptive_growth": bool(runtime.get("adaptive_growth", True)),
            "growth_factor": float(runtime.get("growth_factor", 1.25)),
            "maximum_gpu_memory_gib": float(runtime.get("maximum_gpu_memory_gib", 91.0)),
            "target_gpu_memory_fraction": float(runtime.get("target_gpu_memory_fraction", 0.94)),
        },
        "gates": {
            "maximum_empty_output_rate": float(raw.get("maximum_empty_output_rate", 0.01)),
            "maximum_token_ceiling_rate": float(raw.get("maximum_token_ceiling_rate", 0.01)),
            "maximum_mean_repetition_4gram_fraction": float(raw.get("maximum_mean_repetition_4gram_fraction", 0.02)),
            "minimum_primary_events": int(raw.get("minimum_primary_events", 30)),
        },
        "lexical": {
            "rule_version": LEXICAL_DETECTOR_RULE,
            "minimum_latin_run_words": int(lexical.get("minimum_latin_run_words", 3)),
            "audit_negative_per_language": int(lexical.get("audit_negative_per_language", 40)),
            "minimum_precision": float(lexical.get("minimum_precision", 0.90)),
            "minimum_recall": float(lexical.get("minimum_recall", 0.80)),
            "maximum_false_positive_rate": float(lexical.get("maximum_false_positive_rate", 0.02)),
        },
        "cpu_threads": int(raw.get("cpu_threads", 24)),
    }


def task_path(cfg, split):
    if split not in {"calibration", "formal"}:
        raise ValueError("split must be calibration or formal")
    return paths(cfg).data / f"{split}_tasks.jsonl"


def generation_path(cfg, split):
    return paths(cfg).generations / f"{split}_generations.jsonl"


def item_path(cfg, split):
    return paths(cfg).metrics / f"{split}_item_results.csv"


def load_tasks(cfg, split):
    rows = read_jsonl(task_path(cfg, split))
    protocol = settings(cfg)
    expected_ids = protocol["counts"][split]
    expected_rows = expected_ids * len(protocol["target_languages"])
    if len(rows) != expected_rows or len({row["task_id"] for row in rows}) != expected_rows:
        raise ValueError(f"{split} task cardinality or uniqueness failed")
    if {row.get("split") for row in rows} != {split}:
        raise ValueError(f"{split} task file contains the wrong split")
    return rows


def read_checkpoint(cfg):
    path = Path(cfg["output_dir"]) / "checkpoint_identity.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("checkpoint_sha256"):
        raise ValueError("checkpoint identity is incomplete")
    return payload
