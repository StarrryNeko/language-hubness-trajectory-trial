"""Shared contracts for the script-aware behavior_v2 pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from behavior_v2 import PROTOCOL_VERSION, REPRESENTATION_PROTOCOL
from common import json_dumps_strict, read_jsonl, write_json


ENGLISH_FUNCTION_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "in",
    "i", "is", "it", "its", "not", "of", "on", "or", "she", "that", "the",
    "their", "they", "this", "to", "was", "were", "which", "with", "you",
})
ENGLISH_STRONG_MARKERS = frozenset({
    "and", "are", "that", "the", "their", "they", "this", "was", "were", "which", "with",
})
LEXICAL_DETECTOR_RULE = "strong_marker_or_two_function_words_v1"
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True)
class BehaviorV2Paths:
    root: Path
    data: Path
    geometry: Path
    generations: Path
    metrics: Path
    measurement: Path
    figures: Path
    validation: Path


def ensure_paths(cfg) -> BehaviorV2Paths:
    configured = cfg.get("behavior_v2", {})
    root = Path(cfg["output_dir"]) / str(configured.get("result_directory", PROTOCOL_VERSION))
    paths = BehaviorV2Paths(
        root=root,
        data=root / "data",
        geometry=root / "geometry",
        generations=root / "generations",
        metrics=root / "metrics",
        measurement=root / "measurement",
        figures=root / "figures",
        validation=root / "validation",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value) -> str:
    payload = json_dumps_strict(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path, values):
    payload = {"protocol_version": PROTOCOL_VERSION, **values}
    write_json(path, payload)
    return payload


def configure_cpu_threads(count):
    value = str(int(count))
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[name] = value
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def settings(cfg):
    raw = cfg.get("behavior_v2", {})
    if not raw.get("enabled", False):
        raise ValueError("behavior_v2.enabled must be true")
    languages = list(cfg.get("dataset", {}).get("languages", {}))
    metadata = cfg.get("dataset", {}).get("language_metadata", {})
    if len(languages) < 20 or set(languages) != set(metadata):
        raise ValueError("behavior_v2 requires complete metadata for the full language inventory")
    if cfg.get("metrics", {}).get("representations") != ["mean_pool"]:
        raise ValueError("behavior_v2 permits mean_pool as the only representation")
    source_languages = list(raw.get("source_languages", []))
    target_languages = list(raw.get("non_latin_target_languages", []))
    if "en" in source_languages or "en" in target_languages:
        raise ValueError("behavior_v2 formal tasks must not use English as source or target")
    if not source_languages or not target_languages:
        raise ValueError("behavior_v2 source and target inventories are required")
    missing = set(source_languages + target_languages + ["en"]) - set(languages)
    if missing:
        raise ValueError(f"behavior_v2 languages are absent from dataset inventory: {sorted(missing)}")
    if any(metadata[language]["script"] == "Latin" for language in target_languages):
        raise ValueError("behavior_v2 confirmatory targets must be non-Latin-script languages")
    latin_languages = [
        language for language in languages if metadata[language]["script"] == "Latin"
    ]
    other_latin = [language for language in latin_languages if language != "en"]
    non_latin = [language for language in languages if language not in latin_languages]
    if "en" not in latin_languages or len(other_latin) < 3 or len(non_latin) < 3:
        raise ValueError("behavior_v2 cannot form Latin and non-Latin comparison groups")

    primary_layer = int(raw["primary_layer"])
    analysis_layers = sorted(set(map(int, raw.get("analysis_layers", [primary_layer]))))
    if primary_layer not in analysis_layers:
        raise ValueError("behavior_v2 primary layer must be included in analysis_layers")
    demos = int(raw.get("demonstration_semantic_ids", 4))
    evaluations = int(raw.get("evaluation_semantic_ids", 800))
    if demos < 2 or evaluations < 100:
        raise ValueError("behavior_v2 requires at least two demonstrations and 100 evaluation IDs")
    if int(cfg["dataset"]["sample_size_per_language"]) != demos + evaluations:
        raise ValueError("dataset sample size must equal V2 demonstrations plus evaluation IDs")

    decoding = raw.get("decoding", {})
    if bool(decoding.get("do_sample", False)) or int(decoding.get("num_beams", 1)) != 1:
        raise ValueError("behavior_v2 requires deterministic greedy decoding")
    if not bool(decoding.get("allow_natural_termination", True)):
        raise ValueError("behavior_v2 requires natural termination")
    if bool(decoding.get("prompt_add_special_tokens", False)):
        raise ValueError("behavior_v2 prompts must not add tokenizer control tokens")
    maximum_new_tokens = int(decoding.get("max_new_tokens", 256))
    maximum_prompt_tokens = int(decoding.get("max_prompt_tokens", 2048))
    if maximum_new_tokens < 1 or maximum_prompt_tokens < 1:
        raise ValueError("behavior_v2 token limits must be positive")

    runtime = raw.get("generation_runtime", {})
    batch_size = int(runtime.get("batch_size", decoding.get("batch_size", 8)))
    minimum_batch_size = int(runtime.get("minimum_batch_size", 1))
    maximum_gpu_memory_gib = float(runtime.get("maximum_gpu_memory_gib", 76.0))
    if batch_size < 1 or minimum_batch_size < 1 or minimum_batch_size > batch_size:
        raise ValueError("behavior_v2 generation batch sizes are invalid")
    if maximum_gpu_memory_gib <= 0:
        raise ValueError("behavior_v2 maximum_gpu_memory_gib must be positive")

    detector = raw.get("lexical_leakage", {})
    minimum_run = int(detector.get("minimum_latin_run_words", 3))
    if minimum_run < 2:
        raise ValueError("lexical leakage minimum run must be at least two words")
    if not bool(detector.get("require_english_function_word", True)):
        raise ValueError("formal V2 requires an English function word in a leakage span")
    if detector.get("rule_version", LEXICAL_DETECTOR_RULE) != LEXICAL_DETECTOR_RULE:
        raise ValueError("behavior_v2 lexical detector rule version changed")
    audit_negative_per_language = int(detector.get("audit_negative_per_language", 60))
    if audit_negative_per_language < 20:
        raise ValueError("behavior_v2 requires at least 20 audited negatives per target language")
    local_scaling_k = int(raw.get("local_scaling_k", 5))
    k_sensitivity = sorted(set(map(int, raw.get("k_sensitivity", [3, 5, 10]))))
    if local_scaling_k not in k_sensitivity:
        raise ValueError("behavior_v2 local_scaling_k must be included in k_sensitivity")

    resources = raw.get("resources", {})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "representation_protocol": REPRESENTATION_PROTOCOL,
        "seed": int(raw.get("seed", cfg.get("seed", 42))),
        "languages": languages,
        "language_metadata": metadata,
        "latin_languages": latin_languages,
        "other_latin_languages": other_latin,
        "non_latin_languages": non_latin,
        "source_languages": source_languages,
        "target_languages": target_languages,
        "language_names": dict(raw.get("language_names", {})),
        "demonstration_semantic_ids": demos,
        "evaluation_semantic_ids": evaluations,
        "role_assignment_seed": int(raw.get("role_assignment_seed", 20260809)),
        "prompt_template": str(raw.get(
            "prompt_template",
            "{demonstrations}\n\n{source_name}: {source_text}\n{target_name}:",
        )),
        "primary_layer": primary_layer,
        "analysis_layers": analysis_layers,
        "local_scaling_k": local_scaling_k,
        "k_sensitivity": k_sensitivity,
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": maximum_new_tokens,
            "use_cache": bool(decoding.get("use_cache", True)),
            "prompt_add_special_tokens": False,
            "allow_natural_termination": True,
            "max_prompt_tokens": maximum_prompt_tokens,
        },
        "generation_runtime": {
            "batch_size": batch_size,
            "minimum_batch_size": minimum_batch_size,
            "oom_backoff": bool(runtime.get("oom_backoff", True)),
            "maximum_gpu_memory_gib": maximum_gpu_memory_gib,
        },
        "resources": {
            "cpu_threads": int(resources.get("cpu_threads", 24)),
            "geometry_device": str(resources.get("geometry_device", "cuda")),
            "geometry_dtype": str(resources.get("geometry_dtype", "float32")),
            "allow_tf32": bool(resources.get("allow_tf32", False)),
        },
        "lexical_leakage": {
            "minimum_latin_run_words": minimum_run,
            "require_english_function_word": True,
            "rule_version": LEXICAL_DETECTOR_RULE,
            "audit_negative_per_language": audit_negative_per_language,
            "validation_report": str(detector.get(
                "validation_report", "lexical_detector_validation.json"
            )),
            "minimum_precision": float(detector.get("minimum_precision", 0.90)),
            "minimum_recall": float(detector.get("minimum_recall", 0.80)),
            "maximum_false_positive_rate": float(
                detector.get("maximum_false_positive_rate", 0.01)
            ),
        },
        "maximum_empty_output_rate": float(raw.get("maximum_empty_output_rate", 0.01)),
        "maximum_token_budget_rate": float(raw.get("maximum_token_budget_rate", 0.01)),
        "bootstrap_samples": int(raw.get("bootstrap_samples", 1000)),
        "confidence_level": float(raw.get("confidence_level", 0.95)),
    }


def task_file(cfg):
    return ensure_paths(cfg).data / "behavior_v2_tasks.jsonl"


def generation_file(cfg):
    return ensure_paths(cfg).generations / "generations.jsonl"


def read_checkpoint_identity(cfg):
    path = Path(cfg["output_dir"]) / "checkpoint_identity.json"
    if not path.exists():
        raise FileNotFoundError(f"behavior_v2 requires checkpoint identity: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint identity has no checkpoint_sha256: {path}")
    return payload


def split_natural_completion(token_ids, termination_ids, maximum):
    """Remove the native termination control and classify the finish reason."""
    ids = list(map(int, token_ids))
    for index, token_id in enumerate(ids):
        if token_id in termination_ids:
            return ids[:index], "natural_stop", token_id
    if len(ids) != int(maximum):
        raise ValueError("unterminated V2 generation did not consume the token ceiling")
    return ids, "token_budget", None


def load_tasks(cfg):
    rows = read_jsonl(task_file(cfg))
    validate_tasks(rows, settings(cfg))
    return rows


def validate_tasks(rows, protocol):
    required = {
        "task_id", "semantic_id", "condition", "source_lang", "target_lang",
        "source_text", "reference_text", "prompt", "prompt_template_sha256",
    }
    if not rows or required - set(rows[0]):
        raise ValueError("behavior_v2 task file is empty or incomplete")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("behavior_v2 task IDs must be unique")
    semantic_ids = sorted({str(row["semantic_id"]) for row in rows})
    if len(semantic_ids) != protocol["evaluation_semantic_ids"]:
        raise ValueError("behavior_v2 evaluation semantic-ID count changed")
    target_set = set(protocol["target_languages"])
    source_set = set(protocol["source_languages"])
    if {row["condition"] for row in rows} != {"non_english_to_non_latin"}:
        raise ValueError("behavior_v2 task condition changed")
    if any(row["source_lang"] not in source_set for row in rows):
        raise ValueError("behavior_v2 task contains a source outside the frozen inventory")
    if any(row["target_lang"] not in target_set for row in rows):
        raise ValueError("behavior_v2 task contains a target outside the frozen inventory")
    if any(not str(row["prompt"]).strip() for row in rows):
        raise ValueError("behavior_v2 task contains an empty prompt")
    for semantic_id in semantic_ids:
        group = [row for row in rows if str(row["semantic_id"]) == semantic_id]
        if len(group) != len(target_set) or {row["target_lang"] for row in group} != target_set:
            raise ValueError(f"behavior_v2 target balance failed for semantic ID {semantic_id}")
        if any(row["source_lang"] == row["target_lang"] for row in group):
            raise ValueError("behavior_v2 source and target languages must differ")
    for target in sorted(target_set):
        group = [row for row in rows if row["target_lang"] == target]
        counts = [
            sum(row["source_lang"] == source for row in group)
            for source in sorted(source_set - {target})
        ]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"behavior_v2 sources are unbalanced for target {target}")
    if len({row["prompt_template_sha256"] for row in rows}) != 1:
        raise ValueError("behavior_v2 prompt-template hash changed within task file")
    return {
        "rows": len(rows),
        "semantic_ids": len(semantic_ids),
        "task_sha256": stable_sha256(rows),
    }


def is_latin_word(token):
    letters = [char for char in str(token) if char.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(char, "") for char in letters)


def lexical_script_features(text, minimum_run=3):
    """Measure Latin-script intrusion and high-precision English spans."""
    text = str(text)
    letters = [char for char in text if char.isalpha()]
    latin_letters = [char for char in letters if "LATIN" in unicodedata.name(char, "")]
    tokens = WORD_PATTERN.findall(text)
    runs, current = [], []
    for token in tokens:
        if is_latin_word(token):
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    qualifying = [run for run in runs if len(run) >= int(minimum_run)]
    english_runs = []
    for run in qualifying:
        normalized = [token.lower().strip("'- ") for token in run]
        function_hits = sum(token in ENGLISH_FUNCTION_WORDS for token in normalized)
        if any(token in ENGLISH_STRONG_MARKERS for token in normalized) or function_hits >= 2:
            english_runs.append(run)
    return {
        "alphabetic_character_count": len(letters),
        "latin_character_count": len(latin_letters),
        "latin_script_fraction": len(latin_letters) / max(len(letters), 1),
        "maximum_latin_run_words": max((len(run) for run in runs), default=0),
        "has_latin_span": int(bool(qualifying)),
        "english_lexical_leakage": int(bool(english_runs)),
    }


def detector_report_path(cfg, protocol=None):
    protocol = protocol or settings(cfg)
    configured = Path(protocol["lexical_leakage"]["validation_report"])
    if configured.is_absolute():
        return configured
    return ensure_paths(cfg).measurement / configured


def load_detector_report(cfg, required=True):
    protocol = settings(cfg)
    path = detector_report_path(cfg, protocol)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"behavior_v2 lexical detector report is missing: {path}")
        return path, None
    report = json.loads(path.read_text(encoding="utf-8"))
    thresholds = protocol["lexical_leakage"]
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("lexical detector report protocol mismatch")
    if report.get("passed") is not True:
        raise ValueError("lexical detector report is not marked as passed")
    if report.get("rule_version") != thresholds["rule_version"]:
        raise ValueError("lexical detector report rule version mismatch")
    item_path = ensure_paths(cfg).metrics / "behavior_v2_item_results.csv"
    if item_path.exists() and report.get("item_results_sha256") != sha256_file(item_path):
        raise ValueError("lexical detector report belongs to different item results")
    if float(report.get("precision", -1)) < thresholds["minimum_precision"]:
        raise ValueError("lexical detector precision gate failed")
    if float(report.get("recall", -1)) < thresholds["minimum_recall"]:
        raise ValueError("lexical detector recall gate failed")
    if float(report.get("false_positive_rate", 2)) > thresholds["maximum_false_positive_rate"]:
        raise ValueError("lexical detector false-positive gate failed")
    return path, report
