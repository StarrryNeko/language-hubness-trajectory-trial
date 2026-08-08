"""Evaluate script intrusion, English lexical leakage, and chrF++ for V2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_v1.common import clustered_bootstrap_mean
from behavior_v1.evaluate import quality_scorer
from behavior_v2.common import (
    ensure_paths, generation_file, lexical_script_features, load_detector_report,
    load_tasks, settings, sha256_file, write_manifest,
)
from common import load_config, read_jsonl


def repetition_4gram_fraction(text):
    tokens = str(text).split()
    grams = [tuple(tokens[index:index + 4]) for index in range(max(0, len(tokens) - 3))]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def main():
    parser = argparse.ArgumentParser(description="Evaluate behavior_v2 outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--allow-unvalidated-detector", action="store_true",
        help="Development-only pass used to create the blinded annotation sample.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    paths = ensure_paths(cfg)
    tasks = load_tasks(cfg)
    generations = read_jsonl(generation_file(cfg))
    by_id = {str(row["task_id"]): row for row in generations}
    if len(by_id) != len(generations) or set(by_id) != {str(row["task_id"]) for row in tasks}:
        raise ValueError("behavior_v2 generations do not match frozen tasks")
    quality_backend, quality = quality_scorer()
    records = []
    for task in tasks:
        generation = by_id[str(task["task_id"])]
        text = str(generation.get("generated_text", "")).strip()
        features = lexical_script_features(
            text, protocol["lexical_leakage"]["minimum_latin_run_words"]
        )
        records.append({
            "task_id": task["task_id"],
            "semantic_id": task["semantic_id"],
            "model": generation["model"],
            "condition": task["condition"],
            "source_lang": task["source_lang"],
            "target_lang": task["target_lang"],
            "source_script": task["source_script"],
            "target_script": task["target_script"],
            "generated_text": text,
            "reference_text": task["reference_text"],
            **features,
            "semantic_quality_chrfpp": quality(text, task["reference_text"]),
            "repetition_4gram_fraction": repetition_4gram_fraction(text),
            "empty_output": int(not text),
            "prompt_token_count": int(generation["prompt_token_count"]),
            "generated_token_count": int(generation["generated_token_count"]),
            "finish_reason": generation["finish_reason"],
        })
    items = pd.DataFrame.from_records(records)
    item_path = paths.metrics / "behavior_v2_item_results.csv"
    items.to_csv(item_path, index=False, encoding="utf-8")
    detector_path, detector_report = load_detector_report(
        cfg, required=not args.allow_unvalidated_detector
    )
    summary_records = []
    for target, group in items.groupby("target_lang", sort=True):
        for metric in (
            "latin_script_fraction", "has_latin_span", "english_lexical_leakage",
            "semantic_quality_chrfpp", "repetition_4gram_fraction", "empty_output",
        ):
            mean, low, high = clustered_bootstrap_mean(
                group[metric].to_numpy(dtype=float),
                group.semantic_id.astype(str).to_numpy(),
                seed=protocol["seed"] + sum(map(ord, target + metric)),
                n_boot=protocol["bootstrap_samples"],
                confidence=protocol["confidence_level"],
            )
            summary_records.append({
                "target_lang": target,
                "metric": metric,
                "mean": mean,
                "ci_lower": low,
                "ci_upper": high,
                "rows": len(group),
                "semantic_ids": group.semantic_id.nunique(),
            })
    summary = pd.DataFrame.from_records(summary_records)
    summary.to_csv(paths.metrics / "behavior_v2_model_summary.csv", index=False, encoding="utf-8")
    generation_manifest_path = paths.generations / "generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    write_manifest(paths.metrics / "evaluation_manifest.json", {
        "config_path": str(config_path),
        "rows": len(items),
        "item_results_sha256": sha256_file(item_path),
        "generation_file_sha256": sha256_file(generation_file(cfg)),
        "task_file_sha256": sha256_file(paths.data / "behavior_v2_tasks.jsonl"),
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "quality_backend": quality_backend,
        "lexical_detector": {
            "minimum_latin_run_words": protocol["lexical_leakage"]["minimum_latin_run_words"],
            "require_english_function_word": True,
            "rule_version": protocol["lexical_leakage"]["rule_version"],
            "validation_report_path": str(detector_path),
            "validation_report_sha256": (
                sha256_file(detector_path) if detector_report is not None else None
            ),
            "validated": detector_report is not None,
        },
        "natural_stop_rate": generation_manifest["natural_stop_rate"],
        "token_budget_rate": generation_manifest["token_budget_rate"],
        "empty_output_rate": float(items.empty_output.mean()),
        "activation_intervention": False,
        "formal_evaluation_ready": bool(
            detector_report is not None and quality_backend == "sacrebleu_chrfpp"
        ),
    })
    print(
        f"behavior_v2 evaluation saved to {item_path}; "
        f"detector_validated={detector_report is not None}"
    )


if __name__ == "__main__":
    main()
