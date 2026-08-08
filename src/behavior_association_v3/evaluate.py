"""Evaluate a V3 split and enforce pre-registered generation-quality gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from behavior_association_v3.common import (
    generation_path, item_path, lexical_script_features, load_tasks, paths,
    settings, sha256_file, task_path, write_manifest,
)
from behavior_v1.evaluate import LanguageIdentifier, quality_scorer
from common import load_config, read_jsonl


def repetition_4gram_fraction(text):
    words = str(text).split()
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(value - 1 for value in counts.values() if value > 1) / len(grams)


def detector_report(cfg, required):
    path = paths(cfg).measurement / "lexical_detector_validation.json"
    if not path.exists():
        if required:
            raise FileNotFoundError("formal V3 evaluation requires calibration-set detector validation")
        return path, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = settings(cfg)
    gates = protocol["lexical"]
    valid = (
        payload.get("protocol_version") == "behavior_association_v3"
        and payload.get("passed") is True
        and payload.get("source_split") == "calibration"
        and payload.get("rule_version") == gates["rule_version"]
        and float(payload.get("precision", -1)) >= gates["minimum_precision"]
        and float(payload.get("recall", -1)) >= gates["minimum_recall"]
        and float(payload.get("false_positive_rate", 2)) <= gates["maximum_false_positive_rate"]
    )
    calibration_items = item_path(cfg, "calibration")
    if calibration_items.exists():
        valid = valid and payload.get("calibration_item_results_sha256") == sha256_file(calibration_items)
    if not valid:
        raise ValueError("calibration lexical-detector report failed the frozen V3 gates")
    return path, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("calibration", "formal"), required=True)
    parser.add_argument("--allow-unvalidated-detector", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    tasks = load_tasks(cfg, args.split)
    generations = read_jsonl(generation_path(cfg, args.split))
    by_id = {row["task_id"]: row for row in generations}
    if len(by_id) != len(generations) or set(by_id) != {row["task_id"] for row in tasks}:
        raise ValueError("V3 generations do not exactly match tasks")
    detector_path, detector = detector_report(
        cfg, required=(args.split == "formal" and not args.allow_unvalidated_detector)
    )
    identifier = LanguageIdentifier({"language_id": protocol["target_language_gate"]})
    language_id_model_path = Path(protocol["target_language_gate"]["model_path"])
    quality_backend, quality = quality_scorer()
    records = []
    for task in tasks:
        generation = by_id[task["task_id"]]
        text = str(generation.get("generated_text", "")).strip()
        predicted_language, predicted_confidence = identifier.predict(text)
        reference_language, reference_confidence = identifier.predict(task["reference_text"])
        records.append({
            "task_id": task["task_id"], "split": args.split,
            "semantic_id": task["semantic_id"], "condition": task["condition"],
            "source_lang": task["source_lang"], "target_lang": task["target_lang"],
            "generated_text": text, "reference_text": task["reference_text"],
            "predicted_language": predicted_language,
            "predicted_language_confidence": predicted_confidence,
            "target_language_retention_top1": int(predicted_language == task["target_lang"]),
            "reference_predicted_language": reference_language,
            "reference_language_confidence": reference_confidence,
            "reference_language_top1_correct": int(reference_language == task["target_lang"]),
            **lexical_script_features(text, protocol["lexical"]["minimum_latin_run_words"]),
            "semantic_quality_chrfpp": quality(text, task["reference_text"]),
            "repetition_4gram_fraction": repetition_4gram_fraction(text),
            "empty_output": int(not text),
            "prompt_token_count": int(generation["prompt_token_count"]),
            "generated_token_count": int(generation["generated_token_count"]),
            "finish_reason": generation["finish_reason"],
        })
    frame = pd.DataFrame(records)
    destination = item_path(cfg, args.split)
    frame.to_csv(destination, index=False, encoding="utf-8")
    generation_manifest_path = paths(cfg).generations / f"{args.split}_generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    observed = {
        "empty_output_rate": float(frame.empty_output.mean()),
        "token_ceiling_rate": float((frame.finish_reason == "token_ceiling").mean()),
        "mean_repetition_4gram_fraction": float(frame.repetition_4gram_fraction.mean()),
        "english_lexical_leakage_events": int(frame.english_lexical_leakage.sum()),
        "repetition_boundary_rate": float((frame.finish_reason == "repetition_boundary").mean()),
        "reference_language_top1_accuracy": float(frame.reference_language_top1_correct.mean()),
        "output_target_language_retention": float(frame.target_language_retention_top1.mean()),
    }
    retention_by_target = {
        str(language): float(group.target_language_retention_top1.mean())
        for language, group in frame.groupby("target_lang", sort=True)
    }
    gates = protocol["gates"]
    blockers = []
    allowed_finish_reasons = {
        "native_eos", "text_boundary", "repetition_boundary", "token_ceiling"
    }
    if not set(frame.finish_reason).issubset(allowed_finish_reasons):
        blockers.append("generation contains an unknown finish reason")
    native_rows = [row for row in generations if row.get("finish_reason") == "native_eos"]
    if any(row.get("termination_token_id") is None or row.get("termination_position") is None for row in native_rows):
        blockers.append("native-EOS rows lack termination token audit fields")
    ceiling_rows = [row for row in generations if row.get("finish_reason") == "token_ceiling"]
    if any(int(row.get("raw_generated_token_count", -1)) < protocol["decoding"]["max_new_tokens"] for row in ceiling_rows):
        blockers.append("token-ceiling row ended below the safety limit")
    if generation_manifest.get("forced_eos_token_id_disabled") is not True:
        blockers.append("generation did not disable length-forced EOS")
    if not generation_manifest.get("native_eos_token_ids"):
        blockers.append("generation manifest lacks native EOS token IDs")
    if generation_manifest.get("prompt_special_tokens_added") is not False:
        blockers.append("prompt tokenization unexpectedly added control tokens")
    if generation_manifest.get("generated_forbidden_special_token_count") != 0:
        blockers.append("generated output contains forbidden non-EOS special tokens")
    if observed["empty_output_rate"] > gates["maximum_empty_output_rate"]:
        blockers.append("empty-output gate failed")
    if observed["token_ceiling_rate"] > gates["maximum_token_ceiling_rate"]:
        blockers.append("token-ceiling gate failed")
    if observed["mean_repetition_4gram_fraction"] > gates["maximum_mean_repetition_4gram_fraction"]:
        blockers.append("repetition gate failed")
    if observed["repetition_boundary_rate"] > gates["maximum_repetition_boundary_rate"]:
        blockers.append("repetition-boundary rate gate failed")
    target_gate = protocol["target_language_gate"]
    if observed["reference_language_top1_accuracy"] < target_gate["minimum_reference_top1_accuracy"]:
        blockers.append("reference target-language identification gate failed")
    if observed["output_target_language_retention"] < target_gate["minimum_output_target_retention"]:
        blockers.append("overall output target-language retention gate failed")
    failed_targets = [
        language for language, value in retention_by_target.items()
        if value < target_gate["minimum_per_target_retention"]
    ]
    if failed_targets:
        blockers.append(f"per-target output retention gate failed: {failed_targets}")
    if quality_backend != "sacrebleu_chrfpp":
        blockers.append("formal chrF++ backend is unavailable")
    if args.split == "formal" and detector is None:
        blockers.append("formal split lacks a passed calibration detector")
    status = "PASS" if not blockers else "FAIL"
    write_manifest(paths(cfg).metrics / f"{args.split}_evaluation_manifest.json", {
        "config_path": str(config_path), "split": args.split, "rows": len(frame),
        "task_file_sha256": sha256_file(task_path(cfg, args.split)),
        "generation_file_sha256": sha256_file(generation_path(cfg, args.split)),
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "item_results_sha256": sha256_file(destination),
        "quality_backend": quality_backend,
        "detector_validation_sha256": sha256_file(detector_path) if detector else None,
        "observed": observed, "retention_by_target": retention_by_target,
        "gates": gates, "target_language_gate": target_gate,
        "language_id_backend": identifier.backend,
        "language_id_model_path": str(language_id_model_path.resolve()),
        "language_id_model_sha256": (
            sha256_file(language_id_model_path) if identifier.backend == "fasttext" else None
        ),
        "status": status, "blockers": blockers,
        "primary_estimable": observed["english_lexical_leakage_events"] >= gates["minimum_primary_events"],
    })
    print(f"V3 {args.split} evaluation: {status}; {observed}")
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
