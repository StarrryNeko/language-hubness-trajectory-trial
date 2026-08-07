"""Evaluate target-language retention, English leakage, and translation quality."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_common import (
    behavior_settings,
    clustered_bootstrap_mean,
    ensure_behavior_dirs,
    generation_file,
    load_tasks,
    sha256_file,
    write_manifest,
)
from common import load_config, read_jsonl


SCRIPT_RANGES = {
    "zh": ((0x3400, 0x9FFF),),
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "hi": ((0x0900, 0x097F),),
    "ru": ((0x0400, 0x052F),),
    "ja": ((0x3040, 0x30FF), (0x3400, 0x9FFF)),
}
ENGLISH_MARKERS = {
    "the", "and", "is", "are", "was", "were", "to", "of", "in", "that", "for",
    "with", "this", "from", "it", "as", "be", "on", "by", "an", "a",
}


def script_count(text, ranges):
    return sum(any(low <= ord(char) <= high for low, high in ranges) for char in text)


class LanguageIdentifier:
    def __init__(self, settings):
        cfg = settings.get("language_id", {})
        self.backend = str(cfg.get("backend", "script_heuristic"))
        self.threshold = float(cfg.get("confidence_threshold", 0.70))
        self.english_span_threshold = float(cfg.get("english_span_threshold", 0.15))
        self.label_map = dict(cfg.get("label_map", {}))
        self.model = None
        if self.backend == "fasttext":
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                raise RuntimeError(
                    "fasttext-wheel is incompatible with NumPy 2.x copy semantics; "
                    "install the frozen dependency with: python -m pip install 'numpy==1.26.4'"
                )
            model_path = cfg.get("model_path")
            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError("formal fastText LID requires behavior_v1.language_id.model_path")
            try:
                import fasttext
            except ImportError as exc:
                raise ImportError("install fasttext-wheel for formal behavior evaluation") from exc
            self.model = fasttext.load_model(str(model_path))
        elif self.backend != "script_heuristic":
            raise ValueError("language_id.backend must be fasttext or script_heuristic")

    def predict(self, text):
        normalized = " ".join(str(text).split())
        if not normalized:
            return "unknown", 0.0
        if self.model is not None:
            labels, probabilities = self.model.predict(normalized, k=1)
            label = str(labels[0]).replace("__label__", "")
            return self.label_map.get(label, label), float(probabilities[0])
        counts = {language: script_count(normalized, ranges) for language, ranges in SCRIPT_RANGES.items()}
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            visible = sum(char.isalpha() for char in normalized)
            return best, counts[best] / max(visible, 1)
        words = re.findall(r"[A-Za-z]+", normalized.lower())
        marker_fraction = sum(word in ENGLISH_MARKERS for word in words) / max(len(words), 1)
        if marker_fraction >= 0.08:
            return "en", min(1.0, 0.5 + marker_fraction)
        return "unknown", 0.0

    def english_span_fraction(self, text):
        tokens = re.findall(r"[^\W\d_]+", str(text), flags=re.UNICODE)
        eligible = [token for token in tokens if len(token) >= 2]
        if not eligible:
            return 0.0
        english = 0
        for token in eligible:
            label, confidence = self.predict(token)
            english += int(label == "en" and confidence >= self.threshold)
        return english / len(eligible)


def character_fscore(hypothesis, reference):
    from collections import Counter
    left, right = Counter(str(hypothesis)), Counter(str(reference))
    overlap = sum((left & right).values())
    precision = overlap / max(sum(left.values()), 1)
    recall = overlap / max(sum(right.values()), 1)
    return 100 * 2 * precision * recall / max(precision + recall, 1e-12)


def quality_scorer():
    try:
        from sacrebleu.metrics import CHRF
        metric = CHRF(word_order=2)
        return "sacrebleu_chrfpp", lambda hypothesis, reference: float(
            metric.sentence_score(str(hypothesis), [str(reference)]).score
        )
    except ImportError:
        return "character_fscore_smoke_only", character_fscore


def main():
    parser = argparse.ArgumentParser(description="Evaluate behavior_v1 generations")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = behavior_settings(cfg)
    paths = ensure_behavior_dirs(cfg)
    tasks = {str(row["task_id"]): row for row in load_tasks(cfg)}
    generations = read_jsonl(generation_file(cfg))
    if {str(row["task_id"]) for row in generations} != set(tasks):
        raise ValueError("generation/task IDs do not match")
    identifier = LanguageIdentifier(settings)
    quality_backend, score_quality = quality_scorer()
    def evaluate_row(row):
        task = tasks[str(row["task_id"])]
        text = str(row.get("generated_text", ""))
        predicted, confidence = identifier.predict(text)
        english_fraction = identifier.english_span_fraction(text)
        target = str(task["target_lang"])
        retention = int(predicted == target and confidence >= identifier.threshold)
        leakage = (
            np.nan if target == "en" else int(
                predicted == "en" or english_fraction >= identifier.english_span_threshold
            )
        )
        reference_label, reference_confidence = identifier.predict(task["reference_text"])
        audit_row = {
            "semantic_id": str(task["semantic_id"]),
            "target_lang": target,
            "correct": int(
                reference_label == target and reference_confidence >= identifier.threshold
            ),
            "confidence": reference_confidence,
        }
        record = {
            "task_id": task["task_id"],
            "semantic_id": task["semantic_id"],
            "model": row["model"],
            "condition": task["condition"],
            "source_lang": task["source_lang"],
            "target_lang": target,
            "generated_text": text,
            "reference_text": task["reference_text"],
            "predicted_language": predicted,
            "language_confidence": confidence,
            "target_language_retention": retention,
            "english_span_fraction": english_fraction,
            "unnecessary_english_leakage": leakage,
            "semantic_quality_chrfpp": score_quality(text, task["reference_text"]),
            "empty_output": int(not text.strip()),
            "prompt_token_count": row.get("prompt_token_count"),
            "generated_token_count": row.get("generated_token_count"),
            "finish_reason": row.get("finish_reason"),
        }
        return record, audit_row

    workers = settings["resources"]["evaluation_workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        evaluated = list(executor.map(evaluate_row, generations, chunksize=32))
    records = [record for record, _ in evaluated]
    reference_audit = [audit for _, audit in evaluated]
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(paths.metrics / "behavior_item_results.csv", index=False, encoding="utf-8")
    summaries = []
    for condition, group in frame.groupby("condition", sort=True):
        for metric in (
            "target_language_retention", "unnecessary_english_leakage", "semantic_quality_chrfpp"
        ):
            values = group[[metric, "semantic_id"]].dropna()
            if values.empty:
                continue
            mean, low, high = clustered_bootstrap_mean(
                values[metric], values.semantic_id, settings["seed"] + len(summaries),
                settings["bootstrap_samples"], settings["confidence_level"],
            )
            summaries.append({
                "condition": condition, "metric": metric,
                "mean": mean, "ci_lower": low, "ci_upper": high,
                "rows": len(values), "semantic_ids": values.semantic_id.nunique(),
            })
    pd.DataFrame(summaries).to_csv(
        paths.metrics / "behavior_model_summary.csv", index=False, encoding="utf-8"
    )
    audit = pd.DataFrame(reference_audit).drop_duplicates(
        ["semantic_id", "target_lang"]
    )
    calibration = audit.groupby("target_lang", as_index=False).agg(
        rows=("correct", "size"), accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
    )
    calibration.to_csv(paths.metrics / "language_id_reference_audit.csv", index=False, encoding="utf-8")
    write_manifest(paths.metrics / "behavior_evaluation_manifest.json", {
        "rows": len(frame),
        "semantic_ids": frame.semantic_id.nunique(),
        "language_id_backend": identifier.backend,
        "language_id_confidence_threshold": identifier.threshold,
        "english_span_threshold": identifier.english_span_threshold,
        "reference_language_id_accuracy": float(audit.correct.mean()),
        "quality_backend": quality_backend,
        "resources": settings["resources"],
        "formal_evaluation_ready": (
            identifier.backend == "fasttext" and quality_backend == "sacrebleu_chrfpp"
        ),
        "generation_file_sha256": sha256_file(generation_file(cfg)),
    })
    print(f"Saved behavior evaluation for {len(frame)} generations")


if __name__ == "__main__":
    main()
