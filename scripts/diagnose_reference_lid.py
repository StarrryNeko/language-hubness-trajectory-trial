"""Read-only diagnosis of fastText LID on frozen behavior_v1 reference texts.

The script never writes experiment results. It loads the frozen reference-text
task file (or any reference JSONL via --task-file) and reports, per target language:

- accuracy under the frozen confidence threshold, computed on the same
  (semantic_id, target_lang) audit units as evaluate_behavior_outputs.py;
- raw fastText labels before and after label_map;
- whether failures are label errors or threshold-only (correct label below
  the confidence threshold);
- confidence percentiles for correctly labeled references and a threshold
  sweep for diagnosis only;
- concrete error examples.

An optional --output path writes a JSON diagnostic report; without it the
script prints to stdout only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavior_common import behavior_settings, task_file
from common import json_dumps_strict, load_config, read_jsonl


def load_fasttext(settings):
    lid = settings.get("language_id", {})
    backend = str(lid.get("backend", "script_heuristic"))
    if backend != "fasttext":
        raise ValueError(
            "diagnose_reference_lid requires behavior_v1.language_id.backend=fasttext; "
            f"got {backend!r}"
        )
    model_path = lid.get("model_path")
    if not model_path:
        raise ValueError("behavior_v1.language_id.model_path is required")
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"fastText LID model does not exist: {path}")
    try:
        import fasttext
    except ImportError as exc:
        raise ImportError(
            "install fasttext-wheel (with NumPy<2) for reference LID diagnosis"
        ) from exc
    return fasttext.load_model(str(path)), path


def normalize(text):
    return " ".join(str(text).split())


def percentile_summary(values):
    if not values:
        return None
    array = np.asarray(sorted(values), dtype=float)
    percentiles = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
    return {
        "count": int(len(array)),
        "min": float(array[0]),
        "p5": float(np.percentile(array, 5.0)),
        "p10": float(np.percentile(array, 10.0)),
        "p25": float(np.percentile(array, 25.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p75": float(np.percentile(array, 75.0)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(array[-1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose fastText LID failures on frozen behavior_v1 reference texts."
    )
    parser.add_argument("--config", required=True, help="behavior_v1 model config")
    parser.add_argument(
        "--task-file",
        default=None,
        help="Optional JSONL override with rows containing semantic_id, "
             "target_lang, and reference_text.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for the diagnostic report; stdout-only when omitted.",
    )
    parser.add_argument(
        "--max-errors-per-language", type=int, default=10,
        help="Maximum printed error examples per target language.",
    )
    parser.add_argument(
        "--top-labels", type=int, default=8,
        help="Number of most frequent predicted labels shown per target language.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    settings = behavior_settings(cfg)
    if args.task_file:
        task_path = Path(args.task_file).resolve()
        if not task_path.exists():
            raise FileNotFoundError(f"reference task file does not exist: {task_path}")
        tasks = read_jsonl(task_path)
    else:
        task_path = task_file(cfg)
        if not task_path.exists():
            raise FileNotFoundError(
                f"behavior task file does not exist; run prepare_behavior_tasks.py first: {task_path}"
            )
        tasks = read_jsonl(task_path)
    if not tasks:
        raise ValueError(f"reference task file is empty: {task_path}")
    missing_fields = {
        field for field in ("semantic_id", "target_lang", "reference_text")
        if field not in tasks[0]
    }
    if missing_fields:
        raise ValueError(f"reference rows are missing fields: {sorted(missing_fields)}")

    lid = settings["language_id"]
    threshold = float(lid.get("confidence_threshold", 0.70))
    label_map = {str(key): str(value) for key, value in lid.get("label_map", {}).items()}
    required_accuracy = float(
        cfg.get("behavior_v1", {}).get("minimum_reference_lid_accuracy", 0.95)
    )
    model, model_path = load_fasttext(settings)

    records = []
    for row in tasks:
        target = str(row["target_lang"])
        text = normalize(row["reference_text"])
        if not text:
            records.append({
                "semantic_id": str(row["semantic_id"]),
                "target_lang": target,
                "raw_label": None,
                "mapped_label": None,
                "confidence": 0.0,
                "label_correct": False,
                "confidence_pass": False,
                "correct": False,
                "empty_reference": True,
            })
            continue
        raw_labels, probabilities = model.predict(text, k=1)
        raw_label = str(raw_labels[0]).replace("__label__", "")
        mapped_label = label_map.get(raw_label, raw_label)
        confidence = float(probabilities[0])
        label_correct = mapped_label == target
        confidence_pass = confidence >= threshold
        records.append({
            "semantic_id": str(row["semantic_id"]),
            "target_lang": target,
            "raw_label": raw_label,
            "mapped_label": mapped_label,
            "confidence": confidence,
            "label_correct": bool(label_correct),
            "confidence_pass": bool(confidence_pass),
            "correct": bool(label_correct and confidence_pass),
            "empty_reference": False,
        })

    # Match the frozen evaluation gate: one audit unit per (semantic_id, target_lang).
    unique_records = []
    seen_units = set()
    for record in records:
        unit = (record["semantic_id"], record["target_lang"])
        if unit not in seen_units:
            seen_units.add(unit)
            unique_records.append(record)
    records = unique_records

    by_language = {}
    error_examples = {}
    for target in sorted({record["target_lang"] for record in records}):
        group = [record for record in records if record["target_lang"] == target]
        correct = sum(record["correct"] for record in group)
        label_correct = sum(record["label_correct"] for record in group)
        confidence_pass = sum(record["confidence_pass"] for record in group)
        failures = [record for record in group if not record["correct"]]
        label_errors = [
            record for record in failures
            if not record["label_correct"] and not record["empty_reference"]
        ]
        threshold_only = [
            record for record in failures
            if record["label_correct"] and not record["confidence_pass"]
        ]
        raw_counts = {}
        mapped_counts = {}
        for record in group:
            raw_counts[record["raw_label"]] = raw_counts.get(record["raw_label"], 0) + 1
            mapped_counts[record["mapped_label"]] = mapped_counts.get(
                record["mapped_label"], 0
            ) + 1
        top_raw = sorted(raw_counts.items(), key=lambda item: (-item[1], str(item[0])))
        top_mapped = sorted(mapped_counts.items(), key=lambda item: (-item[1], str(item[0])))
        mean_confidence = sum(record["confidence"] for record in group) / len(group)
        correct_label_confidences = [
            record["confidence"] for record in group if record["label_correct"]
        ]
        confidence_percentiles = percentile_summary(correct_label_confidences)
        examples = []
        for record in sorted(
            failures, key=lambda item: (-item["confidence"], item["semantic_id"])
        )[: max(1, args.max_errors_per_language)]:
            row = next(
                candidate for candidate in tasks
                if str(candidate["semantic_id"]) == record["semantic_id"]
                and str(candidate["target_lang"]) == target
            )
            text = normalize(row["reference_text"])
            examples.append({
                "semantic_id": record["semantic_id"],
                "raw_label": record["raw_label"],
                "mapped_label": record["mapped_label"],
                "confidence": record["confidence"],
                "failure_type": (
                    "empty_reference" if record["empty_reference"]
                    else "label_error" if not record["label_correct"]
                    else "threshold_only"
                ),
                "reference_excerpt": text[:160],
            })
        by_language[target] = {
            "rows": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "mean_confidence": mean_confidence,
            "label_correct_rate": label_correct / len(group),
            "confidence_pass_rate": confidence_pass / len(group),
            "failures": len(failures),
            "label_errors": len(label_errors),
            "threshold_only": len(threshold_only),
            "correct_label_confidence_percentiles": confidence_percentiles,
            "top_raw_labels": top_raw[: max(1, args.top_labels)],
            "top_mapped_labels": top_mapped[: max(1, args.top_labels)],
        }
        if examples:
            error_examples[target] = examples

    total = len(records)
    total_correct = sum(record["correct"] for record in records)
    total_failures = total - total_correct
    overall_accuracy = total_correct / total
    failure_languages = {
        target: summary
        for target, summary in by_language.items()
        if summary["failures"] > 0
    }

    threshold_candidates = sorted({
        0.0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60,
        0.625, 0.65, 0.675, 0.687, 0.69, 0.70, 0.75,
    })
    threshold_sweep = []
    for candidate in threshold_candidates:
        passed = sum(
            1 for record in records
            if record["label_correct"] and record["confidence"] >= candidate
        )
        accuracy = passed / total
        threshold_sweep.append({
            "threshold": candidate,
            "accuracy": accuracy,
            "pass": bool(accuracy >= required_accuracy),
            "recovered_correct": int(passed),
        })

    report = {
        "protocol_version": "behavior_v1_reference_lid_diagnostic_v1",
        "config_path": str(config_path),
        "task_file": str(task_path),
        "task_file_override": bool(args.task_file),
        "model_path": str(model_path),
        "confidence_threshold": threshold,
        "required_accuracy": required_accuracy,
        "label_map": label_map,
        "task_rows": len(tasks),
        "audit_rows": total,
        "semantic_ids": len({record["semantic_id"] for record in records}),
        "overall_accuracy": overall_accuracy,
        "overall_pass": bool(overall_accuracy >= required_accuracy),
        "total_failures": total_failures,
        "failure_languages": failure_languages,
        "by_language": by_language,
        "failure_examples": error_examples,
        "threshold_sweep": threshold_sweep,
        "note": (
            "Diagnostic threshold sweep is computed on the frozen evaluation "
            "references and is for diagnosis only; the final frozen threshold "
            "must be chosen on an independent calibration set before rerunning."
        ),
    }

    print(f"Reference LID diagnosis: {task_path}")
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold:.3f} | Required accuracy: {required_accuracy:.3f}")
    print(f"Audit units: {total} (task rows: {len(tasks)}; duplicate condition rows removed)")
    print(f"Overall: {total_correct}/{total} = {overall_accuracy:.4f} "
          f"({'PASS' if report['overall_pass'] else 'FAIL'})")
    print()
    header = (
        f"{'lang':<6}{'rows':>6}{'correct':>8}{'accuracy':>10}"
        f"{'mean_conf':>11}{'label_ok':>9}{'conf_ok':>9}{'fail':>7}"
    )
    print(header)
    print("-" * len(header))
    for target, summary in sorted(by_language.items()):
        print(
            f"{target:<6}{summary['rows']:>6}{summary['correct']:>8}"
            f"{summary['accuracy']:>10.3f}{summary['mean_confidence']:>11.3f}"
            f"{summary['label_correct_rate']:>9.3f}{summary['confidence_pass_rate']:>9.3f}"
            f"{summary['failures']:>7}"
        )
    print()
    print("Correct-label confidence percentiles:")
    for target, summary in sorted(by_language.items()):
        percentile = summary["correct_label_confidence_percentiles"]
        if percentile is None:
            continue
        print(
            f"  {target:<4} n={percentile['count']:<4} "
            f"min={percentile['min']:.3f} p5={percentile['p5']:.3f} "
            f"p10={percentile['p10']:.3f} p25={percentile['p25']:.3f} "
            f"p50={percentile['p50']:.3f} p75={percentile['p75']:.3f} "
            f"p90={percentile['p90']:.3f} max={percentile['max']:.3f}"
        )
    print()
    print("Threshold sweep (diagnostic only, frozen references):")
    print(f"{'threshold':>9}{'accuracy':>11}{'pass':>6}")
    for item in threshold_sweep:
        print(f"{item['threshold']:>9.3f}{item['accuracy']:>11.4f}{'YES' if item['pass'] else 'NO':>6}")
    print()
    for target, summary in sorted(failure_languages.items()):
        print(
            f"== {target}: {summary['failures']} failures "
            f"(label errors={summary['label_errors']}, threshold-only={summary['threshold_only']}) =="
        )
        print("  top raw labels:", summary["top_raw_labels"])
        print("  top mapped labels:", summary["top_mapped_labels"])
        for example in error_examples.get(target, []):
            print(
                f"  id={example['semantic_id']} raw={example['raw_label']!r} "
                f"mapped={example['mapped_label']!r} conf={example['confidence']:.3f} "
                f"type={example['failure_type']}"
            )
            print(f"    ref={example['reference_excerpt']!r}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json_dumps_strict(report, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nDiagnostic report written to {output_path}")


if __name__ == "__main__":
    main()