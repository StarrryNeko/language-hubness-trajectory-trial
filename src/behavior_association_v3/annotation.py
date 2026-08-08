"""Create and validate the calibration-only blinded lexical-detector audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_association_v3.common import item_path, paths, settings, sha256_file, write_manifest
from common import load_config


def create(cfg, output):
    protocol = settings(cfg)
    items = pd.read_csv(item_path(cfg, "calibration"))
    positives = items.loc[items.english_lexical_leakage == 1].copy()
    rng = np.random.default_rng(protocol["seed"] + 701)
    negatives = []
    for _, group in items.loc[items.english_lexical_leakage == 0].groupby("target_lang"):
        take = min(protocol["lexical"]["audit_negative_per_language"], len(group))
        negatives.append(group.loc[rng.choice(group.index.to_numpy(), take, replace=False)])
    frame = pd.concat([positives, *negatives], ignore_index=True)
    frame = frame.sample(frac=1, random_state=protocol["seed"] + 702).reset_index(drop=True)
    frame = frame[["task_id", "semantic_id", "source_lang", "target_lang", "generated_text"]]
    frame["human_english_leakage"] = ""
    frame["annotator_notes"] = ""
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(frame)} blinded calibration audit rows to {output}")


def validate(cfg, annotations, output):
    protocol = settings(cfg)
    audit = pd.read_csv(annotations)
    items = pd.read_csv(item_path(cfg, "calibration"))
    labels = items[["task_id", "target_lang", "english_lexical_leakage"]].rename(columns={
        "target_lang": "frozen_target_lang", "english_lexical_leakage": "automatic",
    })
    if audit.task_id.duplicated().any():
        raise ValueError("audit contains duplicate task IDs")
    frame = audit.merge(labels, on="task_id", how="left", validate="one_to_one")
    if frame.automatic.isna().any() or not (frame.target_lang == frame.frozen_target_lang).all():
        raise ValueError("audit rows changed or fall outside calibration results")
    automatic = pd.to_numeric(frame.automatic, errors="raise").astype(int)
    human = pd.to_numeric(frame.human_english_leakage, errors="raise").astype(int)
    expected_positive = set(items.loc[items.english_lexical_leakage == 1, "task_id"])
    if set(frame.loc[automatic == 1, "task_id"]) != expected_positive:
        raise ValueError("every automatic calibration positive must be audited")
    for language, group in items.groupby("target_lang"):
        expected = min(
            protocol["lexical"]["audit_negative_per_language"],
            int((group.english_lexical_leakage == 0).sum()),
        )
        observed = int(((frame.target_lang == language) & (automatic == 0)).sum())
        if observed != expected:
            raise ValueError(f"calibration negative-audit count changed for {language}")
    weights = np.ones(len(frame), dtype=float)
    for language, group in frame.loc[automatic == 0].groupby("target_lang"):
        total = int(((items.target_lang == language) & (items.english_lexical_leakage == 0)).sum())
        weights[group.index] = total / len(group)
    tp = float(weights[(automatic == 1) & (human == 1)].sum())
    fp = float(weights[(automatic == 1) & (human == 0)].sum())
    fn = float(weights[(automatic == 0) & (human == 1)].sum())
    tn = float(weights[(automatic == 0) & (human == 0)].sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    gates = protocol["lexical"]
    passed = precision >= gates["minimum_precision"] and recall >= gates["minimum_recall"] and fpr <= gates["maximum_false_positive_rate"]
    write_manifest(output, {
        "source_split": "calibration", "annotation_file": str(Path(annotations).resolve()),
        "annotation_file_sha256": sha256_file(annotations),
        "calibration_item_results_sha256": sha256_file(item_path(cfg, "calibration")),
        "precision": precision, "recall": recall, "false_positive_rate": fpr,
        "rule_version": protocol["lexical"]["rule_version"],
        "weighted_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "gates": gates, "passed": bool(passed),
    })
    print(f"V3 calibration detector validation passed={passed}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--config", required=True)
    create_parser.add_argument("--output", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--annotations", required=True)
    validate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "create":
        create(cfg, Path(args.output))
    else:
        validate(cfg, Path(args.annotations), Path(args.output))


if __name__ == "__main__":
    main()
