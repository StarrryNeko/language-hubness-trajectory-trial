"""Create and validate a blinded manual audit for the V2 lexical detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_v2.common import ensure_paths, settings, sha256_file, write_manifest
from common import load_config


def create_sample(cfg, output, negative_per_language):
    protocol = settings(cfg)
    frozen_negative_count = protocol["lexical_leakage"]["audit_negative_per_language"]
    if int(negative_per_language) != frozen_negative_count:
        raise ValueError(
            f"formal V2 freezes negative audit rows per language at {frozen_negative_count}"
        )
    paths = ensure_paths(cfg)
    items = pd.read_csv(paths.metrics / "behavior_v2_item_results.csv")
    positives = items.loc[items.english_lexical_leakage == 1].copy()
    rng = np.random.default_rng(protocol["seed"] + 701)
    negatives = []
    for _, group in items.loc[items.english_lexical_leakage == 0].groupby("target_lang"):
        take = min(int(negative_per_language), len(group))
        indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
        negatives.append(group.loc[indices])
    sample = pd.concat([positives, *negatives], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=protocol["seed"] + 702).reset_index(drop=True)
    # The automatic decision is deliberately omitted from the annotation file.
    # Validation restores it from immutable item results by task_id.
    sample = sample[[
        "task_id", "semantic_id", "source_lang", "target_lang", "generated_text",
    ]]
    sample["human_english_leakage"] = ""
    sample["annotator_notes"] = ""
    sample.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(sample)} blinded-audit rows to {output}")


def validate_annotations(cfg, annotations, output):
    protocol = settings(cfg)
    frame = pd.read_csv(annotations)
    required = {"task_id", "human_english_leakage", "target_lang"}
    if required - set(frame):
        raise ValueError("annotation CSV is missing required columns")
    if frame.task_id.duplicated().any():
        raise ValueError("annotation CSV contains duplicate task IDs")
    item_path = ensure_paths(cfg).metrics / "behavior_v2_item_results.csv"
    items = pd.read_csv(item_path)
    labels = items[["task_id", "target_lang", "english_lexical_leakage"]].rename(
        columns={
            "target_lang": "frozen_target_lang",
            "english_lexical_leakage": "automatic_english_leakage",
        }
    )
    frame = frame.merge(labels, on="task_id", how="left", validate="one_to_one")
    if frame.automatic_english_leakage.isna().any():
        raise ValueError("annotation CSV contains task IDs outside item results")
    if not (frame.target_lang.astype(str) == frame.frozen_target_lang.astype(str)).all():
        raise ValueError("annotation CSV changed one or more frozen target languages")
    automatic = pd.to_numeric(frame.automatic_english_leakage, errors="raise").astype(int)
    human = pd.to_numeric(frame.human_english_leakage, errors="raise").astype(int)
    if not set(automatic.unique()).issubset({0, 1}) or not set(human.unique()).issubset({0, 1}):
        raise ValueError("annotation labels must be binary 0/1")
    expected_positive_ids = set(
        items.loc[items.english_lexical_leakage == 1, "task_id"].astype(str)
    )
    audited_positive_ids = set(frame.loc[automatic == 1, "task_id"].astype(str))
    if audited_positive_ids != expected_positive_ids:
        raise ValueError("annotation CSV must contain every automatic positive exactly once")
    frozen_negative_count = protocol["lexical_leakage"]["audit_negative_per_language"]
    for language, full_group in items.groupby("target_lang"):
        total_negative = int((full_group.english_lexical_leakage == 0).sum())
        audited_negative = int(
            ((frame.target_lang.astype(str) == str(language)) & (automatic == 0)).sum()
        )
        if audited_negative != min(frozen_negative_count, total_negative):
            raise ValueError(f"annotation negative sample count changed for {language}")
    # All automatic positives are audited. Automatic negatives are a stratified
    # random sample, so weight them back to the full target-language strata.
    weights = np.ones(len(frame), dtype=float)
    for language, group in frame.loc[automatic == 0].groupby("target_lang"):
        sampled = len(group)
        total = int(
            ((items.target_lang.astype(str) == str(language))
             & (items.english_lexical_leakage == 0)).sum()
        )
        if sampled < 1 or total < sampled:
            raise ValueError(f"invalid negative audit stratum for {language}")
        weights[group.index.to_numpy()] = total / sampled
    weighted_tp = float(weights[(automatic == 1) & (human == 1)].sum())
    weighted_fp = float(weights[(automatic == 1) & (human == 0)].sum())
    weighted_fn = float(weights[(automatic == 0) & (human == 1)].sum())
    weighted_tn = float(weights[(automatic == 0) & (human == 0)].sum())
    precision = weighted_tp / max(weighted_tp + weighted_fp, 1.0)
    recall = weighted_tp / max(weighted_tp + weighted_fn, 1.0)
    false_positive_rate = weighted_fp / max(weighted_fp + weighted_tn, 1.0)
    raw_confusion = {
        "tp": int(((automatic == 1) & (human == 1)).sum()),
        "fp": int(((automatic == 1) & (human == 0)).sum()),
        "fn": int(((automatic == 0) & (human == 1)).sum()),
        "tn": int(((automatic == 0) & (human == 0)).sum()),
    }
    by_language = {}
    weighted_frame = frame.assign(_a=automatic, _h=human, _w=weights)
    for language, group in weighted_frame.groupby("target_lang"):
        lang_fp = float(group.loc[(group._a == 1) & (group._h == 0), "_w"].sum())
        lang_tn = float(group.loc[(group._a == 0) & (group._h == 0), "_w"].sum())
        by_language[str(language)] = {
            "rows": len(group),
            "false_positive_rate": lang_fp / max(lang_fp + lang_tn, 1.0),
        }
    maximum_language_fpr = max(
        (value["false_positive_rate"] for value in by_language.values()), default=1.0
    )
    gates = protocol["lexical_leakage"]
    passed = bool(
        precision >= gates["minimum_precision"]
        and recall >= gates["minimum_recall"]
        and false_positive_rate <= gates["maximum_false_positive_rate"]
        and maximum_language_fpr <= gates["maximum_false_positive_rate"]
    )
    write_manifest(output, {
        "annotation_file": str(Path(annotations).resolve()),
        "annotation_file_sha256": sha256_file(annotations),
        "item_results_sha256": sha256_file(item_path),
        "rows": len(frame),
        "sampling": "all automatic positives plus stratified random automatic negatives",
        "rule_version": gates["rule_version"],
        "audited_confusion": raw_confusion,
        "weighted_confusion_estimate": {
            "tp": weighted_tp, "fp": weighted_fp,
            "fn": weighted_fn, "tn": weighted_tn,
        },
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "maximum_language_false_positive_rate": maximum_language_fpr,
        "by_language": by_language,
        "gates": {
            "minimum_precision": gates["minimum_precision"],
            "minimum_recall": gates["minimum_recall"],
            "maximum_false_positive_rate": gates["maximum_false_positive_rate"],
        },
        "passed": passed,
    })
    print(f"Lexical detector validation passed={passed}: {output}")


def main():
    parser = argparse.ArgumentParser(description="Behavior_v2 lexical detector audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--config", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--negative-per-language", type=int, default=60)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--annotations", required=True)
    validate.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "create":
        create_sample(cfg, args.output, args.negative_per_language)
    else:
        validate_annotations(cfg, args.annotations, args.output)


if __name__ == "__main__":
    main()
