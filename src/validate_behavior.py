"""Integrity, protocol, and interpretation gates for behavior_v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_common import behavior_settings, ensure_behavior_dirs, load_tasks, sha256_file
from common import load_config, read_jsonl, write_json


def read_json(path, blockers):
    path = Path(path)
    if not path.exists():
        blockers.append(f"missing file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"invalid JSON: {path}: {exc}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Validate behavior_v1 outputs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = behavior_settings(cfg)
    paths = ensure_behavior_dirs(cfg)
    blockers, caveats = [], []
    tasks = load_tasks(cfg)
    expected_tasks = (
        settings["evaluation_semantic_ids"] * len(settings["evaluation_languages"]) * 3
    )
    if len(tasks) != expected_tasks:
        blockers.append(f"task count mismatch: expected {expected_tasks}, got {len(tasks)}")

    output_root = Path(cfg["output_dir"])
    data_manifest = read_json(output_root / "data" / "dataset_manifest.json", blockers)
    task_manifest = read_json(paths.data / "behavior_task_manifest.json", blockers)
    checkpoint = read_json(output_root / "checkpoint_identity.json", blockers)
    extraction = read_json(output_root / "extraction_manifest.json", blockers)
    generation = read_json(paths.generations / "generation_manifest.json", blockers)
    evaluation = read_json(paths.metrics / "behavior_evaluation_manifest.json", blockers)
    geometry = read_json(paths.metrics / "behavior_geometry_manifest.json", blockers)
    association_manifest = read_json(paths.metrics / "behavior_association_manifest.json", blockers)
    association = read_json(paths.validation / "behavior_association_status.json", blockers)

    for name, payload in {
        "tasks": task_manifest, "generation": generation, "evaluation": evaluation,
        "geometry": geometry, "association": association_manifest,
    }.items():
        if payload and payload.get("protocol_version") != "behavior_v1":
            blockers.append(f"{name} manifest has the wrong protocol version")

    selected = set(map(int, data_manifest.get("selected_semantic_indices", [])))
    excluded = set(map(int, data_manifest.get("excluded_semantic_indices", [])))
    if selected & excluded:
        blockers.append("held-out behavior semantic IDs overlap excluded geometry IDs")
    if len(selected) != settings["demonstration_semantic_ids"] + settings["evaluation_semantic_ids"]:
        blockers.append("held-out dataset does not contain the frozen demo+evaluation count")
    expected_selected_hash = cfg["behavior_v1"].get(
        "expected_selected_semantic_indices_sha256"
    )
    if data_manifest.get("selected_semantic_indices_sha256") != expected_selected_hash:
        blockers.append("held-out semantic-ID hash differs from the preregistered behavior sample")
    expected_excluded_hash = cfg["behavior_v1"].get(
        "expected_excluded_semantic_indices_sha256"
    )
    if data_manifest.get("excluded_semantic_indices_sha256") != expected_excluded_hash:
        blockers.append("geometry-seed exclusion hash differs from the frozen exclusion protocol")
    demo_ids = set(map(str, task_manifest.get("demonstration_semantic_ids", [])))
    evaluation_ids = {str(row["semantic_id"]) for row in tasks}
    if demo_ids & evaluation_ids:
        blockers.append("demonstration and evaluation semantic IDs overlap")

    checkpoint_hash = checkpoint.get("checkpoint_sha256")
    for name, payload, key in (
        ("extraction", extraction, "checkpoint_identity_sha256"),
        ("generation", generation, "checkpoint_sha256"),
    ):
        if checkpoint_hash and payload.get(key) != checkpoint_hash:
            blockers.append(f"{name} checkpoint identity does not match checkpoint audit")
    if generation.get("activation_intervention") is not False:
        blockers.append("generation must explicitly record activation_intervention=false")
    if generation.get("prompt_special_tokens_added") is not False:
        blockers.append("generation does not confirm plain-text prompt encoding")
    if generation.get("special_tokens_suppressed") is not True:
        blockers.append("generation does not confirm special-token suppression")
    if generation.get("generated_special_token_count") != 0:
        blockers.append("generation contains forbidden special tokens")
    task_path = paths.data / "behavior_tasks.jsonl"
    generation_path = paths.generations / "generations.jsonl"
    if task_path.exists() and generation.get("task_file_sha256") != sha256_file(task_path):
        blockers.append("generation task hash does not match the frozen task file")
    if generation_path.exists() and generation.get("generation_file_sha256") != sha256_file(generation_path):
        blockers.append("generation manifest hash does not match generation file")
    if generation_path.exists():
        generation_rows = read_jsonl(generation_path)
        expected_budget = int(settings["decoding"]["max_new_tokens"])
        if any(row.get("finish_reason") != "token_budget" for row in generation_rows):
            blockers.append("generation rows contain a non-budget finish reason")
        if any(int(row.get("generated_token_count", -1)) != expected_budget for row in generation_rows):
            blockers.append("generation rows do not all match the frozen token budget")

    item_path = paths.metrics / "behavior_item_results.csv"
    predictor_path = paths.metrics / "behavior_geometry_predictors.csv"
    if item_path.exists():
        items = pd.read_csv(item_path)
        if len(items) != expected_tasks or items.task_id.nunique() != expected_tasks:
            blockers.append("behavior item results do not cover every task exactly once")
        empty_rate = float(items.empty_output.mean())
        if empty_rate > float(cfg["behavior_v1"].get("maximum_empty_output_rate", 0.01)):
            blockers.append(f"empty output rate exceeds gate: {empty_rate:.4f}")
    else:
        items, empty_rate = pd.DataFrame(), None
        blockers.append(f"missing file: {item_path}")
    if predictor_path.exists():
        predictors = pd.read_csv(predictor_path)
        expected_predictors = expected_tasks * len(set(settings["analysis_layers"]))
        if len(predictors) != expected_predictors:
            blockers.append(
                f"geometry predictor count mismatch: expected {expected_predictors}, got {len(predictors)}"
            )
        if predictors[["task_id", "layer"]].duplicated().any():
            blockers.append("geometry predictors contain duplicate task/layer rows")
    else:
        blockers.append(f"missing file: {predictor_path}")

    formal_ready = bool(evaluation.get("formal_evaluation_ready", False))
    calibrated_threshold = evaluation.get("lid_calibration_selected_threshold")
    configured_threshold = float(settings["language_id"]["confidence_threshold"])
    if not evaluation.get("lid_calibration_report_sha256"):
        blockers.append("evaluation does not record an independent LID calibration report")
    if calibrated_threshold is None or abs(
        float(calibrated_threshold) - configured_threshold
    ) > 1e-12:
        blockers.append("evaluation LID threshold does not match the calibration report")
    reference_lid_accuracy = evaluation.get("reference_language_id_accuracy")
    required_lid_accuracy = float(cfg["behavior_v1"].get("minimum_reference_lid_accuracy", 0.95))
    if not formal_ready:
        caveats.append("language ID or chrF++ uses a smoke-test backend; results are not formal evidence")
    if reference_lid_accuracy is None or float(reference_lid_accuracy) < required_lid_accuracy:
        blockers.append(
            f"reference-language LID accuracy is below {required_lid_accuracy:.2f}"
        )
    if extraction.get("truncated_inputs", 0) not in (0, None):
        blockers.append("hidden-state extraction contains truncated inputs")

    association_status = association.get("status", "INVALID")
    if blockers:
        overall = "INVALID"
    elif not formal_ready:
        overall = "SMOKE_TEST_ONLY"
    elif association_status.startswith("SUPPORTED"):
        overall = "FORMAL_ASSOCIATION_SUPPORTED"
    else:
        overall = "FORMAL_ASSOCIATION_NOT_SUPPORTED"
    summary = {
        "protocol_version": "behavior_v1",
        "overall_assessment": overall,
        "association_status": association_status,
        "formal_evaluation_ready": formal_ready,
        "tasks": len(tasks),
        "evaluation_semantic_ids": len(evaluation_ids),
        "checkpoint_sha256": checkpoint_hash,
        "reference_language_id_accuracy": reference_lid_accuracy,
        "required_reference_language_id_accuracy": required_lid_accuracy,
        "empty_output_rate": empty_rate,
        "activation_intervention": False,
        "blockers": blockers,
        "caveats": caveats,
        "allowed_claim": (
            "Geometry predicts held-out translation behavior under a frozen observational protocol. "
            "This result does not establish activation-level causality."
        ),
    }
    write_json(paths.validation / "behavior_validation_summary.json", summary)
    lines = [
        "# behavior_v1 validation", "", f"- Overall: `{overall}`",
        f"- Association: `{association_status}`", f"- Formal evaluator: `{formal_ready}`",
        f"- Activation intervention: `False`", "", "## Blockers", "",
        *([f"- {item}" for item in blockers] if blockers else ["- None"]),
        "", "## Interpretation boundary", "", summary["allowed_claim"],
    ]
    (paths.validation / "behavior_validation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"behavior_v1 validation: {overall}; blockers={len(blockers)}")


if __name__ == "__main__":
    main()
