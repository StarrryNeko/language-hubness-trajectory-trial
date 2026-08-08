"""Integrity, measurement, and interpretation gates for behavior_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_v2.common import ensure_paths, load_tasks, settings, sha256_file, stable_sha256
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
    parser = argparse.ArgumentParser(description="Validate behavior_v2 outputs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    protocol = settings(cfg)
    paths = ensure_paths(cfg)
    blockers, caveats = [], []
    tasks = load_tasks(cfg)
    expected_tasks = protocol["evaluation_semantic_ids"] * len(protocol["target_languages"])
    if len(tasks) != expected_tasks:
        blockers.append(f"task count mismatch: expected {expected_tasks}, got {len(tasks)}")

    output_root = Path(cfg["output_dir"])
    checkpoint = read_json(output_root / "checkpoint_identity.json", blockers)
    extraction = read_json(output_root / "extraction_manifest.json", blockers)
    task_manifest = read_json(paths.data / "behavior_v2_task_manifest.json", blockers)
    generation = read_json(paths.generations / "generation_manifest.json", blockers)
    evaluation = read_json(paths.metrics / "evaluation_manifest.json", blockers)
    geometry = read_json(paths.geometry / "geometry_manifest.json", blockers)
    association_manifest = read_json(paths.measurement / "association_manifest.json", blockers)
    association = read_json(paths.validation / "association_status.json", blockers)
    manifests = {
        "tasks": task_manifest, "generation": generation, "evaluation": evaluation,
        "geometry": geometry, "association": association_manifest,
    }
    for name, payload in manifests.items():
        if payload and payload.get("protocol_version") != "behavior_v2":
            blockers.append(f"{name} manifest has the wrong protocol version")

    if geometry.get("representation") != "mean_pool":
        blockers.append("geometry does not use mean_pool exclusively")
    if geometry.get("representation_protocol") != "mean_pool_v1":
        blockers.append("geometry representation protocol mismatch")
    if geometry.get("layers") != protocol["analysis_layers"]:
        blockers.append("geometry layers differ from the frozen V2 config")
    if geometry.get("primary_layer") != protocol["primary_layer"]:
        blockers.append("geometry primary layer differs from the frozen V2 config")
    if extraction.get("protocol_version") != "mean_pool_v1":
        blockers.append("hidden-state extraction protocol is not mean_pool_v1")
    if extraction.get("representations") != ["mean_pool"]:
        blockers.append("hidden-state extraction contains a non-mean-pool representation")
    if extraction.get("text_tokenization_add_special_tokens") is not False:
        blockers.append("hidden-state extraction did not use plain-text tokenization")
    checkpoint_hash = checkpoint.get("checkpoint_sha256")
    for name, payload, key in (
        ("extraction", extraction, "checkpoint_identity_sha256"),
        ("generation", generation, "checkpoint_sha256"),
        ("geometry", geometry, "checkpoint_sha256"),
    ):
        if checkpoint_hash and payload.get(key) != checkpoint_hash:
            blockers.append(f"{name} checkpoint identity does not match checkpoint audit")
    for name, payload in manifests.items():
        if payload.get("activation_intervention") not in (False, None):
            blockers.append(f"{name} unexpectedly records an activation intervention")
    if generation.get("activation_intervention") is not False:
        blockers.append("generation must explicitly record activation_intervention=false")
    if generation.get("prompt_special_tokens_added") is not False:
        blockers.append("generation does not confirm plain-text prompt encoding")
    if generation.get("generated_forbidden_special_token_count") != 0:
        blockers.append("generation contains forbidden control tokens")

    task_path = paths.data / "behavior_v2_tasks.jsonl"
    generation_path = paths.generations / "generations.jsonl"
    if task_manifest.get("task_sha256") != stable_sha256(tasks):
        blockers.append("task manifest semantic hash differs from current frozen tasks")
    if task_manifest.get("source_data_content_sha256") != extraction.get("data_content_sha256"):
        blockers.append("task and extraction data-content hashes differ")
    if geometry.get("task_file_sha256") != sha256_file(task_path):
        blockers.append("geometry task hash differs from the frozen task file")
    if task_path.exists() and generation.get("task_file_sha256") != sha256_file(task_path):
        blockers.append("generation task hash differs from the frozen task file")
    if generation_path.exists() and generation.get("generation_file_sha256") != sha256_file(generation_path):
        blockers.append("generation manifest hash differs from the generation file")
    if evaluation.get("task_file_sha256") != sha256_file(task_path):
        blockers.append("evaluation task hash differs from the frozen task file")
    if generation_path.exists() and evaluation.get("generation_file_sha256") != sha256_file(generation_path):
        blockers.append("evaluation generation hash differs from the generation file")
    if generation_path.exists():
        generation_rows = read_jsonl(generation_path)
        if len(generation_rows) != expected_tasks:
            blockers.append("generation file does not contain every frozen task")
        allowed_reasons = {"natural_stop", "token_budget"}
        if any(row.get("finish_reason") not in allowed_reasons for row in generation_rows):
            blockers.append("generation contains an invalid finish reason")
    token_budget_rate = generation.get("token_budget_rate")
    if token_budget_rate is None or float(token_budget_rate) > protocol["maximum_token_budget_rate"]:
        blockers.append(
            f"token-budget finish rate exceeds gate: {token_budget_rate} > "
            f"{protocol['maximum_token_budget_rate']}"
        )

    item_path = paths.metrics / "behavior_v2_item_results.csv"
    predictor_path = paths.geometry / "behavior_v2_geometry_predictors.csv"
    if item_path.exists():
        items = pd.read_csv(item_path)
        if len(items) != expected_tasks or items.task_id.nunique() != expected_tasks:
            blockers.append("item results do not cover every frozen task exactly once")
        empty_rate = float(items.empty_output.mean())
        if empty_rate > protocol["maximum_empty_output_rate"]:
            blockers.append(
                f"empty output rate exceeds gate: {empty_rate:.4f} > "
                f"{protocol['maximum_empty_output_rate']:.4f}"
            )
    else:
        items, empty_rate = pd.DataFrame(), None
        blockers.append(f"missing file: {item_path}")
    if predictor_path.exists():
        predictors = pd.read_csv(predictor_path)
        expected_predictors = expected_tasks * len(protocol["analysis_layers"])
        if len(predictors) != expected_predictors:
            blockers.append(
                f"geometry predictor count mismatch: expected {expected_predictors}, "
                f"got {len(predictors)}"
            )
        if predictors[["task_id", "layer"]].duplicated().any():
            blockers.append("geometry predictors contain duplicate task/layer rows")
    else:
        blockers.append(f"missing file: {predictor_path}")
    if item_path.exists() and association_manifest.get("item_results_sha256") != sha256_file(item_path):
        blockers.append("association item-results hash differs from current evaluation")
    if predictor_path.exists() and association_manifest.get("geometry_predictors_sha256") != sha256_file(predictor_path):
        blockers.append("association geometry hash differs from current predictors")

    formal_ready = bool(evaluation.get("formal_evaluation_ready", False))
    lexical = evaluation.get("lexical_detector", {})
    if not formal_ready:
        blockers.append("evaluation is not formal-ready")
    if lexical.get("validated") is not True or not lexical.get("validation_report_sha256"):
        blockers.append("English lexical detector lacks a passed blinded validation report")
    if lexical.get("rule_version") != protocol["lexical_leakage"]["rule_version"]:
        blockers.append("English lexical detector rule version changed")
    if (
        lexical.get("validation_report_sha256")
        and association_manifest.get("detector_report_sha256")
        != lexical.get("validation_report_sha256")
    ):
        blockers.append("evaluation and association used different detector reports")
    if evaluation.get("quality_backend") != "sacrebleu_chrfpp":
        blockers.append("chrF++ does not use the formal SacreBLEU backend")
    if extraction.get("truncated_inputs", 0) not in (0, None):
        blockers.append("hidden-state extraction contains truncated inputs")

    association_status = association.get("status", "INVALID")
    if blockers:
        overall = "INVALID"
    elif association_status.startswith("SUPPORTED"):
        overall = "FORMAL_ASSOCIATION_SUPPORTED"
    else:
        overall = "FORMAL_ASSOCIATION_NOT_SUPPORTED"
    summary = {
        "protocol_version": "behavior_v2",
        "overall_assessment": overall,
        "association_status": association_status,
        "formal_evaluation_ready": formal_ready,
        "tasks": len(tasks),
        "evaluation_semantic_ids": len({str(row['semantic_id']) for row in tasks}),
        "checkpoint_sha256": checkpoint_hash,
        "empty_output_rate": empty_rate,
        "token_budget_rate": token_budget_rate,
        "representation": "mean_pool",
        "activation_intervention": False,
        "blockers": blockers,
        "caveats": caveats,
        "allowed_claim": (
            "Whole-language geometry is observationally associated with held-out script/lexical "
            "translation behavior under a frozen mean-pool protocol. This is not a causal claim."
        ),
    }
    write_json(paths.validation / "validation_summary.json", summary)
    lines = [
        "# behavior_v2 validation", "", f"- Overall: `{overall}`",
        f"- Association: `{association_status}`", f"- Formal evaluator: `{formal_ready}`",
        "- Representation: `mean_pool`", "- Activation intervention: `False`",
        "", "## Blockers", "",
        *([f"- {item}" for item in blockers] if blockers else ["- None"]),
        "", "## Interpretation boundary", "", summary["allowed_claim"],
    ]
    (paths.validation / "validation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"behavior_v2 validation: {overall}; blockers={len(blockers)}")


if __name__ == "__main__":
    main()
