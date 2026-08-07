"""Compare frozen behavior_v1 evidence across the three-model suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_common import behavior_settings, ensure_behavior_dirs
from common import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description="Compare behavior_v1 model results")
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    config_paths = [suite_path.parent / value for value in suite["configs"]]
    configs = [load_config(path) for path in config_paths]
    if len(configs) < 3:
        raise ValueError("behavior comparison requires at least three frozen models")
    output = Path(suite["comparison_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records, validation_rows, task_hashes, eval_hashes = [], [], set(), set()
    for config_path, cfg in zip(config_paths, configs):
        settings = behavior_settings(cfg)
        paths = ensure_behavior_dirs(cfg)
        task_manifest = json.loads(
            (paths.data / "behavior_task_manifest.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (paths.validation / "behavior_validation_summary.json").read_text(encoding="utf-8")
        )
        association = json.loads(
            (paths.validation / "behavior_association_status.json").read_text(encoding="utf-8")
        )
        task_hashes.add(task_manifest["task_sha256"])
        eval_hashes.add(task_manifest["evaluation_semantic_id_sha256"])
        primary = association["primary_test"]
        local = association["local_scaled_robustness_test"]
        records.append({
            "experiment_name": cfg["experiment_name"],
            "model": cfg["model"]["name_or_path"],
            "model_family": cfg["model"]["family"],
            "training_stage": cfg["model"]["training_stage"],
            "primary_layer": settings["primary_layer"],
            "association_status": association["status"],
            "primary_coefficient": primary.get("coefficient"),
            "primary_p_value": primary.get("p_value_two_sided"),
            "local_scaled_coefficient": local.get("coefficient"),
            "local_scaled_p_value": local.get("p_value_two_sided"),
            "overall_assessment": validation["overall_assessment"],
            "formal_evaluation_ready": validation["formal_evaluation_ready"],
        })
        summaries = pd.read_csv(paths.metrics / "behavior_model_summary.csv")
        summaries.insert(0, "experiment_name", cfg["experiment_name"])
        validation_rows.append(summaries)
    if len(task_hashes) != 1 or len(eval_hashes) != 1:
        raise ValueError("models did not use identical frozen behavior tasks")
    comparison = pd.DataFrame.from_records(records)
    comparison.to_csv(output / "behavior_model_association_comparison.csv", index=False)
    pd.concat(validation_rows, ignore_index=True).to_csv(
        output / "behavior_model_metric_comparison.csv", index=False
    )
    positives = set(suite.get("confirmatory_positive_models", []))
    controls = set(suite.get("negative_control_models", []))
    names = set(comparison.experiment_name)
    if not positives or not positives.issubset(names) or not controls.issubset(names):
        raise ValueError("suite must freeze valid confirmatory_positive_models and negative_control_models")
    supported = comparison.set_index("experiment_name").association_status.str.startswith("SUPPORTED")
    local_supported = (
        comparison.set_index("experiment_name").association_status
        == "SUPPORTED_RAW_AND_LOCAL_SCALED"
    )
    all_formal = bool(
        comparison.formal_evaluation_ready.all()
        and comparison.overall_assessment.str.startswith("FORMAL_").all()
    )
    positive_replication = bool(supported.loc[sorted(positives)].all())
    positive_local_robustness = bool(local_supported.loc[sorted(positives)].all())
    negative_control_specific = bool((~supported.loc[sorted(controls)]).all())
    if not all_formal:
        status = "INVALID_OR_SMOKE_ONLY"
    elif positive_replication and positive_local_robustness and negative_control_specific:
        status = "REPLICATED_RAW_AND_LOCAL_SCALED_WITH_NEGATIVE_CONTROL_SPECIFICITY"
    elif positive_replication and negative_control_specific:
        status = "REPLICATED_RAW_DENSITY_SENSITIVE_WITH_NEGATIVE_CONTROL_SPECIFICITY"
    elif positive_replication and positive_local_robustness:
        status = "REPLICATED_RAW_AND_LOCAL_SCALED_WITHOUT_NEGATIVE_CONTROL_SPECIFICITY"
    elif positive_replication:
        status = "REPLICATED_RAW_DENSITY_SENSITIVE_WITHOUT_NEGATIVE_CONTROL_SPECIFICITY"
    elif supported.loc[sorted(positives)].any():
        status = "PARTIAL_REPLICATION"
    else:
        status = "NOT_REPLICATED"
    summary = {
        "protocol_version": "behavior_v1",
        "status": status,
        "models": len(comparison),
        "confirmatory_positive_models": sorted(positives),
        "negative_control_models": sorted(controls),
        "positive_model_replication": positive_replication,
        "positive_model_local_scaled_robustness": positive_local_robustness,
        "negative_control_specificity": negative_control_specific,
        "formal_suite": all_formal,
        "task_sha256": next(iter(task_hashes)),
        "evaluation_semantic_id_sha256": next(iter(eval_hashes)),
        "claim_boundary": "cross-model observational replication; no activation intervention or causal claim",
    }
    write_json(output / "behavior_cross_model_status.json", summary)
    print(f"behavior_v1 cross-model comparison: {status}")


if __name__ == "__main__":
    main()
