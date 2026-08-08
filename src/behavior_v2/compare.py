"""Compare frozen behavior_v2 geometry and associations across models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_v2.common import ensure_paths, settings
from common import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description="Compare behavior_v2 model results")
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    config_paths = [suite_path.parent / value for value in suite["configs"]]
    configs = [load_config(path) for path in config_paths]
    if len(configs) < 3:
        raise ValueError("behavior_v2 comparison requires at least three frozen models")
    output = Path(suite["comparison_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records, geometry_frames, metric_frames = [], [], []
    task_hashes, evaluation_hashes = set(), set()
    for config_path, cfg in zip(config_paths, configs):
        protocol = settings(cfg)
        paths = ensure_paths(cfg)
        task_manifest = json.loads(
            (paths.data / "behavior_v2_task_manifest.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (paths.validation / "validation_summary.json").read_text(encoding="utf-8")
        )
        association = json.loads(
            (paths.validation / "association_status.json").read_text(encoding="utf-8")
        )
        task_hashes.add(task_manifest["task_sha256"])
        evaluation_hashes.add(task_manifest["evaluation_semantic_id_sha256"])
        primary = association["primary_test"]
        local = association["local_scaled_robustness_test"]
        records.append({
            "experiment_name": cfg["experiment_name"],
            "model": cfg["model"]["name_or_path"],
            "model_family": cfg["model"]["family"],
            "training_stage": cfg["model"]["training_stage"],
            "primary_layer": protocol["primary_layer"],
            "association_status": association["status"],
            "primary_coefficient": primary.get("coefficient"),
            "primary_p_value": primary.get("p_value_two_sided"),
            "local_scaled_coefficient": local.get("coefficient"),
            "local_scaled_p_value": local.get("p_value_two_sided"),
            "overall_assessment": validation["overall_assessment"],
            "formal_evaluation_ready": validation["formal_evaluation_ready"],
        })
        geometry = pd.read_csv(paths.geometry / "english_advantage.csv")
        geometry.insert(0, "experiment_name", cfg["experiment_name"])
        geometry_frames.append(geometry)
        metrics = pd.read_csv(paths.metrics / "behavior_v2_model_summary.csv")
        metrics.insert(0, "experiment_name", cfg["experiment_name"])
        metric_frames.append(metrics)
    if len(task_hashes) != 1 or len(evaluation_hashes) != 1:
        raise ValueError("models did not use identical frozen behavior_v2 tasks")
    comparison = pd.DataFrame.from_records(records)
    comparison.to_csv(output / "association_comparison.csv", index=False)
    pd.concat(geometry_frames, ignore_index=True).to_csv(
        output / "english_geometry_comparison.csv", index=False
    )
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output / "behavior_metric_comparison.csv", index=False
    )
    positives = set(suite.get("confirmatory_positive_models", []))
    controls = set(suite.get("negative_control_models", []))
    names = set(comparison.experiment_name)
    if not positives or not positives.issubset(names) or not controls.issubset(names):
        raise ValueError("suite must freeze valid positive and negative-control model lists")
    supported = comparison.set_index("experiment_name").association_status.str.startswith(
        "SUPPORTED"
    )
    local_supported = (
        comparison.set_index("experiment_name").association_status
        == "SUPPORTED_RAW_AND_LOCAL_SCALED"
    )
    all_formal = bool(
        comparison.formal_evaluation_ready.all()
        and comparison.overall_assessment.str.startswith("FORMAL_").all()
    )
    positive_replication = bool(supported.loc[sorted(positives)].all())
    local_replication = bool(local_supported.loc[sorted(positives)].all())
    negative_specific = bool((~supported.loc[sorted(controls)]).all())
    if not all_formal:
        status = "INVALID"
    elif positive_replication and local_replication and negative_specific:
        status = "REPLICATED_RAW_AND_LOCAL_SCALED_WITH_NEGATIVE_CONTROL_SPECIFICITY"
    elif positive_replication and negative_specific:
        status = "REPLICATED_RAW_ONLY_WITH_NEGATIVE_CONTROL_SPECIFICITY"
    elif supported.loc[sorted(positives)].any():
        status = "PARTIAL_REPLICATION"
    else:
        status = "NOT_REPLICATED"
    summary = {
        "protocol_version": "behavior_v2", "status": status,
        "models": len(comparison),
        "confirmatory_positive_models": sorted(positives),
        "negative_control_models": sorted(controls),
        "positive_model_replication": positive_replication,
        "positive_model_local_scaled_robustness": local_replication,
        "negative_control_specificity": negative_specific,
        "formal_suite": all_formal,
        "task_sha256": next(iter(task_hashes)),
        "evaluation_semantic_id_sha256": next(iter(evaluation_hashes)),
        "representation": "mean_pool",
        "claim_boundary": "cross-model observational replication; no activation intervention or causal claim",
    }
    write_json(output / "cross_model_status.json", summary)
    print(f"behavior_v2 cross-model comparison: {status}")


if __name__ == "__main__":
    main()
