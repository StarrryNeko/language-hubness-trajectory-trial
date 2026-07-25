import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config
from evidence_rules import (
    REQUIRED_EVIDENCE_METRICS,
    max_consecutive_layers,
    validate_evidence_grid,
)


VALID_CROSS_MODEL_STATUSES = {"NOT_SUPPORTED", "DENSITY_SENSITIVE", "ROBUST"}


def classify_cross_model(validation):
    """Re-evaluate one mean-pool model under primary and density criteria.

    The primary layer list already includes the four-metric and source-breadth
    intersection produced by run_validations.py. Density robustness is stricter:
    primary and local-scaled evidence must hold on the same consecutive layers.
    """
    source_status = validation.get("model_status")
    evidence = validation.get("joint_evidence")
    if source_status == "INVALID":
        reason = evidence.get("reason") if isinstance(evidence, dict) else None
        raise ValueError(reason or "model validation marked INVALID")
    if not isinstance(evidence, dict):
        raise ValueError("validation joint_evidence is missing or invalid")
    required = {
        "primary_joint_layers",
        "density_joint_layers",
        "min_consecutive_layers",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"validation joint_evidence is missing fields: {sorted(missing)}")

    minimum = int(evidence["min_consecutive_layers"])
    if minimum < 1:
        raise ValueError("min_consecutive_layers must be at least 1")
    primary_layers = sorted(set(map(int, evidence["primary_joint_layers"])))
    density_layers = sorted(set(map(int, evidence["density_joint_layers"])))
    overlap_layers = sorted(set(primary_layers) & set(density_layers))
    primary_run = max_consecutive_layers(primary_layers)
    density_run = max_consecutive_layers(density_layers)
    overlap_run = max_consecutive_layers(overlap_layers)
    if primary_run < minimum:
        status = "NOT_SUPPORTED"
    elif overlap_run >= minimum:
        status = "ROBUST"
    else:
        status = "DENSITY_SENSITIVE"

    return {
        "status": status,
        "source_validation_status": source_status,
        "primary_supported": primary_run >= minimum,
        "primary_joint_layers": primary_layers,
        "primary_joint_longest_run": primary_run,
        "density_joint_layers": density_layers,
        "density_joint_longest_run": density_run,
        "primary_density_overlap_layers": overlap_layers,
        "primary_density_overlap_longest_run": overlap_run,
        "min_consecutive_layers": minimum,
    }


def load_model_result(config_path):
    cfg = load_config(config_path)
    model_name = cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path"))
    experiment = cfg["experiment_name"]
    output = Path(cfg["output_dir"])
    evidence_path = output / "metrics" / "english_hubness_evidence.csv"
    validation_path = output / "validation" / "validation_summary.json"
    extraction_path = output / "extraction_manifest.json"
    for path in (evidence_path, validation_path, extraction_path):
        if not path.exists():
            raise ValueError(f"required result file is missing: {path}")
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"result JSON cannot be parsed: {error}") from error
    cross_model_evidence = classify_cross_model(validation)
    try:
        if extraction.get("protocol_version") != "mean_pool_no_eos_v1":
            raise ValueError("extraction manifest is not from the active protocol")
        if extraction.get("representations") != ["mean_pool"]:
            raise ValueError("extraction manifest is not mean-pool-only")
        if extraction.get("appended_eos") is not False:
            raise ValueError("extraction manifest does not confirm appended_eos=False")
        layer_count = int(extraction["layers"])
        if layer_count < 1:
            raise ValueError("layer count must be positive")
        expected_layers = list(range(layer_count))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("extraction manifest has no valid layer count") from error
    frame = pd.read_csv(evidence_path)
    primary = cfg["metrics"].get("primary_representation", "mean_pool")
    frame = frame[
        (frame.representation == primary) & (frame.similarity_method == "cosine")
    ].copy()
    validate_evidence_grid(frame, expected_layers=expected_layers)
    frame = frame[frame.metric.isin(REQUIRED_EVIDENCE_METRICS)].copy()
    frame["model"] = model_name
    frame["experiment_name"] = experiment
    maximum = max(1, expected_layers[-1])
    frame["normalized_layer"] = frame.layer.astype(int) / maximum
    return frame, {
        "model": model_name,
        "experiment_name": experiment,
        **cross_model_evidence,
        "reason": None,
    }


def compare_suite(suite_path):
    suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    output = Path(suite["comparison_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    model_statuses = []
    for relative in suite["configs"]:
        config_path = suite_path.parent / relative
        try:
            frame, record = load_model_result(config_path)
            frames.append(frame)
            model_statuses.append(record)
        except (KeyError, OSError, TypeError, ValueError) as error:
            try:
                cfg = load_config(config_path)
                model = cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path"))
                experiment = cfg.get("experiment_name", config_path.stem)
            except (OSError, ValueError, json.JSONDecodeError):
                model, experiment = config_path.stem, config_path.stem
            model_statuses.append({
                "model": model,
                "experiment_name": experiment,
                "status": "INVALID",
                "reason": str(error),
            })

    trajectory_columns = [
        "representation", "similarity_method", "layer", "metric", "mean", "ci_lower",
        "ci_upper", "model", "experiment_name", "normalized_layer",
    ]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=trajectory_columns)
    combined.to_csv(output / "model_english_hubness_trajectories.csv", index=False)

    summary_rows = []
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for (model, experiment, metric), group in combined.groupby(["model", "experiment_name", "metric"]):
        group = group.sort_values("normalized_layer")
        peak = group.loc[group["mean"].idxmax()]
        positive_layers = group.loc[group.ci_lower > 0, "layer"].astype(int).tolist()
        summary_rows.append({
            "model": model,
            "experiment_name": experiment,
            "metric": metric,
            "trajectory_auc": float(integrate(group["mean"], group.normalized_layer)),
            "positive_ci_layer_fraction": float((group.ci_lower > 0).mean()),
            "positive_ci_longest_run": max_consecutive_layers(positive_layers),
            "peak_value": float(peak["mean"]),
            "peak_normalized_layer": float(peak.normalized_layer),
        })
    summary = pd.DataFrame(summary_rows, columns=[
        "model", "experiment_name", "metric", "trajectory_auc",
        "positive_ci_layer_fraction", "positive_ci_longest_run", "peak_value",
        "peak_normalized_layer",
    ])
    summary.to_csv(output / "model_comparison_summary.csv", index=False)

    if not combined.empty:
        metrics = list(REQUIRED_EVIDENCE_METRICS)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, metric in zip(axes.flat, metrics):
            part = combined[combined.metric == metric]
            for model, group in part.groupby("model", sort=True):
                group = group.sort_values("normalized_layer")
                ax.plot(group.normalized_layer, group["mean"], marker="o", label=model)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel("English advantage")
            if not part.empty:
                ax.legend(fontsize=8)
        fig.suptitle("English Same-semantics Hubness across Model Families")
        fig.tight_layout()
        fig.savefig(output / "model_hubness_comparison.png", dpi=180)
        plt.close(fig)

    primary_supported_models = sorted({
        row["model"] for row in model_statuses if row.get("primary_supported") is True
    })
    robust_models = sorted({row["model"] for row in model_statuses if row["status"] == "ROBUST"})
    conditional_models = sorted({
        row["model"] for row in model_statuses if row["status"] == "DENSITY_SENSITIVE"
    })
    verdict = {
        "models_compared": [row["model"] for row in model_statuses],
        "model_statuses": model_statuses,
        "valid_model_count": sum(
            row["status"] in VALID_CROSS_MODEL_STATUSES for row in model_statuses
        ),
        "primary_supported_models": primary_supported_models,
        "primary_only_replication_status": (
            "REPLICATED" if len(primary_supported_models) >= 2 else "NOT_REPLICATED"
        ),
        "robust_models": robust_models,
        "conditional_models": conditional_models,
        "replication_status": "REPLICATED" if len(robust_models) >= 2 else "NOT_REPLICATED",
        "evaluation_policy": {
            "representation_protocol": "mean_pool_only_without_appended_eos",
            "primary_rule": (
                "Four English hubness CIs and source breadth must jointly hold for "
                "the configured minimum consecutive layers."
            ),
            "density_robust_rule": (
                "Primary and local-scaled four-metric evidence must overlap on the "
                "same configured minimum consecutive layers."
            ),
            "cross_model_rule": (
                "At least two distinct models must satisfy the density-robust rule; "
                "INVALID models are excluded."
            ),
        },
        "rule": (
            "Formal replication requires at least two distinct mean-pool models with "
            "same-layer primary and density-controlled evidence."
        ),
    }
    (output / "model_comparison_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return verdict


def main():
    parser = argparse.ArgumentParser(description="Compare normalized hubness trajectories across models")
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    verdict = compare_suite(args.suite)
    print(
        "Mean-pool primary-only comparison: "
        f"{verdict['primary_only_replication_status']}"
    )
    print(
        "Mean-pool density-robust comparison: "
        f"{verdict['replication_status']}"
    )


if __name__ == "__main__":
    main()
