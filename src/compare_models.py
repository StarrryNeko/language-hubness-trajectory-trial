import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config
from evidence_rules import (
    MODEL_STATUSES,
    REQUIRED_EVIDENCE_METRICS,
    max_consecutive_layers,
    validate_evidence_grid,
    validate_model_status_payload,
)


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
    status = validate_model_status_payload(validation)
    if status == "INVALID":
        reason = validation.get("joint_evidence", {}).get("reason", "model validation marked INVALID")
        raise ValueError(reason)
    try:
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
        "status": status,
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

    robust_models = sorted({row["model"] for row in model_statuses if row["status"] == "ROBUST"})
    conditional_models = sorted({
        row["model"] for row in model_statuses if row["status"] == "REPRESENTATION_SENSITIVE"
    })
    valid_statuses = MODEL_STATUSES - {"INVALID"}
    verdict = {
        "models_compared": [row["model"] for row in model_statuses],
        "model_statuses": model_statuses,
        "valid_model_count": sum(row["status"] in valid_statuses for row in model_statuses),
        "robust_models": robust_models,
        "conditional_models": conditional_models,
        "replication_status": "REPLICATED" if len(robust_models) >= 2 else "NOT_REPLICATED",
        "rule": "At least two models must independently be ROBUST under joint four-metric, breadth, EOS, and density-control rules; INVALID models are excluded.",
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
    print(f"Model comparison: {verdict['replication_status']}")


if __name__ == "__main__":
    main()
