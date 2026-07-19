import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import load_config


def max_consecutive(values):
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def main():
    parser = argparse.ArgumentParser(description="Compare normalized hubness trajectories across models")
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    frames = []
    for relative in suite["configs"]:
        config_path = suite_path.parent / relative
        cfg = load_config(config_path)
        model_name = cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path"))
        path = Path(cfg["output_dir"]) / "metrics" / "english_hubness_evidence.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; finish the model run before comparison")
        frame = pd.read_csv(path)
        frame = frame[
            (frame.representation == cfg["metrics"].get("primary_representation", "mean_pool"))
            & (frame.similarity_method == "cosine")
        ].copy()
        frame["model"] = model_name
        frame["experiment_name"] = cfg["experiment_name"]
        maximum = max(1, int(frame.layer.max()))
        frame["normalized_layer"] = frame.layer / maximum
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output = Path(suite["comparison_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "model_english_hubness_trajectories.csv", index=False)

    summary_rows = []
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for (model, experiment, metric), group in combined.groupby(["model", "experiment_name", "metric"]):
        group = group.sort_values("normalized_layer")
        peak = group.loc[group["mean"].idxmax()]
        summary_rows.append({
            "model": model,
            "experiment_name": experiment,
            "metric": metric,
            "trajectory_auc": float(integrate(group["mean"], group.normalized_layer)),
            "positive_ci_layer_fraction": float((group.ci_lower > 0).mean()),
            "positive_ci_longest_run": max_consecutive((group.ci_lower > 0).tolist()),
            "peak_value": float(peak["mean"]),
            "peak_normalized_layer": float(peak.normalized_layer),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "model_comparison_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    metrics = sorted(combined.metric.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, metric in zip(axes.flat, metrics):
        part = combined[combined.metric == metric]
        sns.lineplot(data=part, x="normalized_layer", y="mean", hue="model", ax=ax, errorbar=None)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("English advantage")
    fig.suptitle("English Same-semantics Hubness across Model Families")
    fig.tight_layout()
    fig.savefig(output / "model_hubness_comparison.png", dpi=180)
    plt.close(fig)

    metrics_by_model = summary.pivot(index="model", columns="metric", values="positive_ci_longest_run")
    replicated = (metrics_by_model >= 3).all(axis=1)
    verdict = {
        "models_compared": list(metrics_by_model.index),
        "models_with_all_four_evidence_dimensions": replicated[replicated].index.tolist(),
        "replication_status": "REPLICATED" if replicated.sum() >= 2 else "NOT_REPLICATED",
        "rule": "At least two model families must show >=3 consecutive positive-CI layers for all four English hubness dimensions.",
    }
    (output / "model_comparison_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Model comparison: {verdict['replication_status']} -> {output}")


if __name__ == "__main__":
    main()
