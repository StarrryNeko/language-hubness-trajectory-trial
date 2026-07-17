import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ensure_dirs, load_config


def main():
    parser = argparse.ArgumentParser(description="Run kNN robustness metrics without re-extracting hidden states.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    src_dir = Path(__file__).resolve().parent
    summaries = []

    for k in args.k_values:
        tag = f"k{k}"
        print(f"\n=== k robustness run: k={k} ===", flush=True)
        subprocess.run([
            sys.executable, str(src_dir / "compute_metrics.py"),
            "--config", args.config, "--k", str(k), "--result-tag", tag,
        ], check=True)
        subprocess.run([
            sys.executable, str(src_dir / "plot_trajectories.py"),
            "--config", args.config, "--result-tag", tag,
        ], check=True)
        frame = pd.read_csv(Path(paths["metrics"]) / tag / "pooling_robustness_summary.csv")
        frame["k"] = k
        summaries.append(frame)

    summary = pd.concat(summaries, ignore_index=True)
    summary_path = Path(paths["metrics"]) / "k_robustness_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    primary = cfg["metrics"].get("primary_representation", "last_token")
    plot_data = summary[summary.representation == primary]
    max_layer = max(
        1,
        int(pd.read_csv(Path(paths["metrics"]) / f"k{args.k_values[0]}" / "alignment_gain.csv").layer.max()),
    )
    same_claims = all(
        plot_data[column].nunique(dropna=False) == 1
        for column in [
            "retrieval_directions_above_random_ci",
            "english_hub_supported_languages",
            "languages_with_positive_re_separation_ci",
        ]
    )
    specificity_support = (
        plot_data.english_specificity_longest_run >= int(cfg["metrics"].get("min_consecutive_layers", 3))
    )
    same_claims = same_claims and specificity_support.nunique(dropna=False) == 1
    peak_stability = all(
        (plot_data[column].max() - plot_data[column].min()) / max_layer <= 0.25
        for column in ["alignment_peak_layer", "retrieval_peak_layer", "english_hub_peak_layer"]
    )
    k_status = "CONSISTENT" if same_claims and peak_stability else "SENSITIVE"
    verdict = pd.DataFrame([{
        "primary_representation": primary,
        "k_values": ",".join(map(str, sorted(plot_data.k.unique()))),
        "status": k_status,
        "same_claim_support": bool(same_claims),
        "peak_layer_stable_within_25pct": bool(peak_stability),
    }])
    verdict_path = Path(paths["metrics"]) / "k_robustness_verdict.csv"
    verdict.to_csv(verdict_path, index=False, encoding="utf-8")
    long = plot_data.melt(
        id_vars=["k"],
        value_vars=["english_hub_peak", "mean_re_separation_strength"],
        var_name="metric", value_name="value",
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=long, x="k", y="value", hue="metric", marker="o")
    plt.title("kNN Robustness Summary")
    plt.tight_layout()
    figure_path = Path(paths["figures"]) / "k_robustness_summary.png"
    plt.savefig(figure_path, dpi=180)
    plt.close()
    print(f"Saved {summary_path}")
    print(f"Saved {verdict_path} ({k_status})")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
