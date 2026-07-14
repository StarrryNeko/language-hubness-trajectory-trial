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
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
