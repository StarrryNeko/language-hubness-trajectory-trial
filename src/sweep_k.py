import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ensure_dirs, load_config
from evidence_rules import joint_positive_layers, max_consecutive_layers


def main():
    parser = argparse.ArgumentParser(description="Same-semantics kNN robustness sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    src = Path(__file__).resolve().parent
    primary = cfg["metrics"].get("primary_representation", "mean_pool")
    rows = []
    for k in args.k_values:
        tag = f"k{k}"
        subprocess.run([
            sys.executable, str(src / "compute_metrics.py"), "--config", args.config,
            "--k", str(k), "--result-tag", tag,
        ], check=True)
        frame = pd.read_csv(Path(paths["metrics"]) / tag / "english_hubness_evidence.csv")
        frame = frame[(frame.representation == primary) & (frame.similarity_method == "cosine")]
        joint_layers = joint_positive_layers(frame)
        joint_run = max_consecutive_layers(joint_layers)
        for metric, group in frame.groupby("metric"):
            peak = group.loc[group["mean"].idxmax()]
            rows.append({
                "k": k,
                "metric": metric,
                "positive_ci_layers": int((group.ci_lower > 0).sum()),
                "positive_ci_longest_run": max_consecutive_layers(
                    group.loc[group.ci_lower > 0, "layer"].astype(int).tolist()
                ),
                "joint_positive_ci_layers": len(joint_layers),
                "joint_positive_ci_longest_run": joint_run,
                "peak_value": float(peak["mean"]),
                "peak_layer": int(peak.layer),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(paths["metrics"]) / "k_robustness_summary.csv", index=False)
    min_run = int(cfg["metrics"].get("min_consecutive_layers", 3))
    joint_support = summary.groupby("k").joint_positive_ci_longest_run.first() >= min_run
    status = "CONSISTENT" if joint_support.nunique() == 1 else "SENSITIVE"
    pd.DataFrame([{
        "status": status,
        "k_values": ",".join(map(str, args.k_values)),
        "criterion": f"same presence/absence of a >={min_run}-layer run where all four CIs are jointly positive",
    }]).to_csv(Path(paths["metrics"]) / "k_robustness_verdict.csv", index=False)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=summary, x="k", y="positive_ci_layers", hue="metric", marker="o")
    plt.title(f"Same-semantics k Robustness ({status})")
    plt.tight_layout()
    plt.savefig(Path(paths["figures"]) / "k_robustness_summary.png", dpi=180)
    plt.close()
    print(f"k robustness: {status}")


if __name__ == "__main__":
    main()
