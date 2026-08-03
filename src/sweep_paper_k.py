"""Run the complete paper_v1 hubness decision at every predeclared k."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import load_config
from paper_common import ensure_paper_dirs, write_manifest


def main():
    parser = argparse.ArgumentParser(description="paper_v1 k robustness sweep")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    paths = ensure_paper_dirs(cfg)
    hubness = paths.metrics / "hubness"
    primary_k = int(cfg["metrics"].get("nearest_neighbors_k", 5))
    k_values = sorted(set(map(int, cfg.get("robustness", {}).get("k_values", [1, 3, 5, 10]))))
    invalid = [value for value in k_values if not 1 <= value < len(cfg["dataset"]["languages"])]
    if invalid:
        raise ValueError(f"invalid paper_v1 k values: {invalid}")
    script = Path(__file__).resolve().parent / "compute_hubness_paper.py"
    records = []
    for k in k_values:
        result_dir = hubness if k == primary_k else hubness / f"k_{k}"
        if k != primary_k:
            subprocess.run([
                sys.executable, str(script), "--config", str(config_path),
                "--k", str(k), "--result-tag", f"k_{k}",
                "--summary-only",
            ], check=True)
        status = json.loads((result_dir / "paper_model_status.json").read_text(encoding="utf-8"))
        specificity = pd.read_csv(result_dir / "target_rotation_summary.csv")
        permutation = pd.read_csv(result_dir / "target_rotation_permutation.csv")
        raw = specificity[specificity.similarity_method == "cosine"]
        density = specificity[specificity.similarity_method == "local_scaled_cosine"]
        global_p = permutation.groupby("similarity_method").global_max_target_layer_p_value.first()
        records.append({
            "k": k,
            "status": status["status"],
            "primary_joint_layers": ",".join(map(str, status["primary_joint_layers"])),
            "primary_joint_longest_run": status["primary_joint_longest_run"],
            "density_joint_layers": ",".join(map(str, status["density_joint_layers"])),
            "density_joint_longest_run": status["density_joint_longest_run"],
            "overlap_layers": ",".join(map(str, status["primary_density_overlap_layers"])),
            "overlap_longest_run": status["primary_density_overlap_longest_run"],
            "raw_best_english_rank": int(raw.english_rank.min()),
            "raw_peak_layer": int(raw.loc[raw.english_mean.idxmax(), "layer"]),
            "raw_peak_effect": float(raw.english_mean.max()),
            "density_best_english_rank": int(density.english_rank.min()),
            "density_peak_layer": int(density.loc[density.english_mean.idxmax(), "layer"]),
            "density_peak_effect": float(density.english_mean.max()),
            "raw_global_permutation_p": float(global_p.get("cosine", 1.0)),
            "density_global_permutation_p": float(global_p.get("local_scaled_cosine", 1.0)),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(hubness / "k_robustness_summary.csv", index=False, encoding="utf-8")
    status_agreement = frame.status.nunique() == 1
    write_manifest(hubness / "k_robustness_manifest.json", {
        "k_values": k_values,
        "primary_k": primary_k,
        "full_status_reported_per_k": True,
        "status_agreement": bool(status_agreement),
        "note": "The per-k table, not a single consistency label, is the evidence artifact.",
        "output_files": ["k_robustness_summary.csv"],
    })
    validation = {
        "module": "08_k_robustness",
        "status": "PASS" if status_agreement else "WARN",
        "k_values": k_values,
        "statuses": frame[["k", "status"]].to_dict("records"),
        "interpretation": "Inspect effect size, rank, peak layer, breadth-aware status and p-values per k.",
    }
    (paths.validation / "08_k_robustness.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved full paper_v1 k sweep for k={k_values}; status_agreement={status_agreement}")


if __name__ == "__main__":
    main()
