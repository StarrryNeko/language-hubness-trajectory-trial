"""Compare completed paper_v1 analyses without requiring positive replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import load_config, model_metadata


def compare_suite(suite_path):
    suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    output = Path(suite["comparison_output_dir"]) / "paper_v1"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []
    for relative in suite["configs"]:
        config_path = suite_path.parent / relative
        cfg = load_config(config_path)
        paper_root = Path(cfg["output_dir"]) / cfg.get("paper_v1", {}).get("result_directory", "paper_v1")
        summary_path = paper_root / "validation" / "paper_validation_summary.json"
        specificity_path = paper_root / "metrics" / "hubness" / "target_rotation_summary.csv"
        competition_path = (
            paper_root / "metrics" / "similarity_competition" / "english_competition_by_layer.csv"
        )
        trajectory_path = (
            paper_root / "metrics" / "norm_trajectory" / "english_trajectory_events_summary.json"
        )
        geometry_path = (
            paper_root / "metrics" / "norm_trajectory" / "language_geometry_by_layer.csv"
        )
        if not summary_path.exists() or not specificity_path.exists():
            missing.append(str(config_path))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        specificity = pd.read_csv(specificity_path)
        raw = specificity[specificity.similarity_method == "cosine"]
        density = specificity[specificity.similarity_method == "local_scaled_cosine"]
        metadata = model_metadata(cfg, require=bool(cfg.get("comparison_metadata_required", False)))
        record = {
            "experiment_name": cfg.get("experiment_name", Path(relative).stem),
            "model": cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path")),
            "family": metadata.get("model_family"),
            "generation": metadata.get("model_generation"),
            "parameter_count_billions": metadata.get("parameter_count_billions"),
            "size_class": metadata.get("size_class"),
            "overall_assessment": summary.get("overall_assessment"),
            "claim_level": summary.get("claim_level"),
            **summary.get("statuses", {}),
            "raw_best_english_rank": int(raw.english_rank.min()),
            "raw_top_rank_layer_count": int((raw.english_rank == 1).sum()),
            "density_best_english_rank": int(density.english_rank.min()),
            "density_top_rank_layer_count": int((density.english_rank == 1).sum()),
        }
        if competition_path.exists():
            competition = pd.read_csv(competition_path)
            hard = competition[
                (competition.similarity_method == "cosine")
                & (competition.metric == "hard_margin")
            ]
            wins = competition[
                (competition.similarity_method == "cosine")
                & (competition.metric == "pairwise_win_rate")
            ]
            record.update({
                "raw_best_hard_margin": float(hard["mean"].max()),
                "raw_positive_hard_margin_layer_count": int((hard.ci_lower > 0).sum()),
                "raw_best_pairwise_win_rate": float(wins["mean"].max()),
            })
        if trajectory_path.exists():
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            record.update({
                "median_hub_entry_normalized_depth": trajectory.get(
                    "median_entry_normalized_depth"
                ),
                "median_hub_peak_normalized_depth": trajectory.get(
                    "median_peak_normalized_depth"
                ),
                "late_reversal_fraction": trajectory.get("late_reversal_fraction"),
            })
        if geometry_path.exists():
            geometry = pd.read_csv(geometry_path)
            english_norm = geometry[
                (geometry.language == cfg["metrics"].get("english_language", "en"))
                & (geometry.feature == "norm_rank")
            ]
            record["english_best_norm_rank"] = float(english_norm["mean"].min())
        records.append(record)
    if not records:
        raise FileNotFoundError("no completed paper_v1 model analyses were found")
    frame = pd.DataFrame(records)
    frame.to_csv(output / "paper_model_comparison.csv", index=False, encoding="utf-8")
    status_counts = {
        column: frame[column].value_counts(dropna=False).to_dict()
        for column in [
            "alignment_status", "english_specificity_status",
            "geometry_robustness_status", "sample_robustness_status",
        ]
        if column in frame
    }
    summary = {
        "protocol_version": "paper_v1",
        "models_completed": len(frame),
        "models_missing": missing,
        "status_counts": status_counts,
        "cross_model_status": "AVAILABLE" if len(frame) >= 2 else "INSUFFICIENT_MODELS",
        "interpretation": (
            "Model heterogeneity is reported directly; no minimum number of positive models is imposed."
        ),
    }
    (output / "paper_model_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figure_frame = frame.sort_values("raw_best_english_rank")
    fig, ax = plt.subplots(figsize=(max(7, len(frame) * 1.5), 4.5))
    positions = range(len(figure_frame))
    ax.scatter(positions, figure_frame.raw_best_english_rank, label="raw cosine", s=55)
    ax.scatter(positions, figure_frame.density_best_english_rank, label="local-scaled", s=55)
    ax.set_xticks(list(positions), figure_frame.experiment_name, rotation=25, ha="right")
    ax.set_ylabel("Best English target rank (1 is highest)")
    ax.set_title("English target rank across paper_v1 model analyses")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "english_rank_cross_model.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description="Compare completed paper_v1 model outputs")
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    frame, summary = compare_suite(args.suite)
    print(f"Compared {len(frame)} models; cross_model_status={summary['cross_model_status']}")


if __name__ == "__main__":
    main()
