"""Aggregate module-level evidence into bounded paper_v1 claim statuses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import load_config
from evidence_rules import max_consecutive_layers
from paper_common import ensure_paper_dirs, load_hidden_dataset, semantic_id_hash


def require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} is empty")


def main():
    parser = argparse.ArgumentParser(description="Validate and summarize paper_v1 evidence")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_paper_dirs(cfg)
    dataset = load_hidden_dataset(cfg)
    expected_hash = semantic_id_hash(dataset.semantic_ids)
    min_run = int(cfg["metrics"].get("min_consecutive_layers", 3))
    blockers = []
    caveats = []
    dataset_manifest_path = Path(cfg["output_dir"]) / "data" / "dataset_manifest.json"
    if dataset_manifest_path.exists():
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        sampling_strategy = dataset_manifest.get("sample_selection_strategy", "unknown")
    else:
        sampling_strategy = cfg.get("dataset", {}).get("sample_selection", {}).get(
            "strategy", "unknown"
        )
    experiment_stage = (
        "FORMAL_RANDOM_SAMPLE" if sampling_strategy == "random_without_replacement"
        else "METHOD_REANALYSIS" if sampling_strategy == "first_n"
        else "SAMPLING_UNVERIFIED"
    )
    if experiment_stage != "FORMAL_RANDOM_SAMPLE":
        caveats.append(
            "The extracted semantic pool was not verified as a random sample; use it for method reanalysis, not the final confirmatory claim."
        )

    manifest_paths = [
        paths.metrics / "alignment" / "alignment_manifest.json",
        paths.metrics / "hubness" / "hubness_manifest.json",
        paths.metrics / "geometry_controls" / "geometry_manifest.json",
        paths.metrics / "similarity_competition" / "similarity_competition_manifest.json",
        paths.metrics / "norm_trajectory" / "norm_trajectory_manifest.json",
        paths.metrics / "language_structure" / "language_structure_manifest.json",
        paths.metrics / "language_probe" / "probe_manifest.json",
    ]
    for path in manifest_paths:
        if not path.exists():
            blockers.append(f"missing manifest: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != "paper_v1":
            blockers.append(f"wrong protocol version: {path}")
        if payload.get("semantic_id_sha256") != expected_hash:
            blockers.append(f"semantic-ID hash mismatch: {path}")

    alignment_status = "INVALID"
    alignment_details = {}
    alignment_path = paths.metrics / "alignment" / "alignment_summary.csv"
    retrieval_path = paths.metrics / "alignment" / "semantic_retrieval.csv"
    if alignment_path.exists() and retrieval_path.exists():
        alignment = pd.read_csv(alignment_path)
        retrieval = pd.read_csv(retrieval_path)
        require_columns(alignment, ["layer", "pair_group", "ci_lower"], "alignment_summary")
        require_columns(
            retrieval,
            ["layer", "query_lang", "target_lang", "recall1_ci_lower", "random_recall_at_1"],
            "semantic_retrieval",
        )
        positive_layers = alignment.loc[
            (alignment.pair_group == "all_pairs") & (alignment.layer > 0) & (alignment.ci_lower > 0),
            "layer",
        ].astype(int).tolist()
        direction_support = retrieval[retrieval.layer > 0].groupby(
            ["query_lang", "target_lang"]
        ).apply(
            lambda group: bool((group.recall1_ci_lower > group.random_recall_at_1).any()),
            include_groups=False,
        )
        support_fraction = float(direction_support.mean())
        longest = max_consecutive_layers(positive_layers)
        alignment_status = (
            "SUPPORTED" if longest >= min_run and support_fraction >= 0.8 else "NOT_SUPPORTED"
        )
        alignment_details = {
            "overall_positive_layers": positive_layers,
            "longest_run": longest,
            "required_run": min_run,
            "retrieval_directions_supported": int(direction_support.sum()),
            "retrieval_directions_total": int(len(direction_support)),
            "retrieval_support_fraction": support_fraction,
            "required_retrieval_fraction": 0.8,
        }
    else:
        blockers.append("alignment outputs are incomplete")

    hubness_status = "INVALID"
    hubness_details = {}
    hub_status_path = paths.metrics / "hubness" / "paper_model_status.json"
    permutation_path = paths.metrics / "hubness" / "target_rotation_permutation.csv"
    specificity_path = paths.metrics / "hubness" / "target_rotation_summary.csv"
    if hub_status_path.exists() and permutation_path.exists() and specificity_path.exists():
        hubness_details = json.loads(hub_status_path.read_text(encoding="utf-8"))
        hubness_status = hubness_details.get("status", "INVALID")
        permutation = pd.read_csv(permutation_path)
        specificity = pd.read_csv(specificity_path)
        require_columns(
            permutation, ["similarity_method", "global_max_target_layer_p_value"],
            "target_rotation_permutation",
        )
        require_columns(
            specificity, ["similarity_method", "layer", "english_rank"],
            "target_rotation_summary",
        )
        global_p = permutation.groupby("similarity_method").global_max_target_layer_p_value.first()
        selection_corrected = bool(
            global_p.get("cosine", 1.0) <= 0.05
            and global_p.get("local_scaled_cosine", 1.0) <= 0.05
        )
        english_specificity_status = (
            "SELECTION_CORRECTED" if selection_corrected else "NOT_SUPPORTED"
        )
        hubness_details["global_permutation_p_values"] = {
            key: float(value) for key, value in global_p.items()
        }
        hubness_details["english_top_rank_layers"] = {
            method: group.loc[group.english_rank == 1, "layer"].astype(int).tolist()
            for method, group in specificity.groupby("similarity_method")
        }
    else:
        english_specificity_status = "INVALID"
        blockers.append("hubness target-rotation outputs are incomplete")

    competition_status = "INVALID"
    competition_details = {}
    competition_status_path = (
        paths.metrics / "similarity_competition" / "similarity_competition_status.json"
    )
    competition_summary_path = (
        paths.metrics / "similarity_competition" / "english_competition_by_layer.csv"
    )
    competition_matrix_path = (
        paths.metrics / "similarity_competition" / "source_candidate_similarity.csv"
    )
    if (
        competition_status_path.exists()
        and competition_summary_path.exists()
        and competition_matrix_path.exists()
    ):
        competition_details = json.loads(
            competition_status_path.read_text(encoding="utf-8")
        )
        competition_status = competition_details.get("status", "INVALID")
        competition_summary = pd.read_csv(competition_summary_path)
        competition_matrix = pd.read_csv(competition_matrix_path)
        require_columns(
            competition_summary,
            ["similarity_method", "layer", "metric", "mean", "ci_lower", "ci_upper"],
            "english_competition_by_layer",
        )
        require_columns(
            competition_matrix,
            ["similarity_method", "layer", "source_lang", "candidate_lang", "mean"],
            "source_candidate_similarity",
        )
        expected_metrics = {
            "english_similarity", "best_non_english_similarity", "hard_margin",
            "pairwise_win_rate", "english_rank",
        }
        if set(competition_summary.metric) != expected_metrics:
            blockers.append("competitive similarity metrics are incomplete")
        if competition_matrix.source_lang.nunique() != len(dataset.languages):
            blockers.append("competitive similarity source-language matrix is incomplete")
    else:
        blockers.append("competitive similarity outputs are incomplete")

    norm_status = "INVALID"
    trajectory_status = "INVALID"
    norm_details = {}
    norm_summary_path = paths.metrics / "norm_trajectory" / "language_geometry_by_layer.csv"
    dynamics_path = (
        paths.metrics / "norm_trajectory" / "adjacent_layer_dynamics_by_language.csv"
    )
    trajectory_path = (
        paths.metrics / "norm_trajectory" / "english_trajectory_events_summary.json"
    )
    if norm_summary_path.exists() and dynamics_path.exists() and trajectory_path.exists():
        norm_summary = pd.read_csv(norm_summary_path)
        dynamics = pd.read_csv(dynamics_path)
        require_columns(
            norm_summary,
            ["layer", "normalized_depth", "language", "feature", "mean", "ci_lower"],
            "language_geometry_by_layer",
        )
        require_columns(
            dynamics,
            ["layer", "normalized_depth", "language", "feature", "mean"],
            "adjacent_layer_dynamics_by_language",
        )
        norm_details = json.loads(trajectory_path.read_text(encoding="utf-8"))
        norm_manifest = json.loads(
            (paths.metrics / "norm_trajectory" / "norm_trajectory_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        norm_status = "AVAILABLE"
        trajectory_status = norm_details.get("status", "INVALID")
        required_geometry = {
            "raw_norm", "norm_rank", "norm_zscore", "distance_to_global_centroid",
            "distance_to_loo_semantic_centroid", "pc1_projection", "local_density",
        }
        if not required_geometry.issubset(set(norm_summary.feature)):
            blockers.append("norm/centroid/density geometry features are incomplete")
        if experiment_stage == "FORMAL_RANDOM_SAMPLE" and not norm_manifest.get(
            "token_count_saved", False
        ):
            blockers.append("formal mechanism analysis is missing sentence token counts")
    else:
        blockers.append("norm or layer-trajectory outputs are incomplete")

    sample_manifest_path = paths.metrics / "hubness" / "sample_robustness_manifest.json"
    if sample_manifest_path.exists():
        sample_payload = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
        sample_status = sample_payload.get("sample_robustness_status", "INVALID")
        sample_details = sample_payload
        caveats.append(sample_payload.get("limitation"))
    else:
        sample_status = "NOT_RUN"
        sample_details = {}
        blockers.append("sample robustness has not been run")

    k_path = paths.metrics / "hubness" / "k_robustness_summary.csv"
    if k_path.exists():
        k_frame = pd.read_csv(k_path)
        k_status = "STABLE" if k_frame.status.nunique() == 1 else "HETEROGENEOUS"
        k_details = k_frame.to_dict("records")
    else:
        k_status, k_details = "NOT_RUN", []
        blockers.append("paper_v1 k robustness has not been run")

    probe_path = paths.metrics / "language_probe" / "probe_scores.csv"
    if probe_path.exists():
        probe = pd.read_csv(probe_path)
        require_columns(probe, ["layer", "macro_f1", "empirical_p_value"], "probe_scores")
        best_probe = probe.loc[probe.macro_f1.idxmax()]
        probe_status = "AVAILABLE"
        probe_details = {
            "peak_macro_f1": float(best_probe.macro_f1),
            "peak_layer": int(best_probe.layer),
            "peak_empirical_p_value": float(best_probe.empirical_p_value),
        }
    else:
        probe_status, probe_details = "NOT_RUN", {}
        blockers.append("language probe has not been run")

    structure_path = paths.metrics / "language_structure" / "centroid_separation.csv"
    language_structure_status = "AVAILABLE" if structure_path.exists() else "NOT_RUN"
    if language_structure_status == "NOT_RUN":
        blockers.append("language structure has not been run")

    if blockers:
        overall = "NEEDS_REVISION"
        claim_level = "NO_PAPER_CLAIM"
    elif alignment_status != "SUPPORTED":
        overall = "SHARE_WITH_CAVEATS"
        claim_level = "GEOMETRY_ONLY_NO_SHARED_SEMANTIC_CLAIM"
    elif english_specificity_status != "SELECTION_CORRECTED":
        overall = "READY_FOR_MODEL_COMPARISON"
        claim_level = "MULTILINGUAL_HUB_STRUCTURE_WITHOUT_ENGLISH_SPECIFICITY"
    elif competition_status == "NOT_SUPPORTED":
        overall = "READY_FOR_MODEL_COMPARISON"
        claim_level = "ENGLISH_HUB_WITHOUT_STRONGEST_COMPETITOR_ADVANTAGE"
    elif hubness_status == "ROBUST" and competition_status == "ROBUST" and sample_status == "REPLICATED":
        overall = "READY_FOR_MODEL_COMPARISON"
        claim_level = "GEOMETRY_ROBUST_COMPETITIVE_ENGLISH_SPECIFIC_HUB"
    elif hubness_status == "DENSITY_SENSITIVE":
        overall = "READY_FOR_MODEL_COMPARISON"
        claim_level = "DENSITY_SENSITIVE_ENGLISH_PATTERN"
    else:
        overall = "READY_FOR_MODEL_COMPARISON"
        claim_level = "ENGLISH_SPECIFICITY_WITH_LIMITED_ROBUSTNESS"
    scientific_claim_level = claim_level
    if not blockers and experiment_stage != "FORMAL_RANDOM_SAMPLE":
        overall = "METHOD_REANALYSIS_ONLY"
        claim_level = f"PROVISIONAL_{scientific_claim_level}"

    summary = {
        "protocol_version": "paper_v1",
        "overall_assessment": overall,
        "claim_level": claim_level,
        "scientific_claim_level_before_sampling_qualification": scientific_claim_level,
        "experiment_stage": experiment_stage,
        "sample_selection_strategy": sampling_strategy,
        "statuses": {
            "alignment_status": alignment_status,
            "english_specificity_status": english_specificity_status,
            "similarity_competition_status": competition_status,
            "geometry_robustness_status": hubness_status,
            "norm_mechanism_status": norm_status,
            "layer_trajectory_status": trajectory_status,
            "sample_robustness_status": sample_status,
            "k_robustness_status": k_status,
            "language_structure_status": language_structure_status,
            "language_probe_status": probe_status,
            "cross_model_status": "NOT_RUN",
            "behavior_association_status": "NOT_RUN",
            "intervention_status": "NOT_RUN",
        },
        "alignment": alignment_details,
        "hubness": hubness_details,
        "similarity_competition": competition_details,
        "norm_and_trajectory": norm_details,
        "sample_robustness": sample_details,
        "k_robustness": k_details,
        "language_probe": probe_details,
        "blockers": blockers,
        "caveats": [item for item in caveats if item],
        "allowed_claim": (
            "Offline analyses can describe competitive English hubness, radial/directional "
            "geometry and layer trajectories. They do not establish behavioral causality."
        ),
    }
    (paths.validation / "paper_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# paper_v1 validation summary", "",
        f"- Overall assessment: `{overall}`",
        f"- Claim level: `{claim_level}`", "",
        "## Module statuses", "",
        *[f"- {name}: `{value}`" for name, value in summary["statuses"].items()],
        "", "## Blocking issues", "",
        *([f"- {item}" for item in blockers] if blockers else ["- None"]),
        "", "## Required interpretation boundary", "",
        "The current offline analyses describe representation geometry. They do not establish a behavioral or causal effect.",
    ]
    (paths.validation / "paper_validation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"paper_v1 validation: {overall}; claim={claim_level}")


if __name__ == "__main__":
    main()
