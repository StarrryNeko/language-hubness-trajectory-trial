"""Estimate observational links between frozen geometry predictors and behavior."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from behavior_common import behavior_settings, ensure_behavior_dirs, sha256_file, write_manifest
from common import load_config, write_json


OUTCOMES = {
    "unnecessary_english_leakage": 1,
    "target_language_retention": -1,
    "semantic_quality_chrfpp": -1,
}
PREDICTORS = {
    "english_minus_target_cosine": ("competitive_raw", 1),
    "english_minus_target_local_scaled": ("competitive_local_scaled", 1),
    "source_english_cosine": ("english_similarity_raw", 1),
    "source_english_local_scaled": ("english_similarity_local_scaled", 1),
    "english_k_occurrence": ("hubness_raw", 1),
    "english_local_scaled_k_occurrence": ("hubness_local_scaled", 1),
    "english_norm": ("mechanism_norm", -1),
    "english_distance_to_global_centroid": ("mechanism_centroid", -1),
    "english_distance_to_loo_semantic_centroid": ("mechanism_loo_centroid", -1),
    "english_pc1_projection": ("mechanism_pc1", 1),
    "english_local_density": ("mechanism_density", 1),
}


def benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.where(np.isfinite(values))[0]
    if not len(finite):
        return result
    order = finite[np.argsort(values[finite])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def standardize(frame, columns):
    result = frame.copy()
    usable = []
    for column in columns:
        if column not in result:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            continue
        result[f"{column}_z"] = (values - float(values.mean())) / scale
        usable.append(f"{column}_z")
    return result, usable


def fit_clustered_model(frame, outcome, predictor, expected_sign):
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("behavior association requires statsmodels>=0.14") from exc
    controls = ["source_sentence_token_count", "target_sentence_token_count"]
    if predictor in {"source_english_cosine", "source_english_local_scaled"}:
        controls.append(
            "source_target_cosine"
            if predictor == "source_english_cosine" else "source_target_local_scaled"
        )
    selected = [outcome, predictor, "semantic_id", "source_lang", "target_lang", *controls]
    data = frame[[column for column in selected if column in frame]].dropna().copy()
    if len(data) < 40 or data.semantic_id.nunique() < 10:
        raise ValueError("fewer than 40 rows or 10 semantic clusters")
    if data[outcome].nunique() < 2:
        raise ValueError("outcome has no variation")
    data, standardized = standardize(data, [predictor, *controls])
    predictor_z = f"{predictor}_z"
    if predictor_z not in standardized:
        raise ValueError("predictor has no variation")
    control_terms = [name for name in standardized if name != predictor_z]
    formula = " + ".join(
        [f"{outcome} ~ {predictor_z}", *control_terms, "C(source_lang)", "C(target_lang)"]
    )
    try:
        if outcome in {"unnecessary_english_leakage", "target_language_retention"}:
            estimator = smf.glm(formula, data=data, family=sm.families.Binomial())
            fit = estimator.fit(cov_type="cluster", cov_kwds={"groups": data.semantic_id})
            family = "binomial_logit_cluster_robust"
        else:
            estimator = smf.ols(formula, data=data)
            fit = estimator.fit(cov_type="cluster", cov_kwds={"groups": data.semantic_id})
            family = "ols_cluster_robust"
    except Exception as exc:
        if exc.__class__.__name__ in {
            "PerfectSeparationError", "PerfectSeparationWarning", "PatsyError"
        }:
            raise ValueError(str(exc)) from exc
        raise
    coefficient = float(fit.params[predictor_z])
    standard_error = float(fit.bse[predictor_z])
    return {
        "coefficient": coefficient,
        "standard_error": standard_error,
        "ci_lower": coefficient - 1.96 * standard_error,
        "ci_upper": coefficient + 1.96 * standard_error,
        "p_value_two_sided": float(fit.pvalues[predictor_z]),
        "direction_consistent": bool(np.sign(coefficient) == int(expected_sign)),
        "rows": len(data),
        "semantic_ids": int(data.semantic_id.nunique()),
        "model_family": family,
        "formula": formula,
        "status": "OK",
    }


def main():
    parser = argparse.ArgumentParser(description="Compute behavior_v1 observational associations")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = behavior_settings(cfg)
    paths = ensure_behavior_dirs(cfg)
    item_path = paths.metrics / "behavior_item_results.csv"
    predictor_path = paths.metrics / "behavior_geometry_predictors.csv"
    items = pd.read_csv(item_path)
    predictors = pd.read_csv(predictor_path)
    if items.task_id.nunique() != len(items):
        raise ValueError("behavior item results must have one row per task")
    merged = predictors.merge(items, on=[
        "task_id", "semantic_id", "condition", "source_lang", "target_lang"
    ], how="left", validate="many_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("geometry predictors and behavior outputs do not align")
    merged = merged.drop(columns="_merge")
    analysis = merged.loc[merged.condition == "non_english_to_non_english"].copy()
    records = []
    for layer in sorted(analysis.layer.unique()):
        layer_frame = analysis.loc[analysis.layer == layer]
        for predictor, (predictor_family, predictor_orientation) in PREDICTORS.items():
            for outcome, outcome_sign in OUTCOMES.items():
                expected_sign = predictor_orientation * outcome_sign
                record = {
                    "layer": int(layer), "normalized_depth": float(layer_frame.normalized_depth.iloc[0]),
                    "predictor": predictor, "predictor_family": predictor_family,
                    "outcome": outcome, "predictor_orientation": predictor_orientation,
                    "expected_sign": expected_sign,
                }
                try:
                    record.update(fit_clustered_model(
                        layer_frame, outcome, predictor, expected_sign
                    ))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    record.update({
                        "coefficient": np.nan, "standard_error": np.nan,
                        "ci_lower": np.nan, "ci_upper": np.nan,
                        "p_value_two_sided": np.nan, "direction_consistent": False,
                        "rows": 0, "semantic_ids": 0, "model_family": None,
                        "formula": None, "status": f"UNAVAILABLE: {exc}",
                    })
                records.append(record)
    results = pd.DataFrame.from_records(records)
    primary_layer = int(settings["primary_layer"])
    primary_mask = (
        (results.layer == primary_layer)
        & (results.predictor == "english_minus_target_cosine")
        & (results.outcome == "unnecessary_english_leakage")
    )
    if primary_mask.sum() != 1:
        raise ValueError("frozen primary association test is missing or duplicated")
    secondary = ~primary_mask
    results["q_value_secondary_bh"] = np.nan
    results.loc[secondary, "q_value_secondary_bh"] = benjamini_hochberg(
        results.loc[secondary, "p_value_two_sided"]
    )
    results["is_frozen_primary"] = primary_mask
    output = paths.metrics / "behavior_association_results.csv"
    results.to_csv(output, index=False, encoding="utf-8")
    primary = results.loc[primary_mask].iloc[0]
    local_check = results.loc[
        (results.layer == primary_layer)
        & (results.predictor == "english_minus_target_local_scaled")
        & (results.outcome == "unnecessary_english_leakage")
    ].iloc[0]
    primary_supported = bool(
        primary.status == "OK"
        and primary.direction_consistent
        and float(primary.p_value_two_sided) < 0.05
    )
    local_robust = bool(
        local_check.status == "OK"
        and local_check.direction_consistent
        and float(local_check.p_value_two_sided) < 0.05
    )
    summary = {
        "protocol_version": "behavior_v1",
        "status": (
            "SUPPORTED_RAW_AND_LOCAL_SCALED" if primary_supported and local_robust
            else "SUPPORTED_RAW_ONLY_DENSITY_SENSITIVE" if primary_supported
            else "NOT_SUPPORTED"
        ),
        "claim_type": "OBSERVATIONAL_ASSOCIATION_NOT_CAUSAL",
        "primary_layer": primary_layer,
        "primary_predictor": "english_minus_target_cosine",
        "primary_outcome": "unnecessary_english_leakage",
        "primary_test": primary.to_dict(),
        "local_scaled_robustness_test": local_check.to_dict(),
        "multiple_testing": "BH correction over all non-primary tests within model",
    }
    write_json(paths.validation / "behavior_association_status.json", summary)
    write_manifest(paths.metrics / "behavior_association_manifest.json", {
        "rows": len(results), "primary_layer": primary_layer,
        "analysis_condition": "non_english_to_non_english",
        "item_results_sha256": sha256_file(item_path),
        "geometry_predictors_sha256": sha256_file(predictor_path),
        "inference": "cluster-robust by semantic ID with source/target language fixed effects",
        "resources": settings["resources"],
    })
    print(f"behavior_v1 association: {summary['status']} -> {output}")


if __name__ == "__main__":
    main()
