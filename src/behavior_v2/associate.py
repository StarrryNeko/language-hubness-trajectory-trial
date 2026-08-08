"""Estimate frozen observational associations for behavior_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_v1.associate import benjamini_hochberg, standardize
from behavior_v2.common import ensure_paths, load_detector_report, settings, sha256_file, write_manifest
from common import load_config, write_json


OUTCOMES = {
    "english_lexical_leakage": {"family": "binomial", "signs": {"english": 1, "latin": 1}},
    "latin_script_fraction": {"family": "ols", "signs": {"english": 1, "latin": 1}},
    "semantic_quality_chrfpp": {"family": "ols", "signs": {"english": -1, "latin": -1}},
}

RAW_PREDICTORS = {
    "english": "english_specific_advantage",
    "latin": "latin_attraction",
    "alignment": "source_target_cosine",
}
LOCAL_PREDICTORS = {
    "english": "english_specific_advantage_local_scaled",
    "latin": "latin_attraction_local_scaled",
    "alignment": "source_target_local_scaled",
}


def fit_joint_model(frame, outcome, family, predictors):
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("behavior_v2 association requires statsmodels>=0.14") from exc
    controls = ["source_sentence_token_count", "target_sentence_token_count"]
    selected = [outcome, "semantic_id", "source_lang", "target_lang", *predictors.values(), *controls]
    data = frame[selected].dropna().copy()
    if len(data) < 100 or data.semantic_id.nunique() < 50:
        raise ValueError("fewer than 100 rows or 50 semantic clusters")
    if data[outcome].nunique() < 2:
        raise ValueError("outcome has no variation")
    data, standardized = standardize(data, [*predictors.values(), *controls])
    required = {f"{name}_z" for name in predictors.values()}
    if not required.issubset(standardized):
        raise ValueError("one or more frozen predictors have no variation")
    terms = [f"{name}_z" for name in predictors.values()]
    terms.extend(name for name in standardized if name not in terms)
    formula = f"{outcome} ~ " + " + ".join([*terms, "C(source_lang)", "C(target_lang)"])
    try:
        if family == "binomial":
            estimator = smf.glm(formula, data=data, family=sm.families.Binomial())
        else:
            estimator = smf.ols(formula, data=data)
        fit = estimator.fit(cov_type="cluster", cov_kwds={"groups": data.semantic_id})
    except Exception as exc:
        if exc.__class__.__name__ in {
            "PerfectSeparationError", "PerfectSeparationWarning", "PatsyError"
        }:
            raise ValueError(str(exc)) from exc
        raise
    results = {}
    for role, predictor in predictors.items():
        term = f"{predictor}_z"
        coefficient = float(fit.params[term])
        standard_error = float(fit.bse[term])
        results[role] = {
            "coefficient": coefficient,
            "standard_error": standard_error,
            "ci_lower": coefficient - 1.96 * standard_error,
            "ci_upper": coefficient + 1.96 * standard_error,
            "p_value_two_sided": float(fit.pvalues[term]),
        }
    return results, {
        "rows": len(data),
        "semantic_ids": int(data.semantic_id.nunique()),
        "model_family": f"{family}_cluster_robust",
        "formula": formula,
    }


def unavailable_record(layer, depth, outcome, scale, role, predictor, expected, message):
    return {
        "layer": int(layer), "normalized_depth": float(depth), "outcome": outcome,
        "geometry_scale": scale, "predictor_role": role, "predictor": predictor,
        "expected_sign": expected, "coefficient": np.nan, "standard_error": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan, "p_value_two_sided": np.nan,
        "direction_consistent": False, "rows": 0, "semantic_ids": 0,
        "model_family": None, "formula": None, "status": f"UNAVAILABLE: {message}",
    }


def main():
    parser = argparse.ArgumentParser(description="Compute behavior_v2 observational associations")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    paths = ensure_paths(cfg)
    detector_path, _ = load_detector_report(cfg, required=True)
    item_path = paths.metrics / "behavior_v2_item_results.csv"
    predictor_path = paths.geometry / "behavior_v2_geometry_predictors.csv"
    evaluation_manifest = json.loads(
        (paths.metrics / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    if evaluation_manifest.get("formal_evaluation_ready") is not True:
        raise ValueError("behavior_v2 association requires a formal-ready evaluation")
    items = pd.read_csv(item_path)
    predictors = pd.read_csv(predictor_path)
    keys = ["task_id", "semantic_id", "condition", "source_lang", "target_lang"]
    if items.task_id.duplicated().any():
        raise ValueError("behavior_v2 item results must have one row per task")
    merged = predictors.merge(items, on=keys, how="left", validate="many_to_one", indicator=True)
    if not (merged._merge == "both").all():
        raise ValueError("behavior_v2 geometry predictors and outcomes do not align")
    merged = merged.drop(columns="_merge")
    records = []
    for layer in sorted(merged.layer.unique()):
        layer_frame = merged.loc[merged.layer == layer]
        depth = float(layer_frame.normalized_depth.iloc[0])
        for scale, predictor_set in (("raw", RAW_PREDICTORS), ("local_scaled", LOCAL_PREDICTORS)):
            for outcome, specification in OUTCOMES.items():
                try:
                    estimates, fit_info = fit_joint_model(
                        layer_frame, outcome, specification["family"], predictor_set
                    )
                    for role, predictor in predictor_set.items():
                        expected = specification["signs"].get(role, 0)
                        estimate = estimates[role]
                        records.append({
                            "layer": int(layer), "normalized_depth": depth,
                            "outcome": outcome, "geometry_scale": scale,
                            "predictor_role": role, "predictor": predictor,
                            "expected_sign": expected, **estimate,
                            "direction_consistent": bool(
                                expected == 0 or np.sign(estimate["coefficient"]) == expected
                            ),
                            **fit_info, "status": "OK",
                        })
                except (ValueError, np.linalg.LinAlgError) as exc:
                    for role, predictor in predictor_set.items():
                        records.append(unavailable_record(
                            layer, depth, outcome, scale, role, predictor,
                            specification["signs"].get(role, 0), str(exc),
                        ))
    results = pd.DataFrame.from_records(records)
    primary_mask = (
        (results.layer == protocol["primary_layer"])
        & (results.outcome == "english_lexical_leakage")
        & (results.geometry_scale == "raw")
        & (results.predictor_role == "english")
    )
    if primary_mask.sum() != 1:
        raise ValueError("frozen behavior_v2 primary test is missing or duplicated")
    results["is_frozen_primary"] = primary_mask
    results["q_value_secondary_bh"] = np.nan
    secondary = ~primary_mask
    results.loc[secondary, "q_value_secondary_bh"] = benjamini_hochberg(
        results.loc[secondary, "p_value_two_sided"]
    )
    output = paths.measurement / "behavior_v2_association_results.csv"
    results.to_csv(output, index=False, encoding="utf-8")
    primary = results.loc[primary_mask].iloc[0]
    local = results.loc[
        (results.layer == protocol["primary_layer"])
        & (results.outcome == "english_lexical_leakage")
        & (results.geometry_scale == "local_scaled")
        & (results.predictor_role == "english")
    ].iloc[0]
    primary_supported = bool(
        primary.status == "OK" and primary.direction_consistent
        and float(primary.p_value_two_sided) < 0.05
    )
    local_supported = bool(
        local.status == "OK" and local.direction_consistent
        and float(local.p_value_two_sided) < 0.05
    )
    summary = {
        "protocol_version": protocol["protocol_version"],
        "status": (
            "SUPPORTED_RAW_AND_LOCAL_SCALED" if primary_supported and local_supported
            else "SUPPORTED_RAW_ONLY" if primary_supported else "NOT_SUPPORTED"
        ),
        "claim_type": "OBSERVATIONAL_ASSOCIATION_NOT_CAUSAL",
        "primary_layer": protocol["primary_layer"],
        "primary_predictor": "english_specific_advantage",
        "primary_outcome": "english_lexical_leakage",
        "primary_test": primary.to_dict(),
        "local_scaled_robustness_test": local.to_dict(),
        "multiple_testing": "BH correction over all non-primary tests within model",
    }
    write_json(paths.validation / "association_status.json", summary)
    write_manifest(paths.measurement / "association_manifest.json", {
        "config_path": str(config_path), "rows": len(results),
        "primary_layer": protocol["primary_layer"],
        "item_results_sha256": sha256_file(item_path),
        "geometry_predictors_sha256": sha256_file(predictor_path),
        "detector_report_sha256": sha256_file(detector_path),
        "inference": "joint models with cluster-robust SE by semantic ID and source/target fixed effects",
        "activation_intervention": False,
    })
    print(f"behavior_v2 association: {summary['status']} -> {output}")


if __name__ == "__main__":
    main()
