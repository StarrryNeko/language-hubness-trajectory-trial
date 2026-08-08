"""Run the frozen formal V3 GEE association and density robustness test."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

from behavior_association_v3.common import item_path, paths, settings, sha256_file, write_manifest
from behavior_v1.associate import benjamini_hochberg
from common import load_config, write_json


def standardize(frame, columns):
    result = frame.copy()
    for column in columns:
        scale = float(result[column].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"predictor/control has no variation: {column}")
        result[f"{column}_z"] = (result[column] - result[column].mean()) / scale
    return result


def fit_gee(frame, predictor):
    columns = [predictor, "source_token_count", "target_token_count"]
    data = standardize(frame.dropna(subset=["english_lexical_leakage", *columns]), columns)
    formula = (
        f"english_lexical_leakage ~ {predictor}_z + source_token_count_z + "
        "target_token_count_z + C(source_lang) + C(target_lang)"
    )
    model = smf.gee(
        formula, groups="semantic_id", data=data, family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )
    fit = model.fit(maxiter=200)
    term = f"{predictor}_z"
    return {
        "predictor": predictor, "coefficient": float(fit.params[term]),
        "standard_error": float(fit.bse[term]),
        "ci_lower": float(fit.conf_int().loc[term, 0]),
        "ci_upper": float(fit.conf_int().loc[term, 1]),
        "p_value_two_sided": float(fit.pvalues[term]),
        "rows": len(data), "semantic_ids": int(data.semantic_id.nunique()),
        "events": int(data.english_lexical_leakage.sum()),
        "formula": formula, "model_family": "binomial_logit_GEE_exchangeable",
    }


def fit_cluster_ols(frame, outcome, predictor):
    columns = [predictor, "source_token_count", "target_token_count"]
    data = standardize(frame.dropna(subset=[outcome, *columns]), columns)
    formula = (
        f"{outcome} ~ {predictor}_z + source_token_count_z + "
        "target_token_count_z + C(source_lang) + C(target_lang)"
    )
    fit = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data.semantic_id}
    )
    term = f"{predictor}_z"
    return {
        "predictor": predictor, "outcome": outcome,
        "coefficient": float(fit.params[term]), "standard_error": float(fit.bse[term]),
        "ci_lower": float(fit.conf_int().loc[term, 0]),
        "ci_upper": float(fit.conf_int().loc[term, 1]),
        "p_value_two_sided": float(fit.pvalues[term]),
        "rows": len(data), "semantic_ids": int(data.semantic_id.nunique()),
        "formula": formula, "model_family": "OLS_cluster_robust_semantic_id",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    protocol = settings(cfg)
    evaluation_path = paths(cfg).metrics / "formal_evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "PASS":
        raise ValueError("formal V3 evaluation gates must pass before association")
    items = pd.read_csv(item_path(cfg, "formal"))
    predictors_path = paths(cfg).measurement / "geometry_predictors.csv"
    predictors = pd.read_csv(predictors_path)
    formal_predictors = predictors.loc[predictors.split == "formal"]
    merged_all = formal_predictors.merge(
        items, on=["task_id", "split", "semantic_id", "source_lang", "target_lang"],
        how="left", validate="many_to_one", indicator=True,
    )
    if not (merged_all._merge == "both").all():
        raise ValueError("formal V3 predictors and outcomes do not align")
    merged = merged_all.loc[merged_all.layer == protocol["primary_layer"]].copy()
    events = int(merged.english_lexical_leakage.sum())
    minimum = protocol["gates"]["minimum_primary_events"]
    result_records = []
    # Adjacent-layer and density checks keep the same competition hypothesis.
    if events >= minimum:
        for layer in protocol["analysis_layers"]:
            layer_frame = merged_all.loc[merged_all.layer == layer]
            for scale, predictor in (
                ("raw", "english_target_competition"),
                ("local_scaled", "english_target_competition_local_scaled"),
            ):
                estimate = fit_gee(layer_frame, predictor)
                result_records.append({
                    "layer": layer, "geometry_scale": scale,
                    "analysis_role": "primary" if layer == protocol["primary_layer"] and scale == "raw" else (
                        "density_robustness" if layer == protocol["primary_layer"] else "layer_sensitivity"
                    ),
                    "outcome": "english_lexical_leakage", **estimate,
                })
        # Primary-layer decomposition: English-specific vs general Latin attraction.
        for scale, suffix in (("raw", ""), ("local_scaled", "_local_scaled")):
            estimate = fit_gee(merged, f"english_specific_advantage{suffix}")
            result_records.append({
                "layer": protocol["primary_layer"], "geometry_scale": scale,
                "analysis_role": "secondary_decomposition",
                "outcome": "english_lexical_leakage", **estimate,
            })
    for scale, suffix in (("raw", ""), ("local_scaled", "_local_scaled")):
        estimate = fit_cluster_ols(merged, "latin_script_fraction", f"latin_attraction{suffix}")
        result_records.append({
            "layer": protocol["primary_layer"], "geometry_scale": scale,
            "analysis_role": "secondary_decomposition",
            **estimate,
        })
    quality = fit_cluster_ols(
        merged, "semantic_quality_chrfpp", "english_target_competition"
    )
    result_records.append({
        "layer": protocol["primary_layer"], "geometry_scale": "raw",
        "analysis_role": "quality_guardrail", **quality,
    })
    results = pd.DataFrame(result_records)
    results["q_value_secondary_bh"] = np.nan
    secondary_mask = ~results.analysis_role.isin(["primary", "density_robustness"])
    results.loc[secondary_mask, "q_value_secondary_bh"] = benjamini_hochberg(
        results.loc[secondary_mask, "p_value_two_sided"]
    )
    result_path = paths(cfg).measurement / "association_results.csv"
    results.to_csv(result_path, index=False, encoding="utf-8")
    if events < minimum:
        summary = {
            "protocol_version": "behavior_association_v3",
            "status": "PRIMARY_NOT_ESTIMABLE_DUE_TO_LOW_EVENT_COUNT",
            "events": events, "minimum_primary_events": minimum,
            "claim_type": "OBSERVATIONAL_ASSOCIATION_NOT_CAUSAL",
        }
    else:
        raw = results.loc[results.analysis_role == "primary"].iloc[0].to_dict()
        local = results.loc[results.analysis_role == "density_robustness"].iloc[0].to_dict()
        raw_supported = raw["coefficient"] > 0 and raw["ci_lower"] > 0
        local_supported = local["coefficient"] > 0 and local["ci_lower"] > 0
        summary = {
            "protocol_version": "behavior_association_v3",
            "status": (
                "SUPPORTED_RAW_AND_LOCAL_SCALED" if raw_supported and local_supported
                else "SUPPORTED_RAW_ONLY" if raw_supported else "NOT_SUPPORTED"
            ),
            "claim_type": "OBSERVATIONAL_ASSOCIATION_NOT_CAUSAL",
            "primary_layer": protocol["primary_layer"],
            "primary_predictor": "english_target_competition",
            "primary_outcome": "english_lexical_leakage",
            "primary_test": raw, "local_scaled_robustness_test": local,
        }
    output = paths(cfg).validation / "association_status.json"
    write_json(output, summary)
    write_manifest(paths(cfg).measurement / "association_manifest.json", {
        "item_results_sha256": sha256_file(item_path(cfg, "formal")),
        "geometry_predictors_sha256": sha256_file(predictors_path),
        "association_results_sha256": sha256_file(result_path),
        "formal_evaluation_manifest_sha256": sha256_file(evaluation_path),
        "inference": "primary binomial GEE by semantic ID; secondary OLS uses semantic-ID cluster-robust SE",
        "multiple_testing": "BH over secondary decomposition, layer sensitivity, and quality guardrail; primary and local robustness excluded",
        "activation_intervention": False,
    })
    print(f"V3 association: {summary['status']}")


if __name__ == "__main__":
    main()
