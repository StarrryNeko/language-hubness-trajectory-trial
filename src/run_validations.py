import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import configured_representations, ensure_dirs, load_config, representation_file_map
from evidence_rules import classify_model_status, joint_positive_layers, max_consecutive_layers
from numerical_validation import validate_representation_array


ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def report(name, status, method, evidence, interpretation, actions=None):
    return {
        "name": name,
        "status": status,
        "validation_method": method,
        "evidence": evidence,
        "interpretation": interpretation,
        "required_actions": actions or [],
    }


def write_report(folder, number, slug, payload):
    stem = f"{number:02d}_{slug}"
    (folder / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# {payload['name']}", "", f"**Status:** {payload['status']}", "",
        "## Method", "", payload["validation_method"], "", "## Evidence", "",
        *[f"- {item}" for item in payload["evidence"]], "", "## Interpretation", "",
        payload["interpretation"],
    ]
    if payload["required_actions"]:
        lines.extend(["", "## Required actions", "", *[f"- {x}" for x in payload["required_actions"]]])
    (folder / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    validation = Path(paths["validation"])
    metrics = Path(paths["metrics"])
    hidden = Path(paths["hidden"])
    languages = list(cfg["dataset"]["languages"])
    minimum = int(cfg["dataset"].get("minimum_languages_per_semantic_group", 20))
    reports = []

    data_manifest_path = Path(paths["data"]) / "dataset_manifest.json"
    if data_manifest_path.exists():
        manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        passed = (
            len(languages) >= minimum >= 20
            and manifest.get("languages_per_semantic_group") == len(languages)
            and manifest.get("complete_parallel_groups") is True
        )
        evidence = [
            f"Configured languages={len(languages)}; minimum={minimum}",
            f"Semantic groups={manifest.get('semantic_groups')}",
            f"Complete groups={manifest.get('complete_parallel_groups')}",
            f"Candidate scope={manifest.get('candidate_scope')}",
        ]
    else:
        passed, evidence = False, ["dataset_manifest.json is missing"]
    reports.append(report(
        "Balanced >=20-language parallel data", "PASS" if passed else "FAIL",
        "Require every semantic ID to contain exactly one sentence in every configured language.",
        evidence,
        "The same-semantics candidate sets are balanced." if passed else "The language graph is incomplete.",
        [] if passed else ["Rerun prepare_flores.py and repair incomplete semantic groups."],
    ))

    reps = configured_representations(cfg)
    metadata_path = hidden / "metadata.csv"
    extraction_ok = metadata_path.exists()
    extraction_evidence = []
    extraction_layer_counts = set()
    if extraction_ok:
        meta = pd.read_csv(metadata_path)
        extraction_evidence.append(f"Metadata rows={len(meta)}")
        extraction_ok &= not meta.was_truncated.astype(str).str.lower().isin({"true", "1", "yes"}).any()
        eos_ids = meta.sentinel_eos_token_id.nunique()
        extraction_ok &= eos_ids == 1
        extraction_evidence.append(f"Unique sentinel EOS IDs={eos_ids}")
        for name in reps:
            path = hidden / representation_file_map()[name]
            if not path.exists():
                extraction_ok = False
                extraction_evidence.append(f"Missing {path.name}")
                continue
            values = np.load(path, mmap_mode="r")
            try:
                validate_representation_array(values, len(meta), f"validation representation={name}")
                finite = True
            except ValueError as error:
                finite = False
                extraction_evidence.append(str(error))
            extraction_ok &= finite
            if values.ndim == 3:
                extraction_layer_counts.add(int(values.shape[1]))
            extraction_evidence.append(f"{name}: shape={tuple(values.shape)}, all_finite={finite}")
    else:
        extraction_evidence.append("metadata.csv is missing")
    if len(extraction_layer_counts) != 1:
        extraction_ok = False
        expected_metric_layers = None
        extraction_evidence.append(
            f"Representations must share exactly one layer count; found={sorted(extraction_layer_counts)}"
        )
    else:
        expected_metric_layers = list(range(next(iter(extraction_layer_counts))))
    reports.append(report(
        "Mean-pool and sentinel-EOS extraction", "PASS" if extraction_ok else "FAIL",
        "Verify that only the two approved representations exist in the active config, EOS is identical within a model, and arrays are finite/aligned.",
        extraction_evidence,
        "Sentence representations follow the revised protocol." if extraction_ok else "Extraction is unsafe.",
        [] if extraction_ok else ["Rerun extract_hidden.py before computing metrics."],
    ))
    metric_manifest_path = metrics / "metrics_manifest.json"
    if metric_manifest_path.exists():
        metric_manifest = json.loads(metric_manifest_path.read_text(encoding="utf-8"))
        isolated = (
            metric_manifest.get("candidate_scope") == "same_semantic_id_only"
            and metric_manifest.get("cross_semantic_similarity_computed") is False
            and metric_manifest.get("bootstrap_unit") == "semantic_id"
        )
        isolation_evidence = [
            f"candidate_scope={metric_manifest.get('candidate_scope')}",
            f"cross_semantic_similarity_computed={metric_manifest.get('cross_semantic_similarity_computed')}",
            f"bootstrap_unit={metric_manifest.get('bootstrap_unit')}",
        ]
    else:
        isolated, isolation_evidence = False, ["metrics_manifest.json is missing"]
    reports.append(report(
        "Strict same-semantics comparison scope", "PASS" if isolated else "FAIL",
        "Audit the metrics manifest: every candidate set must be one semantic ID and uncertainty must resample semantic IDs.",
        isolation_evidence,
        "No target sentence was ranked against a different meaning." if isolated else "Metric scope is not auditable.",
    ))

    evidence_path = metrics / "english_hubness_evidence.csv"
    hub_status = "FAIL"
    hub_evidence = ["english_hubness_evidence.csv is missing"]
    min_run = int(cfg["metrics"].get("min_consecutive_layers", 3))
    primary = cfg["metrics"].get("primary_representation", "mean_pool")
    validation_representation = cfg["metrics"].get("validation_representation", "sentinel_eos")
    primary_joint_layers = []
    broad_layer_numbers = []
    eos_joint_layers = []
    density_joint_layers = []
    evidence_error = None
    frame = None
    if evidence_path.exists():
        try:
            frame = pd.read_csv(evidence_path)
            main = frame[(frame.representation == primary) & (frame.similarity_method == "cosine")]
            primary_joint_layers = joint_positive_layers(main, expected_layers=expected_metric_layers)
            breadth_path = metrics / "english_hubness_breadth.csv"
            if not breadth_path.exists():
                raise ValueError("english_hubness_breadth.csv is missing")
            breadth = pd.read_csv(breadth_path)
            breadth = breadth[
                (breadth.representation == primary) & (breadth.similarity_method == "cosine")
            ].copy()
            if breadth.empty or breadth.duplicated("layer").any():
                raise ValueError("primary breadth grid is empty or contains duplicate layers")
            if expected_metric_layers is not None and set(breadth.layer.astype(int)) != set(expected_metric_layers):
                raise ValueError("primary breadth grid does not cover every expected layer")
            numeric = breadth[[
                "supported_source_languages", "total_source_languages",
                "supported_source_scripts", "supported_non_latin_languages",
            ]].to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError("primary breadth grid contains non-finite values")
            broad = (
                (breadth.supported_source_languages >= np.ceil(breadth.total_source_languages / 2))
                & (breadth.supported_source_scripts >= 4)
                & (breadth.supported_non_latin_languages >= 3)
            )
            broad_layer_numbers = breadth.loc[broad, "layer"].astype(int).tolist()
            primary_with_breadth = sorted(set(primary_joint_layers) & set(broad_layer_numbers))
            primary_run = max_consecutive_layers(primary_with_breadth)
            hub_status = "PASS" if primary_run >= min_run else "WARN"
            hub_evidence = [
                f"Four-metric joint-positive layers={primary_joint_layers}",
                f"Broad-source layers={broad_layer_numbers}",
                f"Joint evidence + breadth layers={primary_with_breadth}; longest run={primary_run}; required={min_run}",
            ]
        except (KeyError, TypeError, ValueError) as error:
            evidence_error = str(error)
            hub_evidence = [f"Invalid primary evidence: {error}"]
    reports.append(report(
        "Multi-dimensional English hubness evidence", hub_status,
        "Require English to exceed the balanced null in reverse-kNN occurrence, centrality, rank, and medoid rate, with support from at least half of source languages and multiple scripts; proximity alone is insufficient.",
        hub_evidence,
        "All four dimensions contain some supported layers." if hub_status == "PASS" else "Do not make a general English-hub claim yet.",
    ))

    agreement_path = metrics / "representation_agreement.csv"
    agreement_status = "FAIL"
    agreement_evidence = ["English evidence is missing"]
    if frame is not None:
        try:
            eos = frame[
                (frame.representation == validation_representation)
                & (frame.similarity_method == "cosine")
            ]
            eos_joint_layers = joint_positive_layers(eos, expected_layers=expected_metric_layers)
            eos_run = max_consecutive_layers(eos_joint_layers)
            agreement_status = "PASS" if eos_run >= min_run else "WARN"
            agreement_evidence = [
                f"Four-metric joint-positive EOS layers={eos_joint_layers}",
                f"Longest run={eos_run}; required={min_run}",
            ]
            if agreement_path.exists():
                agree = pd.read_csv(agreement_path)
                median_r = float(agree.pairwise_similarity_pearson.median())
                agreement_evidence.append(f"Secondary pair-geometry median Pearson r={median_r:.3f}")
        except (KeyError, TypeError, ValueError) as error:
            evidence_error = evidence_error or str(error)
            agreement_evidence = [f"Invalid EOS evidence: {error}"]
    reports.append(report(
        "Sentinel-EOS validation", agreement_status,
        "Compare layer-wise English evidence signs and all within-semantics language-pair similarities between mean pooling and sentinel EOS.",
        agreement_evidence,
        "EOS broadly validates the mean-pool trajectory." if agreement_status == "PASS" else "The result is representation-sensitive.",
    ))

    global_path = metrics / "hubness_global.csv"
    if global_path.exists():
        global_frame = pd.read_csv(global_path)
        primary = cfg["metrics"].get("primary_representation", "mean_pool")
        ties = global_frame[
            (global_frame.representation == primary)
            & (global_frame.similarity_method == "cosine")
            & (global_frame.metric == "topk_boundary_tie_rate")
        ]
        high_tie_layers = ties[ties["mean"] >= 0.25].layer.astype(int).tolist()
        tie_status = "PASS" if not high_tie_layers else "WARN"
        tie_evidence = [
            f"Mean boundary-tie rate range={ties['mean'].min():.3f}..{ties['mean'].max():.3f}",
            f"Layers with tie rate >=25%: {high_tie_layers}",
        ]
    else:
        tie_status, tie_evidence = "FAIL", ["hubness_global.csv is missing"]
    reports.append(report(
        "Tie-safe neighborhood geometry", tie_status,
        "Fractionally divide the kth-neighbor slot across exact/near ties and flag layers where at least 25% of queries hit a boundary tie.",
        tie_evidence,
        "Top-k selection is not dominated by collapsed/tied geometry." if tie_status == "PASS" else
        "Exclude high-tie layers from substantive hubness claims even though fractional ranking prevents order bias.",
    ))

    if frame is not None:
        try:
            scaled = frame[
                (frame.representation == primary)
                & (frame.similarity_method == "local_scaled_cosine")
            ]
            density_joint_layers = joint_positive_layers(scaled, expected_layers=expected_metric_layers)
            density_run = max_consecutive_layers(density_joint_layers)
            density_status = "PASS" if density_run >= min_run else "WARN"
            density_evidence = [
                f"Four-metric joint-positive local-scaled layers={density_joint_layers}",
                f"Longest run={density_run}; required={min_run}",
            ]
        except (KeyError, TypeError, ValueError) as error:
            evidence_error = evidence_error or str(error)
            density_status = "FAIL"
            density_evidence = [f"Invalid density-control evidence: {error}"]
    else:
        density_status, density_evidence = "FAIL", ["English evidence is missing"]
    reports.append(report(
        "Local-density hubness control", density_status,
        "Re-rank only the same semantic group with a CSLS-style local density correction and compare evidence signs.",
        density_evidence,
        "English evidence is not solely a local-density artifact." if density_status == "PASS" else "Density correction changes the conclusion.",
    ))

    k_path = metrics / "k_robustness_verdict.csv"
    if k_path.exists():
        row = pd.read_csv(k_path).iloc[0]
        k_status = "PASS" if row.status == "CONSISTENT" else "WARN"
        k_evidence = [f"k={row.k_values}; status={row.status}"]
    else:
        k_status, k_evidence = "WARN", ["k sweep not run yet"]
    reports.append(report(
        "k-value robustness", k_status,
        "Repeat the same-semantics graph at several valid k values.", k_evidence,
        "Hubness support is stable across neighborhood sizes." if k_status == "PASS" else "Run or inspect the k sweep before reporting.",
    ))

    slugs = [
        "dataset", "representations", "semantic_scope", "hubness_evidence",
        "eos_validation", "tie_geometry", "density_control", "k_robustness",
    ]
    stems = [write_report(validation, i, slug, payload) for i, (slug, payload) in enumerate(zip(slugs, reports), 1)]
    overall = max(reports, key=lambda item: ORDER[item["status"]])["status"]
    critical_invalid = evidence_error or any(
        item["status"] == "FAIL" for item in reports[:7]
    )
    if critical_invalid:
        model_rule = {
            "status": "INVALID",
            "reason": evidence_error or "required dataset/extraction/metric validation failed",
            "primary_joint_layers": primary_joint_layers,
            "eos_joint_layers": eos_joint_layers,
            "density_joint_layers": density_joint_layers,
            "min_consecutive_layers": min_run,
        }
    else:
        model_rule = classify_model_status(
            primary_joint_layers,
            broad_layer_numbers,
            eos_joint_layers,
            density_joint_layers,
            min_run,
        )
    summary = {
        "overall_status": overall,
        "model_status": model_rule["status"],
        "joint_evidence": model_rule,
        "reports": [{"file_stem": stem, "name": item["name"], "status": item["status"]} for stem, item in zip(stems, reports)],
        "claim_rule": "English hubness requires convergent reverse-kNN, centrality/rank, medoid, source-breadth, EOS, density, k, and multi-model evidence.",
    }
    (validation / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Validation summary", "", f"**Overall status:** {overall}",
        f"**Model status:** {model_rule['status']}", "",
    ]
    lines += [f"- {item['status']}: {item['name']} (`{item['file_stem']}.md`)" for item in summary["reports"]]
    lines += ["", "## Claim rule", "", summary["claim_rule"], ""]
    (validation / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation overall: {overall}; reports saved to {validation}")


if __name__ == "__main__":
    main()
