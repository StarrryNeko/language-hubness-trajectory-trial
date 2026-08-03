"""Recompute paper_v1 hubness status on frozen random semantic subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except (ImportError, PermissionError):
    def tqdm(iterable, **_kwargs):
        return iterable

from common import load_config
from compute_hubness_paper import HUB_METRICS
from evidence_rules import classify_model_status
from paper_common import bootstrap_mean_ci, ensure_paper_dirs, paper_settings, write_manifest


def breadth_pass(selected, english_index, languages, language_metadata, rng, n_boot, confidence, k):
    supported_languages = []
    for source_index, source_lang in enumerate(languages):
        if source_index == english_index:
            continue
        values = selected[:, source_index, english_index]
        _, low, _ = bootstrap_mean_ci(values, rng, n_boot, confidence)
        if low > k / (len(languages) - 1):
            supported_languages.append(source_lang)
    scripts = {
        language_metadata.get(lang, {}).get("script", "unknown") for lang in supported_languages
    }
    non_latin = [
        lang for lang in supported_languages
        if language_metadata.get(lang, {}).get("script", "unknown") != "Latin"
    ]
    return {
        "supported_source_languages": len(supported_languages),
        "total_source_languages": len(languages) - 1,
        "supported_source_scripts": len(scripts),
        "supported_non_latin_languages": len(non_latin),
        "pass": (
            len(supported_languages) >= np.ceil((len(languages) - 1) / 2)
            and len(scripts) >= 4 and len(non_latin) >= 3
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="paper_v1 random semantic-subset robustness")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = paper_settings(cfg)
    paths = ensure_paper_dirs(cfg)
    hubness = paths.metrics / "hubness"
    arrays = sorted((hubness / "sample_arrays").glob("layer_*.npz"))
    if not arrays:
        raise FileNotFoundError("hubness sample arrays are missing; run compute_hubness_paper.py first")
    first = np.load(arrays[0])
    semantic_ids = first["semantic_ids"].astype(str).tolist()
    languages = first["languages"].astype(str).tolist()
    english = cfg["metrics"].get("english_language", "en")
    english_index = languages.index(english)
    n_subsets = int(cfg.get("paper_v1", {}).get("sample_robustness_subsets", 10))
    requested_size = int(cfg.get("paper_v1", {}).get("sample_robustness_size", 80))
    subset_size = min(requested_size, len(semantic_ids))
    if subset_size < 20:
        raise ValueError("sample robustness requires at least 20 semantic IDs per subset")
    subset_rng = np.random.default_rng(settings["seed"] + 601)
    subsets = [
        np.sort(subset_rng.choice(len(semantic_ids), subset_size, replace=False))
        for _ in range(n_subsets)
    ]
    language_metadata = cfg["dataset"].get("language_metadata", {})
    n_boot = settings["bootstrap_samples"]
    confidence = settings["confidence_level"]
    k = int(cfg["metrics"].get("nearest_neighbors_k", 5))
    min_run = int(cfg["metrics"].get("min_consecutive_layers", 3))
    layer_records = []

    for layer_path in tqdm(arrays, desc="paper_v1 sample robustness"):
        layer = int(layer_path.stem.split("_")[-1])
        payload = np.load(layer_path)
        for subset_index, indices in enumerate(subsets):
            rng = np.random.default_rng(settings["seed"] + 7000 + subset_index * 100 + layer)
            record = {"subset": subset_index, "layer": layer, "n_semantic_ids": len(indices)}
            for method in ("cosine", "local_scaled_cosine"):
                metric_passes = []
                for metric in HUB_METRICS:
                    values = payload[f"{method}__{metric}"][indices, english_index]
                    mean, low, high = bootstrap_mean_ci(values, rng, n_boot, confidence)
                    record[f"{method}__{metric}__mean"] = mean
                    record[f"{method}__{metric}__ci_lower"] = low
                    record[f"{method}__{metric}__ci_upper"] = high
                    metric_passes.append(low > 0)
                selected = payload[f"{method}__selected"][indices].astype(np.float32)
                breadth = breadth_pass(
                    selected, english_index, languages, language_metadata,
                    rng, n_boot, confidence, k,
                )
                record[f"{method}__four_metric_pass"] = all(metric_passes)
                record[f"{method}__breadth_pass"] = breadth.pop("pass")
                for key, value in breadth.items():
                    record[f"{method}__{key}"] = value
                primary_values = payload[f"{method}__k_occurrence_excess"][indices]
                means = primary_values.mean(axis=0)
                record[f"{method}__english_rank"] = int(
                    pd.Series(-means).rank(method="min").iloc[english_index]
                )
            layer_records.append(record)

    layer_frame = pd.DataFrame(layer_records)
    summary_records = []
    for subset_index, group in layer_frame.groupby("subset"):
        raw_layers = group.loc[
            group["cosine__four_metric_pass"] & group["cosine__breadth_pass"], "layer"
        ].astype(int).tolist()
        density_layers = group.loc[
            group["local_scaled_cosine__four_metric_pass"]
            & group["local_scaled_cosine__breadth_pass"], "layer"
        ].astype(int).tolist()
        status = classify_model_status(raw_layers, raw_layers, density_layers, min_run)
        summary_records.append({
            "subset": int(subset_index),
            "semantic_ids": ",".join(semantic_ids[index] for index in subsets[subset_index]),
            "n_semantic_ids": subset_size,
            "status": status["status"],
            "primary_joint_layers": ",".join(map(str, status["primary_joint_layers"])),
            "primary_joint_longest_run": status["primary_joint_longest_run"],
            "density_joint_layers": ",".join(map(str, status["density_joint_layers"])),
            "density_joint_longest_run": status["density_joint_longest_run"],
            "overlap_layers": ",".join(map(str, status["primary_density_overlap_layers"])),
            "overlap_longest_run": status["primary_density_overlap_longest_run"],
        })
    summary = pd.DataFrame(summary_records)
    full_status = json.loads((hubness / "paper_model_status.json").read_text(encoding="utf-8"))["status"]
    agreement = float((summary.status == full_status).mean())
    robustness_status = "REPLICATED" if agreement >= 0.8 else "SENSITIVE"
    layer_frame.to_csv(hubness / "sample_robustness_layers.csv", index=False, encoding="utf-8")
    summary.to_csv(hubness / "sample_robustness_summary.csv", index=False, encoding="utf-8")
    write_manifest(hubness / "sample_robustness_manifest.json", {
        "source": "saved per-semantic hubness arrays",
        "subsets": n_subsets,
        "subset_size": subset_size,
        "sampling_without_replacement": True,
        "full_sample_status": full_status,
        "status_agreement_rate": agreement,
        "sample_robustness_status": robustness_status,
        "limitation": "Internal subset stability does not replace a random/full-dev extraction.",
        "output_files": ["sample_robustness_layers.csv", "sample_robustness_summary.csv"],
    })
    validation = {
        "module": "09_sample_robustness",
        "status": "PASS" if robustness_status == "REPLICATED" else "WARN",
        "sample_robustness_status": robustness_status,
        "status_agreement_rate": agreement,
        "limitation": "The source pool remains the currently extracted semantic IDs.",
    }
    (paths.validation / "09_sample_robustness.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Sample robustness={robustness_status}; agreement={agreement:.1%}")


if __name__ == "__main__":
    main()

