import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dirs, load_config, read_jsonl, representation_file_map


STATUS_ORDER = {"PASS": 0, "WARN": 1, "NOT_RUN": 1, "FAIL": 2}


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def max_consecutive(values):
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def bool_series(series):
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])
    return series.astype(bool)


def result(name, status, method, evidence, interpretation, required_actions=None):
    return {
        "name": name,
        "status": status,
        "validation_method": method,
        "evidence": evidence,
        "interpretation": interpretation,
        "required_actions": required_actions or [],
    }


def write_report(validation_dir, number, slug, payload):
    stem = f"{number:02d}_{slug}"
    payload = native(payload)
    (validation_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# {payload['name']}",
        "",
        f"**Status:** {payload['status']}",
        "",
        "## Validation method",
        "",
        payload["validation_method"],
        "",
        "## Evidence",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["evidence"]] or ["- No evidence available."])
    lines.extend(["", "## Interpretation", "", payload["interpretation"]])
    if payload["required_actions"]:
        lines.extend(["", "## Required actions", ""])
        lines.extend([f"- {item}" for item in payload["required_actions"]])
    (validation_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stem


def validate_dataset(paths, cfg):
    path = Path(paths["data"]) / "parallel_samples.jsonl"
    manifest_path = Path(paths["data"]) / "dataset_manifest.json"
    if not path.exists():
        return result(
            "Dataset completeness and parallel alignment",
            "FAIL",
            "Check row counts, language balance, unique (semantic ID, language) keys, and complete parallel groups.",
            [f"Missing {path}"],
            "No dataset can be validated.",
            ["Run prepare_flores.py."],
        )
    rows = pd.DataFrame(read_jsonl(path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    expected_languages = sorted(cfg["dataset"]["languages"])
    expected_per_language = int(cfg["dataset"]["sample_size_per_language"])
    counts = rows.groupby("lang").size().to_dict()
    duplicate_keys = int(rows.duplicated(["id", "lang"]).sum())
    duplicate_texts = int(rows.duplicated(["lang", "text"]).sum())
    group_sizes = rows.groupby("id").lang.nunique()
    complete_groups = int((group_sizes == len(expected_languages)).sum())
    expected_rows = expected_per_language * len(expected_languages)
    passed = (
        len(rows) == expected_rows and
        all(counts.get(language) == expected_per_language for language in expected_languages) and
        duplicate_keys == 0 and
        duplicate_texts == 0 and
        complete_groups == expected_per_language and
        manifest is not None and
        manifest.get("source") == cfg["dataset"].get("source") and
        manifest.get("split") == cfg["dataset"].get("split") and
        int(manifest.get("known_suspicious_suffix_count", 0)) == 0
    )
    return result(
        "Dataset completeness and parallel alignment",
        "PASS" if passed else "FAIL",
        "Validate the configured FLORES sample at semantic-ID × language grain before any vector analysis.",
        [
            f"Rows: {len(rows)}/{expected_rows}",
            f"Language counts: {counts}",
            f"Complete semantic groups: {complete_groups}/{expected_per_language}",
            f"Duplicate keys: {duplicate_keys}; within-language duplicate texts: {duplicate_texts}",
            f"Dataset manifest: {'present' if manifest is not None else 'missing'}",
            f"Manifest source/split: {None if manifest is None else manifest.get('source')}/"
            f"{None if manifest is None else manifest.get('split')}",
            f"Known suspicious suffix count: {None if manifest is None else manifest.get('known_suspicious_suffix_count')}",
            f"Source rows cleaned by configured suffix policy: "
            f"{None if manifest is None else manifest.get('source_rows_matching_known_suffix')}",
            f"Suffix policy: {None if manifest is None else manifest.get('known_suffix_policy')}",
        ],
        "The dataset is structurally suitable for parallel-sentence comparisons." if passed else
        "Dataset structure can bias or invalidate downstream comparisons.",
        [] if passed else ["Run prepare_flores.py from the same config and do not reuse unmanifested toy data."],
    )


def validate_sentence_tokens(paths):
    metadata_path = Path(paths["hidden"]) / "metadata.csv"
    index_path = Path(paths["hidden"]) / "hidden_state_sentence_index.jsonl"
    if not metadata_path.exists():
        return result(
            "Sentence-to-hidden-state traceability and terminal-token audit",
            "FAIL",
            "Map every hidden-state row to its sentence and inspect language-specific terminal characters and token IDs.",
            [f"Missing {metadata_path}"],
            "Hidden states cannot be traced back to source sentences.",
        ), None, None
    metadata = pd.read_csv(metadata_path)
    required = {
        "row_idx", "id", "lang", "text", "terminal_char", "last_token_id",
        "terminal_unicode_category", "ends_with_ascii_period", "last_token_decoded",
        "last_token_is_content", "last_content_token_id", "shared_sentinel_token_id",
    }
    missing = sorted(required - set(metadata.columns))
    distribution = None
    if not missing:
        distribution = (
            metadata.groupby(["lang", "terminal_char"], dropna=False).size()
            .rename("count").reset_index()
        )
        distribution["share_within_language"] = distribution.groupby("lang")["count"].transform(
            lambda values: values / values.sum()
        )
    index_rows = read_jsonl(index_path) if index_path.exists() else []
    traceable = len(index_rows) == len(metadata) and not missing
    period_shares = {}
    punctuation_last_shares = {}
    if not missing:
        for language, group in metadata.groupby("lang"):
            period_shares[str(language)] = float(bool_series(group.ends_with_ascii_period).mean())
            punctuation_last_shares[str(language)] = float((~bool_series(group.last_token_is_content)).mean())
    sentinel_count = (
        int(metadata.shared_sentinel_token_id.dropna().astype(str).nunique()) if not missing else 0
    )
    suspicious = None
    if not missing:
        suspicious_mask = (
            metadata.terminal_unicode_category.astype(str).str[:1].isin(["L", "N"]) |
            metadata.text.astype(str).str.contains(r"\.x\s*$", regex=True)
        )
        suspicious = metadata.loc[suspicious_mask, [
            "row_idx", "id", "lang", "text", "terminal_char", "last_token_decoded"
        ]]
    suspicious_count = 0 if suspicious is None else len(suspicious)
    if not traceable or sentinel_count != 1:
        status = "FAIL"
    elif suspicious_count:
        status = "WARN"
    else:
        status = "PASS"
    interpretation = (
        "Every vector row is auditable. A high non-content last-token share is not an extraction error; "
        "it is evidence that raw last-token results must be compared with last-content-token and shared-sentinel controls."
        if traceable else
        "The vector-to-sentence mapping or token audit is incomplete."
    )
    return result(
        "Sentence-to-hidden-state traceability and terminal-token audit",
        status,
        "For every vector row, save the full sentence, token sequence, original final token, final content token, "
        "and a language-invariant shared sentinel. Compare ASCII-period and punctuation-terminal rates by language.",
        [
            f"Traceable rows: {len(index_rows)}/{len(metadata)}",
            f"Missing audit columns: {missing}",
            f"ASCII period share by language: {period_shares}",
            f"Non-content original-last-token share by language: {punctuation_last_shares}",
            f"Distinct shared sentinel token IDs: {sentinel_count}",
            f"Suspicious alphanumeric or '.x' terminal rows: {suspicious_count}",
        ],
        interpretation,
        (
            [] if status == "PASS" else
            ["Inspect 02_suspicious_terminal_rows.csv and repair confirmed source artifacts before rerunning."]
            if status == "WARN" else
            ["Rerun extract_hidden.py with the updated extractor."]
        ),
    ), distribution, suspicious


def validate_hidden_integrity(paths, cfg):
    metadata_path = Path(paths["hidden"]) / "metadata.csv"
    if not metadata_path.exists():
        return result("Hidden-state file integrity", "FAIL", "Check vector shapes and finite values.",
                      ["metadata.csv is missing"], "No hidden-state audit is possible.")
    metadata = pd.read_csv(metadata_path)
    files = representation_file_map()
    requested = cfg["metrics"].get("representations", ["last_token"])
    evidence = []
    failures = []
    reference_shape = None
    for name in requested:
        path = Path(paths["hidden"]) / files[name]
        if not path.exists():
            failures.append(f"Missing {path.name}")
            continue
        values = np.load(path, mmap_mode="r")
        evidence.append(f"{name}: shape={tuple(values.shape)}, dtype={values.dtype}")
        if reference_shape is None:
            reference_shape = values.shape
        if values.shape[0] != len(metadata) or values.shape != reference_shape:
            failures.append(f"{name} shape does not match metadata/reference")
        sample = np.asarray(values[: min(32, len(values))])
        if not np.isfinite(sample).all():
            failures.append(f"{name} sample contains NaN/Inf")
        if sample.shape[1] > 1 and np.allclose(sample[:, 0], sample[:, 1]):
            failures.append(f"{name} adjacent layers appear duplicated")
    truncated = int(bool_series(metadata.was_truncated).sum())
    if truncated:
        failures.append(f"{truncated} inputs were truncated")
    evidence.append(f"Metadata rows={len(metadata)}; truncated={truncated}")
    return result(
        "Hidden-state file integrity",
        "PASS" if not failures else "FAIL",
        "Check each configured representation for row/layer/dimension agreement, finite sampled values, "
        "non-duplicated adjacent layers, and zero truncation.",
        evidence + failures,
        "Saved representations are structurally valid." if not failures else
        "One or more representation files are unsafe for analysis.",
        [] if not failures else ["Fix extraction and rerun before interpreting metrics."],
    )


def validate_semantics(paths, cfg):
    metrics = Path(paths["metrics"])
    required = [metrics / "alignment_gain.csv", metrics / "semantic_retrieval.csv"]
    if not all(path.exists() for path in required):
        return result("Cross-lingual semantic validity", "FAIL",
                      "Require AlignmentGain and retrieval CIs.", ["Required metric files are missing."],
                      "Semantic validity is not testable.")
    primary = cfg["metrics"].get("primary_representation", "last_token")
    min_run = int(cfg["metrics"].get("min_consecutive_layers", 3))
    alignment = pd.read_csv(required[0])
    retrieval = pd.read_csv(required[1])
    pair_runs = {}
    for pair, group in alignment[alignment.representation == primary].groupby(["lang_a", "lang_b"]):
        pair_runs["-".join(pair)] = max_consecutive((group.sort_values("layer").ci_lower > 0).tolist())
    direction_support = {}
    for direction, group in retrieval[retrieval.representation == primary].groupby(["query_lang", "target_lang"]):
        direction_support["->".join(direction)] = bool(
            (group.recall1_ci_lower > group.random_recall_at_1).any()
        )
    passed = all(value >= min_run for value in pair_runs.values()) and all(direction_support.values())
    return result(
        "Cross-lingual semantic validity",
        "PASS" if passed else "WARN",
        "Require multi-shuffle AlignmentGain CI > 0 for a continuous layer run and Recall@1 CI above random "
        "for every language direction.",
        [f"Alignment positive-CI longest runs: {pair_runs}", f"Retrieval directions above random CI: {direction_support}"],
        "The primary representation contains auditable sentence semantics." if passed else
        "Semantic evidence is partial across language pairs or directions.",
    )


def validate_specificity(paths, cfg):
    path = Path(paths["metrics"]) / "anchor_specificity_contrasts.csv"
    if not path.exists():
        return result("English specificity", "FAIL", "Directly test English minus every pseudo-anchor.",
                      [f"Missing {path.name}"], "English specificity is not auditable.")
    frame = pd.read_csv(path)
    primary = cfg["metrics"].get("primary_representation", "last_token")
    min_run = int(cfg["metrics"].get("min_consecutive_layers", 3))
    primary_frame = frame[frame.representation == primary]
    layer_support = primary_frame.groupby("layer").ci_lower.min().sort_index() > 0
    longest = max_consecutive(layer_support.tolist())
    passed = longest >= min_run
    return result(
        "English specificity",
        "PASS" if passed else "WARN",
        "At each layer, bootstrap semantic-ID-clustered differences between English specificity and each "
        "rotated pseudo-anchor; require all lower CIs > 0 for a continuous run.",
        [f"Supported layers: {int(layer_support.sum())}/{len(layer_support)}", f"Longest continuous run: {longest}"],
        "English is specifically closer than every pseudo-anchor over a stable layer segment." if passed else
        "English is not consistently above every pseudo-anchor.",
    )


def validate_hubness(paths, cfg):
    summary_path = Path(paths["metrics"]) / "pooling_robustness_summary.csv"
    occurrence_path = Path(paths["metrics"]) / "hubness_occurrence.csv"
    if not summary_path.exists() or not occurrence_path.exists():
        return result("English attraction and k-occurrence hubness", "FAIL",
                      "Separate neighbor-language attraction from point-level k-occurrence hubness.",
                      ["Required attraction/hubness files are missing."], "Hub evidence is not auditable.")
    primary = cfg["metrics"].get("primary_representation", "last_token")
    summary = pd.read_csv(summary_path)
    row = summary[summary.representation == primary].iloc[0]
    occurrence = pd.read_csv(occurrence_path)
    en_occurrence = occurrence[
        (occurrence.representation == primary) & (occurrence.candidate_lang == cfg["metrics"].get("english_language", "en"))
    ]
    supported_count = int(row.english_hub_supported_language_count)
    status = "PASS" if supported_count >= 2 else ("WARN" if supported_count == 1 else "WARN")
    occurrence_range = (
        f"{float(en_occurrence.mean_k_occurrence.min()):.3f}..{float(en_occurrence.mean_k_occurrence.max()):.3f}"
        if not en_occurrence.empty else "unavailable"
    )
    supported_names = "none" if pd.isna(row.english_hub_supported_languages) else str(row.english_hub_supported_languages)
    return result(
        "English attraction and k-occurrence hubness",
        status,
        "For each non-English language, require both English-neighbor attraction CI above the balanced baseline "
        "and paired directional-asymmetry CI above zero for a continuous run. Separately save point k-occurrence.",
        [
            f"Supported source languages: {supported_names}",
            f"Supported language count: {supported_count}",
            f"English mean k-occurrence range: {occurrence_range}",
        ],
        "Evidence supports a broad English hub." if supported_count >= 2 else
        "English attraction is language-pair-specific and must not be generalized as a universal hub.",
    )


def validate_reseparation(paths, cfg):
    path = Path(paths["metrics"]) / "re_separation_summary.csv"
    if not path.exists():
        return result("Late language re-separation", "FAIL", "Compare pre-specified late and mid layer windows.",
                      [f"Missing {path.name}"], "Re-separation is not auditable.")
    primary = cfg["metrics"].get("primary_representation", "last_token")
    frame = pd.read_csv(path)
    primary_frame = frame[frame.representation == primary]
    positive = primary_frame[primary_frame.ci_lower > 0].lang.astype(str).tolist()
    status = "PASS" if len(positive) == len(primary_frame) else ("WARN" if positive else "WARN")
    values = {
        str(row.lang): {
            "mid": float(row.mid_purity), "late": float(row.late_purity),
            "late_minus_mid": float(row.re_separation_strength), "ci_lower": float(row.ci_lower),
        }
        for row in primary_frame.itertuples()
    }
    return result(
        "Late language re-separation",
        status,
        "Exclude layer 0 and bootstrap the pre-specified late-window minus mid-window purity contrast by sentence; "
        "inspect every language rather than only the mean.",
        [f"Positive-CI languages: {positive}", f"Per-language contrasts: {values}"],
        "All languages show late re-separation." if status == "PASS" else
        "Re-separation is partial or representation-dependent.",
    )


def validate_representation_robustness(paths):
    verdict_path = Path(paths["metrics"]) / "pooling_robustness_verdict.csv"
    summary_path = Path(paths["metrics"]) / "pooling_robustness_summary.csv"
    if not verdict_path.exists() or not summary_path.exists():
        return result("Representation robustness", "NOT_RUN",
                      "Compare raw final token, final content token, shared sentinel, and pooling controls.",
                      ["Representation robustness outputs are missing."], "Robustness has not been evaluated.")
    verdict = pd.read_csv(verdict_path).iloc[0]
    summary = pd.read_csv(summary_path)
    geometry = dict(zip(summary.representation.astype(str), summary.geometry_warning_layers.astype(int)))
    return result(
        "Representation robustness",
        "PASS" if verdict.status == "CONSISTENT" else "WARN",
        "Require claim-level agreement, peak-layer proximity, and no low-variance geometry warning across all "
        "configured representations. Do not accept agreement based only on effect signs.",
        [f"Verdict: {verdict.status}", f"Geometry-warning layers: {geometry}"],
        "Findings are representation-robust." if verdict.status == "CONSISTENT" else
        "At least one claim is sensitive to terminal-token choice or pooling geometry.",
    )


def validate_k_robustness(paths):
    verdict_path = Path(paths["metrics"]) / "k_robustness_verdict.csv"
    if not verdict_path.exists():
        return result("kNN k-value robustness", "NOT_RUN", "Compare k=5,10,20 claim support and peak layers.",
                      [f"Missing {verdict_path.name}"], "k robustness has not been run.")
    verdict = pd.read_csv(verdict_path).iloc[0]
    return result(
        "kNN k-value robustness",
        "PASS" if verdict.status == "CONSISTENT" else "WARN",
        "Require the same claim support across k and peak-layer movement below 25% of model depth.",
        [f"k values: {verdict.k_values}", f"Verdict: {verdict.status}"],
        "kNN conclusions are stable." if verdict.status == "CONSISTENT" else
        "At least one kNN conclusion or peak location is k-sensitive.",
    )


def main():
    parser = argparse.ArgumentParser(description="Save each validation idea and its evidence as a separate report.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    validation_dir = Path(paths["validation"])

    reports = []
    reports.append(validate_dataset(paths, cfg))
    token_report, terminal_distribution, suspicious_terminals = validate_sentence_tokens(paths)
    reports.append(token_report)
    reports.append(validate_hidden_integrity(paths, cfg))
    reports.append(validate_semantics(paths, cfg))
    reports.append(validate_specificity(paths, cfg))
    reports.append(validate_hubness(paths, cfg))
    reports.append(validate_reseparation(paths, cfg))
    reports.append(validate_representation_robustness(paths))
    reports.append(validate_k_robustness(paths))

    slugs = [
        "dataset", "sentence_token_audit", "hidden_integrity", "semantic_alignment",
        "english_specificity", "english_attraction_hubness", "language_reseparation",
        "representation_robustness", "k_robustness",
    ]
    stems = [
        write_report(validation_dir, index, slug, report)
        for index, (slug, report) in enumerate(zip(slugs, reports), start=1)
    ]
    if terminal_distribution is not None:
        terminal_distribution.to_csv(
            validation_dir / "02_terminal_character_distribution.csv", index=False, encoding="utf-8"
        )
    if suspicious_terminals is not None and not suspicious_terminals.empty:
        suspicious_terminals.to_csv(
            validation_dir / "02_suspicious_terminal_rows.csv", index=False, encoding="utf-8"
        )

    overall = max(reports, key=lambda item: STATUS_ORDER[item["status"]])["status"]
    if overall == "NOT_RUN":
        overall = "WARN"
    summary = {
        "overall_status": overall,
        "reports": [
            {"file_stem": stem, "name": report["name"], "status": report["status"]}
            for stem, report in zip(stems, reports)
        ],
        "decision_rule": (
            "Expand the experiment only when data, hidden integrity, semantics, representation robustness, "
            "and k robustness pass; treat language-pair-specific research findings as partial rather than failures."
        ),
    }
    (validation_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_lines = ["# Validation summary", "", f"**Overall status:** {overall}", ""]
    summary_lines.extend([f"- {item['status']}: {item['name']} (`{item['file_stem']}.md`)" for item in summary["reports"]])
    summary_lines.extend(["", "## Decision rule", "", summary["decision_rule"], ""])
    (validation_dir / "validation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n=== Validation outputs ===")
    for item in summary["reports"]:
        print(f"{item['status']:>7}  {item['file_stem']}.md")
    print(f"Overall: {overall}")
    print(f"Read {validation_dir / 'validation_summary.md'}")


if __name__ == "__main__":
    main()
