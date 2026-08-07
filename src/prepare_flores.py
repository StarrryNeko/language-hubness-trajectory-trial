import argparse
import hashlib
import json
import os
import shutil
import unicodedata
from pathlib import Path

# Keep dataset downloads on the lower-memory HTTP path in AutoDL containers.
# huggingface_hub reads this variable when it is imported by datasets below.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset
from tqdm import tqdm

from common import (
    ensure_dirs,
    load_config,
    read_jsonl,
    select_semantic_indices_excluding,
    set_seed,
    validate_language_inventory,
    write_jsonl,
)


FLORES_PLUS_CONFIG_ALIASES = {
    # FLORES-200 used zho_Hans; FLORES+ identifies Mandarin as cmn_Hans.
    "zho_Hans": "cmn_Hans",
}


def excluded_semantic_indices(selection_cfg, config_directory):
    """Resolve explicit indices and frozen dataset manifests used as exclusions."""
    excluded = {int(value) for value in selection_cfg.get("exclude_indices", [])}
    manifests = []
    for configured in selection_cfg.get("exclude_manifest_paths", []):
        path = Path(configured)
        if not path.is_absolute():
            candidates = [Path.cwd() / path, Path(config_directory) / path]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not path.exists():
            raise FileNotFoundError(f"excluded sample manifest does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("selected_semantic_indices")
        if not isinstance(values, list):
            raise ValueError(f"excluded sample manifest has no selected_semantic_indices list: {path}")
        excluded.update(int(value) for value in values)
        manifests.append({
            "path": str(path.resolve()),
            "selected_count": len(values),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return sorted(excluded), manifests


def pick_sentence(row):
    for key in ["sentence", "text", "translation"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Cannot find sentence field in row keys: {list(row.keys())}")


def apply_known_suffix_policy(rows, validation_cfg):
    policy = validation_cfg.get("known_suffix_policy", "error")
    replacements = validation_cfg.get("known_suffix_replacements", {".x": "."})
    if policy not in {"error", "warn", "replace"}:
        raise ValueError("data_validation.known_suffix_policy must be error, warn, or replace")

    processed = []
    audit = []
    for source_row in rows:
        row = dict(source_row)
        text = str(row["text"])
        for suffix, replacement in replacements.items():
            if text.endswith(suffix):
                cleaned = text[: -len(suffix)] + replacement if suffix else text
                audit.append({
                    "id": row["id"],
                    "lang": row["lang"],
                    "matched_suffix": suffix,
                    "replacement": replacement,
                    "original_text": text,
                    "cleaned_text": cleaned,
                })
                if policy == "replace":
                    row["source_text"] = text
                    row["text"] = cleaned
                    row["data_cleaning"] = f"terminal suffix {suffix!r} replaced with {replacement!r}"
                break
        processed.append(row)
    return processed, audit, policy, replacements


def validate_parallel_rows(rows, expected_languages):
    groups = {}
    duplicate_keys = 0
    seen = set()
    for row in rows:
        key = (str(row["id"]), str(row["lang"]))
        duplicate_keys += int(key in seen)
        seen.add(key)
        groups.setdefault(str(row["id"]), set()).add(str(row["lang"]))
    incomplete = [semantic_id for semantic_id, langs in groups.items() if langs != set(expected_languages)]
    duplicate_texts = len(rows) - len({(str(row["lang"]), row["text"]) for row in rows})
    suspicious_suffix_rows = []
    alphanumeric_terminal_rows = []
    for row in rows:
        text = str(row["text"]).rstrip()
        terminal = text[-1] if text else ""
        if text.endswith(".x"):
            suspicious_suffix_rows.append({"id": row["id"], "lang": row["lang"], "text": row["text"]})
        if terminal and unicodedata.category(terminal)[0] in {"L", "N"}:
            alphanumeric_terminal_rows.append({"id": row["id"], "lang": row["lang"], "text": row["text"]})
    if duplicate_keys or incomplete:
        raise ValueError(
            f"Invalid parallel data: duplicate keys={duplicate_keys}, incomplete semantic groups={len(incomplete)}"
        )
    return {
        "rows": len(rows),
        "semantic_groups": len(groups),
        "languages": list(expected_languages),
        "rows_per_language": {
            lang: sum(str(row["lang"]) == lang for row in rows) for lang in expected_languages
        },
        "duplicate_within_language_texts": duplicate_texts,
        "known_suspicious_suffix_count": len(suspicious_suffix_rows),
        "known_suspicious_suffix_examples": suspicious_suffix_rows[:20],
        "alphanumeric_terminal_count": len(alphanumeric_terminal_rows),
        "alphanumeric_terminal_examples": alphanumeric_terminal_rows[:20],
        "complete_parallel_groups": True,
        "candidate_scope": "same_semantic_id_only",
        "languages_per_semantic_group": len(expected_languages),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_language_inventory(cfg)
    set_seed(cfg.get("seed", 42))
    paths = ensure_dirs(cfg)

    dataset_cfg = cfg["dataset"]
    source = dataset_cfg.get("source", "flores")
    split = dataset_cfg.get("split", "dev")
    sample_size = int(dataset_cfg.get("sample_size_per_language", 200))
    languages = dataset_cfg["languages"]

    out_path = Path(paths["data"]) / "parallel_samples.jsonl"

    if source == "local_jsonl":
        input_path = dataset_cfg.get("local_path")
        if not input_path:
            raise ValueError("dataset.source='local_jsonl' requires dataset.local_path")
        shutil.copyfile(input_path, out_path)
        rows = read_jsonl(str(out_path))
        print(f"Copied local dataset from {input_path} to {out_path}")
    else:
        rows = []
        selected_indices = None
        selection_cfg = dataset_cfg.get("sample_selection", {})
        selection_strategy = selection_cfg.get("strategy", "first_n")
        selection_seed = int(selection_cfg.get("seed", cfg.get("seed", 42)))
        dataset_name = dataset_cfg.get(
            "dataset_name",
            "openlanguagedata/flores_plus" if source == "flores_plus" else "facebook/flores",
        )
        cache_dir = dataset_cfg.get("cache_dir", cfg.get("huggingface_cache_dir"))
        excluded_indices, exclusion_manifests = excluded_semantic_indices(
            selection_cfg, Path(args.config).resolve().parent
        )
        for short_lang, configured_flores_lang in tqdm(languages.items(), desc="Loading FLORES languages"):
            flores_lang = (
                FLORES_PLUS_CONFIG_ALIASES.get(configured_flores_lang, configured_flores_lang)
                if source == "flores_plus" else configured_flores_lang
            )
            load_kwargs = {"split": split}
            if cache_dir:
                load_kwargs["cache_dir"] = cache_dir
            if bool(dataset_cfg.get("use_auth_token", source == "flores_plus")):
                load_kwargs["token"] = True
            ds = load_dataset(dataset_name, flores_lang, **load_kwargs)
            limit = min(sample_size, len(ds))
            if selected_indices is None:
                selected_indices = select_semantic_indices_excluding(
                    len(ds), limit, selection_strategy, selection_seed, excluded_indices
                )
            if len(selected_indices) != limit or max(selected_indices, default=-1) >= len(ds):
                raise ValueError(
                    "FLORES language splits do not support one shared semantic-index sample: "
                    f"language={short_lang}, rows={len(ds)}, requested_indices={len(selected_indices)}"
                )
            for idx in selected_indices:
                row = ds[idx]
                rows.append(
                    {
                        "id": f"{idx:05d}",
                        "lang": short_lang,
                        "flores_lang": flores_lang,
                        "configured_flores_lang": configured_flores_lang,
                        "text": pick_sentence(row),
                    }
                )
    data_validation_cfg = cfg.get("data_validation", {})
    rows, cleaning_audit, suffix_policy, suffix_replacements = apply_known_suffix_policy(
        rows, data_validation_cfg
    )
    write_jsonl(str(out_path), rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    manifest = validate_parallel_rows(rows, list(languages))
    manifest.update({
        "source": source,
        "dataset_name": dataset_cfg.get("dataset_name"),
        "huggingface_cache_dir": dataset_cfg.get("cache_dir", cfg.get("huggingface_cache_dir")),
        "split": split,
        "output": str(out_path),
        "known_suffix_policy": suffix_policy,
        "known_suffix_replacements": suffix_replacements,
        "source_rows_matching_known_suffix": len(cleaning_audit),
        "suffix_cleaning_audit": cleaning_audit,
        "sample_selection_strategy": (
            dataset_cfg.get("sample_selection", {}).get("strategy", "local_file_order")
            if source == "local_jsonl" else selection_strategy
        ),
        "sample_selection_seed": (
            dataset_cfg.get("sample_selection", {}).get("seed")
            if source == "local_jsonl" else selection_seed
        ),
        "selected_semantic_indices": (
            None if source == "local_jsonl" else selected_indices
        ),
        "selected_semantic_indices_sha256": (
            None if source == "local_jsonl" else hashlib.sha256(
                "\n".join(map(str, selected_indices)).encode("utf-8")
            ).hexdigest()
        ),
        "excluded_semantic_indices": (
            [] if source == "local_jsonl" else excluded_indices
        ),
        "excluded_semantic_indices_sha256": (
            None if source == "local_jsonl" else hashlib.sha256(
                "\n".join(map(str, excluded_indices)).encode("utf-8")
            ).hexdigest()
        ),
        "exclusion_manifests": (
            [] if source == "local_jsonl" else exclusion_manifests
        ),
        "data_content_sha256": hashlib.sha256(
            "\n".join(
                json.dumps(
                    {"id": str(row["id"]), "lang": str(row["lang"]), "text": str(row["text"])},
                    sort_keys=True, ensure_ascii=False,
                )
                for row in rows
            ).encode("utf-8")
        ).hexdigest(),
    })
    manifest_path = Path(paths["data"]) / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Dataset validation ===")
    print(f"Complete semantic groups: {manifest['semantic_groups']}")
    print(f"Rows per language: {manifest['rows_per_language']}")
    print(f"Duplicate within-language texts: {manifest['duplicate_within_language_texts']}")
    print(f"Known suspicious suffixes: {manifest['known_suspicious_suffix_count']}")
    print(f"Source rows matching configured suffixes: {len(cleaning_audit)} (policy={suffix_policy})")
    print(f"Alphanumeric terminal rows: {manifest['alphanumeric_terminal_count']}")
    print(f"Saved {manifest_path}")
    if suffix_policy == "error" and cleaning_audit:
        raise ValueError(
            "Known suspicious terminal suffixes were found. Inspect dataset_manifest.json or use the "
            "audited 'replace' policy after confirming the configured replacements."
        )


if __name__ == "__main__":
    main()
