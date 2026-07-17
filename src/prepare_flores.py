import argparse
import json
import shutil
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common import ensure_dirs, load_config, read_jsonl, set_seed, write_jsonl


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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
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
        for short_lang, flores_lang in tqdm(languages.items(), desc="Loading FLORES languages"):
            ds = load_dataset("facebook/flores", flores_lang, split=split)
            limit = min(sample_size, len(ds))
            for idx in range(limit):
                row = ds[idx]
                rows.append(
                    {
                        "id": f"{idx:05d}",
                        "lang": short_lang,
                        "flores_lang": flores_lang,
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
        "split": split,
        "output": str(out_path),
        "known_suffix_policy": suffix_policy,
        "known_suffix_replacements": suffix_replacements,
        "source_rows_matching_known_suffix": len(cleaning_audit),
        "suffix_cleaning_audit": cleaning_audit,
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
