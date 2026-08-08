"""Build an independent LID calibration set from held-out parallel sentences.

The behavior evaluation references are frozen and must not be reused to choose
the LID threshold. This script instead assembles reference rows for semantic
IDs that were excluded from behavior_v1 (the union of the two seed selections
in configs/behavior_exclusions_seed1_seed2.json) from one or more
parallel_samples.jsonl files.

Output rows contain only the fields the LID diagnostic needs:

- semantic_id
- target_lang
- reference_text

The script never touches experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavior_common import behavior_settings
from behavior_common import sha256_file, stable_json_sha256
from common import load_config, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Build an independent reference-text LID calibration JSONL."
    )
    parser.add_argument("--config", required=True, help="behavior_v1 model config")
    parser.add_argument(
        "--parallel-samples",
        required=True,
        action="append",
        help="parallel_samples.jsonl path (repeatable to merge seed files).",
    )
    parser.add_argument(
        "--exclusions",
        default=str(REPO_ROOT / "configs" / "behavior_exclusions_seed1_seed2.json"),
        help="Frozen exclusion manifest path.",
    )
    parser.add_argument("--output", required=True, help="Calibration JSONL output path.")
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Audit manifest path; defaults to <output>.manifest.json.",
    )
    parser.add_argument(
        "--formal-data-manifest",
        default=None,
        help="Formal behavior dataset_manifest.json; defaults to output_dir/data.",
    )
    parser.add_argument(
        "--min-complete-groups", type=int, default=20,
        help="Minimum complete semantic groups before refusing to proceed.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    settings = behavior_settings(cfg)
    required_languages = ["en", *settings["evaluation_languages"]]

    formal_manifest_path = Path(
        args.formal_data_manifest
        or (Path(cfg["output_dir"]) / "data" / "dataset_manifest.json")
    ).resolve()
    if not formal_manifest_path.exists():
        raise FileNotFoundError(
            f"formal behavior dataset manifest does not exist: {formal_manifest_path}"
        )
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    raw_formal_ids = formal_manifest.get("selected_semantic_indices")
    if not isinstance(raw_formal_ids, list) or not raw_formal_ids:
        raise ValueError(
            "formal behavior dataset manifest has no selected_semantic_indices"
        )
    formal_ids = {int(value) for value in raw_formal_ids}

    exclusion_path = Path(args.exclusions)
    if not exclusion_path.exists():
        raise FileNotFoundError(f"exclusion manifest does not exist: {exclusion_path}")
    manifest = json.loads(exclusion_path.read_text(encoding="utf-8"))
    raw_indices = manifest.get("selected_semantic_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError(f"exclusion manifest has no selected_semantic_indices: {exclusion_path}")
    excluded_ids = {int(value) for value in raw_indices}
    if int(manifest.get("union_count", -1)) != len(excluded_ids):
        raise ValueError("exclusion manifest union_count does not match its unique IDs")
    overlap = sorted(excluded_ids & formal_ids)
    if overlap:
        raise ValueError(
            f"calibration and formal behavior semantic IDs overlap: {overlap[:5]}"
        )

    texts = {}
    source_files = []
    for value in args.parallel_samples:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"parallel samples file does not exist: {path}")
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"parallel samples file is empty: {path}")
        for row in rows:
            missing = {"id", "lang", "text"} - set(row)
            if missing:
                raise ValueError(
                    f"parallel sample row is missing fields {sorted(missing)} in {path}: {row}"
                )
            key = (int(row["id"]), str(row["lang"]))
            text = str(row["text"])
            if key in texts and texts[key] != text:
                raise ValueError(
                    f"conflicting parallel text for semantic_id={key[0]} lang={key[1]}: {path}"
                )
            texts[key] = text
        source_files.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(rows),
        })

    available_ids = {key[0] for key in texts}
    excluded_available = sorted(excluded_ids & available_ids)
    missing_ids = sorted(excluded_ids - available_ids)
    complete_ids = []
    incomplete_ids = []
    for semantic_id in excluded_available:
        missing_languages = sorted(
            language for language in required_languages
            if (semantic_id, language) not in texts
        )
        if missing_languages:
            incomplete_ids.append({"semantic_id": semantic_id, "missing": missing_languages})
        else:
            complete_ids.append(semantic_id)

    if len(complete_ids) < args.min_complete_groups:
        raise ValueError(
            f"only {len(complete_ids)} complete calibration groups "
            f"(required >= {args.min_complete_groups}); add more parallel sample files"
        )
    if missing_ids or incomplete_ids or set(complete_ids) != excluded_ids:
        raise ValueError(
            "calibration sources must contain every excluded semantic ID with every "
            f"required language: missing_ids={len(missing_ids)}, "
            f"incomplete_ids={len(incomplete_ids)}"
        )

    rows = []
    for semantic_id in sorted(complete_ids):
        for language in required_languages:
            rows.append({
                "semantic_id": str(semantic_id),
                "target_lang": language,
                "reference_text": texts[(semantic_id, language)],
            })

    output_path = Path(args.output)
    write_jsonl(output_path, rows)
    manifest_path = Path(args.manifest_output or f"{output_path}.manifest.json")
    calibration_manifest = {
        "protocol_version": "behavior_v1_lid_calibration_v1",
        "config_path": str(Path(args.config).resolve()),
        "task_file": str(output_path.resolve()),
        "task_file_sha256": sha256_file(output_path),
        "rows": len(rows),
        "languages": required_languages,
        "rows_per_language": {
            language: sum(row["target_lang"] == language for row in rows)
            for language in required_languages
        },
        "calibration_semantic_ids": sorted(complete_ids),
        "calibration_semantic_ids_sha256": stable_json_sha256(sorted(complete_ids)),
        "calibration_semantic_id_count": len(complete_ids),
        "exclusion_manifest": str(exclusion_path.resolve()),
        "exclusion_manifest_sha256": sha256_file(exclusion_path),
        "formal_data_manifest": str(formal_manifest_path),
        "formal_data_manifest_sha256": sha256_file(formal_manifest_path),
        "formal_selected_semantic_ids_sha256": stable_json_sha256(sorted(formal_ids)),
        "formal_selected_semantic_id_count": len(formal_ids),
        "calibration_formal_overlap_count": 0,
        "source_files": source_files,
    }
    write_json(manifest_path, calibration_manifest)

    print(f"Calibration LID tasks: {output_path}")
    print(f"Calibration manifest: {manifest_path}")
    print(f"Excluded IDs: {len(excluded_ids)} | available in samples: {len(excluded_available)}")
    print(f"Complete groups: {len(complete_ids)} | rows: {len(rows)}")
    if missing_ids:
        print(f"Excluded IDs absent from samples: {len(missing_ids)} "
              f"(first {min(5, len(missing_ids))}: {missing_ids[:5]})")
    if incomplete_ids:
        print(f"Incomplete groups: {len(incomplete_ids)}")
        for item in incomplete_ids[:5]:
            print(f"  id={item['semantic_id']} missing={item['missing']}")
    print("Languages:", required_languages)
    print("Sources:")
    for source in source_files:
        print(f"  {source['path']} sha256={source['sha256']}")


if __name__ == "__main__":
    main()
