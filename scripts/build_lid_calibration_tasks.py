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
from common import load_config, read_jsonl, write_jsonl


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
        "--min-complete-groups", type=int, default=20,
        help="Minimum complete semantic groups before refusing to proceed.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    settings = behavior_settings(cfg)
    required_languages = ["en", *settings["evaluation_languages"]]

    exclusion_path = Path(args.exclusions)
    if not exclusion_path.exists():
        raise FileNotFoundError(f"exclusion manifest does not exist: {exclusion_path}")
    manifest = json.loads(exclusion_path.read_text(encoding="utf-8"))
    raw_indices = manifest.get("selected_semantic_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError(f"exclusion manifest has no selected_semantic_indices: {exclusion_path}")
    excluded_ids = {int(value) for value in raw_indices}

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
        source_files.append(str(path.resolve()))

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

    print(f"Calibration LID tasks: {output_path}")
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
        print(f"  {source}")


if __name__ == "__main__":
    main()