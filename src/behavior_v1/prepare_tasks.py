"""Build frozen few-shot translation tasks from held-out parallel sentences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from behavior_v1.common import (
    behavior_settings,
    ensure_behavior_dirs,
    stable_json_sha256,
    validate_tasks,
    write_manifest,
)
from common import load_config, read_jsonl, write_jsonl


def parallel_groups(rows):
    groups = {}
    for row in rows:
        semantic_id = str(row["id"])
        language = str(row["lang"])
        if language in groups.setdefault(semantic_id, {}):
            raise ValueError(f"duplicate parallel row: semantic_id={semantic_id}, language={language}")
        groups[semantic_id][language] = str(row["text"])
    return groups


def render_prompt(settings, groups, demo_ids, semantic_id, source_lang, target_lang):
    names = settings["language_names"]
    missing_names = sorted({source_lang, target_lang} - set(names))
    if missing_names:
        raise ValueError(f"behavior_v1.language_names is missing: {missing_names}")
    values = {
        "source_name": names[source_lang],
        "target_name": names[target_lang],
        "demonstrations": "\n\n".join(
            f"{names[source_lang]}: {groups[demo_id][source_lang]}\n"
            f"{names[target_lang]}: {groups[demo_id][target_lang]}"
            for demo_id in demo_ids
        ),
        "source_text": groups[semantic_id][source_lang],
    }
    return settings["prompt_template"].format(**values)


def build_tasks(cfg, rows):
    settings = behavior_settings(cfg)
    groups = parallel_groups(rows)
    required_languages = ["en", *settings["evaluation_languages"]]
    incomplete = [
        semantic_id for semantic_id, values in groups.items()
        if not set(required_languages).issubset(values)
    ]
    if incomplete:
        raise ValueError(f"held-out data has incomplete behavior language groups: {incomplete[:5]}")
    semantic_ids = sorted(groups, key=lambda value: int(value))
    required_count = settings["demonstration_semantic_ids"] + settings["evaluation_semantic_ids"]
    if len(semantic_ids) != required_count:
        raise ValueError(
            f"behavior dataset must contain exactly {required_count} semantic IDs; got {len(semantic_ids)}"
        )
    demo_ids = semantic_ids[: settings["demonstration_semantic_ids"]]
    evaluation_ids = semantic_ids[settings["demonstration_semantic_ids"] :]
    template_hash = stable_json_sha256({
        "template": settings["prompt_template"],
        "language_names": settings["language_names"],
        "demo_ids": demo_ids,
    })
    languages = settings["evaluation_languages"]
    tasks = []
    for semantic_position, semantic_id in enumerate(evaluation_ids):
        pairs = []
        target_offset = 1 + semantic_position % (len(languages) - 1)
        for index, source_lang in enumerate(languages):
            pairs.append((
                "non_english_to_non_english",
                source_lang,
                languages[(index + target_offset) % len(languages)],
            ))
        pairs.extend(("english_to_non_english", "en", language) for language in languages)
        pairs.extend(("non_english_to_english", language, "en") for language in languages)
        for condition, source_lang, target_lang in pairs:
            task_id = f"{semantic_id}__{condition}__{source_lang}__{target_lang}"
            tasks.append({
                "task_id": task_id,
                "semantic_id": semantic_id,
                "condition": condition,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": groups[semantic_id][source_lang],
                "reference_text": groups[semantic_id][target_lang],
                "prompt": render_prompt(
                    settings, groups, demo_ids, semantic_id, source_lang, target_lang
                ),
                "prompt_template_sha256": template_hash,
                "demonstration_semantic_ids": demo_ids,
            })
    summary = validate_tasks(tasks, settings)
    return tasks, demo_ids, evaluation_ids, summary


def main():
    parser = argparse.ArgumentParser(description="Prepare frozen behavior_v1 tasks")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    paths = ensure_behavior_dirs(cfg)
    source = Path(cfg["output_dir"]) / "data" / "parallel_samples.jsonl"
    data_manifest_path = source.parent / "dataset_manifest.json"
    if not source.exists() or not data_manifest_path.exists():
        raise FileNotFoundError("prepare held-out FLORES data before behavior tasks")
    tasks, demo_ids, evaluation_ids, summary = build_tasks(cfg, read_jsonl(source))
    destination = paths.data / "behavior_tasks.jsonl"
    write_jsonl(destination, tasks)
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    write_manifest(paths.data / "behavior_task_manifest.json", {
        **summary,
        "source_parallel_samples": str(source),
        "source_data_content_sha256": data_manifest.get("data_content_sha256"),
        "source_selected_semantic_indices_sha256": data_manifest.get(
            "selected_semantic_indices_sha256"
        ),
        "excluded_semantic_indices_sha256": data_manifest.get(
            "excluded_semantic_indices_sha256"
        ),
        "demonstration_semantic_ids": demo_ids,
        "evaluation_semantic_id_sha256": stable_json_sha256(evaluation_ids),
        "conditions_per_semantic_id": len(tasks) // len(evaluation_ids),
    })
    print(f"Saved {len(tasks)} behavior tasks to {destination}")


if __name__ == "__main__":
    main()
