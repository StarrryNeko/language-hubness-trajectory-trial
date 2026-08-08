"""Build balanced non-English to non-Latin behavior_v2 translation tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from behavior_v2.common import (
    ensure_paths, settings, stable_sha256, validate_tasks, write_manifest,
)
from common import load_config, read_jsonl, write_jsonl


def parallel_groups(rows):
    groups = {}
    for row in rows:
        semantic_id = str(row["id"])
        language = str(row["lang"])
        values = groups.setdefault(semantic_id, {})
        if language in values:
            raise ValueError(f"duplicate V2 parallel row: {semantic_id}/{language}")
        values[language] = str(row["text"])
    return groups


def role_order(semantic_ids, seed):
    def key(value):
        return hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).hexdigest()
    return sorted(map(str, semantic_ids), key=key)


def render_prompt(protocol, groups, demo_ids, semantic_id, source, target):
    names = protocol["language_names"]
    if source not in names or target not in names:
        raise ValueError(f"language display name is missing: {source}/{target}")
    demonstrations = "\n\n".join(
        f"{names[source]}: {groups[demo][source]}\n"
        f"{names[target]}: {groups[demo][target]}"
        for demo in demo_ids
    )
    return protocol["prompt_template"].format(
        demonstrations=demonstrations,
        source_name=names[source],
        target_name=names[target],
        source_text=groups[semantic_id][source],
    )


def build_tasks(cfg, rows):
    protocol = settings(cfg)
    groups = parallel_groups(rows)
    required = set(protocol["languages"])
    incomplete = [key for key, value in groups.items() if set(value) != required]
    if incomplete:
        raise ValueError(f"V2 parallel groups are incomplete: {incomplete[:5]}")
    ordered = role_order(groups, protocol["role_assignment_seed"])
    demo_count = protocol["demonstration_semantic_ids"]
    evaluation_count = protocol["evaluation_semantic_ids"]
    if len(ordered) != demo_count + evaluation_count:
        raise ValueError("V2 prepared data does not match frozen demo/evaluation counts")
    demo_ids = ordered[:demo_count]
    evaluation_ids = ordered[demo_count:]
    template_hash = stable_sha256({
        "template": protocol["prompt_template"],
        "language_names": protocol["language_names"],
        "demonstrations": demo_ids,
    })
    tasks = []
    sources = protocol["source_languages"]
    targets = protocol["target_languages"]
    for semantic_position, semantic_id in enumerate(evaluation_ids):
        for target_position, target in enumerate(targets):
            candidates = [language for language in sources if language != target]
            source = candidates[(semantic_position + target_position) % len(candidates)]
            task_id = f"{semantic_id}__non_english_to_non_latin__{source}__{target}"
            tasks.append({
                "task_id": task_id,
                "semantic_id": semantic_id,
                "condition": "non_english_to_non_latin",
                "source_lang": source,
                "target_lang": target,
                "source_script": protocol["language_metadata"][source]["script"],
                "target_script": protocol["language_metadata"][target]["script"],
                "source_text": groups[semantic_id][source],
                "reference_text": groups[semantic_id][target],
                "prompt": render_prompt(
                    protocol, groups, demo_ids, semantic_id, source, target
                ),
                "prompt_template_sha256": template_hash,
                "demonstration_semantic_ids": demo_ids,
            })
    summary = validate_tasks(tasks, protocol)
    return tasks, demo_ids, evaluation_ids, summary


def main():
    parser = argparse.ArgumentParser(description="Prepare frozen behavior_v2 tasks")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    paths = ensure_paths(cfg)
    source = Path(cfg["output_dir"]) / "data" / "parallel_samples.jsonl"
    source_manifest_path = source.parent / "dataset_manifest.json"
    if not source.exists() or not source_manifest_path.exists():
        raise FileNotFoundError("prepare V2 devtest parallel data before building tasks")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("split") != "devtest":
        raise ValueError("behavior_v2 confirmation data must come from FLORES+ devtest")
    tasks, demos, evaluation_ids, summary = build_tasks(cfg, read_jsonl(source))
    destination = paths.data / "behavior_v2_tasks.jsonl"
    write_jsonl(destination, tasks)
    write_manifest(paths.data / "behavior_v2_task_manifest.json", {
        **summary,
        "config_path": str(config_path),
        "representation_protocol": protocol["representation_protocol"],
        "source_parallel_samples": str(source),
        "source_data_content_sha256": source_manifest.get("data_content_sha256"),
        "source_split": source_manifest.get("split"),
        "source_selected_semantic_indices_sha256": source_manifest.get(
            "selected_semantic_indices_sha256"
        ),
        "role_assignment_seed": protocol["role_assignment_seed"],
        "demonstration_semantic_ids": demos,
        "demonstration_semantic_id_sha256": stable_sha256(demos),
        "evaluation_semantic_id_sha256": stable_sha256(evaluation_ids),
        "conditions_per_semantic_id": len(protocol["target_languages"]),
        "target_languages": protocol["target_languages"],
        "source_languages": protocol["source_languages"],
        "activation_intervention": False,
    })
    print(f"Saved {len(tasks)} behavior_v2 tasks to {destination}")


if __name__ == "__main__":
    main()
