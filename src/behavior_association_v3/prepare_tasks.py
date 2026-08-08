"""Freeze disjoint demonstration, calibration, and formal V3 task sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from behavior_association_v3.common import paths, settings, stable_sha256, task_path, write_manifest
from behavior_v2.prepare_tasks import parallel_groups
from common import load_config, read_jsonl, write_jsonl


def ordered_ids(values, seed):
    return sorted(map(str, values), key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())


def render_prompt(protocol, groups, demo_ids, semantic_id, source, target):
    names = protocol["language_names"]
    demos = "\n\n".join(
        f"{names[source]}: {groups[item][source]}\n{names[target]}: {groups[item][target]}"
        for item in demo_ids
    )
    return protocol["prompt_template"].format(
        demonstrations=demos, source_name=names[source], target_name=names[target],
        source_text=groups[semantic_id][source],
    )


def build(cfg, rows):
    protocol = settings(cfg)
    groups = parallel_groups(rows)
    required = set(protocol["languages"])
    if any(set(group) != required for group in groups.values()):
        raise ValueError("parallel groups are incomplete")
    ordered = ordered_ids(groups, protocol["role_seed"])
    d = protocol["counts"]["demonstration"]
    c = protocol["counts"]["calibration"]
    demo_ids = ordered[:d]
    split_ids = {"calibration": ordered[d:d + c], "formal": ordered[d + c:]}
    result = {}
    for split, semantic_ids in split_ids.items():
        tasks = []
        for semantic_position, semantic_id in enumerate(semantic_ids):
            for target_position, target in enumerate(protocol["target_languages"]):
                candidates = [lang for lang in protocol["source_languages"] if lang != target]
                source = candidates[(semantic_position + target_position) % len(candidates)]
                prompt = render_prompt(protocol, groups, demo_ids, semantic_id, source, target)
                tasks.append({
                    "task_id": f"v3__{split}__{semantic_id}__{source}__{target}",
                    "split": split, "semantic_id": semantic_id,
                    "condition": "non_english_to_non_latin",
                    "source_lang": source, "target_lang": target,
                    "source_text": groups[semantic_id][source],
                    "reference_text": groups[semantic_id][target],
                    "prompt": prompt,
                    "prompt_template_sha256": stable_sha256({
                        "template": protocol["prompt_template"], "demo_ids": demo_ids,
                        "language_names": protocol["language_names"],
                    }),
                    "demonstration_semantic_ids": demo_ids,
                })
        expected_rows = len(semantic_ids) * len(protocol["target_languages"])
        if len(tasks) != expected_rows or len({row["task_id"] for row in tasks}) != expected_rows:
            raise ValueError(f"{split} V3 task cardinality failed")
        for target in protocol["target_languages"]:
            target_rows = [row for row in tasks if row["target_lang"] == target]
            counts = [
                sum(row["source_lang"] == source for row in target_rows)
                for source in protocol["source_languages"] if source != target
            ]
            if max(counts) - min(counts) > 1:
                raise ValueError(f"{split} source rotation is unbalanced for {target}")
        result[split] = tasks
    return demo_ids, split_ids, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    source = Path(cfg["output_dir"]) / "data" / "parallel_samples.jsonl"
    source_manifest = source.parent / "dataset_manifest.json"
    if not source.exists() or not source_manifest.exists():
        raise FileNotFoundError("V3 reuses prepared 804-ID parallel data")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("split") != "devtest":
        raise ValueError("V3 requires FLORES+ devtest")
    demos, split_ids, tasks = build(cfg, read_jsonl(source))
    for split, rows in tasks.items():
        write_jsonl(task_path(cfg, split), rows)
    write_manifest(paths(cfg).data / "task_manifest.json", {
        "config_path": str(config_path),
        "source_data_content_sha256": manifest.get("data_content_sha256"),
        "demonstration_semantic_ids": demos,
        "demonstration_semantic_id_sha256": stable_sha256(demos),
        "calibration_semantic_id_sha256": stable_sha256(split_ids["calibration"]),
        "formal_semantic_id_sha256": stable_sha256(split_ids["formal"]),
        "calibration_formal_overlap": len(set(split_ids["calibration"]) & set(split_ids["formal"])),
        "calibration_task_sha256": stable_sha256(tasks["calibration"]),
        "formal_task_sha256": stable_sha256(tasks["formal"]),
        "counts": settings(cfg)["counts"],
    })
    print("Saved V3 calibration and formal tasks")


if __name__ == "__main__":
    main()
