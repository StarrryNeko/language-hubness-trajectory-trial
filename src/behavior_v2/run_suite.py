"""Run the frozen three-model behavior_v2 pipeline without intervention."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from behavior_v2.common import configure_cpu_threads, ensure_paths, settings, sha256_file
from common import load_config
from run_extraction_suite import validate_frozen_suite


def call(script, *arguments):
    command = [sys.executable, str(script), *map(str, arguments)]
    print(f"\n{' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def geometry_reusable(cfg):
    paths = ensure_paths(cfg)
    predictor = paths.geometry / "behavior_v2_geometry_predictors.csv"
    manifest_path = paths.geometry / "geometry_manifest.json"
    task_path = paths.data / "behavior_v2_tasks.jsonl"
    checkpoint_path = Path(cfg["output_dir"]) / "checkpoint_identity.json"
    if not all(path.exists() for path in (predictor, manifest_path, task_path, checkpoint_path)):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        protocol = settings(cfg)
        return (
            manifest.get("protocol_version") == "behavior_v2"
            and manifest.get("representation") == "mean_pool"
            and manifest.get("layers") == protocol["analysis_layers"]
            and manifest.get("primary_layer") == protocol["primary_layer"]
            and manifest.get("checkpoint_sha256") == checkpoint.get("checkpoint_sha256")
            and manifest.get("task_file_sha256") == sha256_file(task_path)
            and manifest.get("predictor_file_sha256") == sha256_file(predictor)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Run behavior_v2 three-model suite")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument(
        "--stage", choices=("all", "prepare", "generate", "audit", "analyze"),
        default="all",
    )
    parser.add_argument("--negative-audit-per-language", type=int, default=60)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    config_paths = [suite_path.parent / value for value in suite["configs"]]
    configs = [load_config(path) for path in config_paths]
    validate_frozen_suite(suite, configs)
    protocols = [settings(cfg) for cfg in configs]
    configure_cpu_threads(protocols[0]["resources"]["cpu_threads"])
    for protocol in protocols[1:]:
        if protocol["resources"] != protocols[0]["resources"]:
            raise ValueError("all behavior_v2 configs must use identical resource settings")
    src = Path(__file__).resolve().parents[1]

    if args.stage in {"all", "prepare"} and not args.skip_extraction:
        extraction_args = ["--suite", suite_path]
        if args.resume:
            extraction_args.append("--resume")
        call(src / "run_extraction_suite.py", *extraction_args)
        call(src / "audit_checkpoint_identity.py", "--suite", suite_path)
    task_hashes = set()
    if args.stage in {"all", "prepare", "analyze"}:
        for config_path, cfg in zip(config_paths, configs):
            call(src / "prepare_behavior_v2_tasks.py", "--config", config_path)
            if args.resume and geometry_reusable(cfg):
                print(f"Resume: reused behavior_v2 geometry for {cfg['experiment_name']}")
            else:
                call(src / "export_behavior_v2_geometry.py", "--config", config_path)
            call(src / "plot_behavior_v2.py", "--config", config_path)
            task_hashes.add(sha256_file(ensure_paths(cfg).data / "behavior_v2_tasks.jsonl"))
        if len(task_hashes) != 1:
            raise ValueError("behavior_v2 task files differ across models")
    if args.stage == "prepare":
        print("behavior_v2 prepare stage complete: extraction, checkpoint audit, tasks, geometry.")
        return

    if args.stage in {"all", "generate"} and not args.skip_generation:
        for config_path in config_paths:
            generation_args = ["--config", config_path]
            if args.resume:
                generation_args.append("--resume")
            call(src / "generate_behavior_v2.py", *generation_args)
    if args.stage == "generate":
        print("behavior_v2 generation stage complete.")
        return

    if args.stage == "audit":
        for config_path, cfg in zip(config_paths, configs):
            call(
                src / "evaluate_behavior_v2.py", "--config", config_path,
                "--allow-unvalidated-detector",
            )
            annotation_path = ensure_paths(cfg).measurement / "lexical_detector_audit.csv"
            call(
                src / "annotate_behavior_v2.py", "create", "--config", config_path,
                "--output", annotation_path, "--negative-per-language",
                args.negative_audit_per_language,
            )
        print("behavior_v2 audit samples created. Human 0/1 labels are required before analysis.")
        return

    for config_path in config_paths:
        call(src / "evaluate_behavior_v2.py", "--config", config_path)
        call(src / "compute_behavior_v2_association.py", "--config", config_path)
        call(src / "validate_behavior_v2.py", "--config", config_path)
    if len(configs) >= 3:
        call(src / "compare_behavior_v2_models.py", "--suite", suite_path)
        print("behavior_v2 suite and cross-model comparison complete.")
    else:
        print(
            "behavior_v2 single-model analysis complete; cross-model comparison "
            "was intentionally skipped."
        )
    print("No activation intervention was run.")


if __name__ == "__main__":
    main()
