"""Run the frozen three-model behavior_v1 pipeline without intervention."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from behavior_common import (
    behavior_settings, configure_cpu_environment, ensure_behavior_dirs,
    read_checkpoint_identity, sha256_file,
)
from common import load_config
from run_extraction_suite import validate_frozen_suite


def call(script, *arguments):
    command = [sys.executable, str(script), *map(str, arguments)]
    print(f"\n{' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def predictors_reusable(cfg):
    paths = ensure_behavior_dirs(cfg)
    predictor = paths.metrics / "behavior_geometry_predictors.csv"
    manifest_path = paths.metrics / "behavior_geometry_manifest.json"
    tasks = paths.data / "behavior_tasks.jsonl"
    if not predictor.exists() or not manifest_path.exists() or not tasks.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = behavior_settings(cfg)
        checkpoint = read_checkpoint_identity(cfg)
        return (
            manifest.get("protocol_version") == "behavior_v1"
            and manifest.get("layers") == sorted(set(settings["analysis_layers"]))
            and manifest.get("primary_layer") == settings["primary_layer"]
            and manifest.get("resources") == settings["resources"]
            and manifest.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"]
            and manifest.get("task_file_sha256") == sha256_file(tasks)
            and manifest.get("predictor_file_sha256") == sha256_file(predictor)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Run behavior_v1 three-model suite")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument(
        "--stage", choices=("all", "prepare", "generate", "analyze"), default="all",
        help="Run the complete pipeline or one restartable stage.",
    )
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    config_paths = [suite_path.parent / value for value in suite["configs"]]
    configs = [load_config(path) for path in config_paths]
    validate_frozen_suite(suite, configs)
    resource_protocols = [behavior_settings(cfg)["resources"] for cfg in configs]
    if any(value != resource_protocols[0] for value in resource_protocols[1:]):
        raise ValueError("all behavior suite configs must use identical resources")
    configure_cpu_environment(resource_protocols[0]["cpu_threads"])
    print(
        f"Resources: CPU threads={resource_protocols[0]['cpu_threads']}; "
        f"evaluation workers={resource_protocols[0]['evaluation_workers']}; "
        f"geometry device={resource_protocols[0]['geometry_device']}",
        flush=True,
    )
    src = Path(__file__).resolve().parent
    run_extraction = args.stage in {"all", "prepare"} and not args.skip_extraction
    if run_extraction:
        extraction_args = ["--suite", suite_path]
        if args.resume:
            extraction_args.append("--resume")
        call(src / "run_extraction_suite.py", *extraction_args)
        call(src / "audit_checkpoint_identity.py", "--suite", suite_path)

    task_hashes = set()
    for config_path, cfg in zip(config_paths, configs):
        call(src / "prepare_behavior_tasks.py", "--config", config_path)
        needs_predictors = args.stage in {"all", "prepare", "analyze"}
        if needs_predictors:
            if args.resume and predictors_reusable(cfg):
                print(f"Resume: reused GPU geometry predictors for {cfg['experiment_name']}")
            else:
                call(src / "export_behavior_predictors.py", "--config", config_path)
        task_hashes.add(sha256_file(ensure_behavior_dirs(cfg).data / "behavior_tasks.jsonl"))
    if len(task_hashes) != 1:
        raise ValueError("behavior task files differ across models")
    if args.stage == "prepare":
        print("behavior_v1 prepare stage complete: data, checkpoint audit, tasks, and predictors.")
        return

    for config_path in config_paths:
        if args.stage in {"all", "generate"} and not args.skip_generation:
            generation_args = ["--config", config_path]
            if args.resume:
                generation_args.append("--resume")
            call(src / "generate_behavior.py", *generation_args)
        if args.stage == "generate":
            continue
        call(src / "evaluate_behavior_outputs.py", "--config", config_path)
        call(src / "compute_behavior_association.py", "--config", config_path)
        call(src / "validate_behavior.py", "--config", config_path)
    if args.stage == "generate":
        print("behavior_v1 generation stage complete.")
        return
    call(src / "compare_behavior_models.py", "--suite", suite_path)
    print("behavior_v1 suite complete; no activation intervention was run.")


if __name__ == "__main__":
    main()
