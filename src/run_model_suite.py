import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import configured_representations, load_config, representation_file_map
from evidence_rules import validate_model_status_payload


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reuse_data(source_output, target_output):
    source = Path(source_output) / "data"
    target = Path(target_output) / "data"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["parallel_samples.jsonl", "dataset_manifest.json"]:
        source_file = source / name
        target_file = target / name
        if not source_file.exists():
            raise FileNotFoundError(f"First model did not produce {source_file}")
        if target_file.exists() and digest(target_file) != digest(source_file):
            raise ValueError(f"Refusing to overwrite non-identical prepared data: {target_file}")
        if not target_file.exists():
            shutil.copy2(source_file, target_file)


def completed_for_config(cfg):
    output = Path(cfg["output_dir"])
    snapshot = output / "config_snapshot.json"
    required = [
        output / "extraction_manifest.json",
        output / "metrics" / "english_hubness_evidence.csv",
        output / "validation" / "validation_summary.json",
    ]
    if not snapshot.exists() or not all(path.exists() for path in required):
        return False
    try:
        summary = json.loads(required[-1].read_text(encoding="utf-8"))
        model_status = validate_model_status_payload(summary)
        metric_manifest = json.loads(
            (output / "metrics" / "metrics_manifest.json").read_text(encoding="utf-8")
        )
        return (
            json.loads(snapshot.read_text(encoding="utf-8")) == cfg
            and model_status != "INVALID"
            and metric_manifest.get("joint_evidence_rule")
            == "all_four_ci_lower_gt_zero_on_same_layer"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def prepared_data_reusable(cfg):
    output = Path(cfg["output_dir"])
    data = output / "data"
    return all((data / name).exists() for name in ("parallel_samples.jsonl", "dataset_manifest.json"))


def extraction_reusable(cfg):
    output = Path(cfg["output_dir"])
    manifest_path = output / "extraction_manifest.json"
    metadata_path = output / "hidden" / "metadata.csv"
    required_vectors = [
        output / "hidden" / representation_file_map()[name]
        for name in configured_representations(cfg)
    ]
    if not manifest_path.exists() or not metadata_path.exists() or not all(path.exists() for path in required_vectors):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_representations = set(manifest.get("representations", []))
        truncated_inputs = int(manifest.get("truncated_inputs", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    expected_storage = cfg.get("storage_dtype", "float16")
    expected_model = cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path"))
    return (
        manifest.get("model") == expected_model
        and manifest.get("storage_dtype") == expected_storage
        and manifest_representations == set(configured_representations(cfg))
        and truncated_inputs == 0
    )


def append_reuse_flags(command, cfg, allow_skip_prepare):
    if allow_skip_prepare and prepared_data_reusable(cfg):
        command.append("--skip-prepare")
    if extraction_reusable(cfg):
        command.append("--skip-extract")


def main():
    parser = argparse.ArgumentParser(description="Run identical 24-language data through multiple models")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--skip-k-sweep", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip only runs whose completed config snapshot exactly matches the current resolved config.",
    )
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    configs = [suite_path.parent / item for item in suite["configs"]]
    if len(configs) < 2:
        raise ValueError("A model comparison suite requires at least two configs")
    pilot = Path(__file__).resolve().parent / "run_pilot.py"
    first_cfg = load_config(configs[0])
    first_command = [sys.executable, str(pilot), "--config", str(configs[0])]
    append_reuse_flags(first_command, first_cfg, allow_skip_prepare=True)
    if args.skip_k_sweep:
        first_command.append("--skip-k-sweep")
    if not (args.resume and completed_for_config(first_cfg)):
        subprocess.run(first_command, check=True)
    else:
        print(f"Resume: verified and skipped {first_cfg['experiment_name']}")
    for config_path in configs[1:]:
        cfg = load_config(config_path)
        reuse_data(first_cfg["output_dir"], cfg["output_dir"])
        if args.resume and completed_for_config(cfg):
            print(f"Resume: verified and skipped {cfg['experiment_name']}")
        else:
            command = [sys.executable, str(pilot), "--config", str(config_path), "--skip-prepare"]
            if extraction_reusable(cfg):
                command.append("--skip-extract")
            if args.skip_k_sweep:
                command.append("--skip-k-sweep")
            subprocess.run(command, check=True)
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parent / "compare_models.py"),
        "--suite", str(suite_path),
    ], check=True)


if __name__ == "__main__":
    main()
