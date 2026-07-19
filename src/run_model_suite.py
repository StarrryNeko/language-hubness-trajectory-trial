import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import load_config


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
        return json.loads(snapshot.read_text(encoding="utf-8")) == cfg
    except (OSError, json.JSONDecodeError):
        return False


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
            if args.skip_k_sweep:
                command.append("--skip-k-sweep")
            subprocess.run(command, check=True)
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parent / "compare_models.py"),
        "--suite", str(suite_path),
    ], check=True)


if __name__ == "__main__":
    main()
