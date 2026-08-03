"""Prepare shared data and extract hidden states, without running legacy metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import load_config
from run_model_suite import extraction_reusable, prepared_data_reusable, reuse_data


def call(script, config):
    command = [sys.executable, str(script), "--config", str(config)]
    print(f"\n{' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def validate_frozen_suite(suite, configs):
    """Reject silent sample/model drift in formal one-week suites."""
    if len({cfg["output_dir"] for cfg in configs}) != len(configs):
        raise ValueError("suite configs must use distinct output_dir values")
    model_ids = [
        cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path"))
        for cfg in configs
    ]
    if None in model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("suite model IDs must be present and unique")
    reference_languages = list(configs[0]["dataset"]["languages"])
    for cfg in configs[1:]:
        if list(cfg["dataset"]["languages"]) != reference_languages:
            raise ValueError("all suite configs must use identical ordered languages")
    expectations = {
        "sample_size_per_language": suite.get("sample_size_per_language"),
        "sample_selection_strategy": suite.get("sample_selection_strategy"),
        "sample_selection_seed": suite.get("sample_selection_seed"),
    }
    for cfg in configs:
        dataset = cfg.get("dataset", {})
        selection = dataset.get("sample_selection", {})
        actual = {
            "sample_size_per_language": dataset.get("sample_size_per_language"),
            "sample_selection_strategy": selection.get("strategy"),
            "sample_selection_seed": selection.get("seed"),
        }
        for key, expected in expectations.items():
            if expected is not None and actual[key] != expected:
                raise ValueError(
                    f"suite/config mismatch for {key}: expected={expected!r}, "
                    f"actual={actual[key]!r}, model={cfg.get('experiment_name')}"
                )
    if suite.get("model_list_frozen_before_results") is False:
        raise ValueError("formal suite must freeze the model list before results")


def main():
    parser = argparse.ArgumentParser(description="Run only preparation and hidden-state extraction")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    config_paths = [suite_path.parent / item for item in suite["configs"]]
    if not config_paths:
        raise ValueError("suite has no configs")
    configs = [load_config(path) for path in config_paths]
    validate_frozen_suite(suite, configs)
    src = Path(__file__).resolve().parent

    first_path, first_cfg = config_paths[0], configs[0]
    if not (args.resume and prepared_data_reusable(first_cfg)):
        call(src / "prepare_flores.py", first_path)
    else:
        print(f"Resume: reused prepared data for {first_cfg['experiment_name']}")
    if not (args.resume and extraction_reusable(first_cfg)):
        call(src / "extract_hidden.py", first_path)
    else:
        print(f"Resume: reused extraction for {first_cfg['experiment_name']}")

    for config_path, cfg in zip(config_paths[1:], configs[1:]):
        reuse_data(first_cfg["output_dir"], cfg["output_dir"])
        if args.resume and extraction_reusable(cfg):
            print(f"Resume: reused extraction for {cfg['experiment_name']}")
        else:
            call(src / "extract_hidden.py", config_path)
    print("Extraction suite complete. No hubness result was evaluated in this stage.")


if __name__ == "__main__":
    main()
