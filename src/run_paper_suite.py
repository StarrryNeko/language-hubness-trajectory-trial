"""Run paper_v1 only for models with complete, reusable hidden-state extraction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import load_config
from run_model_suite import extraction_reusable


def main():
    parser = argparse.ArgumentParser(
        description="Offline paper_v1 suite; never downloads or loads model weights"
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--require-all", action="store_true",
        help="Fail instead of skipping configs whose extraction is incomplete or incompatible.",
    )
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    runner = Path(__file__).resolve().parent / "run_paper_analysis.py"
    completed = []
    skipped = []
    for relative in suite["configs"]:
        config_path = suite_path.parent / relative
        cfg = load_config(config_path)
        if not extraction_reusable(cfg):
            message = (
                f"No reusable mean_pool_v1 extraction for {config_path}; "
                "paper suite will not start the model."
            )
            if args.require_all:
                raise RuntimeError(message)
            print(f"Skip: {message}")
            skipped.append(str(config_path))
            continue
        command = [sys.executable, str(runner), "--config", str(config_path)]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)
        completed.append(str(config_path))
    if not completed:
        raise RuntimeError("No configs had a reusable extraction; no model was started")
    subprocess.run([
        sys.executable,
        str(Path(__file__).resolve().parent / "compare_paper_models.py"),
        "--suite", str(suite_path),
    ], check=True)
    output = Path(suite["comparison_output_dir"]) / "paper_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_suite_run.json").write_text(json.dumps({
        "protocol_version": "paper_v1",
        "model_weights_loaded": False,
        "completed_configs": completed,
        "skipped_incomplete_extractions": skipped,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"paper_v1 suite complete for {len(completed)} models; skipped={len(skipped)}")


if __name__ == "__main__":
    main()
