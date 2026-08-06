"""Repair derived paper artifacts from reusable hidden states; never load weights."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import load_config
from run_model_suite import extraction_reusable


def strict_json_audit(root):
    failures = []
    for path in Path(root).rglob("*.json"):
        try:
            json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-standard numeric token {token}")
                ),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate norm/trajectory JSON, corrected validation claims and a unified "
            "comparison without loading model weights or using a GPU"
        )
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--rerun-all-offline",
        action="store_true",
        help="Recompute every paper_v1 metric from hidden arrays instead of only affected outputs.",
    )
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    src = Path(__file__).resolve().parent
    completed = []
    for relative in suite["configs"]:
        config_path = suite_path.parent / relative
        cfg = load_config(config_path)
        if not extraction_reusable(cfg):
            raise RuntimeError(f"No compatible reusable hidden states for {config_path}")
        if args.rerun_all_offline:
            scripts = [src / "run_paper_analysis.py"]
        else:
            scripts = [src / "compute_norm_trajectory.py", src / "validate_paper.py"]
        for script in scripts:
            subprocess.run(
                [sys.executable, str(script), "--config", str(config_path)], check=True
            )
        failures = strict_json_audit(
            Path(cfg["output_dir"]) / cfg.get("paper_v1", {}).get("result_directory", "paper_v1")
        )
        if failures:
            raise ValueError("Strict JSON audit failed:\n" + "\n".join(failures))
        completed.append(str(config_path))
    subprocess.run(
        [sys.executable, str(src / "compare_paper_models.py"), "--suite", str(suite_path)],
        check=True,
    )
    print(f"Repaired and strictly validated paper outputs for {len(completed)} models")


if __name__ == "__main__":
    main()
