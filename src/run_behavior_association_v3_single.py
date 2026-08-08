"""Explicit single-model stages with an unavoidable human-audit breakpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from behavior_association_v3.common import configure_cpu_threads, paths, settings
from common import load_config


def call(script, *arguments):
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", required=True,
        choices=("structure", "prepare", "calibrate", "formal-generate", "analyze"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    protocol = settings(cfg)
    configure_cpu_threads(max(1, protocol["cpu_threads"] // 3))
    src = Path(__file__).resolve().parent
    common = ["--config", args.config]
    resume = ["--resume"] if args.resume else []
    if args.stage == "structure":
        call(src / "run_structure_v2.py", *common)
    elif args.stage == "prepare":
        structure_status = Path(cfg["output_dir"]) / "structure_v2" / "validation" / "validation_summary.json"
        if not structure_status.exists() or json.loads(structure_status.read_text(encoding="utf-8")).get("status") != "VALID":
            raise ValueError("run and validate this model's structure_v2 stage before V3 preparation")
        call(src / "prepare_behavior_association_v3_tasks.py", *common)
        call(src / "export_behavior_association_v3_predictors.py", *common)
    elif args.stage == "calibrate":
        call(src / "generate_behavior_association_v3.py", *common, "--split", "calibration", *resume)
        call(
            src / "evaluate_behavior_association_v3.py", *common, "--split", "calibration",
            "--allow-unvalidated-detector",
        )
        audit = paths(cfg).measurement / "lexical_detector_calibration_audit.csv"
        call(src / "annotate_behavior_association_v3.py", "create", *common, "--output", audit)
        print(f"Human labels are now required in: {audit}")
    elif args.stage == "formal-generate":
        report = paths(cfg).measurement / "lexical_detector_validation.json"
        if not report.exists():
            raise FileNotFoundError(
                "validate the calibration audit before formal generation: "
                f"python src/annotate_behavior_association_v3.py validate --config {args.config} "
                f"--annotations {paths(cfg).measurement / 'lexical_detector_calibration_audit.csv'} "
                f"--output {report}"
            )
        report_payload = json.loads(report.read_text(encoding="utf-8"))
        if report_payload.get("passed") is not True or report_payload.get("source_split") != "calibration":
            raise ValueError("formal generation requires a passed calibration-only detector report")
        call(src / "generate_behavior_association_v3.py", *common, "--split", "formal", *resume)
    else:
        call(src / "evaluate_behavior_association_v3.py", *common, "--split", "formal")
        call(src / "compute_behavior_association_v3.py", *common)
        call(src / "validate_behavior_association_v3.py", *common)


if __name__ == "__main__":
    main()
