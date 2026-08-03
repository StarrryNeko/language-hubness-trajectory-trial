"""Read-only audit of server artifacts and the safe next action for each config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config
from run_model_suite import completed_for_config, extraction_reusable, prepared_data_reusable


def classify_run_state(cfg):
    output = Path(cfg["output_dir"])
    paper_root = output / cfg.get("paper_v1", {}).get("result_directory", "paper_v1")
    paper_summary = paper_root / "validation" / "paper_validation_summary.json"
    if paper_summary.exists():
        return "PAPER_ANALYSIS_COMPLETE", "Use --resume; inspect paper_v1 validation summary."
    if extraction_reusable(cfg):
        if completed_for_config(cfg):
            return "READY_FOR_PAPER_REANALYSIS", "Run run_paper_analysis.py; do not re-extract."
        return "EXTRACTION_REUSABLE", "Run paper_v1 offline analysis; legacy metrics may be absent."
    hidden = output / "hidden"
    hidden_artifacts = list(hidden.glob("*")) if hidden.exists() else []
    if hidden_artifacts:
        return (
            "PARTIAL_OR_INCOMPATIBLE_EXTRACTION",
            "Do not merge with formal output. Inspect manifest; current extractor must restart this model.",
        )
    if prepared_data_reusable(cfg):
        return "DATA_READY_EXTRACTION_MISSING", "Run extraction for this model only."
    if output.exists():
        return "OUTPUT_INCOMPLETE", "Audit files; prepare data in this isolated output before extraction."
    return "NOT_STARTED", "No compatible artifacts; start only after this suite is authorized."


def main():
    parser = argparse.ArgumentParser(description="Audit model/suite artifacts without changing them")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--suite")
    args = parser.parse_args()
    if args.config:
        config_paths = [Path(args.config).resolve()]
    else:
        suite_path = Path(args.suite).resolve()
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        config_paths = [suite_path.parent / item for item in suite["configs"]]
    records = []
    for config_path in config_paths:
        cfg = load_config(config_path)
        state, action = classify_run_state(cfg)
        records.append({
            "config": str(config_path),
            "experiment": cfg.get("experiment_name", config_path.stem),
            "output_dir": cfg["output_dir"],
            "sampling_strategy": cfg.get("dataset", {}).get("sample_selection", {}).get(
                "strategy", "legacy_unspecified"
            ),
            "state": state,
            "safe_next_action": action,
        })
    for record in records:
        print(
            f"{record['experiment']}: {record['state']}\n"
            f"  output={record['output_dir']}\n"
            f"  sampling={record['sampling_strategy']}\n"
            f"  next={record['safe_next_action']}"
        )
    print("\n" + json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

