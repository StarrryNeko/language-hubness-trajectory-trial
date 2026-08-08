"""Compare the frozen three-model V3 results without treating non-significance as equivalence."""

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_association_v3.common import paths
from common import load_config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    records = []
    calibration_hashes, formal_hashes = set(), set()
    for relative in suite["configs"]:
        cfg = load_config(suite_path.parent / relative)
        destination = paths(cfg)
        validation = json.loads((destination.validation / "validation_summary.json").read_text(encoding="utf-8"))
        association = json.loads((destination.validation / "association_status.json").read_text(encoding="utf-8"))
        task_manifest = json.loads((destination.data / "task_manifest.json").read_text(encoding="utf-8"))
        calibration_hashes.add(task_manifest["calibration_task_sha256"])
        formal_hashes.add(task_manifest["formal_task_sha256"])
        primary = association.get("primary_test", {})
        local = association.get("local_scaled_robustness_test", {})
        records.append({
            "experiment_name": cfg["experiment_name"], "model": cfg["model"]["name_or_path"],
            "frozen_role": suite["model_roles"][cfg["experiment_name"]],
            "validation_status": validation["status"],
            "association_status": association["status"],
            "primary_coefficient": primary.get("coefficient"),
            "primary_ci_lower": primary.get("ci_lower"),
            "primary_ci_upper": primary.get("ci_upper"),
            "local_coefficient": local.get("coefficient"),
            "local_ci_lower": local.get("ci_lower"),
            "local_ci_upper": local.get("ci_upper"),
        })
    frame = pd.DataFrame(records)
    if len(calibration_hashes) != 1 or len(formal_hashes) != 1:
        raise ValueError("three V3 models did not use identical frozen task sets")
    output = Path(suite["comparison_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8")
    positives = frame.loc[frame.frozen_role == "expected_positive"]
    all_valid = bool((frame.validation_status == "VALID").all())
    raw_replicated = bool(positives.association_status.str.startswith("SUPPORTED").all())
    local_replicated = bool((positives.association_status == "SUPPORTED_RAW_AND_LOCAL_SCALED").all())
    status = (
        "INVALID" if not all_valid else
        "REPLICATED_RAW_AND_LOCAL_SCALED" if raw_replicated and local_replicated else
        "REPLICATED_RAW_ONLY" if raw_replicated else
        "PARTIAL_OR_NOT_REPLICATED"
    )
    write_json(output / "cross_model_status.json", {
        "protocol_version": "behavior_association_v3", "status": status,
        "all_models_valid": all_valid, "expected_positive_raw_replication": raw_replicated,
        "expected_positive_local_scaled_replication": local_replicated,
        "mistral_interpretation": "frozen contrast reported descriptively; non-significance is not equivalence",
        "calibration_task_sha256": next(iter(calibration_hashes)),
        "formal_task_sha256": next(iter(formal_hashes)),
        "claim_type": "CROSS_MODEL_OBSERVATIONAL_ASSOCIATION_NOT_CAUSAL",
    })
    print(f"V3 cross-model comparison: {status}")


if __name__ == "__main__":
    main()
