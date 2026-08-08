"""Final integrity and interpretation gates for a single V3 model."""

import argparse
import json

from behavior_association_v3.common import paths, settings, sha256_file
from common import load_config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings(cfg)
    destination = paths(cfg)
    blockers = []
    required = [
        destination.data / "task_manifest.json",
        destination.generations / "formal_generation_manifest.json",
        destination.metrics / "formal_evaluation_manifest.json",
        destination.measurement / "lexical_detector_validation.json",
        destination.measurement / "predictor_manifest.json",
        destination.measurement / "association_manifest.json",
        destination.validation / "association_status.json",
    ]
    for path in required:
        if not path.exists():
            blockers.append(f"missing file: {path}")
    payloads = {}
    for path in required:
        if path.exists():
            try:
                payloads[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                blockers.append(f"invalid JSON: {path}")
    tasks = payloads.get("task_manifest.json", {})
    if tasks.get("calibration_formal_overlap") != 0:
        blockers.append("calibration and formal semantic IDs overlap")
    if payloads.get("formal_evaluation_manifest.json", {}).get("status") != "PASS":
        blockers.append("formal evaluation gates did not pass")
    if payloads.get("lexical_detector_validation.json", {}).get("passed") is not True:
        blockers.append("calibration detector validation did not pass")
    association = payloads.get("association_status.json", {})
    if association.get("claim_type") != "OBSERVATIONAL_ASSOCIATION_NOT_CAUSAL":
        blockers.append("association claim boundary changed")
    summary = {
        "protocol_version": "behavior_association_v3",
        "status": "VALID" if not blockers else "INVALID",
        "association_status": association.get("status", "NOT_RUN"),
        "blockers": blockers,
    }
    write_json(destination.validation / "validation_summary.json", summary)
    print(f"V3 validation: {summary['status']}; association={summary['association_status']}")
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
