"""Validate the geometry-only structure_v2 artifact."""

import argparse
import json
from pathlib import Path

import pandas as pd

from common import load_config, write_json
from structure_v2.common import paths, settings, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    protocol = settings(cfg)
    destination = paths(cfg)
    result_path = destination.geometry / "confirmatory_geometry.csv"
    manifest_path = destination.geometry / "geometry_manifest.json"
    blockers = []
    if not result_path.exists() or not manifest_path.exists():
        blockers.append("missing geometry result or manifest")
        frame, manifest = pd.DataFrame(), {}
    else:
        frame = pd.read_csv(result_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"layer", "metric", "mean", "ci_lower", "ci_upper", "semantic_ids"}
    if not frame.empty and required - set(frame):
        blockers.append("geometry table is missing required columns")
    expected = len(protocol["analysis_layers"]) * (2 + 2 * len(protocol["k_sensitivity"]))
    if len(frame) != expected:
        blockers.append(f"geometry row count mismatch: expected {expected}, got {len(frame)}")
    if not frame.empty and not set(frame.layer) == set(protocol["analysis_layers"]):
        blockers.append("geometry layer set changed")
    if manifest.get("result_sha256") != (sha256_file(result_path) if result_path.exists() else None):
        blockers.append("geometry result hash mismatch")
    if manifest.get("bootstrap_unit") != "semantic_id":
        blockers.append("uncertainty is not semantic-ID bootstrap")
    payload = {
        "protocol_version": "structure_v2",
        "status": "VALID" if not blockers else "INVALID",
        "blockers": blockers,
        "primary_layer": protocol["primary_layer"],
        "rows": len(frame),
    }
    write_json(destination.validation / "validation_summary.json", payload)
    print(f"structure_v2 validation: {payload['status']}")
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
