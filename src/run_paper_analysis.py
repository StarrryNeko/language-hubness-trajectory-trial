"""Run all offline paper_v1 analyses against reusable hidden-state arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import load_config
from paper_common import ensure_paper_dirs, load_hidden_dataset, semantic_id_hash


def resolved_digest(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def run_step(script, config):
    command = [sys.executable, str(script), "--config", str(config)]
    print(f"\n=== {script.stem} ===\n{' '.join(command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return {
        "script": script.name,
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run paper_v1 analyses without loading model weights or re-extracting hidden states"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-language-structure", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-k-sweep", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if not cfg.get("paper_v1", {}).get("enabled", False):
        raise ValueError("paper_v1.enabled must be true")
    dataset = load_hidden_dataset(cfg)
    paths = ensure_paper_dirs(cfg)
    digest = resolved_digest(cfg)
    snapshot_path = paths.root / "config_snapshot.json"
    previous_matches = False
    if snapshot_path.exists():
        try:
            previous_matches = json.loads(snapshot_path.read_text(encoding="utf-8")) == cfg
        except (OSError, json.JSONDecodeError):
            previous_matches = False
    snapshot_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    src = Path(__file__).resolve().parent
    steps = [
        (src / "compute_alignment.py", paths.metrics / "alignment" / "alignment_manifest.json"),
        (src / "compute_geometry_controls.py", paths.metrics / "geometry_controls" / "geometry_manifest.json"),
        (src / "compute_hubness_paper.py", paths.metrics / "hubness" / "hubness_manifest.json"),
        (
            src / "compute_similarity_competition.py",
            paths.metrics / "similarity_competition" / "similarity_competition_manifest.json",
        ),
        (
            src / "compute_norm_trajectory.py",
            paths.metrics / "norm_trajectory" / "norm_trajectory_manifest.json",
        ),
    ]
    if not args.skip_k_sweep:
        steps.append((
            src / "sweep_paper_k.py", paths.metrics / "hubness" / "k_robustness_manifest.json"
        ))
    steps.append((
        src / "compute_sample_robustness.py",
        paths.metrics / "hubness" / "sample_robustness_manifest.json",
    ))
    if not args.skip_language_structure:
        steps.append((
            src / "compute_language_structure.py",
            paths.metrics / "language_structure" / "language_structure_manifest.json",
        ))
    if not args.skip_probe:
        steps.append((
            src / "run_language_probe.py",
            paths.metrics / "language_probe" / "probe_manifest.json",
        ))
    if not args.skip_figures and not args.skip_language_structure and not args.skip_probe:
        steps.append((src / "plot_paper_results.py", paths.figures / "english_competition_by_layer.png"))
    steps.append((src / "validate_paper.py", paths.validation / "paper_validation_summary.json"))

    started = time.perf_counter()
    records = []
    for script, marker in steps:
        if args.resume and previous_matches and marker.exists():
            print(f"Resume: verified config snapshot and skipped {script.name}")
            records.append({"script": script.name, "status": "reused", "elapsed_seconds": 0.0})
        else:
            records.append(run_step(script, config_path))
    manifest = {
        "protocol_version": "paper_v1",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config_path": str(config_path),
        "resolved_config_sha256": digest,
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "hidden_states_reused": True,
        "model_weights_loaded": False,
        "rows": len(dataset.meta),
        "semantic_ids": len(dataset.semantic_ids),
        "languages": len(dataset.languages),
        "sample_selection_strategy": cfg.get("dataset", {}).get("sample_selection", {}).get(
            "strategy", "unknown"
        ),
        "steps": records,
    }
    (paths.root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\npaper_v1 offline analysis complete: {paths.root}")


if __name__ == "__main__":
    main()
