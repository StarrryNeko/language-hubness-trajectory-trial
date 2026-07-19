import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import load_config


def run_step(label, script, config, extra_args=None):
    command = [sys.executable, str(script), "--config", config]
    command.extend(extra_args or [])
    print(f"\n{'=' * 72}\n{label}\nCommand: {' '.join(command)}\n{'=' * 72}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return {
        "label": label,
        "script": str(script),
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "completed",
    }


def git_provenance(project_dir):
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_dir, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def main():
    parser = argparse.ArgumentParser(description="Run the complete sentence-level pilot and validation pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-k-sweep", action="store_true")
    parser.add_argument("--skip-validations", action="store_true")
    parser.add_argument(
        "--show-sentences",
        choices=["none", "preview", "all"],
        default=None,
        help="Override extraction-time sentence display. Use 'all' to print every hidden-state row's sentence.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_text = json.dumps(cfg, indent=2, ensure_ascii=False)

    started_at = time.perf_counter()
    started_iso = datetime.now(timezone.utc).isoformat()
    src_dir = Path(__file__).resolve().parent
    steps = []
    if not args.skip_prepare:
        steps.append(("Prepare and structurally validate parallel data", src_dir / "prepare_flores.py", []))
    if not args.skip_extract:
        extract_args = ["--show-sentences", args.show_sentences] if args.show_sentences else []
        steps.append((
            "Extract mean-pool vectors and sentinel-EOS validation vectors",
            src_dir / "extract_hidden.py",
            extract_args,
        ))
    steps.append((
        "Compute strictly same-semantics multilingual hubness metrics",
        src_dir / "compute_metrics.py",
        [],
    ))
    if not args.skip_figures:
        steps.append(("Generate diagnostic and research figures", src_dir / "plot_trajectories.py", []))

    robustness_cfg = cfg.get("robustness", {})
    run_k_sweep = bool(robustness_cfg.get("run_k_sweep_in_full_pipeline", True)) and not args.skip_k_sweep
    if run_k_sweep:
        k_values = [str(value) for value in robustness_cfg.get("k_values", [5, 10, 20])]
        steps.append(("Run kNN k-value robustness sweep", src_dir / "sweep_k.py", ["--k-values", *k_values]))
    if not args.skip_validations:
        steps.append((
            "Save each validation idea, evidence, verdict, and required action separately",
            src_dir / "run_validations.py",
            [],
        ))

    step_results = []
    for index, (label, script, extra_args) in enumerate(steps, start=1):
        step_results.append(run_step(f"STEP {index}/{len(steps)} - {label}", script, args.config, extra_args))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": started_iso,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started_at,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "experiment_name": cfg.get("experiment_name"),
        "model": cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path")),
        "dataset": cfg.get("dataset"),
        "metrics": cfg.get("metrics"),
        "representation_controls": cfg.get("representation_controls"),
        "robustness": cfg.get("robustness"),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_provenance(src_dir.parent),
        "skipped_prepare": args.skip_prepare,
        "skipped_extract": args.skip_extract,
        "skipped_figures": args.skip_figures,
        "skipped_k_sweep": not run_k_sweep,
        "skipped_validations": args.skip_validations,
        "steps": step_results,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPILOT PIPELINE COMPLETE")
    if not args.skip_validations:
        print("Start with validation/validation_summary.md, then inspect the numbered validation reports.")
    else:
        print("Validation reports were skipped. Read metrics/validation_report.txt and inspect figures/*.png.")


if __name__ == "__main__":
    main()
