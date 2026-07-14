import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run_step(label, script, config):
    command = [sys.executable, str(script), "--config", config]
    print(f"\n{'=' * 72}\n{label}\nCommand: {' '.join(command)}\n{'=' * 72}", flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the complete sentence-level pilot and validation pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    started_at = time.perf_counter()
    started_iso = datetime.now(timezone.utc).isoformat()
    src_dir = Path(__file__).resolve().parent
    steps = []
    if not args.skip_prepare:
        steps.append(("STEP 1/4 - Prepare parallel data", src_dir / "prepare_flores.py"))
    if not args.skip_extract:
        steps.append(("STEP 2/4 - Extract last-token and mean-pool hidden states", src_dir / "extract_hidden.py"))
    steps.extend([
        ("STEP 3/4 - Validate representations and compute research metrics", src_dir / "compute_metrics.py"),
        ("STEP 4/4 - Generate diagnostic and research figures", src_dir / "plot_trajectories.py"),
    ])

    for label, script in steps:
        run_step(label, script, args.config)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": started_iso,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started_at,
        "config_path": str(Path(args.config).resolve()),
        "experiment_name": cfg.get("experiment_name"),
        "model": cfg.get("model_name_or_path"),
        "dataset": cfg.get("dataset"),
        "metrics": cfg.get("metrics"),
        "python": sys.version,
        "platform": platform.platform(),
        "skipped_prepare": args.skip_prepare,
        "skipped_extract": args.skip_extract,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPILOT PIPELINE COMPLETE")
    print("Read metrics/validation_report.txt first, then inspect figures/*.png.")


if __name__ == "__main__":
    main()
