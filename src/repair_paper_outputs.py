"""Repair derived paper artifacts from reusable hidden states; never load weights."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
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


def worker_environment(threads_per_worker):
    """Cap native math libraries so parallel model jobs do not oversubscribe CPUs."""
    environment = os.environ.copy()
    threads = str(int(threads_per_worker))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = threads
    return environment


def repair_config(config_path, rerun_all_offline, threads_per_worker, src):
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    if not extraction_reusable(cfg):
        raise RuntimeError(f"No compatible reusable hidden states for {config_path}")
    scripts = (
        [src / "run_paper_analysis.py"]
        if rerun_all_offline
        else [src / "compute_norm_trajectory.py", src / "validate_paper.py"]
    )
    environment = worker_environment(threads_per_worker)
    for script in scripts:
        subprocess.run(
            [sys.executable, str(script), "--config", str(config_path)],
            check=True,
            env=environment,
        )
    paper_root = (
        Path(cfg["output_dir"])
        / cfg.get("paper_v1", {}).get("result_directory", "paper_v1")
    )
    failures = strict_json_audit(paper_root)
    if failures:
        raise ValueError("Strict JSON audit failed:\n" + "\n".join(failures))
    return str(config_path)


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
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of model configs to process concurrently (recommended: 4 on 24 vCPU).",
    )
    parser.add_argument(
        "--threads-per-worker", type=int, default=None,
        help="Native BLAS/OpenMP thread cap per model worker (default: cpu_count/workers).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cpu_count = os.cpu_count() or 1
    threads_per_worker = args.threads_per_worker or max(1, cpu_count // args.workers)
    if threads_per_worker < 1:
        parser.error("--threads-per-worker must be positive")
    if args.workers * threads_per_worker > cpu_count:
        print(
            f"Warning: {args.workers} workers x {threads_per_worker} threads exceeds "
            f"detected {cpu_count} CPUs; oversubscription may slow the run.",
            flush=True,
        )
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    src = Path(__file__).resolve().parent
    config_paths = [suite_path.parent / relative for relative in suite["configs"]]
    print(
        f"Repairing {len(config_paths)} models with {args.workers} workers x "
        f"{threads_per_worker} native threads",
        flush=True,
    )
    if args.workers == 1:
        completed = [
            repair_config(path, args.rerun_all_offline, threads_per_worker, src)
            for path in config_paths
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    repair_config,
                    path,
                    args.rerun_all_offline,
                    threads_per_worker,
                    src,
                ): path
                for path in config_paths
            }
            completed = []
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
                print(f"Completed: {futures[future]}", flush=True)
    subprocess.run(
        [sys.executable, str(src / "compare_paper_models.py"), "--suite", str(suite_path)],
        check=True,
    )
    print(f"Repaired and strictly validated paper outputs for {len(completed)} models")


if __name__ == "__main__":
    main()
