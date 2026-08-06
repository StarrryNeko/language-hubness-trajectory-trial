"""Download portable model bundles with deterministic HTTP range resumption.

This is intended for unstable connections where the Hub client's temporary
download name changes after a process restart. Large weights are written to a
stable ``<filename>.part`` path and resumed with an HTTP Range request.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_url, snapshot_download

from common import portable_model_directory_name
from download_model_weights import (
    PORTABLE_ALLOW_PATTERNS,
    PORTABLE_IGNORE_PATTERNS,
    models_from_suite,
    portable_inventory,
)


def select_weight_files(info):
    safetensors = [
        item for item in info.siblings
        if item.rfilename.endswith(".safetensors")
    ]
    if safetensors:
        return "safetensors", safetensors
    pytorch_bins = [
        item for item in info.siblings
        if Path(item.rfilename).name.startswith("pytorch_model")
        and item.rfilename.endswith(".bin")
    ]
    if pytorch_bins:
        return "pytorch_bin", pytorch_bins
    raise FileNotFoundError(f"No supported weights in {info.id} at {info.sha}")


def download_with_ranges(repo_id, revision, repo_file, target, max_retries):
    destination = target / repo_file.rfilename
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = repo_file.size
    if expected is None:
        raise ValueError(f"Hub did not report a size for {repo_file.rfilename}")
    if destination.exists():
        if destination.stat().st_size == expected:
            print(f"Already complete: {destination.name}", flush=True)
            return
        raise ValueError(
            f"Existing final file has the wrong size: {destination} "
            f"({destination.stat().st_size} != {expected})"
        )

    partial = destination.with_name(destination.name + ".part")
    url = hf_hub_url(repo_id, repo_file.rfilename, revision=revision)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected:
        raise ValueError(f"Partial file exceeds expected size: {partial}")
    if offset == expected:
        os.replace(partial, destination)
        return
    print(
        f"{repo_file.rfilename}: {offset / 10**9:.2f}/{expected / 10**9:.2f} GB",
        flush=True,
    )

    curl = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"
    if not curl.exists():
        raise FileNotFoundError(f"curl.exe not found at {curl}")
    token = get_token()
    header_path = None
    try:
        command = [
            str(curl), "--location", "--fail-with-body",
            "--connect-timeout", "30", "--max-time", "120",
            "--speed-time", "60", "--speed-limit", "1024",
            "--continue-at", "-", "--output", str(partial),
        ]
        if token:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".headers"
            ) as header_file:
                header_file.write(f"Authorization: Bearer {token}\n")
                header_path = Path(header_file.name)
            command.extend(["--header", f"@{header_path}"])
        command.append(url)
        for attempt in range(1, max_retries + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            print(
                f"curl attempt {attempt}/{max_retries}: "
                f"{offset / 10**9:.2f}/{expected / 10**9:.2f} GB",
                flush=True,
            )
            result = subprocess.run(command, check=False)
            current_size = partial.stat().st_size if partial.exists() else 0
            if result.returncode == 0 and current_size == expected:
                break
            if result.returncode == 0:
                print(
                    f"curl returned success before expected size: "
                    f"{current_size} != {expected}",
                    flush=True,
                )
            print(
                f"curl exit {result.returncode}; preserving {partial} and retrying",
                flush=True,
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"curl exhausted retries; rerun to resume {partial}"
                )
            time.sleep(min(5 * attempt, 60))
    finally:
        if header_path and header_path.exists():
            header_path.unlink()

    actual = partial.stat().st_size
    if actual != expected:
        raise IOError(
            f"Downloaded size mismatch for {repo_file.rfilename}: {actual} != {expected}"
        )
    os.replace(partial, destination)
    print(f"Complete: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-retries", type=int, default=100)
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    records = []

    for index, item in enumerate(models_from_suite(args.suite), 1):
        print(f"\n[{index}] {item['model_id']}", flush=True)
        info = api.model_info(
            item["model_id"], revision=item["revision"], files_metadata=True
        )
        revision = info.sha
        weight_format, weight_files = select_weight_files(info)
        target = output_root / portable_model_directory_name(item["model_id"])
        target.mkdir(parents=True, exist_ok=True)

        # Fetch only small configuration/tokenizer files with the Hub client.
        snapshot_download(
            repo_id=item["model_id"],
            revision=revision,
            local_dir=str(target),
            allow_patterns=PORTABLE_ALLOW_PATTERNS,
            ignore_patterns=PORTABLE_IGNORE_PATTERNS
            + ["*.safetensors", "pytorch_model*.bin"],
        )
        for repo_file in weight_files:
            download_with_ranges(
                item["model_id"], revision, repo_file, target, args.max_retries
            )

        model_manifest = {
            "format": "language-hubness-portable-model-v1",
            "model_id": item["model_id"],
            "configured_revision": item["revision"],
            "resolved_revision": revision,
            "weight_format": weight_format,
        }
        (target / ".lht_model_manifest.json").write_text(
            json.dumps(model_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        inventory = portable_inventory(target)
        records.append({
            **item,
            **model_manifest,
            "local_path": str(target),
            **inventory,
        })
        print(
            f"Verified {item['model_id']}: "
            f"{inventory['weight_bytes'] / 10**9:.2f} GB weights",
            flush=True,
        )

    root_manifest = {
        "format": "language-hubness-portable-model-root-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": str(Path(args.suite).resolve()),
        "models": records,
        "server_environment": {"LHT_MODEL_ROOT": str(output_root)},
    }
    (output_root / "portable_models_manifest.json").write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("ALL_MODEL_DOWNLOADS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
