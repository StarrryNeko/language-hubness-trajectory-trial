"""Download every model referenced by an experiment suite without running inference."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Keep downloads on the regular HTTP path used by the extraction pipeline.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download

from common import load_config, portable_model_directory_name


PORTABLE_ALLOW_PATTERNS = [
    "*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin",
    "*.json", "*.py", "*.model", "*.tiktoken", "tokenizer.*",
    "tokenizer*", "vocab.*", "merges.txt", "added_tokens.json",
]
PORTABLE_IGNORE_PATTERNS = [
    "*.h5", "*.msgpack", "*.onnx", "*.ot", "*.ckpt", "original/*",
    "onnx/*", "flax_model*", "tf_model*",
]


def portable_inventory(directory):
    directory = Path(directory)
    files = [path for path in directory.rglob("*") if path.is_file()]
    weight_files = [
        path for path in files
        if path.suffix == ".safetensors" or path.name.startswith("pytorch_model")
    ]
    if not weight_files:
        raise FileNotFoundError(f"No model weight files found in {directory}")
    if not (directory / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {directory}")
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "files": [str(path.relative_to(directory)).replace("\\", "/") for path in files],
    }


def models_from_suite(suite_path):
    suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    models = []
    seen = set()
    for relative_config in suite["configs"]:
        config_path = suite_path.parent / relative_config
        cfg = load_config(config_path)
        model_cfg = cfg.get("model", {})
        model_id = model_cfg.get("name_or_path", cfg.get("model_name_or_path"))
        if not model_id:
            raise ValueError(f"Model ID is missing in {config_path}")
        key = (model_id, model_cfg.get("revision"))
        if key in seen:
            continue
        seen.add(key)
        models.append({
            "model_id": model_id,
            "revision": model_cfg.get("revision"),
            "cache_dir": model_cfg.get(
                "cache_dir", cfg.get("huggingface_cache_dir")
            ),
            "config_path": str(config_path),
        })
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights/tokenizers for a configured suite."
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Download only this exact Hugging Face model ID; repeat for several.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download plan without downloading.",
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Create portable model folders below this directory instead of relying on "
            "the Hugging Face cache. Upload this directory and set LHT_MODEL_ROOT on the server."
        ),
    )
    parser.add_argument(
        "--verify-root",
        help="Only validate an already downloaded portable model root; do not use the network.",
    )
    args = parser.parse_args()

    models = models_from_suite(args.suite)
    selected = set(args.model)
    if selected:
        models = [item for item in models if item["model_id"] in selected]
        missing = selected - {item["model_id"] for item in models}
        if missing:
            raise ValueError(
                f"Requested models are not present in the suite: {sorted(missing)}"
            )
    if not models:
        raise ValueError("No models selected for download")

    print(f"Resolved {len(models)} model repositories:")
    for index, item in enumerate(models, 1):
        print(
            f"{index}. {item['model_id']} "
            f"revision={item['revision'] or 'main'} "
            f"cache={item['cache_dir'] or 'Hugging Face default'}"
        )
    if args.verify_root:
        root = Path(args.verify_root).expanduser().resolve()
        records = []
        for item in models:
            target = root / portable_model_directory_name(item["model_id"])
            inventory = portable_inventory(target)
            records.append({**item, "local_path": str(target), **inventory})
            print(
                f"Verified {item['model_id']}: {inventory['weight_file_count']} weight files, "
                f"{inventory['total_bytes'] / 10**9:.2f} GB"
            )
        print("PORTABLE_MODEL_ROOT_VERIFIED")
        return
    if args.dry_run:
        return

    portable_root = Path(args.output_root).expanduser().resolve() if args.output_root else None
    if portable_root:
        portable_root.mkdir(parents=True, exist_ok=True)
    records = []
    api = HfApi()
    for index, item in enumerate(models, 1):
        print(f"\n[{index}/{len(models)}] Downloading {item['model_id']}")
        resolved_revision = api.model_info(
            item["model_id"], revision=item["revision"]
        ).sha
        if not resolved_revision:
            raise ValueError(f"Could not resolve immutable revision for {item['model_id']}")
        kwargs = {
            "repo_id": item["model_id"],
            "revision": resolved_revision,
        }
        if portable_root:
            target = portable_root / portable_model_directory_name(item["model_id"])
            target.mkdir(parents=True, exist_ok=True)
            kwargs.update({
                "local_dir": str(target),
                "allow_patterns": PORTABLE_ALLOW_PATTERNS,
                "ignore_patterns": PORTABLE_IGNORE_PATTERNS,
            })
        else:
            kwargs["cache_dir"] = item["cache_dir"]
        local_path = snapshot_download(**kwargs)
        print(f"Complete: {item['model_id']} -> {local_path}")
        if portable_root:
            (Path(local_path) / ".lht_model_manifest.json").write_text(
                json.dumps({
                    "format": "language-hubness-portable-model-v1",
                    "model_id": item["model_id"],
                    "configured_revision": item["revision"],
                    "resolved_revision": resolved_revision,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        record = {
            **item,
            "configured_revision": item["revision"],
            "resolved_revision": resolved_revision,
            "local_path": str(Path(local_path).resolve()),
        }
        if portable_root:
            record.update(portable_inventory(local_path))
        records.append(record)

    if portable_root:
        manifest = {
            "format": "language-hubness-portable-model-root-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "suite": str(Path(args.suite).resolve()),
            "models": records,
            "server_environment": {"LHT_MODEL_ROOT": str(portable_root)},
        }
        (portable_root / "portable_models_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Portable manifest: {portable_root / 'portable_models_manifest.json'}")

    print("\nALL_MODEL_DOWNLOADS_COMPLETE")


if __name__ == "__main__":
    main()
