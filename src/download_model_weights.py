"""Download every model referenced by an experiment suite without running inference."""

import argparse
import json
import os
from pathlib import Path

# Keep downloads on the regular HTTP path used by the extraction pipeline.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from common import load_config


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
    if args.dry_run:
        return

    for index, item in enumerate(models, 1):
        print(f"\n[{index}/{len(models)}] Downloading {item['model_id']}")
        local_path = snapshot_download(
            repo_id=item["model_id"],
            revision=item["revision"],
            cache_dir=item["cache_dir"],
            resume_download=True,
        )
        print(f"Complete: {item['model_id']} -> {local_path}")

    print("\nALL_MODEL_DOWNLOADS_COMPLETE")


if __name__ == "__main__":
    main()
