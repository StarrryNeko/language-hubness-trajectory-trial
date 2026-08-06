"""Record immutable content hashes for local checkpoints without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from common import load_config, resolve_model_source, write_json


WEIGHT_SUFFIXES = (".safetensors", ".bin")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_inventory(directory):
    root = Path(directory).resolve()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No .safetensors/.bin checkpoint files in {root}")
    records = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        records.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
    return {
        "algorithm": "sha256",
        "checkpoint_sha256": aggregate.hexdigest(),
        "weight_file_count": len(records),
        "weight_bytes": sum(item["bytes"] for item in records),
        "weight_files": records,
    }


def audit_config(config_path, update_extraction_manifest=True):
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("name_or_path", cfg.get("model_name_or_path"))
    canonical, source, is_local = resolve_model_source(
        model_id,
        explicit_local_path=model_cfg.get("local_path"),
        model_root=model_cfg.get("model_root"),
    )
    if not is_local:
        raise ValueError(
            f"Checkpoint audit requires a local directory for {canonical}; "
            "set LHT_MODEL_ROOT or model.local_path"
        )
    portable_manifest = Path(source) / ".lht_model_manifest.json"
    portable = (
        json.loads(portable_manifest.read_text(encoding="utf-8"))
        if portable_manifest.exists() else {}
    )
    inventory = checkpoint_inventory(source)
    payload = {
        "format": "language-hubness-checkpoint-identity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": canonical,
        "configured_revision": model_cfg.get("revision"),
        "resolved_revision": portable.get("resolved_revision"),
        "local_source": source,
        **inventory,
    }
    output = Path(cfg["output_dir"]) / "checkpoint_identity.json"
    write_json(output, payload)
    extraction_path = Path(cfg["output_dir"]) / "extraction_manifest.json"
    if update_extraction_manifest and extraction_path.exists():
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        if extraction.get("model") != canonical:
            raise ValueError(
                f"Extraction model mismatch: {extraction.get('model')} != {canonical}"
            )
        extraction["checkpoint_identity_sha256"] = inventory["checkpoint_sha256"]
        extraction["checkpoint_identity_file"] = output.name
        if not extraction.get("resolved_revision") and portable.get("resolved_revision"):
            extraction["resolved_revision"] = portable["resolved_revision"]
        write_json(extraction_path, extraction)
    return output, payload


def main():
    parser = argparse.ArgumentParser(
        description="Hash local checkpoint files and attach their identity to extraction manifests"
    )
    parser.add_argument("--config")
    parser.add_argument("--suite")
    parser.add_argument("--no-update-manifest", action="store_true")
    args = parser.parse_args()
    if bool(args.config) == bool(args.suite):
        parser.error("provide exactly one of --config or --suite")
    if args.config:
        configs = [Path(args.config).resolve()]
    else:
        suite_path = Path(args.suite).resolve()
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        configs = [suite_path.parent / item for item in suite["configs"]]
    for config in configs:
        output, payload = audit_config(config, not args.no_update_manifest)
        print(
            f"Audited {payload['model_id']}: {payload['weight_file_count']} files, "
            f"{payload['weight_bytes'] / 10**9:.2f} GB -> {output}"
        )


if __name__ == "__main__":
    main()
