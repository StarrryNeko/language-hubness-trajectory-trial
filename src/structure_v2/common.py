"""Shared contracts for the geometry-only structure_v2 pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from common import json_dumps_strict, write_json
from structure_v2 import PROTOCOL_VERSION, REPRESENTATION_PROTOCOL


@dataclass(frozen=True)
class StructurePaths:
    root: Path
    geometry: Path
    figures: Path
    validation: Path


def paths(cfg):
    root = Path(cfg["output_dir"]) / "structure_v2"
    result = StructurePaths(root, root / "geometry", root / "figures", root / "validation")
    for value in result.__dict__.values():
        value.mkdir(parents=True, exist_ok=True)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value):
    return hashlib.sha256(
        json_dumps_strict(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_manifest(path, values):
    payload = {"protocol_version": PROTOCOL_VERSION, **values}
    write_json(path, payload)
    return payload


def settings(cfg):
    raw = cfg.get("structure_v2", {})
    if not raw.get("enabled", False):
        raise ValueError("structure_v2.enabled must be true")
    if cfg.get("metrics", {}).get("representations") != ["mean_pool"]:
        raise ValueError("structure_v2 permits only mean_pool")
    languages = list(cfg["dataset"]["languages"])
    metadata = cfg["dataset"]["language_metadata"]
    if set(languages) != set(metadata) or len(languages) < 20:
        raise ValueError("structure_v2 requires complete full-language metadata")
    latin = [lang for lang in languages if metadata[lang]["script"] == "Latin"]
    non_latin = [lang for lang in languages if lang not in latin]
    if "en" not in latin:
        raise ValueError("English must be a Latin-script language")
    primary = int(raw["primary_layer"])
    layers = sorted(set(map(int, raw.get("analysis_layers", [primary]))))
    if primary not in layers:
        raise ValueError("primary_layer must be present in analysis_layers")
    ks = sorted(set(map(int, raw.get("k_sensitivity", [3, 5, 10]))))
    local_k = int(raw.get("local_scaling_k", 5))
    if local_k not in ks:
        raise ValueError("local_scaling_k must be included in k_sensitivity")
    return {
        "representation_protocol": REPRESENTATION_PROTOCOL,
        "languages": languages,
        "metadata": metadata,
        "latin": latin,
        "other_latin": [lang for lang in latin if lang != "en"],
        "non_latin": non_latin,
        "primary_layer": primary,
        "analysis_layers": layers,
        "k_sensitivity": ks,
        "local_scaling_k": local_k,
        "bootstrap_samples": int(raw.get("bootstrap_samples", 1000)),
        "confidence_level": float(raw.get("confidence_level", 0.95)),
        "seed": int(raw.get("seed", 20260809)),
        "resources": dict(raw.get("resources", {})),
    }
