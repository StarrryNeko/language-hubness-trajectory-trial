"""Safely expose archived model outputs at the paths expected by a suite."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from common import load_config


def import_outputs(source_root, suite_path, mode):
    source_root = Path(source_root).resolve()
    suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    records = []
    for relative in suite["configs"]:
        cfg = load_config(suite_path.parent / relative)
        destination = Path(cfg["output_dir"]).resolve()
        source = source_root / destination.name
        if not source.is_dir():
            records.append({"source": str(source), "destination": str(destination), "status": "SOURCE_MISSING"})
            continue
        if destination.exists():
            if destination.resolve() == source.resolve():
                records.append({"source": str(source), "destination": str(destination), "status": "ALREADY_SAME"})
                continue
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite non-empty destination: {destination}"
                )
            destination.rmdir()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            os.symlink(source, destination, target_is_directory=True)
        elif mode == "copy":
            shutil.copytree(source, destination)
        else:
            raise ValueError("mode must be symlink or copy")
        records.append({"source": str(source), "destination": str(destination), "status": mode.upper()})
    return records


def main():
    parser = argparse.ArgumentParser(description="Import archived outputs without overwriting data")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()
    records = import_outputs(args.source_root, args.suite, args.mode)
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

