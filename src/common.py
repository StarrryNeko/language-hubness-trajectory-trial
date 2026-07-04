import json
import os
import random
from pathlib import Path

import numpy as np


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def ensure_dirs(cfg):
    output_dir = Path(cfg["output_dir"])
    paths = {
        "output": output_dir,
        "data": output_dir / "data",
        "hidden": output_dir / "hidden",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

