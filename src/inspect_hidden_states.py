import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dirs, load_config, representation_file_map


def parse_int_list(value):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Display the sentence, token audit, and vector summary behind saved hidden-state rows."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--rows", default=None, help="Comma-separated metadata row indices; default samples each language.")
    parser.add_argument("--languages", default=None, help="Optional comma-separated language filter.")
    parser.add_argument("--layers", default=None, help="Comma-separated layers; default first, middle, and final.")
    parser.add_argument("--representations", default=None, help="Comma-separated representation names.")
    parser.add_argument("--per-language", type=int, default=2)
    parser.add_argument("--vector-head", type=int, default=6)
    parser.add_argument("--show-token-sequence", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)
    hidden_dir = Path(paths["hidden"])
    metadata = pd.read_csv(hidden_dir / "metadata.csv")
    index_path = hidden_dir / "hidden_state_sentence_index.jsonl"
    sentence_index = {}
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                sentence_index[int(record["row_idx"])] = record

    requested_rows = parse_int_list(args.rows)
    languages = set(args.languages.split(",")) if args.languages else None
    candidates = metadata
    if languages:
        candidates = candidates[candidates.lang.astype(str).isin(languages)]
    if requested_rows is None:
        selected = candidates.groupby("lang", sort=True).head(args.per_language)
    else:
        selected = candidates[candidates.row_idx.astype(int).isin(requested_rows)]
    if selected.empty:
        raise ValueError("No metadata rows matched the requested filters")

    file_map = representation_file_map()
    requested_representations = (
        [item.strip() for item in args.representations.split(",") if item.strip()]
        if args.representations
        else cfg.get("metrics", {}).get("representations", ["last_token"])
    )
    arrays = {}
    for name in requested_representations:
        if name not in file_map:
            raise ValueError(f"Unknown representation: {name}")
        path = hidden_dir / file_map[name]
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")
    if not arrays:
        raise FileNotFoundError("None of the requested representation files exists")

    n_layers = next(iter(arrays.values())).shape[1]
    layers = parse_int_list(args.layers)
    if layers is None:
        layers = sorted(set([0, n_layers // 2, n_layers - 1]))
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if invalid_layers:
        raise ValueError(f"Layer indices out of range 0..{n_layers - 1}: {invalid_layers}")

    print("=== Hidden-state sentence inspector ===")
    print(f"Representations: {list(arrays)} | layers: {layers} | rows: {len(selected)}")
    for _, row in selected.sort_values("row_idx").iterrows():
        row_idx = int(row.row_idx)
        print("\n" + "-" * 88)
        print(f"row={row_idx} id={row.id} lang={row.lang}")
        print(f"sentence: {row.text}")
        print(
            f"terminal={getattr(row, 'terminal_char', '')!r} "
            f"original_last={getattr(row, 'last_token_decoded', row.last_token)!r} "
            f"last_content={getattr(row, 'last_content_token_decoded', '')!r} "
            f"sentinel={getattr(row, 'shared_sentinel_decoded', '')!r}"
        )
        record = sentence_index.get(row_idx)
        if args.show_token_sequence and record:
            for position, (token_id, token, decoded) in enumerate(
                zip(record["token_ids"], record["tokens"], record["decoded_tokens"])
            ):
                markers = []
                if position == int(record["last_token_position"]):
                    markers.append("ORIGINAL_LAST")
                if position == int(record["last_content_position"]):
                    markers.append("LAST_CONTENT")
                sentinel_position = record.get("shared_sentinel_position")
                if sentinel_position != "" and position == int(sentinel_position):
                    markers.append("SHARED_SENTINEL")
                print(
                    f"  token[{position:03d}] id={token_id:<8} raw={token!r:<22} "
                    f"decoded={decoded!r} {' '.join(markers)}"
                )
        for name, values in arrays.items():
            for layer in layers:
                vector = np.asarray(values[row_idx, layer], dtype=np.float64)
                head = np.array2string(vector[: args.vector_head], precision=5, separator=", ")
                print(
                    f"  {name:>20} layer={layer:02d} norm={np.linalg.norm(vector):.6f} "
                    f"mean={vector.mean():.6f} std={vector.std():.6f} head={head}"
                )


if __name__ == "__main__":
    main()
