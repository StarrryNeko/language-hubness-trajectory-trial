import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    ensure_dirs,
    load_config,
    read_jsonl,
    representation_file_map,
    set_seed,
    write_jsonl,
)


def get_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def masked_mean(hidden, attention_mask):
    # hidden: [seq, dim], attention_mask: [seq]
    mask = attention_mask.to(hidden.device).to(hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=0).clamp_min(1.0)
    return (hidden * mask).sum(dim=0) / denom


def decoded_token(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def is_content_token(tokenizer, token_id):
    """Return True for tokens containing a Unicode letter, number, or mark."""
    if int(token_id) in set(tokenizer.all_special_ids):
        return False
    piece = decoded_token(tokenizer, token_id).strip()
    return bool(piece) and any(unicodedata.category(ch)[0] in {"L", "N", "M"} for ch in piece)


def select_last_content_position(tokenizer, token_ids):
    for position in range(len(token_ids) - 1, -1, -1):
        if is_content_token(tokenizer, token_ids[position]):
            return position
    return len(token_ids) - 1


def terminal_character(text):
    stripped = text.rstrip()
    return stripped[-1] if stripped else ""


def print_sentence_audit(meta_rows, mode, preview_per_language):
    if mode == "none":
        return
    if mode == "all":
        selected = meta_rows
    else:
        counts = {}
        selected = []
        for row in meta_rows:
            lang = row["lang"]
            counts.setdefault(lang, 0)
            if counts[lang] < preview_per_language:
                selected.append(row)
                counts[lang] += 1

    print("\n=== Sentence-to-hidden-state audit ===")
    print(f"Display mode: {mode} | rows shown: {len(selected)}/{len(meta_rows)}")
    for row in selected:
        print(
            f"row={row['row_idx']:04d} id={row['id']} lang={row['lang']} "
            f"terminal={row['terminal_char']!r} original_last={row['last_token_decoded']!r} "
            f"last_content={row['last_content_token_decoded']!r} "
            f"sentinel={row['shared_sentinel_decoded']!r}"
        )
        print(f"  sentence: {row['text']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--show-sentences",
        choices=["none", "preview", "all"],
        default=None,
        help="Show the sentence and terminal-token audit for no rows, a preview, or every hidden-state row.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    paths = ensure_dirs(cfg)

    data_path = Path(paths["data"]) / "parallel_samples.jsonl"
    rows = read_jsonl(str(data_path))

    model_name = cfg["model_name_or_path"]
    dtype = get_dtype(cfg.get("dtype", "float16"))
    device = cfg.get("device", "cuda")
    max_length = int(cfg.get("max_length", 128))
    controls = cfg.get("representation_controls", {})
    append_shared_sentinel = bool(controls.get("append_shared_sentinel", True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentinel_token_id = controls.get("shared_sentinel_token_id")
    if sentinel_token_id is None:
        sentinel_token_id = tokenizer.eos_token_id
    if append_shared_sentinel and sentinel_token_id is None:
        raise ValueError("A shared sentinel was requested, but the tokenizer has no EOS/sentinel token id.")
    sentinel_token_id = int(sentinel_token_id) if sentinel_token_id is not None else None
    text_max_length = max_length - 1 if append_shared_sentinel else max_length
    if text_max_length < 1:
        raise ValueError("max_length must leave at least one position for sentence text")

    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device == "auto":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    representation_vectors = {
        "last_token": [],
        "last_content_token": [],
        "mean_pool": [],
        "content_mean_pool": [],
    }
    if append_shared_sentinel:
        representation_vectors["shared_sentinel"] = []
    meta_rows = []
    sentence_index_rows = []

    token_export_cfg = cfg.get("token_export", {})
    token_export_enabled = bool(token_export_cfg.get("enabled", False))
    token_export_limit = int(token_export_cfg.get("max_samples_per_language", 5))
    token_export_counts = {}
    token_dir = Path(paths["hidden"]) / "token_samples"
    token_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for row_idx, row in enumerate(tqdm(rows, desc="Extracting hidden states")):
            original_encoded = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=text_max_length,
                padding=False,
            )
            original_ids = original_encoded["input_ids"][0].tolist()
            original_attention = original_encoded["attention_mask"][0]
            original_length = int(original_attention.sum().item())
            last_position = original_length - 1
            last_content_position = select_last_content_position(tokenizer, original_ids[:original_length])

            content_mask = torch.tensor(
                [int(is_content_token(tokenizer, token_id)) for token_id in original_ids],
                dtype=original_attention.dtype,
            )
            if int(content_mask.sum().item()) == 0:
                content_mask = original_attention.clone()

            if append_shared_sentinel:
                sentinel = torch.tensor([[sentinel_token_id]], dtype=original_encoded["input_ids"].dtype)
                original_encoded["input_ids"] = torch.cat([original_encoded["input_ids"], sentinel], dim=1)
                original_encoded["attention_mask"] = torch.cat(
                    [original_encoded["attention_mask"], torch.ones_like(sentinel)], dim=1
                )
                sentinel_position = original_length
                original_pool_mask = torch.cat([original_attention, torch.zeros(1, dtype=original_attention.dtype)])
                content_pool_mask = torch.cat([content_mask, torch.zeros(1, dtype=content_mask.dtype)])
            else:
                sentinel_position = None
                original_pool_mask = original_attention
                content_pool_mask = content_mask

            encoded = {key: value.to(model.device) for key, value in original_encoded.items()}
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            per_representation = {name: [] for name in representation_vectors}
            for layer_hidden in outputs.hidden_states:
                hidden = layer_hidden[0].detach()
                per_representation["last_token"].append(hidden[last_position].float().cpu().numpy())
                per_representation["last_content_token"].append(
                    hidden[last_content_position].float().cpu().numpy()
                )
                per_representation["mean_pool"].append(
                    masked_mean(hidden, original_pool_mask).float().cpu().numpy()
                )
                per_representation["content_mean_pool"].append(
                    masked_mean(hidden, content_pool_mask).float().cpu().numpy()
                )
                if append_shared_sentinel:
                    per_representation["shared_sentinel"].append(
                        hidden[sentinel_position].float().cpu().numpy()
                    )
            for name, layer_vectors in per_representation.items():
                representation_vectors[name].append(np.stack(layer_vectors, axis=0))

            full_token_count = len(tokenizer(row["text"], add_special_tokens=True)["input_ids"])
            last_token_id = int(original_ids[last_position])
            last_content_token_id = int(original_ids[last_content_position])
            terminal_char = terminal_character(row["text"])
            meta_row = {
                "row_idx": row_idx,
                "id": row["id"],
                "lang": row["lang"],
                "flores_lang": row.get("flores_lang", ""),
                "text": row["text"],
                "num_tokens": original_length,
                "model_num_tokens": int(encoded["attention_mask"].sum().item()),
                "full_num_tokens": int(full_token_count),
                "was_truncated": bool(full_token_count > text_max_length),
                "terminal_char": terminal_char,
                "terminal_unicode_category": unicodedata.category(terminal_char) if terminal_char else "",
                "ends_with_ascii_period": terminal_char == ".",
                "last_token_position": last_position,
                "last_token_id": last_token_id,
                "last_token": tokenizer.convert_ids_to_tokens([last_token_id])[0],
                "last_token_decoded": decoded_token(tokenizer, last_token_id),
                "last_token_is_content": bool(is_content_token(tokenizer, last_token_id)),
                "last_content_position": last_content_position,
                "last_content_token_id": last_content_token_id,
                "last_content_token": tokenizer.convert_ids_to_tokens([last_content_token_id])[0],
                "last_content_token_decoded": decoded_token(tokenizer, last_content_token_id),
                "shared_sentinel_position": sentinel_position if sentinel_position is not None else "",
                "shared_sentinel_token_id": sentinel_token_id if append_shared_sentinel else "",
                "shared_sentinel_decoded": (
                    decoded_token(tokenizer, sentinel_token_id) if append_shared_sentinel else ""
                ),
            }
            meta_rows.append(meta_row)

            all_model_ids = encoded["input_ids"][0].detach().cpu().tolist()
            sentence_index_rows.append(
                {
                    **meta_row,
                    "token_ids": all_model_ids,
                    "tokens": tokenizer.convert_ids_to_tokens(all_model_ids),
                    "decoded_tokens": [decoded_token(tokenizer, token_id) for token_id in all_model_ids],
                }
            )

            lang = row["lang"]
            token_export_counts.setdefault(lang, 0)
            if token_export_enabled and token_export_counts[lang] < token_export_limit:
                token_export_counts[lang] += 1
                token_layers = np.stack(
                    [layer[0].detach().float().cpu().numpy() for layer in outputs.hidden_states], axis=0
                )
                out_file = token_dir / f"{row_idx:05d}_{lang}.npz"
                np.savez_compressed(
                    out_file,
                    hidden=token_layers,
                    attention_mask=encoded["attention_mask"][0].detach().cpu().numpy(),
                    tokens=np.array(sentence_index_rows[-1]["tokens"], dtype=object),
                    meta=json.dumps(sentence_index_rows[-1], ensure_ascii=False),
                )

    file_map = representation_file_map()
    stacked = {}
    for name, vectors in representation_vectors.items():
        stacked[name] = np.stack(vectors, axis=0)
        np.save(Path(paths["hidden"]) / file_map[name], stacked[name])
    pd.DataFrame(meta_rows).to_csv(Path(paths["hidden"]) / "metadata.csv", index=False, encoding="utf-8")
    write_jsonl(Path(paths["hidden"]) / "hidden_state_sentence_index.jsonl", sentence_index_rows)

    audit_cfg = cfg.get("hidden_state_audit", {})
    display_mode = args.show_sentences or audit_cfg.get("display_sentences", "preview")
    preview_per_language = int(audit_cfg.get("preview_per_language", 3))
    print_sentence_audit(meta_rows, display_mode, preview_per_language)

    truncated = sum(row["was_truncated"] for row in meta_rows)
    elapsed = time.perf_counter() - started_at
    peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    first_shape = next(iter(stacked.values())).shape
    print("\n=== Hidden-state extraction summary ===")
    print(f"Rows: {len(meta_rows)} | layers: {first_shape[1]} | dim: {first_shape[2]}")
    for name, values in stacked.items():
        print(f"Saved {name:>20}: {values.shape} -> {file_map[name]}")
    print(f"Truncated inputs: {truncated}/{len(meta_rows)}")
    print(f"Extraction time: {elapsed:.1f}s | peak allocated GPU memory: {peak_gpu_gb:.2f} GiB")
    if truncated:
        print("WARNING: truncated inputs can invalidate parallel-sentence comparisons.")
    print(f"Saved metadata to {Path(paths['hidden']) / 'metadata.csv'}")
    print(f"Saved sentence index to {Path(paths['hidden']) / 'hidden_state_sentence_index.jsonl'}")
    extraction_manifest = {
        "model": model_name,
        "tokenizer": getattr(tokenizer, "name_or_path", model_name),
        "rows": len(meta_rows),
        "layers": int(first_shape[1]),
        "hidden_dim": int(first_shape[2]),
        "representations": list(stacked),
        "representation_files": {name: file_map[name] for name in stacked},
        "shared_sentinel_enabled": append_shared_sentinel,
        "shared_sentinel_token_id": sentinel_token_id if append_shared_sentinel else None,
        "shared_sentinel_decoded": (
            decoded_token(tokenizer, sentinel_token_id) if append_shared_sentinel else None
        ),
        "text_max_length": text_max_length,
        "model_max_length": max_length,
        "truncated_inputs": int(truncated),
        "elapsed_seconds": elapsed,
        "peak_allocated_gpu_gib": peak_gpu_gb,
        "torch_version": torch.__version__,
    }
    (Path(paths["output"]) / "extraction_manifest.json").write_text(
        json.dumps(extraction_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
