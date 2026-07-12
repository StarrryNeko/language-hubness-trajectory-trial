import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ensure_dirs, load_config, read_jsonl, set_seed


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    paths = ensure_dirs(cfg)

    data_path = Path(paths["data"]) / "parallel_samples.jsonl"
    rows = read_jsonl(str(data_path))

    model_name = cfg["model_name_or_path"]
    dtype = get_dtype(cfg.get("dtype", "float16"))
    device = cfg.get("device", "cuda")
    max_length = int(cfg.get("max_length", 128))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    sentence_last_vectors = []
    sentence_mean_vectors = []
    meta_rows = []

    token_export_cfg = cfg.get("token_export", {})
    token_export_enabled = bool(token_export_cfg.get("enabled", False))
    token_export_limit = int(token_export_cfg.get("max_samples_per_language", 5))
    token_export_counts = {}
    token_dir = Path(paths["hidden"]) / "token_samples"
    token_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for row_idx, row in enumerate(tqdm(rows, desc="Extracting hidden states")):
            encoded = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            attention_mask = encoded["attention_mask"][0]
            full_token_count = len(tokenizer(row["text"], add_special_tokens=True)["input_ids"])
            valid_positions = torch.nonzero(attention_mask, as_tuple=False).flatten()
            last_position = int(valid_positions[-1].item())
            layer_means = []
            layer_lasts = []
            for layer_hidden in outputs.hidden_states:
                h = layer_hidden[0].detach()
                layer_means.append(masked_mean(h, attention_mask).float().cpu().numpy())
                layer_lasts.append(h[last_position].float().cpu().numpy())
            layer_means = np.stack(layer_means, axis=0)
            layer_lasts = np.stack(layer_lasts, axis=0)
            sentence_mean_vectors.append(layer_means)
            sentence_last_vectors.append(layer_lasts)

            meta_rows.append(
                {
                    "row_idx": row_idx,
                    "id": row["id"],
                    "lang": row["lang"],
                    "flores_lang": row.get("flores_lang", ""),
                    "text": row["text"],
                    "num_tokens": int(attention_mask.sum().item()),
                    "full_num_tokens": int(full_token_count),
                    "was_truncated": bool(full_token_count > max_length),
                    "last_token": tokenizer.convert_ids_to_tokens(
                        [int(encoded["input_ids"][0, last_position].item())]
                    )[0],
                }
            )

            lang = row["lang"]
            token_export_counts.setdefault(lang, 0)
            if token_export_enabled and token_export_counts[lang] < token_export_limit:
                token_export_counts[lang] += 1
                token_layers = []
                for layer_hidden in outputs.hidden_states:
                    token_layers.append(layer_hidden[0].detach().float().cpu().numpy())
                token_layers = np.stack(token_layers, axis=0)
                tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())
                out_file = token_dir / f"{row_idx:05d}_{lang}.npz"
                np.savez_compressed(
                    out_file,
                    hidden=token_layers,
                    attention_mask=attention_mask.detach().cpu().numpy(),
                    tokens=np.array(tokens, dtype=object),
                    meta=json.dumps(row, ensure_ascii=False),
                )

    sentence_last_vectors = np.stack(sentence_last_vectors, axis=0)
    sentence_mean_vectors = np.stack(sentence_mean_vectors, axis=0)
    np.save(Path(paths["hidden"]) / "sentence_layer_last_token.npy", sentence_last_vectors)
    np.save(Path(paths["hidden"]) / "sentence_layer_mean_pool.npy", sentence_mean_vectors)
    pd.DataFrame(meta_rows).to_csv(Path(paths["hidden"]) / "metadata.csv", index=False, encoding="utf-8")

    truncated = sum(row["was_truncated"] for row in meta_rows)
    print("\n=== Hidden-state extraction summary ===")
    print(f"Rows: {len(meta_rows)} | layers: {sentence_last_vectors.shape[1]} | dim: {sentence_last_vectors.shape[2]}")
    print(f"Saved last-token vectors: {sentence_last_vectors.shape}")
    print(f"Saved mean-pool vectors:  {sentence_mean_vectors.shape}")
    print(f"Truncated inputs: {truncated}/{len(meta_rows)}")
    if truncated:
        print("WARNING: truncated inputs can invalidate parallel-sentence comparisons.")
    print(f"Saved metadata to {Path(paths['hidden']) / 'metadata.csv'}")


if __name__ == "__main__":
    main()
