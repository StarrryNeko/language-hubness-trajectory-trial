import argparse
import json
import os
import time
from pathlib import Path

# AutoDL containers can receive transient CAS 401/OOM failures from hf-xet.
# This must be set before importing transformers/huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    configured_representations,
    ensure_dirs,
    load_config,
    read_jsonl,
    representation_file_map,
    set_seed,
    validate_language_inventory,
    write_jsonl,
)


def get_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def masked_mean(hidden, attention_mask):
    """Mean over original non-padding sentence positions, excluding appended EOS."""
    mask = attention_mask.to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=0) / mask.sum().clamp_min(1.0)


def decoded_token(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def model_settings(cfg):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name_or_path", cfg.get("model_name_or_path"))
    if not name:
        raise ValueError("Set model.name_or_path (or legacy model_name_or_path)")
    return {
        "name": name,
        "tokenizer": model_cfg.get("tokenizer_name_or_path", name),
        "revision": model_cfg.get("revision"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "attn_implementation": model_cfg.get("attn_implementation"),
        "cache_dir": model_cfg.get("cache_dir", cfg.get("huggingface_cache_dir")),
    }


def print_sentence_audit(meta_rows, mode, preview_per_language):
    if mode == "none":
        return
    selected = meta_rows if mode == "all" else []
    if mode != "all":
        counts = {}
        for row in meta_rows:
            counts.setdefault(row["lang"], 0)
            if counts[row["lang"]] < preview_per_language:
                selected.append(row)
                counts[row["lang"]] += 1
    print("\n=== Sentence-to-vector audit ===")
    for row in selected:
        print(
            f"row={row['row_idx']:05d} id={row['id']} lang={row['lang']} "
            f"sentence_tokens={row['sentence_num_tokens']} eos={row['sentinel_eos_decoded']!r}"
        )
        print(f"  sentence: {row['text']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--show-sentences", choices=["none", "preview", "all"], default=None)
    args = parser.parse_args()

    started_at = time.perf_counter()
    cfg = load_config(args.config)
    validate_language_inventory(cfg)
    representations = configured_representations(cfg)
    set_seed(cfg.get("seed", 42))
    paths = ensure_dirs(cfg)
    rows = read_jsonl(Path(paths["data"]) / "parallel_samples.jsonl")

    settings = model_settings(cfg)
    dtype = get_dtype(cfg.get("dtype", "float16"))
    device = cfg.get("device", "cuda")
    max_length = int(cfg.get("max_length", 128))
    if max_length < 2:
        raise ValueError("max_length must leave room for sentence text and one sentinel EOS")

    tokenizer_kwargs = {"trust_remote_code": settings["trust_remote_code"]}
    if settings["cache_dir"]:
        tokenizer_kwargs["cache_dir"] = settings["cache_dir"]
    if settings["revision"]:
        tokenizer_kwargs["revision"] = settings["revision"]
    tokenizer = AutoTokenizer.from_pretrained(settings["tokenizer"], **tokenizer_kwargs)
    eos_token_id = cfg.get("representation_controls", {}).get("sentinel_eos_token_id")
    eos_token_id = tokenizer.eos_token_id if eos_token_id is None else int(eos_token_id)
    if eos_token_id is None:
        raise ValueError("sentinel_eos requires a tokenizer EOS token or sentinel_eos_token_id override")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = eos_token_id
    controls = cfg.get("representation_controls", {})
    prepend_bos = bool(controls.get("prepend_bos_when_available", True)) and tokenizer.bos_token_id is not None
    reserved_positions = 1 + int(prepend_bos)
    text_max_length = max_length - reserved_positions
    if text_max_length < 1:
        raise ValueError("max_length must leave space for text plus BOS/EOS controls")

    load_kwargs = {
        "dtype": dtype,
        "trust_remote_code": settings["trust_remote_code"],
    }
    if settings["cache_dir"]:
        load_kwargs["cache_dir"] = settings["cache_dir"]
    if settings["revision"]:
        load_kwargs["revision"] = settings["revision"]
    if settings["attn_implementation"]:
        load_kwargs["attn_implementation"] = settings["attn_implementation"]
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(settings["name"], **load_kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    storage_dtype_name = cfg.get("storage_dtype", "float16")
    if storage_dtype_name not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    storage_dtype = np.float16 if storage_dtype_name == "float16" else np.float32
    vectors = {name: [] for name in representations}
    meta_rows = []
    index_rows = []

    with torch.inference_mode():
        for row_idx, row in enumerate(tqdm(rows, desc="Extracting mean-pool + EOS vectors")):
            encoded = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=text_max_length,
                padding=False,
                add_special_tokens=False,
            )
            text_ids = encoded["input_ids"]
            text_mask = encoded["attention_mask"]
            text_length = int(text_mask[0].sum().item())
            if text_length == 0:
                raise ValueError(f"Tokenizer produced no tokens for row {row_idx}")
            full_length = len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            prefix = (
                torch.tensor([[tokenizer.bos_token_id]], dtype=text_ids.dtype)
                if prepend_bos else torch.empty((1, 0), dtype=text_ids.dtype)
            )
            eos = torch.tensor([[eos_token_id]], dtype=text_ids.dtype)
            model_ids = torch.cat([prefix, text_ids, eos], dim=1)
            model_inputs = {
                "input_ids": model_ids,
                "attention_mask": torch.ones_like(model_ids),
            }
            model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            per_rep = {name: [] for name in representations}
            for layer_hidden in outputs.hidden_states:
                hidden = layer_hidden[0].detach()
                if "mean_pool" in per_rep:
                    per_rep["mean_pool"].append(
                        masked_mean(
                            hidden[int(prepend_bos):int(prepend_bos) + text_length], text_mask[0]
                        ).float().cpu().numpy()
                    )
                if "sentinel_eos" in per_rep:
                    per_rep["sentinel_eos"].append(
                        hidden[int(prepend_bos) + text_length].float().cpu().numpy()
                    )
            for name, layer_vectors in per_rep.items():
                vectors[name].append(np.stack(layer_vectors).astype(storage_dtype, copy=False))

            token_ids = model_inputs["input_ids"][0].detach().cpu().tolist()
            meta = {
                "row_idx": row_idx,
                "id": str(row["id"]),
                "lang": str(row["lang"]),
                "flores_lang": row.get("flores_lang", ""),
                "text": row["text"],
                "sentence_num_tokens": text_length,
                "model_num_tokens": text_length + reserved_positions,
                "full_num_tokens": full_length,
                "was_truncated": bool(full_length > text_max_length),
                "prepended_bos": prepend_bos,
                "bos_token_id": int(tokenizer.bos_token_id) if prepend_bos else "",
                "sentinel_eos_position": int(prepend_bos) + text_length,
                "sentinel_eos_token_id": int(eos_token_id),
                "sentinel_eos_decoded": decoded_token(tokenizer, eos_token_id),
            }
            meta_rows.append(meta)
            index_rows.append({
                **meta,
                "token_ids": token_ids,
                "tokens": tokenizer.convert_ids_to_tokens(token_ids),
                "decoded_tokens": [decoded_token(tokenizer, token_id) for token_id in token_ids],
            })

    file_map = representation_file_map()
    stacked = {}
    for name, values in vectors.items():
        stacked[name] = np.stack(values)
        np.save(Path(paths["hidden"]) / file_map[name], stacked[name])
    pd.DataFrame(meta_rows).to_csv(Path(paths["hidden"]) / "metadata.csv", index=False, encoding="utf-8")
    write_jsonl(Path(paths["hidden"]) / "hidden_state_sentence_index.jsonl", index_rows)

    audit_cfg = cfg.get("hidden_state_audit", {})
    print_sentence_audit(
        meta_rows,
        args.show_sentences or audit_cfg.get("display_sentences", "preview"),
        int(audit_cfg.get("preview_per_language", 1)),
    )
    elapsed = time.perf_counter() - started_at
    first_shape = next(iter(stacked.values())).shape
    truncated = sum(row["was_truncated"] for row in meta_rows)
    peak_gpu = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    manifest = {
        "model": settings["name"],
        "tokenizer": settings["tokenizer"],
        "huggingface_cache_dir": settings["cache_dir"],
        "rows": len(meta_rows),
        "semantic_groups": len({row["id"] for row in meta_rows}),
        "languages": len({row["lang"] for row in meta_rows}),
        "layers": int(first_shape[1]),
        "hidden_dim": int(first_shape[2]),
        "representations": representations,
        "primary_representation": "mean_pool",
        "validation_representation": cfg["metrics"].get("validation_representation", "sentinel_eos"),
        "sentinel_eos_token_id": int(eos_token_id),
        "sentinel_eos_decoded": decoded_token(tokenizer, eos_token_id),
        "prepend_bos_when_available": prepend_bos,
        "text_tokenization_add_special_tokens": False,
        "text_max_length": text_max_length,
        "mean_pool_excludes_appended_eos": True,
        "mean_pool_excludes_prepended_bos": True,
        "candidate_scope": "same_semantic_id_only",
        "storage_dtype": storage_dtype_name,
        "truncated_inputs": truncated,
        "elapsed_seconds": elapsed,
        "peak_allocated_gpu_gib": peak_gpu,
        "torch_version": torch.__version__,
    }
    (Path(paths["output"]) / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== Extraction summary ===")
    print(f"Rows={len(meta_rows)} languages={manifest['languages']} layers={first_shape[1]} dim={first_shape[2]}")
    print(f"Representations={representations}; truncated={truncated}; storage={storage_dtype_name}")
    print(f"Elapsed={elapsed:.1f}s; peak GPU={peak_gpu:.2f} GiB")


if __name__ == "__main__":
    main()
