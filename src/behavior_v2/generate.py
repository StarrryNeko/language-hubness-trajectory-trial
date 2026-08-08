"""Generate behavior_v2 translations with natural model termination."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    LogitsProcessorList, SuppressTokensLogitsProcessor,
)

from behavior_v1.generate import (
    audit_prompt_lengths, configured_context_window, input_device,
    prompt_forbidden_token_ids,
)
from behavior_v2.common import (
    ensure_paths, generation_file, load_tasks, settings, sha256_file,
    read_checkpoint_identity, split_natural_completion, write_manifest,
)
from common import json_dumps_strict, load_config, model_metadata, set_seed, write_json
from extract_hidden import get_dtype, model_settings


def flatten_token_ids(*values):
    result = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value if item is not None)
        else:
            result.add(int(value))
    return result


def load_existing(path):
    if not path.exists():
        return [], set()
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    ids = [str(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("behavior_v2 generation file has duplicate task IDs")
    return rows, set(ids)


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json_dumps_strict(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate natural-stop behavior_v2 outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    set_seed(protocol["seed"])
    paths = ensure_paths(cfg)
    tasks = load_tasks(cfg)
    checkpoint = read_checkpoint_identity(cfg)
    runtime = model_settings(cfg)
    output = generation_file(cfg)
    existing, completed = load_existing(output) if args.resume else ([], set())
    if output.exists() and not args.resume:
        raise FileExistsError(f"V2 generation exists; use --resume or remove it: {output}")
    frozen = {str(row["task_id"]): row for row in tasks}
    for row in existing:
        if str(row["task_id"]) not in frozen:
            raise ValueError("resume file contains a task outside the frozen V2 task set")
        task = frozen[str(row["task_id"])]
        for field in ("prompt", "source_text", "reference_text", "prompt_template_sha256"):
            if row.get(field) != task.get(field):
                raise ValueError(f"V2 resume refused: existing row changed {field}")
        if row.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
            raise ValueError("V2 resume refused: checkpoint changed")
        if row.get("model") != runtime["model_id"]:
            raise ValueError("V2 resume refused: model identity changed")
        if row.get("decoding") != protocol["decoding"]:
            raise ValueError("V2 resume refused: decoding protocol changed")
        if row.get("activation_intervention") is not False:
            raise ValueError("V2 resume refused: intervention flag is invalid")
        finish_reason = row.get("finish_reason")
        count = int(row.get("generated_token_count", -1))
        maximum = protocol["decoding"]["max_new_tokens"]
        if finish_reason == "natural_stop":
            valid_finish = 0 <= count < maximum and row.get("termination_token_id") is not None
        else:
            valid_finish = (
                finish_reason == "token_budget" and count == maximum
                and row.get("termination_token_id") is None
            )
        if not valid_finish:
            raise ValueError("V2 resume refused: existing finish metadata is invalid")
    pending = [row for row in tasks if str(row["task_id"]) not in completed]
    if not pending:
        manifest = paths.generations / "generation_manifest.json"
        if not manifest.exists():
            raise ValueError("complete V2 rows have no generation manifest")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_version") != "behavior_v2"
            or payload.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
            or payload.get("generation_file_sha256") != sha256_file(output)
            or int(payload.get("rows", -1)) != len(existing)
            or payload.get("decoding") != protocol["decoding"]
        ):
            raise ValueError("complete V2 rows have a stale generation manifest")
        print(f"Resume: all {len(existing)} behavior_v2 tasks are complete")
        return
    configured_memory_limit = protocol["generation_runtime"]["maximum_gpu_memory_gib"]
    memory_fraction = None
    if torch.cuda.is_available() and configured_memory_limit > 0:
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        memory_fraction = min(configured_memory_limit / total_gib, 0.95)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
        print(
            f"CUDA allocator limit: {configured_memory_limit:.1f} GiB "
            f"({memory_fraction:.3f} of {total_gib:.1f} GiB)", flush=True,
        )
    tokenizer_kwargs = {
        "trust_remote_code": runtime["trust_remote_code"],
        "local_files_only": runtime["local_files_only"],
    }
    if runtime["cache_dir"]:
        tokenizer_kwargs["cache_dir"] = runtime["cache_dir"]
    if runtime.get("revision") and not runtime["local_files_only"]:
        tokenizer_kwargs["revision"] = runtime["revision"]
    tokenizer = AutoTokenizer.from_pretrained(runtime["tokenizer"], **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    padding_token_added = False
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|langhub_v2_padding|>"})
            padding_token_added = True
    prompt_audit, prompt_lengths = audit_prompt_lengths(
        tokenizer, tasks, protocol["decoding"], prompt_forbidden_token_ids(tokenizer)
    )
    if prompt_audit["maximum_prompt_tokens_observed"] > protocol["decoding"]["max_prompt_tokens"]:
        write_json(paths.generations / "prompt_length_audit.json", prompt_audit)
        raise ValueError("behavior_v2 prompt exceeds max_prompt_tokens")
    config_kwargs = {
        "trust_remote_code": runtime["trust_remote_code"],
        "local_files_only": runtime["local_files_only"],
    }
    if runtime["cache_dir"]:
        config_kwargs["cache_dir"] = runtime["cache_dir"]
    if runtime.get("revision") and not runtime["local_files_only"]:
        config_kwargs["revision"] = runtime["revision"]
    model_config = AutoConfig.from_pretrained(runtime["name"], **config_kwargs)
    context = configured_context_window(model_config, tokenizer)
    prompt_audit.update({
        "model_context_window": context,
        "max_new_tokens": protocol["decoding"]["max_new_tokens"],
        "maximum_total_tokens": (
            prompt_audit["maximum_prompt_tokens_observed"]
            + protocol["decoding"]["max_new_tokens"]
        ),
    })
    write_json(paths.generations / "prompt_length_audit.json", prompt_audit)
    if context is not None and prompt_audit["maximum_total_tokens"] > context:
        raise ValueError("behavior_v2 prompt plus generation ceiling exceeds context window")
    load_kwargs = {
        "dtype": get_dtype(cfg.get("dtype", "bfloat16")),
        "trust_remote_code": runtime["trust_remote_code"],
        "local_files_only": runtime["local_files_only"],
        "low_cpu_mem_usage": True,
    }
    if runtime["cache_dir"]:
        load_kwargs["cache_dir"] = runtime["cache_dir"]
    if runtime.get("revision") and not runtime["local_files_only"]:
        load_kwargs["revision"] = runtime["revision"]
    if runtime["attn_implementation"]:
        load_kwargs["attn_implementation"] = runtime["attn_implementation"]
    device = cfg.get("device", "cuda")
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    model = AutoModelForCausalLM.from_pretrained(runtime["name"], **load_kwargs)
    if padding_token_added:
        model.resize_token_embeddings(len(tokenizer))
    if device != "auto":
        model.to(device)
    model.eval()
    generation_config = copy.deepcopy(model.generation_config)
    # Native EOS stops decoding and is removed before text output.
    termination_field = "eos_token_id"
    termination_ids = flatten_token_ids(
        getattr(tokenizer, termination_field, None),
        getattr(model_config, termination_field, None),
        getattr(generation_config, termination_field, None),
    )
    if not termination_ids:
        raise ValueError("behavior_v2 natural termination requires at least one termination token ID")
    for field in tuple(vars(generation_config)):
        if field.startswith("forced_") and field.endswith("_token_id"):
            setattr(generation_config, field, None)
    setattr(generation_config, termination_field, sorted(termination_ids))
    all_special_ids = set(map(int, tokenizer.all_special_ids))
    suppressed_ids = sorted(all_special_ids - termination_ids)
    processors = LogitsProcessorList([SuppressTokensLogitsProcessor(suppressed_ids)])
    pending.sort(key=lambda row: (-prompt_lengths[str(row["task_id"])], str(row["task_id"])))
    batch_size = protocol["generation_runtime"]["batch_size"]
    minimum_batch = protocol["generation_runtime"]["minimum_batch_size"]
    started = time.perf_counter()
    batch_sizes = []
    oom_events = 0
    cursor = 0
    progress = tqdm(total=len(pending), desc="behavior_v2 generation", unit="tasks")
    while cursor < len(pending):
        current = min(batch_size, len(pending) - cursor)
        batch = pending[cursor : cursor + current]
        encoded = None
        sequences = None
        try:
            encoded = tokenizer(
                [row["prompt"] for row in batch], return_tensors="pt", padding=True,
                truncation=False, add_special_tokens=False,
            )
            prompt_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
            encoded = {
                key: value.to(input_device(model)) for key, value in encoded.items()
            }
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded,
                    generation_config=generation_config,
                    logits_processor=processors,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=protocol["decoding"]["max_new_tokens"],
                    use_cache=protocol["decoding"]["use_cache"],
                    pad_token_id=tokenizer.pad_token_id,
                )
        except RuntimeError as exc:
            is_oom = torch.cuda.is_available() and (
                isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
            )
            if not is_oom or not protocol["generation_runtime"]["oom_backoff"]:
                progress.close()
                raise
            if current <= minimum_batch:
                progress.close()
                raise
            del encoded, sequences
            gc.collect()
            torch.cuda.empty_cache()
            oom_events += 1
            batch_size = max(minimum_batch, current // 2)
            print(f"CUDA OOM: V2 batch {current} -> {batch_size}", flush=True)
            continue
        padded_width = int(encoded["input_ids"].shape[1])
        records = []
        for task, prompt_count, sequence in zip(batch, prompt_token_counts, sequences):
            generated = sequence[padded_width:].detach().cpu().tolist()
            content, finish_reason, termination_id = split_natural_completion(
                generated, termination_ids, protocol["decoding"]["max_new_tokens"]
            )
            forbidden_content = sorted(set(content) & all_special_ids)
            if forbidden_content:
                progress.close()
                raise ValueError(
                    f"V2 generated content contains forbidden control IDs: {forbidden_content}"
                )
            text = tokenizer.decode(content, skip_special_tokens=False).strip()
            records.append({
                **task,
                "model": runtime["model_id"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "generated_text": text,
                "prompt_token_count": int(prompt_count),
                "generated_token_count": len(content),
                "finish_reason": finish_reason,
                "termination_token_id": termination_id,
                "decoding": protocol["decoding"],
                "activation_intervention": False,
            })
        append_rows(output, records)
        cursor += len(batch)
        batch_sizes.append(len(batch))
        progress.update(len(batch))
        del encoded, sequences
    progress.close()
    final_rows, final_ids = load_existing(output)
    expected_ids = {str(row["task_id"]) for row in tasks}
    if final_ids != expected_ids:
        raise ValueError("behavior_v2 generation does not cover the frozen task set")
    natural = sum(row["finish_reason"] == "natural_stop" for row in final_rows)
    budget = len(final_rows) - natural
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() else 0
    write_manifest(paths.generations / "generation_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "model": runtime["model_id"],
        "model_metadata": model_metadata(cfg, require=True),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "task_file_sha256": sha256_file(paths.data / "behavior_v2_tasks.jsonl"),
        "generation_file_sha256": sha256_file(output),
        "rows": len(final_rows),
        "decoding": protocol["decoding"],
        "generation_runtime": protocol["generation_runtime"],
        "natural_stop_rows": natural,
        "natural_stop_rate": natural / len(final_rows),
        "token_budget_rows": budget,
        "token_budget_rate": budget / len(final_rows),
        "termination_token_ids": sorted(termination_ids),
        "suppressed_special_token_ids": suppressed_ids,
        "generated_forbidden_special_token_count": 0,
        "prompt_special_tokens_added": False,
        "batch_size_min": min(batch_sizes, default=0),
        "batch_size_max": max(batch_sizes, default=0),
        "cuda_oom_events": oom_events,
        "maximum_gpu_memory_gib": configured_memory_limit,
        "cuda_allocator_memory_fraction": memory_fraction,
        "peak_gpu_allocated_gib": peak_allocated / (1024 ** 3),
        "peak_gpu_reserved_gib": peak_reserved / (1024 ** 3),
        "padding_token_added": padding_token_added,
        "prompt_length_audit_sha256": sha256_file(
            paths.generations / "prompt_length_audit.json"
        ),
        "elapsed_seconds": elapsed,
        "tasks_per_second": len(pending) / max(elapsed, 1e-9),
        "activation_intervention": False,
    })
    print(
        f"behavior_v2 generation complete: natural={natural}, budget={budget}, "
        f"rate={len(pending) / max(elapsed, 1e-9):.2f} tasks/s"
    )


if __name__ == "__main__":
    main()
