"""Run deterministic behavior_v1 generation without activation intervention."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    SuppressTokensLogitsProcessor,
)

from behavior_common import (
    behavior_settings,
    ensure_behavior_dirs,
    generation_file,
    load_tasks,
    read_checkpoint_identity,
    sha256_file,
    write_manifest,
)
from common import json_dumps_strict, load_config, model_metadata, set_seed, write_json
from extract_hidden import get_dtype, model_settings


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
        raise ValueError("existing behavior generations contain duplicate task IDs")
    return rows, set(ids)


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json_dumps_strict(row, ensure_ascii=False) + "\n")


def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return model.device


def extract_answer(text, method):
    raw = str(text).strip()
    if method == "full_generation":
        return raw
    if method == "first_nonempty_line":
        return next((line.strip() for line in raw.splitlines() if line.strip()), "")
    raise ValueError(f"unsupported behavior output_extraction: {method}")


def prompt_forbidden_token_ids(tokenizer):
    """Return control-token IDs forbidden in plain-text prompt encoding.

    A tokenizer may emit its unknown-token ID for ordinary text that is outside
    its vocabulary even when add_special_tokens=False.  That is a vocabulary
    coverage event, not an automatically inserted prompt control token.
    """
    forbidden = set(map(int, tokenizer.all_special_ids))
    unknown_id = tokenizer.unk_token_id
    if unknown_id is not None:
        forbidden.discard(int(unknown_id))
    return forbidden


def audit_prompt_lengths(tokenizer, tasks, decoding, forbidden_ids, batch_size=128):
    records = []
    unknown_id = tokenizer.unk_token_id
    unknown_counts = []
    for start in range(0, len(tasks), int(batch_size)):
        batch = tasks[start : start + int(batch_size)]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            padding=False,
            truncation=False,
            add_special_tokens=decoding["prompt_add_special_tokens"],
        )
        for task, token_ids in zip(batch, encoded["input_ids"]):
            integer_ids = list(map(int, token_ids))
            found = sorted(set(integer_ids) & forbidden_ids)
            if found:
                raise ValueError(
                    f"behavior prompt contains tokenizer control tokens: "
                    f"task={task['task_id']}, token_ids={found}"
                )
            unknown_counts.append(
                0 if unknown_id is None else integer_ids.count(int(unknown_id))
            )
            records.append((str(task["task_id"]), len(integer_ids)))
    lengths = np.asarray([length for _, length in records], dtype=int)
    longest_task, maximum = max(records, key=lambda value: value[1])
    total_prompt_tokens = int(lengths.sum())
    total_unknown_tokens = int(sum(unknown_counts))
    summary = {
        "tasks": len(records),
        "total_prompt_tokens": total_prompt_tokens,
        "minimum_prompt_tokens": int(lengths.min()),
        "median_prompt_tokens": float(np.median(lengths)),
        "p95_prompt_tokens": float(np.quantile(lengths, 0.95)),
        "p99_prompt_tokens": float(np.quantile(lengths, 0.99)),
        "maximum_prompt_tokens_observed": int(maximum),
        "longest_task_id": longest_task,
        "configured_max_prompt_tokens": int(decoding["max_prompt_tokens"]),
        "prompt_unknown_token_id": (
            None if unknown_id is None else int(unknown_id)
        ),
        "prompt_unknown_token_count": total_unknown_tokens,
        "prompt_unknown_token_fraction": (
            total_unknown_tokens / total_prompt_tokens if total_prompt_tokens else 0.0
        ),
        "tasks_with_prompt_unknown_tokens": int(
            sum(count > 0 for count in unknown_counts)
        ),
    }
    return summary, {task_id: length for task_id, length in records}


def configured_context_window(model_config, tokenizer):
    candidates = []
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(model_config, name, None)
        if isinstance(value, int) and 0 < value < 10**7:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**7:
        candidates.append(tokenizer_limit)
    return min(candidates) if candidates else None


def memory_guided_batch_size(
    current_batch_size,
    minimum_batch_size,
    maximum_batch_size,
    baseline_allocated_bytes,
    peak_allocated_bytes,
    total_device_bytes,
    target_memory_fraction,
    maximum_growth_factor,
):
    """Estimate a safe larger batch from the measured per-batch CUDA peak."""
    current = int(current_batch_size)
    dynamic_bytes = int(peak_allocated_bytes) - int(baseline_allocated_bytes)
    target_bytes = float(total_device_bytes) * float(target_memory_fraction)
    dynamic_budget = target_bytes - int(baseline_allocated_bytes)
    if dynamic_bytes <= 0 or dynamic_budget <= dynamic_bytes:
        return current
    estimated = int(current * dynamic_budget / dynamic_bytes)
    growth_cap = max(current + 1, int(current * float(maximum_growth_factor)))
    return max(
        int(minimum_batch_size),
        min(int(maximum_batch_size), growth_cap, max(current, estimated)),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate behavior_v1 translations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    settings = behavior_settings(cfg)
    set_seed(settings["seed"])
    tasks = load_tasks(cfg)
    checkpoint = read_checkpoint_identity(cfg)
    paths = ensure_behavior_dirs(cfg)
    output = generation_file(cfg)
    existing, completed = load_existing(output) if args.resume else ([], set())
    if output.exists() and not args.resume:
        raise FileExistsError(f"generation output already exists; use --resume: {output}")
    pending = [row for row in tasks if str(row["task_id"]) not in completed]
    frozen_tasks = {str(row["task_id"]): row for row in tasks}
    for row in existing:
        frozen = frozen_tasks[str(row["task_id"])]
        for field in ("prompt", "source_text", "reference_text", "prompt_template_sha256"):
            if row.get(field) != frozen.get(field):
                raise ValueError(f"resume refused: existing generation changed {field}")
        if row.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
            raise ValueError("resume refused: existing generation checkpoint has changed")
        if row.get("decoding") != settings["decoding"]:
            raise ValueError("resume refused: existing generation decoding protocol has changed")

    runtime = model_settings(cfg)
    if not pending:
        manifest_path = paths.generations / "generation_manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                "complete generations have no audit manifest; refuse to reconstruct "
                "special-token compliance after generation"
            )
        print(f"Resume: all {len(existing)} frozen behavior tasks are already generated")
        return
    tokenizer_kwargs = {
        "trust_remote_code": runtime["trust_remote_code"],
        "local_files_only": runtime["local_files_only"],
    }
    if runtime["cache_dir"]:
        tokenizer_kwargs["cache_dir"] = runtime["cache_dir"]
    if runtime["revision"] and not runtime["local_files_only"]:
        tokenizer_kwargs["revision"] = runtime["revision"]
    tokenizer = AutoTokenizer.from_pretrained(runtime["tokenizer"], **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    padding_token_added = False
    padding_token_reused = False
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
            padding_token_reused = True
        else:
            tokenizer.add_special_tokens({"pad_token": "<|langhub_padding|>"})
            padding_token_added = True

    decoding = settings["decoding"]
    forbidden_ids = set(map(int, tokenizer.all_special_ids))
    if not forbidden_ids:
        raise ValueError("tokenizer exposes no special-token inventory for generation auditing")
    prompt_forbidden_ids = prompt_forbidden_token_ids(tokenizer)
    prompt_audit, prompt_tokens_by_task = audit_prompt_lengths(
        tokenizer, tasks, decoding, prompt_forbidden_ids
    )
    if prompt_audit["maximum_prompt_tokens_observed"] > decoding["max_prompt_tokens"]:
        write_json(paths.generations / "prompt_length_audit.json", prompt_audit)
        raise ValueError(
            "behavior prompt exceeds configured max_prompt_tokens: "
            f"observed={prompt_audit['maximum_prompt_tokens_observed']}, "
            f"configured={decoding['max_prompt_tokens']}, "
            f"task={prompt_audit['longest_task_id']}"
        )
    config_kwargs = {
        "trust_remote_code": runtime["trust_remote_code"],
        "local_files_only": runtime["local_files_only"],
    }
    if runtime["cache_dir"]:
        config_kwargs["cache_dir"] = runtime["cache_dir"]
    if runtime["revision"] and not runtime["local_files_only"]:
        config_kwargs["revision"] = runtime["revision"]
    model_config = AutoConfig.from_pretrained(runtime["name"], **config_kwargs)
    context_window = configured_context_window(model_config, tokenizer)
    prompt_audit["model_context_window"] = context_window
    prompt_audit["max_new_tokens"] = decoding["max_new_tokens"]
    prompt_audit["maximum_total_tokens"] = (
        prompt_audit["maximum_prompt_tokens_observed"] + decoding["max_new_tokens"]
    )
    write_json(paths.generations / "prompt_length_audit.json", prompt_audit)
    if context_window is not None and prompt_audit["maximum_total_tokens"] > context_window:
        raise ValueError(
            "behavior prompt plus generation budget exceeds the model context window: "
            f"required={prompt_audit['maximum_total_tokens']}, context={context_window}, "
            f"task={prompt_audit['longest_task_id']}"
        )

    load_kwargs = {
        "dtype": get_dtype(cfg.get("dtype", "float16")),
        "trust_remote_code": runtime["trust_remote_code"],
        "local_files_only": runtime["local_files_only"],
        "low_cpu_mem_usage": True,
    }
    if runtime["cache_dir"]:
        load_kwargs["cache_dir"] = runtime["cache_dir"]
    if runtime["revision"] and not runtime["local_files_only"]:
        load_kwargs["revision"] = runtime["revision"]
    if runtime["attn_implementation"]:
        load_kwargs["attn_implementation"] = runtime["attn_implementation"]
    device = cfg.get("device", "cuda")
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(runtime["name"], **load_kwargs)
    if padding_token_added:
        model.resize_token_embeddings(len(tokenizer))
    if device != "auto":
        model.to(device)
    model.eval()
    generation_config = copy.deepcopy(model.generation_config)
    for field in tuple(vars(generation_config)):
        if field.startswith("forced_") and field.endswith("_token_id"):
            setattr(generation_config, field, None)
    token_filters = LogitsProcessorList([
        SuppressTokensLogitsProcessor(sorted(forbidden_ids))
    ])
    started = time.perf_counter()
    generation_runtime = settings["generation_runtime"]
    pending_count = len(pending)
    if generation_runtime["length_bucketed_batching"]:
        pending = sorted(
            pending,
            key=lambda row: (-prompt_tokens_by_task[str(row["task_id"])], str(row["task_id"])),
        )
    effective_batch_size = generation_runtime["initial_batch_size"]
    minimum_batch_size = generation_runtime["minimum_batch_size"]
    batch_sizes_used = []
    batch_adjustments = []
    oom_events = 0
    peak_gpu_memory_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cursor = 0
    progress = tqdm(total=len(pending), desc="behavior_v1 generation", unit="tasks")
    while cursor < len(pending):
        current_batch_size = min(effective_batch_size, len(pending) - cursor)
        batch = pending[cursor : cursor + current_batch_size]
        encoded = None
        sequences = None
        baseline_allocated_bytes = 0
        try:
            if torch.cuda.is_available():
                baseline_allocated_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=decoding["prompt_add_special_tokens"],
            )
            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            if max(prompt_lengths, default=0) > decoding["max_prompt_tokens"]:
                raise ValueError(
                    f"behavior prompt exceeds max_prompt_tokens={decoding['max_prompt_tokens']}"
                )
            device_for_inputs = input_device(model)
            encoded = {key: value.to(device_for_inputs) for key, value in encoded.items()}
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded,
                    generation_config=generation_config,
                    logits_processor=token_filters,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=decoding["max_new_tokens"],
                    use_cache=decoding["use_cache"],
                    pad_token_id=tokenizer.pad_token_id,
                )
        except RuntimeError as exc:
            is_cuda_oom = isinstance(exc, torch.OutOfMemoryError) or (
                "out of memory" in str(exc).lower() and torch.cuda.is_available()
            )
            if not is_cuda_oom:
                progress.close()
                raise
            if not generation_runtime["oom_backoff"] or current_batch_size <= minimum_batch_size:
                progress.close()
                raise
            del encoded, sequences
            gc.collect()
            torch.cuda.empty_cache()
            oom_events += 1
            backed_off = int(current_batch_size * generation_runtime["oom_backoff_factor"])
            effective_batch_size = max(
                minimum_batch_size, min(current_batch_size - 1, backed_off)
            )
            batch_adjustments.append({
                "reason": "cuda_oom_backoff",
                "from_batch_size": current_batch_size,
                "to_batch_size": effective_batch_size,
            })
            print(
                f"CUDA OOM at batch={current_batch_size}; retrying with "
                f"batch={effective_batch_size}", flush=True,
            )
            continue
        measured_peak_bytes = 0
        if torch.cuda.is_available():
            measured_peak_bytes = torch.cuda.max_memory_allocated()
            peak_gpu_memory_bytes = max(peak_gpu_memory_bytes, measured_peak_bytes)
        padded_prompt_width = int(encoded["input_ids"].shape[1])
        records = []
        for task, prompt_length, sequence in zip(batch, prompt_lengths, sequences):
            generated_ids = sequence[padded_prompt_width:].detach().cpu().tolist()
            found = sorted(set(generated_ids) & forbidden_ids)
            if found:
                progress.close()
                raise ValueError(
                    f"generated output contains forbidden special tokens: "
                    f"task={task['task_id']}, token_ids={found}"
                )
            if len(generated_ids) != decoding["max_new_tokens"]:
                progress.close()
                raise ValueError(
                    f"generation ended before the frozen token budget: "
                    f"task={task['task_id']}, generated={len(generated_ids)}, "
                    f"required={decoding['max_new_tokens']}"
                )
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            text = extract_answer(raw_text, decoding["output_extraction"])
            records.append({
                **task,
                "model": runtime["model_id"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "raw_generated_text": raw_text,
                "generated_text": text,
                "prompt_token_count": int(prompt_length),
                "generated_token_count": len(generated_ids),
                "finish_reason": "token_budget",
                "decoding": decoding,
            })
        append_rows(output, records)
        del encoded, sequences
        cursor += len(batch)
        batch_sizes_used.append(len(batch))
        progress.update(len(batch))
        if (
            torch.cuda.is_available()
            and generation_runtime["adaptive_batch_sizing"]
            and current_batch_size == effective_batch_size
            and effective_batch_size < generation_runtime["maximum_batch_size"]
        ):
            next_batch_size = memory_guided_batch_size(
                current_batch_size=current_batch_size,
                minimum_batch_size=minimum_batch_size,
                maximum_batch_size=generation_runtime["maximum_batch_size"],
                baseline_allocated_bytes=baseline_allocated_bytes,
                peak_allocated_bytes=measured_peak_bytes,
                total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
                target_memory_fraction=generation_runtime["target_gpu_memory_fraction"],
                maximum_growth_factor=generation_runtime["maximum_batch_growth_factor"],
            )
            if next_batch_size != effective_batch_size:
                batch_adjustments.append({
                    "reason": "measured_memory_growth",
                    "from_batch_size": effective_batch_size,
                    "to_batch_size": next_batch_size,
                    "measured_peak_gpu_memory_gib": measured_peak_bytes / 1024**3,
                })
                print(
                    f"CUDA batch autotune: batch={effective_batch_size} -> "
                    f"{next_batch_size}, measured_peak={measured_peak_bytes / 1024**3:.2f} GiB",
                    flush=True,
                )
                effective_batch_size = next_batch_size
    progress.close()

    final_rows, final_ids = load_existing(output)
    expected_ids = {str(row["task_id"]) for row in tasks}
    if final_ids != expected_ids:
        raise ValueError(f"generation output does not cover frozen tasks: {len(final_ids)}/{len(expected_ids)}")
    elapsed = time.perf_counter() - started
    write_manifest(paths.generations / "generation_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "model": runtime["model_id"],
        "model_metadata": model_metadata(cfg, require=True),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "task_file_sha256": sha256_file(paths.data / "behavior_tasks.jsonl"),
        "generation_file_sha256": sha256_file(output),
        "rows": len(final_rows),
        "decoding": decoding,
        "generation_runtime": generation_runtime,
        "effective_batch_size_max": max(batch_sizes_used, default=0),
        "effective_batch_size_min": min(batch_sizes_used, default=0),
        "batch_adjustments": batch_adjustments,
        "cuda_oom_events": oom_events,
        "peak_gpu_memory_gib": peak_gpu_memory_bytes / 1024**3,
        "tasks_per_second": pending_count / max(elapsed, 1e-9),
        "prompt_length_audit_sha256": sha256_file(
            paths.generations / "prompt_length_audit.json"
        ),
        "prompt_special_tokens_added": False,
        "generated_special_token_count": 0,
        "special_tokens_suppressed": True,
        "padding_token_added": padding_token_added,
        "padding_token_reused": padding_token_reused,
        "elapsed_seconds": elapsed,
        "activation_intervention": False,
    })
    print(f"Saved {len(final_rows)} behavior generations to {output}")


if __name__ == "__main__":
    main()
