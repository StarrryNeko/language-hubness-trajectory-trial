"""Generate one V3 split with native EOS, text boundaries, and adaptive GPU batches."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList, SuppressTokensLogitsProcessor,
)

from behavior_association_v3.common import (
    classify_finish, find_repetition_boundary, generation_path, load_tasks, paths,
    read_checkpoint, settings, sha256_file, strip_framework_padding,
    task_path, trim_completion, write_manifest,
)
from behavior_v1.generate import audit_prompt_lengths, configured_context_window, input_device
from common import json_dumps_strict, load_config, model_metadata, set_seed
from extract_hidden import get_dtype, model_settings


class RepetitionBoundaryCriteria(StoppingCriteria):
    """Stop each sequence after a frozen consecutive repeated-token suffix."""

    def __init__(self, prompt_width, minimum_tokens, maximum_block_tokens):
        self.prompt_width = int(prompt_width)
        self.minimum_tokens = int(minimum_tokens)
        self.maximum_block_tokens = int(maximum_block_tokens)

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[:, self.prompt_width:]
        decisions = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        if generated.shape[1] < self.minimum_tokens:
            return decisions
        maximum = min(self.maximum_block_tokens, generated.shape[1] // 3)
        for block_size in range(1, maximum + 1):
            repeats = 8 if block_size == 1 else 5 if block_size == 2 else 4 if block_size == 3 else 3
            width = block_size * repeats
            if generated.shape[1] < width:
                continue
            segments = generated[:, -width:].reshape(len(input_ids), repeats, block_size)
            decisions |= (segments == segments[:, -1:, :]).all(dim=2).all(dim=1)
        return decisions


def read_existing(path):
    if not path.exists():
        return [], set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("generation resume file has duplicate task IDs")
    return rows, set(ids)


def append(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json_dumps_strict(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("calibration", "formal"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    protocol = settings(cfg)
    set_seed(protocol["seed"])
    tasks = load_tasks(cfg, args.split)
    checkpoint = read_checkpoint(cfg)
    runtime = model_settings(cfg)
    output = generation_path(cfg, args.split)
    existing, completed = read_existing(output) if args.resume else ([], set())
    if output.exists() and not args.resume:
        raise FileExistsError(f"use --resume or archive the existing V3 file: {output}")
    task_map = {row["task_id"]: row for row in tasks}
    for row in existing:
        task = task_map.get(row["task_id"])
        if task is None:
            raise ValueError("resume rows do not belong to the frozen V3 task set")
        for field in (
            "split", "semantic_id", "source_lang", "target_lang", "source_text",
            "reference_text", "prompt", "prompt_template_sha256",
        ):
            if row.get(field) != task.get(field):
                raise ValueError(f"resume refused because frozen task field changed: {field}")
        if row.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
            raise ValueError("resume refused because checkpoint changed")
        if row.get("model") != runtime["model_id"]:
            raise ValueError("resume refused because model identity changed")
        if row.get("decoding") != protocol["decoding"]:
            raise ValueError("resume refused because the V3 decoding/EOS protocol changed")
        if row.get("activation_intervention") is not False:
            raise ValueError("resume row has an invalid activation-intervention flag")
    pending = [row for row in tasks if row["task_id"] not in completed]
    if not pending:
        manifest_path = paths(cfg).generations / f"{args.split}_generation_manifest.json"
        if not manifest_path.exists():
            raise ValueError("complete resume rows have no generation manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("generation_file_sha256") != sha256_file(output)
            or manifest.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
            or manifest.get("task_file_sha256") != sha256_file(task_path(cfg, args.split))
            or manifest.get("decoding") != protocol["decoding"]
            or int(manifest.get("rows", -1)) != len(existing)
        ):
            raise ValueError("complete resume rows have a stale generation manifest")
        print(f"Resume: all {len(existing)} {args.split} rows already exist and match manifest")
        return
    tokenizer_kwargs = {"trust_remote_code": runtime["trust_remote_code"], "local_files_only": runtime["local_files_only"]}
    if runtime["cache_dir"]:
        tokenizer_kwargs["cache_dir"] = runtime["cache_dir"]
    tokenizer = AutoTokenizer.from_pretrained(runtime["tokenizer"], **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    audit, lengths = audit_prompt_lengths(tokenizer, tasks, protocol["decoding"], set())
    if audit["maximum_prompt_tokens_observed"] > protocol["decoding"]["max_prompt_tokens"]:
        raise ValueError("V3 prompt exceeds max_prompt_tokens")
    config_kwargs = dict(tokenizer_kwargs)
    model_config = AutoConfig.from_pretrained(runtime["name"], **config_kwargs)
    context = configured_context_window(model_config, tokenizer)
    if context and audit["maximum_prompt_tokens_observed"] + protocol["decoding"]["max_new_tokens"] > context:
        raise ValueError("V3 prompt plus safety ceiling exceeds context window")
    load_kwargs = {
        "dtype": get_dtype(cfg.get("dtype", "bfloat16")),
        "trust_remote_code": runtime["trust_remote_code"],
        "local_files_only": runtime["local_files_only"],
        "low_cpu_mem_usage": True,
    }
    if runtime["cache_dir"]:
        load_kwargs["cache_dir"] = runtime["cache_dir"]
    if runtime["attn_implementation"]:
        load_kwargs["attn_implementation"] = runtime["attn_implementation"]
    device = cfg.get("device", "cuda")
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    allocator_fraction = min(protocol["runtime"]["maximum_gpu_memory_gib"] / total_gib, 0.98)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=0)
    model = AutoModelForCausalLM.from_pretrained(runtime["name"], **load_kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    eos_ids = set()
    for eos_value in (
        tokenizer.eos_token_id, getattr(model_config, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ):
        if isinstance(eos_value, (list, tuple, set)):
            eos_ids.update(map(int, eos_value))
        elif eos_value is not None:
            eos_ids.add(int(eos_value))
    if not eos_ids:
        raise ValueError("V3 native termination requires at least one EOS token ID")
    generation_config = copy.deepcopy(model.generation_config)
    # A forced EOS at max length is a ceiling event, not native model termination.
    # Disable only length-forced EOS while preserving other model-native controls.
    generation_config.forced_eos_token_id = None
    generation_config.eos_token_id = sorted(eos_ids)
    generation_config.min_length = 0
    generation_config.min_new_tokens = None
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.max_new_tokens = protocol["decoding"]["max_new_tokens"]
    generation_config.use_cache = protocol["decoding"]["use_cache"]
    generation_config.stop_strings = protocol["decoding"]["stop_strings"]
    generation_config.pad_token_id = tokenizer.pad_token_id
    all_special_ids = set(map(int, tokenizer.all_special_ids))
    suppressed_ids = sorted(all_special_ids - eos_ids)
    processors = LogitsProcessorList([SuppressTokensLogitsProcessor(suppressed_ids)])
    pending.sort(key=lambda row: (-lengths[row["task_id"]], row["task_id"]))
    batch_size = protocol["runtime"]["initial_batch_size"]
    maximum_batch = protocol["runtime"]["maximum_batch_size"]
    minimum_batch = protocol["runtime"]["minimum_batch_size"]
    target_reserved = total_gib * protocol["runtime"]["target_gpu_memory_fraction"]
    cursor, oom_events = 0, 0
    framework_padding_tokens_removed = 0
    used_batches, peak_values = [], []
    started = time.perf_counter()
    progress = tqdm(total=len(pending), desc=f"V3 {args.split}", unit="tasks")
    while cursor < len(pending):
        current = min(batch_size, len(pending) - cursor)
        batch = pending[cursor:cursor + current]
        encoded = sequences = None
        try:
            torch.cuda.reset_peak_memory_stats(0)
            encoded = tokenizer(
                [row["prompt"] for row in batch], return_tensors="pt", padding=True,
                truncation=False, add_special_tokens=False,
            )
            prompt_counts = encoded["attention_mask"].sum(dim=1).tolist()
            encoded = {key: value.to(input_device(model)) for key, value in encoded.items()}
            padded_width = int(encoded["input_ids"].shape[1])
            repetition_criteria = RepetitionBoundaryCriteria(
                padded_width, protocol["repetition"]["minimum_tokens"],
                protocol["repetition"]["maximum_block_tokens"],
            )
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded, tokenizer=tokenizer,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList([repetition_criteria]),
                    logits_processor=processors,
                )
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom or current <= minimum_batch:
                progress.close()
                raise
            del encoded, sequences
            gc.collect()
            torch.cuda.empty_cache()
            oom_events += 1
            batch_size = max(minimum_batch, current // 2)
            continue
        records = []
        for task, prompt_count, sequence in zip(batch, prompt_counts, sequences):
            returned_token_ids = sequence[padded_width:].detach().cpu().tolist()
            token_ids, padding_removed = strip_framework_padding(
                returned_token_ids, tokenizer.pad_token_id, eos_ids
            )
            framework_padding_tokens_removed += padding_removed
            eos_position = next((i for i, value in enumerate(token_ids) if value in eos_ids), None)
            content_ids = token_ids[:eos_position] if eos_position is not None else token_ids
            repetition_position = find_repetition_boundary(
                content_ids, protocol["repetition"]["minimum_tokens"],
                protocol["repetition"]["maximum_block_tokens"],
            )
            effective_ids = (
                content_ids[:repetition_position]
                if repetition_position is not None else content_ids
            )
            forbidden = sorted(set(effective_ids) & all_special_ids)
            if forbidden:
                raise ValueError(f"V3 generated forbidden special token IDs: {forbidden}")
            decoded = tokenizer.decode(effective_ids, skip_special_tokens=True)
            text, boundary, marker = trim_completion(decoded, protocol["decoding"]["stop_strings"])
            finish = classify_finish(
                eos_position, boundary, repetition_position, len(token_ids),
                protocol["decoding"]["max_new_tokens"],
            )
            observed_eos_id = None if eos_position is None else int(token_ids[eos_position])
            records.append({
                **task, "model": runtime["model_id"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "generated_text": text,
                "prompt_token_count": int(prompt_count),
                "generated_token_count": len(effective_ids),
                "raw_generated_token_count": len(token_ids),
                "returned_generated_token_count": len(returned_token_ids),
                "framework_padding_tokens_removed": padding_removed,
                "finish_reason": finish,
                "text_boundary": marker,
                "termination_token_id": observed_eos_id if finish == "native_eos" else None,
                "termination_position": int(eos_position) if finish == "native_eos" else None,
                "observed_eos_token_id": observed_eos_id,
                "repetition_boundary_position": (
                    int(repetition_position) if finish == "repetition_boundary" else None
                ),
                "decoding": protocol["decoding"],
                "activation_intervention": False,
            })
        append(output, records)
        reserved = torch.cuda.max_memory_reserved(0) / 1024 ** 3
        peak_values.append(reserved)
        used_batches.append(current)
        cursor += current
        progress.update(current)
        if protocol["runtime"]["adaptive_growth"] and current == batch_size and reserved < target_reserved * 0.88:
            proposed = max(batch_size + 1, math.ceil(batch_size * protocol["runtime"]["growth_factor"]))
            batch_size = min(maximum_batch, proposed)
        del encoded, sequences
    progress.close()
    rows, ids = read_existing(output)
    if ids != set(task_map):
        raise ValueError("V3 generation does not exactly cover its frozen task split")
    elapsed = time.perf_counter() - started
    counts = {
        name: sum(row["finish_reason"] == name for row in rows)
        for name in ("native_eos", "text_boundary", "repetition_boundary", "token_ceiling")
    }
    write_manifest(paths(cfg).generations / f"{args.split}_generation_manifest.json", {
        "config_path": str(config_path), "split": args.split,
        "model": runtime["model_id"], "model_metadata": model_metadata(cfg, require=True),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "task_file_sha256": sha256_file(task_path(cfg, args.split)),
        "generation_file_sha256": sha256_file(output), "rows": len(rows),
        "finish_reason_counts": counts,
        "native_eos_token_ids": sorted(eos_ids),
        "forced_eos_token_id_disabled": True,
        "token_ceiling_rate": counts["token_ceiling"] / len(rows),
        "repetition_boundary_rate": counts["repetition_boundary"] / len(rows),
        "repetition_boundary_protocol": protocol["repetition"],
        "batch_size_min": min(used_batches), "batch_size_max": max(used_batches),
        "peak_gpu_reserved_gib": max(peak_values), "cuda_oom_events": oom_events,
        "allocator_fraction": allocator_fraction, "elapsed_seconds": elapsed,
        "tasks_per_second": len(pending) / max(elapsed, 1e-9),
        "decoding": protocol["decoding"], "activation_intervention": False,
        "suppressed_special_token_ids": suppressed_ids,
        "generated_forbidden_special_token_count": 0,
        "framework_padding_tokens_removed": framework_padding_tokens_removed,
        "prompt_special_tokens_added": False,
    })
    print(f"V3 {args.split} generation complete: {counts}")


if __name__ == "__main__":
    main()
