"""Small hidden-state runtime audit before starting a formal extraction."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from common import load_config


AUDIT_SENTENCES = [
    "The experiment checks representations layer by layer.",
    "这个实验逐层检查模型表示。",
    "Este experimento examina representaciones por capa.",
    "Cette expérience examine les représentations couche par couche.",
    "Dieses Experiment prüft Repräsentationen Schicht für Schicht.",
    "هذه التجربة تفحص التمثيلات طبقةً بعد طبقة.",
    "यह प्रयोग प्रत्येक परत में निरूपण की जाँच करता है।",
    "Этот эксперимент проверяет представления по слоям.",
    "この実験では層ごとの表現を調べます。",
    "이 실험은 층별 표현을 확인합니다.",
]


def suite_configs(suite_path):
    suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    return [suite_path.parent / relative for relative in suite["configs"]]


def audit_one(config_path, sentence_count):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from extract_hidden import get_dtype, model_settings
    except ImportError as exc:
        raise RuntimeError(
            "Runtime audit requires torch and transformers. Install the server inference "
            "environment before running this command."
        ) from exc
    cfg = load_config(config_path)
    settings = model_settings(cfg)
    dtype = get_dtype(cfg.get("dtype", "bfloat16"))
    tokenizer_kwargs = {
        "trust_remote_code": settings["trust_remote_code"],
        "local_files_only": settings["local_files_only"],
    }
    if settings["revision"] and not settings["local_files_only"]:
        tokenizer_kwargs["revision"] = settings["revision"]
    tokenizer = AutoTokenizer.from_pretrained(settings["tokenizer"], **tokenizer_kwargs)
    load_kwargs = {
        "dtype": dtype,
        "trust_remote_code": settings["trust_remote_code"],
        "local_files_only": settings["local_files_only"],
        "low_cpu_mem_usage": True,
    }
    if settings["revision"] and not settings["local_files_only"]:
        load_kwargs["revision"] = settings["revision"]
    if settings["attn_implementation"]:
        load_kwargs["attn_implementation"] = settings["attn_implementation"]
    device = cfg.get("device", "auto")
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(settings["name"], **load_kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    input_device = model.get_input_embeddings().weight.device
    observed_layers = None
    hidden_size = None
    with torch.inference_mode():
        for text in AUDIT_SENTENCES[:sentence_count]:
            encoded = tokenizer(
                text, return_tensors="pt", add_special_tokens=False, truncation=True,
                max_length=min(128, int(cfg.get("max_length", 512))),
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            output = model(
                **encoded, output_hidden_states=True, return_dict=True, use_cache=False
            )
            if not output.hidden_states or len(output.hidden_states) < 2:
                raise ValueError("model did not return embedding plus transformer hidden states")
            current_layers = len(output.hidden_states)
            current_size = int(output.hidden_states[-1].shape[-1])
            if observed_layers is None:
                observed_layers, hidden_size = current_layers, current_size
            if current_layers != observed_layers or current_size != hidden_size:
                raise ValueError("hidden-state shape changed across audit sentences")
            if not all(bool(torch.isfinite(hidden).all().item()) for hidden in output.hidden_states):
                raise ValueError("hidden states contain NaN/Inf")
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    record = {
        "status": "PASS",
        "config": str(Path(config_path).resolve()),
        "model_id": settings["model_id"],
        "resolved_model_source": settings["name"],
        "resolved_revision": settings["resolved_revision"],
        "loaded_from_local_directory": settings["local_files_only"],
        "sentences": int(sentence_count),
        "hidden_state_count": int(observed_layers),
        "hidden_size": int(hidden_size),
        "dtype": str(dtype),
        "peak_allocated_gpu_gib": float(peak),
        "elapsed_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
    }
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description="Audit model hidden-state runtime support")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--sentences", type=int, default=10)
    parser.add_argument("--output", default="audits/model_runtime_audit.json")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.sentences <= len(AUDIT_SENTENCES):
        raise ValueError(f"--sentences must be in 1..{len(AUDIT_SENTENCES)}")
    config_paths = suite_configs(args.suite)
    selected = set(args.model)
    records = []
    for config_path in config_paths:
        cfg = load_config(config_path)
        model_id = cfg.get("model", {}).get("name_or_path", cfg.get("model_name_or_path"))
        if selected and model_id not in selected:
            continue
        print(f"\nAuditing {model_id} from {config_path}", flush=True)
        try:
            record = audit_one(config_path, args.sentences)
            print(
                f"PASS {model_id}: hidden_states={record['hidden_state_count']} "
                f"dim={record['hidden_size']} peak={record['peak_allocated_gpu_gib']:.2f} GiB"
            )
        except Exception as exc:  # audit must preserve the exact technical exclusion reason
            record = {
                "status": "FAIL",
                "config": str(config_path.resolve()),
                "model_id": model_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if not args.continue_on_error:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "records": [*records, record],
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                raise
        records.append(record)
    if not records:
        raise ValueError("No configured models matched the selection")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": str(Path(args.suite).resolve()),
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = [record for record in records if record["status"] != "PASS"]
    print(f"\nRuntime audit complete: pass={len(records) - len(failed)} fail={len(failed)}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
