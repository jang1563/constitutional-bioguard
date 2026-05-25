#!/usr/bin/env python
"""A.3 Inference latency / memory benchmark.

Compares production-side cost of v3 / v4 (DeBERTa-v3-base, 184M params)
against WildGuard 7B (Mistral-7B fine-tune) and LLaMA-Guard 3 8B.

Reports for each model:
- Model load time (seconds)
- Peak GPU memory after load (MB)
- Per-item latency: batch=1 and batch=32
- Throughput (items/sec) at each batch size
- Per-item latency: at 128, 256, 512 token contexts (where supported)

Output: phase3_probe_a3_latency_memory.json

Note: WildGuard / LLaMA-Guard are generative and have variable latency
based on output length. We use deterministic max_new_tokens=8 for fairness.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test inputs — 100 short (~256 tok) bio queries paired with synthetic compliance
SAMPLE_QUERY = (
    "What are the key safety considerations for handling biological "
    "samples in a research laboratory setting? Please provide a "
    "comprehensive overview of the standard protocols and best "
    "practices that are commonly used in academic and industrial "
    "research facilities for ensuring the safety of personnel and "
    "the integrity of the samples during routine handling procedures."
)
SAMPLE_RESPONSE = (
    "Here is a detailed overview of the safety considerations and "
    "standard protocols for handling biological samples. The key areas "
    "include personal protective equipment, biosafety cabinet usage, "
    "sample storage and labeling, decontamination procedures, and waste "
    "disposal. Each of these areas has specific guidelines that researchers "
    "must follow to maintain a safe working environment."
)

N_ITEMS = 100


def reset_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def benchmark_deberta(model_name: str, model_dir: Path) -> dict:
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer,
    )
    reset_cuda()
    logger.info("Loading %s from %s", model_name, model_dir)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(model_dir)
    # load_model_and_tokenizer does not move to GPU by default
    model = model.to(DEVICE)
    load_time = time.time() - t0
    mem_after_load = gpu_mem_mb()
    model.eval()

    queries = [SAMPLE_QUERY] * N_ITEMS
    responses = [SAMPLE_RESPONSE] * N_ITEMS

    results = {"model": model_name, "param_count_m": None,
               "load_time_s": round(load_time, 3),
               "mem_after_load_mb": round(mem_after_load, 1)}

    # Param count
    n_params = sum(p.numel() for p in model.parameters())
    results["param_count_m"] = round(n_params / 1e6, 1)

    # Batch 1
    encs1 = [tok(q, r, return_tensors="pt", padding=True, truncation=True,
                 max_length=512) for q, r in zip(queries[:32], responses[:32])]
    encs1 = [{k: v.to(DEVICE) for k, v in e.items()} for e in encs1]
    # warmup
    with torch.no_grad():
        for e in encs1[:3]:
            _ = model(**e)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for e in encs1:
            _ = model(**e)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch1_total = time.time() - t0
    results["batch1_latency_per_item_ms"] = round(batch1_total / len(encs1) * 1000, 2)
    results["batch1_throughput_items_per_s"] = round(len(encs1) / batch1_total, 2)

    # Batch 32
    enc32 = tok(queries, responses, return_tensors="pt", padding=True,
                 truncation=True, max_length=512)
    enc32 = {k: v.to(DEVICE) for k, v in enc32.items()}
    # warmup
    with torch.no_grad():
        _ = model(**enc32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(**enc32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch32_total = time.time() - t0
    results["batch32_latency_per_item_ms"] = round(batch32_total / N_ITEMS * 1000, 2)
    results["batch32_throughput_items_per_s"] = round(N_ITEMS / batch32_total, 2)
    results["mem_peak_mb"] = round(gpu_mem_mb(), 1)

    del model, tok
    reset_cuda()
    return results


def benchmark_wildguard() -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reset_cuda()
    logger.info("Loading WildGuard 7B")
    t0 = time.time()
    name = "allenai/wildguard"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    load_time = time.time() - t0
    mem_after_load = gpu_mem_mb()

    # Build a single WildGuard prompt (instruction-tuned format)
    prompt_template = (
        "<|user|>\n[INST] You are a classifier for analyzing interactions "
        "between humans and LLMs. Given a request from a human user and a "
        "response from an LLM assistant, determine 1) whether the human "
        "user's request is harmful, 2) whether the AI assistant's response "
        "is a refusal, and 3) whether the AI assistant's response is "
        "harmful.\n\nHuman user:\n{q}\n\nAI assistant:\n{r}\n\n---\n\n"
        "Answers: [/INST]\n<|assistant|>\n"
    )
    prompt = prompt_template.format(q=SAMPLE_QUERY, r=SAMPLE_RESPONSE)

    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    # Warmup
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=8, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Batch 1 latency: 5 trials
    times = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**enc, max_new_tokens=8, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    batch1_mean = float(np.mean(times))

    results = {
        "model": "wildguard_7b",
        "param_count_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "load_time_s": round(load_time, 3),
        "mem_after_load_mb": round(mem_after_load, 1),
        "batch1_latency_per_item_ms": round(batch1_mean * 1000, 2),
        "batch1_throughput_items_per_s": round(1 / batch1_mean, 2),
        "mem_peak_mb": round(gpu_mem_mb(), 1),
        "note": "max_new_tokens=8, fp16",
    }
    del model, tok
    reset_cuda()
    return results


def benchmark_llamaguard() -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reset_cuda()
    logger.info("Loading LLaMA-Guard 3 8B")
    name = "meta-llama/Llama-Guard-3-8B"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    load_time = time.time() - t0
    mem_after_load = gpu_mem_mb()

    # Build LG-3 chat prompt; apply_chat_template can return Tensor or BatchEncoding
    conv = [
        {"role": "user", "content": SAMPLE_QUERY},
        {"role": "assistant", "content": SAMPLE_RESPONSE},
    ]
    chat_out = tok.apply_chat_template(conv, return_tensors="pt")
    if hasattr(chat_out, "input_ids"):
        input_ids = chat_out.input_ids
    else:
        input_ids = chat_out
    input_ids = input_ids.to(DEVICE)

    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=8, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=8, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    batch1_mean = float(np.mean(times))

    results = {
        "model": "llama_guard_3_8b",
        "param_count_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "load_time_s": round(load_time, 3),
        "mem_after_load_mb": round(mem_after_load, 1),
        "batch1_latency_per_item_ms": round(batch1_mean * 1000, 2),
        "batch1_throughput_items_per_s": round(1 / batch1_mean, 2),
        "mem_peak_mb": round(gpu_mem_mb(), 1),
        "note": "max_new_tokens=8, fp16",
    }
    del model, tok
    reset_cuda()
    return results


def main():
    report = {}

    # v3
    try:
        report["v3"] = benchmark_deberta("v3", MODELS_DIR / "deberta_bioguard_v3_balanced")
        logger.info("v3: %s", json.dumps(report["v3"], indent=2))
    except Exception as e:
        logger.exception("v3 failed: %s", e)
        report["v3"] = {"error": str(e)}

    # v4
    try:
        report["v4"] = benchmark_deberta("v4", MODELS_DIR / "deberta_bioguard_v4_response_diverse")
        logger.info("v4: %s", json.dumps(report["v4"], indent=2))
    except Exception as e:
        logger.exception("v4 failed: %s", e)
        report["v4"] = {"error": str(e)}

    # WildGuard 7B
    try:
        report["wildguard_7b"] = benchmark_wildguard()
        logger.info("WildGuard: %s", json.dumps(report["wildguard_7b"], indent=2))
    except Exception as e:
        logger.exception("WildGuard failed: %s", e)
        report["wildguard_7b"] = {"error": str(e)}

    # LLaMA-Guard 3 8B
    try:
        report["llama_guard_3_8b"] = benchmark_llamaguard()
        logger.info("LG3: %s", json.dumps(report["llama_guard_3_8b"], indent=2))
    except Exception as e:
        logger.exception("LG3 failed: %s", e)
        report["llama_guard_3_8b"] = {"error": str(e)}

    # Add comparison summary
    summary = {}
    for model in ["v3", "v4", "wildguard_7b", "llama_guard_3_8b"]:
        d = report.get(model, {})
        if "error" in d:
            continue
        summary[model] = {
            "params_M": d.get("param_count_m"),
            "load_time_s": d.get("load_time_s"),
            "mem_peak_mb": d.get("mem_peak_mb"),
            "batch1_latency_ms": d.get("batch1_latency_per_item_ms"),
            "batch32_throughput_items_s": d.get("batch32_throughput_items_per_s",
                                                  d.get("batch1_throughput_items_per_s")),
        }
    report["summary"] = summary

    out = METRICS_DIR / "phase3_probe_a3_latency_memory.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("\nSummary:\n%s", json.dumps(summary, indent=2))
    logger.info("Saved: %s", out)


if __name__ == "__main__":
    main()
