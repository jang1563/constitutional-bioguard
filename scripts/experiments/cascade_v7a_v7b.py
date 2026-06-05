#!/usr/bin/env python
"""Cascade router: v7.A (Stage 1) → v7.B (Stage 2) for biosafety classification.

Stage 1 (v7.A, BioClinical ModernBERT-large 396M, FPR 0.046):
    - prob_unsafe < threshold_low  → PASS (pred=0), skip v7.B
    - prob_unsafe > threshold_high → FLAG (pred=1), skip v7.B
    - else (borderline)            → escalate to v7.B

Stage 2 (v7.B, Qwen3-4B QLoRA + CoT):
    - Generative inference, parse "Response harm: harmful/unharmful"
    - Use the parsed label as the final prediction

Default thresholds: 0.2 / 0.8. Calibrate on a held-out set before deployment.

See docs/V7B_OVER_REFUSAL_ANALYSIS.md for rationale.

Outputs:
    results/metrics/cascade_v7a_v7b_eval_{bench}.json — per-bench results
    results/metrics/cascade_v7a_v7b_summary.json — aggregate
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import torch

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR


# ── Locked benchmarks (same as v7.B eval) ────────────────────────────────
BENCHMARKS = [
    ("or_bench_hard_1k", "or_bench_hard_1k.jsonl"),
    ("false_reject_test", "false_reject_test.jsonl"),
    ("xstest", "xstest.jsonl"),
    ("simple_safety_bio", "simple_safety_bio.jsonl"),
    ("wildguard_native", "wildguard_test.jsonl"),
    ("saladbench_cbrn", "saladbench_cbrn.jsonl"),
    ("alert_cbrn", "alert_cbrn.jsonl"),
]


# ──────────────────────────────────────────────────────────────────────────
# Stage 1: v7.A loader / predictor (reuse evaluate_classifier)
# ──────────────────────────────────────────────────────────────────────────
def load_v7a():
    """Load v7.A BioClinical ModernBERT-large."""
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer,
    )
    model_dir = MODELS_DIR / "deberta_bioguard_v7a_bioclinical_modernbert_large"
    logger.info(f"Loading v7.A from {model_dir}")
    model, tokenizer = load_model_and_tokenizer(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer


def predict_v7a(model, tokenizer, queries, responses, batch_size=32):
    """Return list of prob_unsafe in [0,1] from v7.A."""
    from constitutional_bioguard.evaluation.evaluate_classifier import predict_batch
    preds = predict_batch(
        queries=list(queries),
        responses=list(responses),
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        normalize=True,
    )
    # predict_batch returns (label, confidence, prob_unsafe)
    return [p[2] for p in preds]


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: v7.B loader / predictor (reuse eval_v7b_qwen3_cot.py)
# ──────────────────────────────────────────────────────────────────────────
def load_v7b():
    """Load v7.B Qwen3-4B + LoRA adapter."""
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_v7b_qwen3_cot import load_model as load_v7b_inner
    model_dir = MODELS_DIR / "qwen3_bioguard_v7b_cot"
    logger.info(f"Loading v7.B from {model_dir}")
    model, tokenizer = load_v7b_inner(model_dir)
    return model, tokenizer


def predict_v7b(model, tokenizer, queries, responses, batch_size=4,
                max_new_tokens=512, think=True):
    """Return list of predicted labels (0/1) from v7.B for given subset.

    think=False uses /no_think (diagnostic-confirmed: cuts borderline FPR
    0.50->0.07 on XSTest; recall trade-off absorbed by v7.A in cascade).
    """
    if not queries:
        return [], []
    from eval_v7b_qwen3_cot import predict_batch
    results = predict_batch(
        model, tokenizer, queries, responses,
        batch_size=batch_size, max_new_tokens=max_new_tokens, think=think,
    )
    preds = [r["pred_label"] for r in results]
    return preds, results


# ──────────────────────────────────────────────────────────────────────────
# Cascade routing
# ──────────────────────────────────────────────────────────────────────────
def cascade_predict(v7a_probs, v7a_model_pair, v7b_model_pair,
                    queries, responses, threshold_low=0.2, threshold_high=0.8,
                    v7b_batch_size=4, v7b_think=True):
    """Apply cascade routing.

    Returns:
        final_preds: list of int (0/1)
        routes: list of str ("pass", "flag", "escalate")
        v7b_results: dict {original_idx: v7b_parsed_result} for escalated items
    """
    n = len(v7a_probs)
    final_preds = [None] * n
    routes = [None] * n
    escalate_idx = []

    for i, p in enumerate(v7a_probs):
        if p < threshold_low:
            final_preds[i] = 0
            routes[i] = "pass"
        elif p > threshold_high:
            final_preds[i] = 1
            routes[i] = "flag"
        else:
            routes[i] = "escalate"
            escalate_idx.append(i)

    n_escalate = len(escalate_idx)
    logger.info(f"  Routing: pass={routes.count('pass')}, "
                f"flag={routes.count('flag')}, escalate={n_escalate} "
                f"({n_escalate/n*100:.1f}%)")

    v7b_results_map = {}
    if escalate_idx:
        esc_q = [queries[i] for i in escalate_idx]
        esc_r = [responses[i] for i in escalate_idx]
        v7b_model, v7b_tok = v7b_model_pair
        logger.info(f"  Running v7.B on {n_escalate} escalated items...")
        v7b_preds, v7b_results = predict_v7b(
            v7b_model, v7b_tok, esc_q, esc_r, batch_size=v7b_batch_size,
            think=v7b_think,
        )
        for j, idx in enumerate(escalate_idx):
            final_preds[idx] = v7b_preds[j]
            v7b_results_map[idx] = v7b_results[j]

    return final_preds, routes, v7b_results_map


# ──────────────────────────────────────────────────────────────────────────
# Per-benchmark evaluation
# ──────────────────────────────────────────────────────────────────────────
def eval_benchmark(bench_name, bench_file, v7a_pair, v7b_pair,
                   threshold_low, threshold_high, v7b_think=True,
                   out_prefix="cascade_v7a_v7b"):
    fp = DATA_EXTERNAL / bench_file
    if not fp.exists():
        logger.warning(f"SKIP {bench_name}: {fp} not found")
        return None

    with open(fp) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    # WildGuard normalization (same as eval_v7b_qwen3_cot)
    if bench_name == "wildguard_native":
        rows = [r for r in rows
                if isinstance(r.get("response_harm_label"), str)
                and r["response_harm_label"].lower() in ("harmful", "unharmful")]
        for r in rows:
            r["query"] = r.get("prompt", r.get("query", ""))
            r["label"] = 1 if r["response_harm_label"].lower() == "harmful" else 0

    queries = [r.get("query", "") for r in rows]
    responses = [r.get("response", "") for r in rows]
    labels = [int(r.get("label", 0)) for r in rows]

    if not queries:
        return None

    logger.info(f"Evaluating {bench_name} ({len(queries)} items)...")
    t0 = time.time()

    # Stage 1: v7.A on all
    v7a_model, v7a_tok = v7a_pair
    v7a_probs = predict_v7a(v7a_model, v7a_tok, queries, responses, batch_size=32)

    # Cascade routing + Stage 2
    preds, routes, v7b_map = cascade_predict(
        v7a_probs, v7a_pair, v7b_pair, queries, responses,
        threshold_low=threshold_low, threshold_high=threshold_high,
        v7b_think=v7b_think,
    )

    elapsed = time.time() - t0
    logger.info(f"  {bench_name}: {elapsed:.1f}s ({elapsed/len(queries)*1000:.0f}ms/item)")

    # Metrics
    n = len(rows)
    n_flag = sum(preds)
    n_escalate = routes.count("escalate")
    overall = {
        "n": n,
        "n_flag": n_flag,
        "flag_rate": round(n_flag / n, 4),
        "escalation_rate": round(n_escalate / n, 4),
        "n_pass": routes.count("pass"),
        "n_flag_by_v7a": routes.count("flag"),
        "n_escalate": n_escalate,
        "inference_time_s": round(elapsed, 1),
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
    }

    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    if labels_arr.sum() > 0 and (labels_arr == 0).sum() > 0:
        from sklearn.metrics import f1_score, precision_score, recall_score
        overall["precision"] = round(float(precision_score(labels, preds, zero_division=0)), 4)
        overall["recall"] = round(float(recall_score(labels, preds, zero_division=0)), 4)
        overall["f1"] = round(float(f1_score(labels, preds, zero_division=0)), 4)
        mask0 = labels_arr == 0
        overall["fpr"] = round(float(preds_arr[mask0].mean()) if mask0.any() else 0, 4)

    # Save detailed
    out = {
        "overall": overall,
        "predictions": [
            {
                "label": int(labels[i]),
                "pred": int(preds[i]),
                "route": routes[i],
                "v7a_prob_unsafe": round(float(v7a_probs[i]), 4),
                "v7b_response_harm": v7b_map.get(i, {}).get("response_harm"),
            }
            for i in range(n)
        ],
    }
    out_path = METRICS_DIR / f"{out_prefix}_eval_{bench_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"  {bench_name}: {overall}")
    return overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold-low", type=float, default=0.2,
                   help="Below this v7.A prob_unsafe → PASS (pred=0)")
    p.add_argument("--threshold-high", type=float, default=0.8,
                   help="Above this v7.A prob_unsafe → FLAG (pred=1)")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Subset of benchmarks to run (default: all 7)")
    p.add_argument("--skip-v7b-load", action="store_true",
                   help="Diagnostic: skip v7.B (just v7.A baseline)")
    p.add_argument("--v7b-no-think", action="store_true",
                   help="Run v7.B stage-2 in /no_think mode (diagnostic-confirmed "
                        "borderline FPR 0.50->0.07). Writes cascade_v7a_v7b_nothink_*.")
    args = p.parse_args()

    v7b_think = not args.v7b_no_think
    out_prefix = "cascade_v7a_v7b" if v7b_think else "cascade_v7a_v7b_nothink"

    selected = BENCHMARKS
    if args.benchmarks:
        selected = [(n, f) for n, f in BENCHMARKS if n in args.benchmarks]

    logger.info(f"Cascade thresholds: low={args.threshold_low}, "
                f"high={args.threshold_high}")
    logger.info(f"Benchmarks: {[n for n, _ in selected]}")

    v7a_pair = load_v7a()
    v7b_pair = (None, None) if args.skip_v7b_load else load_v7b()

    logger.info(f"v7.B stage-2 mode: {'think' if v7b_think else 'no_think'} "
                f"(out_prefix={out_prefix})")

    all_results = {}
    for bench_name, bench_file in selected:
        r = eval_benchmark(bench_name, bench_file, v7a_pair, v7b_pair,
                           args.threshold_low, args.threshold_high,
                           v7b_think=v7b_think, out_prefix=out_prefix)
        if r:
            all_results[bench_name] = r

    summary_path = METRICS_DIR / f"{out_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": out_prefix,
            "v7b_think": v7b_think,
            "threshold_low": args.threshold_low,
            "threshold_high": args.threshold_high,
            "benchmarks": all_results,
        }, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    # Comparison-ready table
    print("\n=== Cascade v7.A→v7.B Evaluation Summary ===")
    print(f"Thresholds: low={args.threshold_low}, high={args.threshold_high}")
    print(f"{'Benchmark':<22} {'N':>5} {'Esc%':>6} {'Flag%':>6} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}")
    print("-" * 70)
    for bench, m in all_results.items():
        print(f"{bench:<22} {m['n']:>5} {m['escalation_rate']*100:>5.1f}% "
              f"{m['flag_rate']*100:>5.1f}% "
              f"{m.get('precision','-'):>6} {m.get('recall','-'):>6} "
              f"{m.get('f1','-'):>6} {m.get('fpr','-'):>6}")


if __name__ == "__main__":
    main()
