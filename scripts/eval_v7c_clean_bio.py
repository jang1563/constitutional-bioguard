#!/usr/bin/env python
"""Phase H: v7.C (Llama-3.1-8B + QLoRA) OOD eval on clean bio benchmark.

v7.C is a generative classifier (Llama-3.1-8B-Instruct + 4-bit QLoRA adapter).
Predict on clean bio eval (120 pos + 881 neg) and parse "Prompt harm: harmful/unharmful".
Compare PROMPT-axis recall to pdual_ho baseline (0.366 learned).

Prompt construction MUST match v7.C training (build_v7c_nocot_data.py keeps the
v7.B2 system message verbatim). The canonical SYSTEM_TMPL / DUAL_LABEL_POLICY /
BIO_TAXONOMY / USER_TMPL_PLAIN and parse_output are therefore imported from the
v7.B eval (single source of truth) instead of re-spelled here -- a reworded system
prompt or a yes/no parser makes train != eval and silently zeroes recall (the
v7.B2 mismatch lesson; see eval_v7b_qwen3_cot.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root -> constitutional_bioguard pkg
sys.path.insert(0, str(Path(__file__).parent))          # scripts/  -> sibling eval module

import numpy as np
import torch

# Canonical, train-matched prompt + parser (see module docstring).
from eval_v7b_qwen3_cot import (
    BIO_TAXONOMY,
    DUAL_LABEL_POLICY,
    SYSTEM_TMPL,
    USER_TMPL_PLAIN,
    parse_output,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from constitutional_bioguard.config import DATA_PROCESSED, MODELS_DIR

NO_RESPONSE = "(no response provided)"


def load_model(base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Load Llama-3.1-8B with v7.C QLoRA adapter.

    Adapter dir defaults to the shipped v7.C; override with env V7C_ADAPTER_DIR
    (e.g. the targeted-retrain adapter) to evaluate a different checkpoint.
    """
    adapter_dir = os.environ.get(
        "V7C_ADAPTER_DIR", str(MODELS_DIR / "llama31_8b_bioguard_v7c" / "adapter"))
    print(f"Loading {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    print("Loading base model with 4-bit quant...")
    # Match v7.C training quantization exactly (train_v7c_llama_nocot.py):
    # nf4 + bf16 compute + double quant, model dtype bf16.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    print(f"Loading v7.C adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def build_messages(query: str, response: str) -> list[dict]:
    """System + user messages matching the v7.C training distribution (train == eval)."""
    system = SYSTEM_TMPL.format(policy=DUAL_LABEL_POLICY, taxonomy=BIO_TAXONOMY)
    user = USER_TMPL_PLAIN.format(query=query[:2000], response=(response or "")[:2000])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def predict_batch(model, tokenizer, rows, use_response: bool, verbose: int = 0):
    """PROMPT-harm prediction per row: 1=harmful, 0=unharmful, None=unparseable."""
    preds = []
    n_unparsed = 0
    for idx, r in enumerate(rows):
        resp = (r.get("response") or "") if use_response else NO_RESPONSE
        messages = build_messages(r["query"], resp)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode ONLY the generated continuation, not the echoed prompt.
        gen = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True)
        ph = parse_output(gen)["prompt_harm"]  # harmful | unharmful | unknown
        if ph == "harmful":
            pred = 1
        elif ph == "unharmful":
            pred = 0
        else:
            pred = None
            n_unparsed += 1
        preds.append(pred)
        if verbose and idx < verbose:
            print(f"  [v idx={idx} label={r.get('label')} src={r.get('source')}] "
                  f"prompt_harm={ph!r} gen={gen!r}")
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(rows)}] processed")
    if n_unparsed:
        print(f"  !! unparseable prompt_harm: {n_unparsed}/{len(rows)}")
    return preds


def _recall(preds):
    """(mean over parseable, n_parsed, n_unparsed). nan if nothing parsed."""
    ok = [p for p in preds if p is not None]
    val = float(np.mean(ok)) if ok else float("nan")
    return val, len(ok), sum(p is None for p in preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, eval only this many pos AND this many neg (smoke test)")
    ap.add_argument("--verbose", type=int, default=0,
                    help="print raw generation for the first N rows of each split")
    ap.add_argument("--response", choices=["none", "real"], default="none",
                    help="'none' = prompt-only screening (matches pdual_ho baseline, default); "
                         "'real' = feed the benchmark response (matches training distribution)")
    args = ap.parse_args()
    use_response = args.response == "real"

    # Load clean bio eval
    p = DATA_PROCESSED / "bio_clean_eval.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    if args.limit:
        pos, neg = pos[:args.limit], neg[:args.limit]
    print(f"bio_clean_eval: {len(pos)} pos, {len(neg)} neg  (response={args.response})")

    # Load v7.C
    print("Loading Llama-3.1-8B + v7.C adapter...")
    model, tokenizer = load_model()

    # Prompt-axis eval
    print("=== PROMPT-AXIS (bio recall + benign-bio over-refusal) ===")
    print(f"Predicting on {len(pos)} positives...")
    pos_preds = predict_batch(model, tokenizer, pos, use_response, args.verbose)
    print(f"Predicting on {len(neg)} negatives...")
    neg_preds = predict_batch(model, tokenizer, neg, use_response, args.verbose)

    recall, n_pos_ok, n_pos_unp = _recall(pos_preds)
    overref, n_neg_ok, n_neg_unp = _recall(neg_preds)
    print(f"  bio recall (n={n_pos_ok}/{len(pos)} parsed, {n_pos_unp} unparsed): {recall:.3f}")
    print(f"  benign-bio over-refusal (n={n_neg_ok}/{len(neg)} parsed, {n_neg_unp} unparsed): {overref:.3f}")

    # Per-source breakdown (None-safe)
    src_flags = {}
    for r, pr in zip(pos, pos_preds):
        if pr is None:
            continue
        src_flags.setdefault(r["source"], []).append(int(pr))
    print("  per-source recall:")
    for s, f in sorted(src_flags.items()):
        print(f"    {s:28s} n={len(f):3d} recall={np.mean(f):.3f}")

    # Compare to pdual_ho
    print("\n=== COMPARISON: pdual_ho vs v7.C ===")
    print("  pdual_ho (DeBERTa-base, saladbench OOD): learned recall 0.366")
    print(f"  v7.C     (Llama-3.1-8B, saladbench OOD): prompt recall {recall:.3f}")
    print(f"  delta: {recall - 0.366:+.3f}")


if __name__ == "__main__":
    main()
