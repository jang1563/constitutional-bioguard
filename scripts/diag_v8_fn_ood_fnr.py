#!/usr/bin/env python
"""Diagnose v8 koaug3 FALSE NEGATIVES on ood_fnr (the should-REFUSE harmful set).

Threshold calibration showed FNR is a flat ~0.073 floor (confident errors, not
borderline). This characterises WHICH harmful records the model confidently
misses, to decide: coverage-gap (augment) vs label-noise (relabel) vs hard
adversarial. ood_fnr = published benchmark harmful (WildGuard / SALAD /
ConstitutionRules) — NOT user session data.

Read-only inference. Text format matches training.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_text(r: dict) -> str:
    q = (r.get("query", "") or "").strip()
    resp = (r.get("response", "") or "").strip()
    return f"{q} [SEP] {resp[:256]}" if resp else q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", default="data/splits/ood_fnr.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    harm = [r for r in recs if r.get("binary_label") == "negative"]  # label 1 = harmful

    texts = [build_text(r) for r in harm]
    probs = []
    for i in range(0, len(texts), args.batch_size):
        enc = tok(texts[i:i + args.batch_size], max_length=512, truncation=True,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
        probs.extend(p)
    for r, p in zip(harm, probs):
        r["_p"] = p
        r["_miss"] = p < args.threshold  # false negative

    miss = [r for r in harm if r["_miss"]]

    # p-distribution of the harmful set (is the miss-mass confident, p~0?)
    buckets = {"p<0.01": 0, "0.01-0.1": 0, "0.1-0.5": 0, "0.5-0.9": 0, "p>=0.9": 0}
    for p in probs:
        if p < 0.01: buckets["p<0.01"] += 1
        elif p < 0.1: buckets["0.01-0.1"] += 1
        elif p < 0.5: buckets["0.1-0.5"] += 1
        elif p < 0.9: buckets["0.5-0.9"] += 1
        else: buckets["p>=0.9"] += 1

    def breakdown(key):
        tot = collections.Counter(r.get(key) for r in harm)
        m = collections.Counter(r.get(key) for r in miss)
        return {str(k): {"miss": m.get(k, 0), "total": tot[k],
                         "miss_rate": round(m.get(k, 0) / tot[k], 3)}
                for k in tot if k is not None}

    out = {
        "model": Path(args.model_dir).parent.name,
        "data": Path(args.data).name,
        "threshold": args.threshold,
        "n_harm": len(harm),
        "n_miss": len(miss),
        "fnr": round(len(miss) / len(harm), 4) if harm else 0.0,
        "p_distribution": buckets,
        "by_source": breakdown("source"),
        "by_source_detail": breakdown("source_detail"),
        "by_content_domain": breakdown("content_domain"),
        "by_cr_domain": breakdown("cr_domain"),
        "by_classification_original": breakdown("classification_original"),
        "miss_examples": sorted(
            [{"p": round(r["_p"], 4), "source": r.get("source"),
              "source_detail": r.get("source_detail"),
              "content_domain": r.get("content_domain"),
              "cr_domain": r.get("cr_domain"),
              "has_response": bool((r.get("response") or "").strip()),
              "query": (r.get("query", "") or "")[:240]} for r in miss],
            key=lambda x: x["p"]),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(json.dumps({k: out[k] for k in ["model", "n_harm", "n_miss", "fnr", "p_distribution"]}, indent=2))
    print("\n=== miss by source (miss/total, rate) ===")
    for k, v in sorted(out["by_source"].items(), key=lambda x: -x[1]["miss"]):
        print(f"  {v['miss']:4d}/{v['total']:5d}  {v['miss_rate']:.3f}  {k}")
    print("\n=== miss by source_detail (top 12) ===")
    for k, v in sorted(out["by_source_detail"].items(), key=lambda x: -x[1]["miss"])[:12]:
        print(f"  {v['miss']:4d}/{v['total']:5d}  {v['miss_rate']:.3f}  {k}")
    print("\n=== miss by content_domain (top 12) ===")
    for k, v in sorted(out["by_content_domain"].items(), key=lambda x: -x[1]["miss"])[:12]:
        print(f"  {v['miss']:4d}/{v['total']:5d}  {v['miss_rate']:.3f}  {k}")
    print("\n=== 15 most-confident misses (lowest p) ===")
    for e in out["miss_examples"][:15]:
        print(f"  p={e['p']:.4f} [{e['source']}/{e.get('source_detail')}] {e['query'][:120]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
