#!/usr/bin/env python
"""Diagnose v8 baseline false positives on ood_fpr (the held-out user session/IDE logs).

Read-only inference. Purpose: identify WHICH legitimate real-usage queries get
over-flagged, to guide EXTERNAL (non-session) augmentation. The session data stays
held-out and is NEVER used for training — this is diagnosis only.

Text format matches training: query + " [SEP] " + response[:256] (query-only if no response).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_text(r: dict) -> str:
    q = (r.get("query", "") or "").strip()
    resp = (r.get("response", "") or "").strip()
    return f"{q} [SEP] {resp[:256]}" if resp else q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
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
    legit = [r for r in recs if r.get("binary_label") == "legitimate"]

    texts = [build_text(r) for r in legit]
    probs: list[float] = []
    for i in range(0, len(texts), args.batch_size):
        enc = tok(texts[i:i + args.batch_size], max_length=512, truncation=True,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
        probs.extend(p)

    for r, p in zip(legit, probs):
        r["_p"] = p
        r["_fp"] = p >= args.threshold

    fps = [r for r in legit if r["_fp"]]
    n_resp = sum(1 for r in legit if (r.get("response") or "").strip())

    def breakdown(key: str) -> dict:
        tot = collections.Counter(r.get(key, "") for r in legit)
        fp = collections.Counter(r.get(key, "") for r in fps)
        return {
            k: {"fp": fp.get(k, 0), "total": tot[k], "fp_rate": round(fp.get(k, 0) / tot[k], 3)}
            for k in tot
        }

    out = {
        "model": Path(args.model_dir).name,
        "data": Path(args.data).name,
        "threshold": args.threshold,
        "n_legit": len(legit),
        "n_with_response": n_resp,
        "n_fp": len(fps),
        "fpr": round(len(fps) / len(legit), 4) if legit else 0.0,
        "by_source": breakdown("source"),
        "by_content_domain": breakdown("content_domain"),
        "by_context_level": breakdown("context_level"),
        "by_legitimacy_tier": breakdown("legitimacy_tier"),
        "by_block_trigger": breakdown("block_trigger"),
        "fp_examples": sorted(
            [
                {
                    "p": round(r["_p"], 3),
                    "source": r.get("source"),
                    "content_domain": r.get("content_domain"),
                    "legitimacy_tier": r.get("legitimacy_tier"),
                    "context_level": r.get("context_level"),
                    "block_trigger": r.get("block_trigger"),
                    "protein_category": r.get("protein_category"),
                    "has_response": bool((r.get("response") or "").strip()),
                    "query": (r.get("query", "") or "")[:300],
                }
                for r in fps
            ],
            key=lambda x: -x["p"],
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(json.dumps({k: out[k] for k in
                      ["model", "data", "n_legit", "n_with_response", "n_fp", "fpr"]}, indent=2))
    print("\n=== FP by source (fp/total, rate) ===")
    for k, v in sorted(out["by_source"].items(), key=lambda x: -x[1]["fp"]):
        print(f"  {v['fp']:4d}/{v['total']:4d}  {v['fp_rate']:.3f}  {k}")
    print("\n=== FP by content_domain ===")
    for k, v in sorted(out["by_content_domain"].items(), key=lambda x: -x[1]["fp"]):
        print(f"  {v['fp']:4d}/{v['total']:4d}  {v['fp_rate']:.3f}  {k}")
    print("\n=== FP by context_level ===")
    for k, v in sorted(out["by_context_level"].items(), key=lambda x: -x[1]["fp"]):
        print(f"  {v['fp']:4d}/{v['total']:4d}  {v['fp_rate']:.3f}  {k}")
    print("\n=== Top 20 FP examples (highest P-harmful) ===")
    for e in out["fp_examples"][:20]:
        print(f"  p={e['p']:.3f} [{e['source']}/{e['content_domain']}] {e['query'][:140]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
