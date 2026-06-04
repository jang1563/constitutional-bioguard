#!/usr/bin/env python
# Did v8bh preserve prompt/response orthogonality (decorrelated FPs)?
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).parent.parent
PROMPT = ROOT / "models" / "deberta_v7c_distill_bioborder" / "final"
V8B = ROOT / "models" / "deberta_bioguard_v8b"
V8BH = ROOT / "models" / "deberta_bioguard_v8bh"


def score(md, qs, rs, dev, bs=64):
    tok = AutoTokenizer.from_pretrained(str(md))
    m = AutoModelForSequenceClassification.from_pretrained(
        str(md), dtype=torch.float32).to(dev).eval()
    out = []
    for i in range(0, len(qs), bs):
        qb, rb = qs[i:i + bs], rs[i:i + bs]
        if any(rb):
            enc = tok(qb, rb, max_length=512, truncation=True, padding=True, return_tensors="pt")
        else:
            enc = tok(qb, max_length=512, truncation=True, padding=True, return_tensors="pt")
        enc = enc.to(dev)
        with torch.no_grad():
            out += torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().tolist()
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(out)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use bio-overrefusal-v0.1 if available (original bridge set)
    src = ROOT / "data/bio_overrefusal_queries.jsonl"
    if not src.exists():
        src = ROOT / "data/external/fortress_safe_heldout.jsonl"
    rows = [json.loads(l) for l in open(src) if l.strip()]
    Q = [r["query"] for r in rows]
    R = [r.get("response") or "" for r in rows]
    print(f"set: {src.name} (n={len(rows)})")

    pp = score(PROMPT, Q, [""] * len(Q), dev)
    has_resp = any(R)
    pb = score(V8B, Q, R if has_resp else [""] * len(Q), dev)
    pbh = score(V8BH, Q, R if has_resp else [""] * len(Q), dev)

    fp = pp >= 0.5
    fb = pb >= 0.5
    fbh = pbh >= 0.5

    def overlap(a, b, lab="a&b"):
        return f"a={a.sum()}, b={b.sum()}, both={int((a&b).sum())}, a_only={int((a&~b).sum())}, b_only={int((~a&b).sum())}"

    print(f"\nover-refusal flags (all items are benign):")
    print(f"  prompt head:  {fp.sum()}/{len(rows)} = {fp.mean():.3f}")
    print(f"  v8b (orig):   {fb.sum()}/{len(rows)} = {fb.mean():.3f}")
    print(f"  v8bh (debias):{fbh.sum()}/{len(rows)} = {fbh.mean():.3f}")
    print(f"\nERROR OVERLAP (do prompt + v8bh fail on SAME items? = NOT orthogonal):")
    print(f"  prompt vs v8b:  {overlap(fp, fb)}")
    print(f"  prompt vs v8bh: {overlap(fp, fbh)}")
    print(f"\nKEY: if v8bh's FPs (b_only) are also flagged by prompt -> orthogonality LOST")
    print(f"     if prompt clears v8bh's FPs (b_only large, both small) -> orthogonality KEPT")


if __name__ == "__main__":
    main()
