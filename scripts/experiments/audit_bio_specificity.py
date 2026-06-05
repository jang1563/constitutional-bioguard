#!/usr/bin/env python
# GAP #1: is v8bh actually BIO-SPECIALIZED, or a general response-harm guard?
# Compare v8bh flag-rate on BIO harmful responses vs NON-BIO harmful responses.
# bio-selectivity = recall(bio-harm) / flag-rate(non-bio-harm). >>1 = bio-specific.
# GAP #2: verify which head the conformal cert was computed on (v8b vs v8bh).
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

ROOT = Path(__file__).parent.parent
V8BH = ROOT / "models" / "deberta_bioguard_v8bh"
V8B = ROOT / "models" / "deberta_bioguard_v8b"
KW = [k.lower() for k in json.load(open(ROOT / "configs/bio_keywords_adv.json"))["keywords"]]


def isbio(q, r):
    t = (str(q) + " " + str(r)).lower()
    return any(k in t for k in KW)


def qh(s):
    return hashlib.md5(str(s).strip().lower().encode()).hexdigest()


def score(md, qs, rs, dev, bs=64):
    tok = AutoTokenizer.from_pretrained(str(md))
    m = AutoModelForSequenceClassification.from_pretrained(str(md), dtype=torch.float32).to(dev).eval()
    out = []
    for i in range(0, len(qs), bs):
        enc = tok(qs[i:i + bs], rs[i:i + bs], max_length=512, truncation=True, padding=True,
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            out += torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().tolist()
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.array(out)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # v8b training query block (avoid leakage in the non-bio sample)
    block = set()
    for fn in ["data/processed/v8b_train.jsonl", "data/processed/v8b_val.jsonl"]:
        for l in open(ROOT / fn):
            if l.strip():
                block.add(qh(json.loads(l).get("query") or ""))
    # BeaverTails 330k_test: collect bio-harmful and non-bio-harmful responses
    bd = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    bio_h, nonbio_h = [], []
    for r in bd:
        if r.get("is_safe") is True:
            continue
        q, resp = r.get("prompt", ""), r.get("response", "")
        if qh(q) in block:
            continue
        (bio_h if isbio(q, resp) else nonbio_h).append((q, resp))
    rng = np.random.RandomState(0)
    nonbio_h = [nonbio_h[i] for i in rng.choice(len(nonbio_h), min(400, len(nonbio_h)), replace=False)]
    bio_h = bio_h[:400]
    print(f"bio-harmful={len(bio_h)}, non-bio-harmful={len(nonbio_h)} (BeaverTails 330k_test, leakage-clean)")

    for tag, md in [("v8bh", V8BH), ("v8b", V8B)]:
        sb = score(md, [q for q, _ in bio_h], [r for _, r in bio_h], dev)
        sn = score(md, [q for q, _ in nonbio_h], [r for _, r in nonbio_h], dev)
        rb, rn = (sb >= 0.5).mean(), (sn >= 0.5).mean()
        sel = rb / rn if rn > 0 else float("inf")
        print(f"  {tag}: bio-harm flag={rb:.3f}  non-bio-harm flag={rn:.3f}  selectivity S={sel:.2f}")
    print("  (S>>1 = bio-specialized; S~1 = general harm guard)")

    # GAP #2: conformal provenance
    print("\nGAP #2 conformal cert provenance:")
    rc = json.load(open(ROOT / "results/realresp_curve_data.json"))
    vc = json.load(open(ROOT / "results/v8bh_compare.json"))
    pr = np.array(rc["p_resp"])
    v8bh_s = np.array(vc["large_v8bd"])
    match = np.allclose(pr[:len(v8bh_s)], v8bh_s[:len(pr)], atol=1e-3) if len(pr) == len(v8bh_s) else False
    print(f"  realresp_curve_data (conformal input) == v8bh scores? {match}")
    print(f"  -> conformal cert was computed on {'v8bh' if match else 'v8b (MISMATCH -- must re-run on v8bh)'}")


if __name__ == "__main__":
    main()
