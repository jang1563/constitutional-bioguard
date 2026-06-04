#!/usr/bin/env python
# Integrity audit: check for train/eval leakage in the claims.
import json
import hashlib
from pathlib import Path


def qh(s):
    return hashlib.md5(str(s).strip().lower().encode()).hexdigest()


def load(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def qset(rows, key="query"):
    return set(qh(r.get(key) or r.get("prompt") or "") for r in rows)


print("=" * 64)
print("(A) v8bh FORTRESS-debiasing: query overlap between train-half and held-out eval")
v8bh_train = load("data/processed/v8bh_train.jsonl")
fort_train_q = qset([r for r in v8bh_train if r.get("source") == "fortress_safe_resp"])
heldout = load("data/external/fortress_safe_heldout.jsonl")
heldout_q = qset(heldout)
overlap = fort_train_q & heldout_q
print(f"  fortress train-half queries: {len(fort_train_q)}")
print(f"  held-out eval queries:       {len(heldout_q)}")
print(f"  QUERY overlap (leakage):     {len(overlap)}  ({len(overlap)/max(1,len(heldout_q)):.1%} of eval)")

print("=" * 64)
print("(B) PROMPT head: training data vs FORTRESS-CBRN eval queries")
# prompt head trained on distill_pool_bioborder.jsonl
pool = load("data/processed/distill_pool_bioborder.jsonl")
pool_q = qset(pool)
fort_cbrn = load("data/external/fortress_cbrn.jsonl")
fort_cbrn_q = qset(fort_cbrn)
ov2 = pool_q & fort_cbrn_q
print(f"  distill_pool_bioborder queries: {len(pool_q)}")
print(f"  fortress_cbrn eval queries:     {len(fort_cbrn_q)}")
print(f"  QUERY overlap (leakage):        {len(ov2)}")

print("=" * 64)
print("(C) generated bio-borderline (prompt head neg) vs FORTRESS-CBRN")
gen = load("data/processed/bio_borderline_benign.jsonl") + load("data/processed/bio_borderline_benign_llm.jsonl")
ov3 = qset(gen) & fort_cbrn_q
print(f"  generated bio-borderline: {len(qset(gen))}, overlap with fortress_cbrn: {len(ov3)}")

print("=" * 64)
print("(D) v8b/v8bh training vs real_response_bio_large eval (query-hash)")
v8b_train = load("data/processed/v8b_train.jsonl") + load("data/processed/v8b_val.jsonl")
v8b_q = qset(v8b_train)
rrbl = load("data/external/real_response_bio_large.jsonl")
ov4 = v8b_q & qset(rrbl)
print(f"  v8b train+val queries: {len(v8b_q)}; real_response_bio_large queries: {len(qset(rrbl))}")
print(f"  QUERY overlap (should be ~0 after decon): {len(ov4)}")

print("=" * 64)
print("(E) prompt head distill pool vs bio_clean_eval (recall-claim eval)")
bce = load("data/processed/bio_clean_eval.jsonl")
ov5 = pool_q & qset(bce)
print(f"  bio_clean_eval queries: {len(qset(bce))}; overlap with distill pool: {len(ov5)}")
