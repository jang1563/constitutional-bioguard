#!/usr/bin/env python
# Health-ORSC unavailable; build a BIO over-refusal proxy from OR-Bench (bench-llm/or-bench)
# using our bio_keywords filter, decontaminated vs our training. Label=0 (all should pass).
import json
import hashlib
from datasets import load_dataset

KW = [k.lower() for k in json.load(open("configs/bio_keywords_adv.json"))["keywords"]]


def isbio(q):
    t = str(q).lower()
    return any(k in t for k in KW)


def qh(s):
    return hashlib.md5(str(s).strip().lower().encode()).hexdigest()


# leakage block vs prompt-head training + existing evals
block = set()
for fn in ["data/processed/distill_pool_bioborder.jsonl",
           "data/external/fortress_cbrn.jsonl",
           "data/external/fortress_safe_heldout.jsonl",
           "data/external/real_response_bio_large.jsonl",
           "data/external/sosbench_bio.jsonl"]:
    try:
        for l in open(fn):
            if l.strip():
                r = json.loads(l)
                block.add(qh(r.get("query") or r.get("prompt") or ""))
    except FileNotFoundError:
        pass
print("leakage block:", len(block))

ds = load_dataset("bench-llm/or-bench", "or-bench-80k")
sp = list(ds.keys())[0]
print("OR-Bench keys:", ds[sp].column_names, "n=", len(ds[sp]))

qcol = "prompt" if "prompt" in ds[sp].column_names else "query"
rows = []
for r in ds[sp]:
    q = r.get(qcol, "")
    if not q or qh(q) in block:
        continue
    if isbio(q):
        rows.append({"query": q, "label": 0, "source": "or_bench_bio"})
# dedup
seen, ded = set(), []
for r in rows:
    h = qh(r["query"])
    if h not in seen:
        seen.add(h)
        ded.append(r)
with open("data/external/orbench_bio.jsonl", "w") as f:
    for r in ded:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("orbench_bio:", len(ded), "unique bio-relevant over-refusal prompts")
print("sample:", ded[0]["query"][:120] if ded else "(none)")
