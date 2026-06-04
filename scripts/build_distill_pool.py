#!/usr/bin/env python
"""Step 1 (release plan): assemble the bio-prompt DISTILLATION POOL for compressing the
v7.C-aug2 8B generative teacher into a small DeBERTa student (HarmAug-style, reuse-only,
no generated harmful content).

Pool = reuse-only bio PROMPTS (harmful + benign) from:
  harmful: data/external/{alert_cbrn_strict,saladbench_cbrn,advbench_bio}.jsonl + v7.C
           training harmful prompts
  benign:  data/splits/train.jsonl legit bio + v7.C training benign prompts
LEAKAGE-EXCLUDED from every eval set (bio_clean_eval, ood_fpr, bio-overrefusal-v0.1) so the
student's recall/over-refusal eval stays valid. Output: {query, hard_label} (hard_label
1=harmful 0=benign, used for the BCE term). The teacher soft labels are added in a separate
GPU pass (score_distill_pool). Deduped, seeded.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path("data")
OUT = DATA / "processed" / "distill_pool.jsonl"
KW = [k.lower() for k in json.load(open("configs/bio_keywords_adv.json")).get("keywords", [])]
QRE = re.compile(r"Human user:\n(.*?)\n\nAI assistant:", re.DOTALL)


def load(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def is_bio(q, r=""):
    t = (str(q or "") + " " + str(r or "")).lower()
    return any(k in t for k in KW)


def main():
    block = set()
    for p in ["data/processed/bio_clean_eval.jsonl", "data/splits/ood_fpr.jsonl",
              "data/bio_overrefusal_queries.jsonl"]:
        for r in load(p):
            block.add((r.get("query") or r.get("prompt") or "").strip())
    block.discard("")
    print(f"eval-exclusion (leakage block): {len(block)}")

    pool = {}  # query -> hard_label (dedupe by query)

    def add(q, label):
        q = (q or "").strip()
        if q and q not in block and q not in pool and is_bio(q):
            pool[q] = label

    # harmful from external CBRN/bio prompt sets
    for p in ["data/external/alert_cbrn_strict.jsonl", "data/external/saladbench_cbrn.jsonl",
              "data/external/advbench_bio.jsonl"]:
        for r in load(p):
            add(r.get("query") or r.get("prompt"), 1)
    # benign from the over-refusal training split
    for r in load("data/splits/train.jsonl"):
        if r.get("binary_label") == "legitimate":
            add(r.get("query"), 0)
    # v7.C training prompts (the teacher's own distribution), by prompt_harm
    for r in load("data/processed/v7c_nocot_train_aug2.jsonl"):
        m = QRE.search(r["messages"][1]["content"])
        if m:
            add(m.group(1), 1 if r.get("prompt_harm") == 1 else 0)

    rows = [{"query": q, "hard_label": lab} for q, lab in pool.items()]
    dist = Counter(r["hard_label"] for r in rows)
    print(f"distill pool: {len(rows)}  (harmful {dist.get(1,0)} / benign {dist.get(0,0)})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
