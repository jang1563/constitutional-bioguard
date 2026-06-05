#!/usr/bin/env python
"""Step 1b: augment the distillation pool with leakage-safe BORDERLINE-BENIGN hard negatives
from FalseReject-Train (HF AmazonScience/FalseReject, COLM 2025), to close the borderline-benign
over-refusal gap (pilot: student 0.83 vs teacher 0.17 on OR-Bench-health). Research basis:
docs/STEP1B_RESEARCH_2026-06-03.md (SFT on FalseReject cuts over-refusal without hurting recall;
hard-label borderline-benign breaks the teacher's 0.166 ceiling).

FalseReject is constructed by a different (multi-agent graph) method than OR-Bench, so it is
distinct from the OR-Bench-health eval, BUT it seeds from existing safety datasets -> we
DECONTAMINATE vs every eval set (exact-normalized + shared 8-gram) before use. We then take a
per-category-capped stratified sample (default ~2500) so the general borderline-benign does not
dominate the 2442 bio pool, and emit them as hard_label=0 (benign), source=falsereject.

Output: data/processed/distill_pool_aug.jsonl = distill_pool.jsonl (bio) + FR borderline-benign.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
POOL = ROOT / "data" / "processed" / "distill_pool.jsonl"
OUT = ROOT / "data" / "processed" / "distill_pool_aug.jsonl"
EVAL_SETS = ["data/processed/bio_clean_eval.jsonl", "data/splits/ood_fpr.jsonl",
             "data/bio_overrefusal_queries.jsonl"]

_norm = re.compile(r"\s+")


def norm(s):
    return _norm.sub(" ", str(s or "").lower().strip())


def ngrams(toks, n=8):
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500, help="target FalseReject negatives to add")
    ap.add_argument("--cap", type=int, default=120, help="per-category cap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)

    # 1) eval decontamination index (exact-normalized strings + 8-grams)
    eval_exact, eval_ngr = set(), set()
    for p in EVAL_SETS:
        fp = ROOT / p
        if not fp.exists():
            print(f"  WARN missing eval set {p}")
            continue
        for r in (json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()):
            q = norm(r.get("query") or r.get("prompt"))
            if q:
                eval_exact.add(q)
                eval_ngr |= ngrams(q.split())
    print(f"decon index: {len(eval_exact)} eval strings, {len(eval_ngr)} eval 8-grams")

    # 2) load FalseReject-Train
    from datasets import load_dataset
    fr = load_dataset("AmazonScience/FalseReject")["train"]
    print(f"FalseReject-Train: {len(fr)} prompts, {len(set(fr['category_text']))} categories")

    # 3) decontaminate
    kept = defaultdict(list)
    n_exact, n_ngram = 0, 0
    seen = set()
    for prompt, cat in zip(fr["prompt"], fr["category_text"]):
        q = norm(prompt)
        if not q or q in seen:
            continue
        seen.add(q)
        if q in eval_exact:
            n_exact += 1
            continue
        if eval_ngr & ngrams(q.split()):
            n_ngram += 1
            continue
        kept[cat].append(prompt)
    total_kept = sum(len(v) for v in kept.values())
    print(f"decon dropped: {n_exact} exact, {n_ngram} 8-gram-overlap; kept {total_kept}")

    # 4) per-category-capped stratified sample to ~args.n
    pick = []
    cats = sorted(kept, key=lambda c: len(kept[c]))  # small categories first (take all)
    remaining_cats = len(cats)
    for c in cats:
        if len(pick) >= args.n:
            break
        budget = max(1, (args.n - len(pick)) // max(1, remaining_cats))
        take = min(len(kept[c]), args.cap, budget)
        pick += [(q, c) for q in rng.sample(kept[c], take)]
        remaining_cats -= 1
    rng.shuffle(pick)
    pick = pick[:args.n]
    print(f"sampled {len(pick)} FR negatives across {len(set(c for _, c in pick))} categories")
    print("  top categories:", dict(Counter(c for _, c in pick).most_common(8)))

    # 5) write augmented pool = bio pool + FR borderline-benign (hard_label 0)
    bio = [json.loads(l) for l in open(POOL, encoding="utf-8") if l.strip()]
    nb_h = sum(r["hard_label"] for r in bio)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in bio:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for q, c in pick:
            f.write(json.dumps({"query": q, "hard_label": 0, "soft_label": 0.0,
                                "source": "falsereject", "category": c}, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT}: bio({len(bio)}: {nb_h} harm / {len(bio)-nb_h} benign) "
          f"+ FR borderline-benign({len(pick)}) = {len(bio)+len(pick)} total "
          f"(harm {nb_h} / benign {len(bio)-nb_h+len(pick)})")


if __name__ == "__main__":
    main()
