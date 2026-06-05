#!/usr/bin/env python
"""P1 (dual-mode) ASSESS — the prompt-harm bio data pool from cache.

Inventory cached external datasets for the NEW prompt-harm (intent) head:
  POSITIVES = bio-harm PROMPTS (harmful-intent benchmarks, bio slice)
  HARD-NEG  = benign-bio PROMPTS (legit research / over-refusal sets, bio slice)
Classify each cached file as positive-source / negative-source / response-only /
mcq (needs relabel), count the bio slice, and flag what is MISSING and must be
fetched (SciSafeEval, ClearHarm). Content-blind: counts only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL

BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]

# source-name -> role (by known dataset type)
POS = ["saladbench", "alert", "harmbench", "advbench", "jailbreakbench", "simple_safety"]
NEG = ["false_reject", "or_bench", "xstest", "pubmed", "lab_bench", "med_qa"]
MCQ = ["wmdp"]                       # MCQ knowledge probe -> relabel to intent
RESP = ["wildguard", "beavertails"]  # response-harm sources (other head)


def is_bio(t):
    t = str(t or "").lower()
    return any(k in t for k in BIO_KW)


def prompt_of(r):
    for k in ("query", "prompt", "question", "behavior", "goal", "text"):
        if k in r and r[k]:
            return r[k]
    return ""


def role(name):
    n = name.lower()
    for k in MCQ:
        if k in n:
            return "mcq"
    for k in RESP:
        if k in n:
            return "response"
    for k in POS:
        if k in n:
            return "positive"
    for k in NEG:
        if k in n:
            return "negative"
    return "other"


def main():
    files = sorted(DATA_EXTERNAL.glob("*.jsonl"))
    tot = {"positive": 0, "negative": 0, "mcq": 0}
    print(f"{'file':34s} {'role':9s} {'n':>6s} {'bio':>6s}")
    for p in files:
        try:
            rows = [json.loads(l) for l in open(p) if l.strip()]
        except Exception:
            continue
        rl = role(p.name)
        if rl in ("response", "other"):
            continue
        bio = sum(1 for r in rows if is_bio(prompt_of(r)))
        if rl in tot:
            tot[rl] += bio
        print(f"{p.name:34s} {rl:9s} {len(rows):6d} {bio:6d}")

    print(f"\n=== cached prompt-harm pool ===")
    print(f"  bio-harm PROMPT positives (cached benchmarks): {tot['positive']}")
    print(f"  + MCQ bio (wmdp, needs relabel to intent):      {tot['mcq']}")
    print(f"  benign-bio PROMPT hard-negatives (cached):      {tot['negative']}")
    print(f"\n=== MISSING (fetch) ===")
    for name, why in [("SciSafeEval bio (~12.6K, largest positive set)", "open, CC-BY-NC"),
                      ("ClearHarm CBRN (cleanest positives)", "open, FAR AI"),
                      ("Health-ORSC-Bench bio (hard-negatives)", "pending release"),
                      ("BioProBench / BioASQ (benign-bio volume)", "open")]:
        print(f"  - {name}  [{why}]")
    print("\n(target composition: ~20-30K positives : ~30-45K hard-neg, negative-heavy)")


if __name__ == "__main__":
    main()
