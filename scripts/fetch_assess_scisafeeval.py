#!/usr/bin/env python
"""Dual-mode P1 fetch-assess: SciSafeEval bio slice (the largest prompt-harm positive set).

Counts the biology harmful-intent prompts available, inspects schema/labels, and
caches the bio slice locally for the prompt-harm head. Content-blind: prints
counts/columns/lengths/label-distribution only, never prompt text. Out (cache):
data/external/scisafeeval_bio.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL

REPO = "Tianhao0x01/SciSafeEval"
BIO_HINT = ("bio", "gene", "protein", "genom", "molec", "dna", "rna", "patho")
PROMPT_FIELDS = ("instruction", "query", "prompt", "question", "input", "jailbreak", "text")


def prompt_of(r):
    for k in PROMPT_FIELDS:
        if k in r and r[k]:
            return str(r[k])
    return ""


def main():
    from datasets import get_dataset_config_names, load_dataset
    try:
        cfgs = get_dataset_config_names(REPO)
    except Exception as e:
        print(f"config-list failed ({type(e).__name__}: {e}); trying default load")
        cfgs = [None]
    print("configs:", cfgs)

    bio_cfgs = [c for c in cfgs if c and any(k in c.lower() for k in BIO_HINT)]
    if not bio_cfgs:
        bio_cfgs = cfgs  # fall back: inspect all, filter rows by bio later

    out = []
    total = 0
    for c in bio_cfgs:
        try:
            ds = load_dataset(REPO, c) if c else load_dataset(REPO)
        except Exception as e:
            print(f"  config={c} load FAILED: {type(e).__name__} {str(e)[:100]}")
            continue
        splits = list(ds.keys())
        for sp in splits:
            d = ds[sp]
            if len(d) == 0:
                continue
            cols = list(d[0].keys())
            # field with longest text = likely the harmful instruction
            has_seq = "sequence" in cols
            lens = []
            for r in d:
                p = prompt_of(r)
                if not c or any(k in str(r.get("category", r.get("domain", c))).lower() for k in BIO_HINT) or any(k in p.lower() for k in BIO_HINT):
                    lens.append(len(p))
                    out.append({"prompt": p, "label": 1, "source": f"scisafeeval/{c}/{sp}",
                                "has_sequence": has_seq})
            total += len(d)
            med = sorted(lens)[len(lens) // 2] if lens else 0
            print(f"  config={c} split={sp} n={len(d)} bio_kept={len(lens)} cols={cols[:8]} prompt_len_med={med} has_sequence={has_seq}")

    # dedup
    seen = set(); ded = []
    for r in out:
        k = r["prompt"][:200]
        if k in seen:
            continue
        seen.add(k); ded.append(r)
    with open(DATA_EXTERNAL / "scisafeeval_bio.jsonl", "w") as f:
        for r in ded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nscanned ~{total} rows; kept BIO harmful-intent prompts: {len(ded)} (deduped)")
    print(f"by source: {dict(Counter(r['source'] for r in ded).most_common(12))}")
    print(f"with sequence field: {sum(1 for r in ded if r['has_sequence'])}")
    print(f"wrote data/external/scisafeeval_bio.jsonl")


if __name__ == "__main__":
    main()
