#!/usr/bin/env python
"""R3 fetch-assess — does WildJailbreak yield real HARMFUL bio responses?

WildJailbreak is a safety-training set: its harmful-type pairs may carry REFUSAL
completions (the intended safe behavior), which are useless as harmful positives.
Measure before committing to a full harvest. Streams the train config (no full
download), counts bio harmful-type items, and checks whether their completions are
genuine harmful responses or refusals (regex + length).

Content-blind: counts only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import CONFIGS_DIR

BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]
REFUSAL = re.compile(
    r"\b(i\s*can'?t|i\s*cannot|i'?m\s+sorry|i\s+am\s+sorry|i\s*won'?t|cannot\s+assist|"
    r"can'?t\s+help|can'?t\s+assist|unable\s+to|not\s+able\s+to|i'?m\s+not\s+able|"
    r"against\s+my|i\s+must\s+decline|cannot\s+provide|can'?t\s+provide)\b", re.IGNORECASE)

CAP = 60000  # streaming scan cap (of ~262k) for a fast representative estimate


def is_bio(*parts):
    t = " ".join(str(p or "") for p in parts).lower()
    return any(k in t for k in BIO_KW)


def main():
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/wildjailbreak", "train", split="train", streaming=True)
    except Exception as e:
        print(f"LOAD FAILED ({type(e).__name__}): {e}")
        print("=> likely gated/auth or config name. Pivot to B (accept ceiling) or HarmBench completions.")
        return

    dt = Counter()
    bio_harm = 0
    bio_harm_refusal = 0
    bio_harm_substantive = 0
    n = 0
    cols = None
    for r in ds:
        if cols is None:
            cols = list(r.keys()); print("columns:", cols)
        n += 1
        d = str(r.get("data_type", ""))
        dt[d] += 1
        if "harmful" in d:
            prompt = r.get("adversarial") or r.get("vanilla") or r.get("prompt") or ""
            comp = r.get("completion", r.get("response", "")) or ""
            if is_bio(prompt, comp):
                bio_harm += 1
                if REFUSAL.search(comp[:400]):
                    bio_harm_refusal += 1
                elif len(comp.strip()) >= 80:
                    bio_harm_substantive += 1
        if n >= CAP:
            break

    print(f"\nscanned {n} rows (cap {CAP})")
    print("data_type counts:", dict(dt))
    print(f"\nbio harmful-type items: {bio_harm}")
    print(f"  of which REFUSAL completions (unusable as positives): {bio_harm_refusal}")
    print(f"  of which substantive non-refusal (USABLE positives): {bio_harm_substantive}")
    frac = n / 262000.0 if n else 0
    if bio_harm_substantive:
        print(f"\nextrapolated full-train usable bio positives ~ {int(bio_harm_substantive / max(frac,1e-9))}")
    print("\nVERDICT: if usable ~0 -> WildJailbreak harmful is refusals; pivot. "
          "If usable >= ~80 -> worth a full harvest toward 140/390.")


if __name__ == "__main__":
    main()
