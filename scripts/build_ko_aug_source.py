#!/usr/bin/env python
"""Select the English TRAIN slice to machine-translate into Korean (KO augmentation).

Why: v8 OOD-FPR (19.3%) is ~80% a LANGUAGE-coverage gap — train.jsonl is 0% Korean,
so the model flags Korean conversational/technical queries as harmful by default
(KO 78.5% FP vs EN 8.4%; diag in data/audit/diag_v8_ood_fpr_falsepos.json). Fix:
add Korean to TRAIN by translating existing English train queries (held-out-safe —
derived from train, NEVER from the session/eval data).

Two design constraints baked in here:
  1. Translate BOTH classes (legit + negative), not just benign. Korean-only-in-legit
     would teach a "Korean => safe" shortcut (a trivial KO jailbreak + OOD-FNR blow-up).
  2. LENGTH-MATCH the harmful sample to the benign length distribution, so the model
     cannot learn "long KO => safe, short KO => harmful" within Korean.

Excludes the 'Query about protein {PDB_ID}' template stubs (79.7% of legit) — translating
those yields degenerate non-natural Korean. We translate only natural-language queries.

Output: data/aug/ko_aug_source.jsonl — original records + provenance fields. The English
text stays in query_en/response_en; the translate step fills query/response with Korean.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "data" / "splits" / "train.jsonl"

TEMPLATE_RE = re.compile(r"^Query about protein \S+\s*$", re.I)
# Bio-relevant content domains preferred (within a length bin) for harmful samples,
# so KO harmful resembles in-domain attempts rather than only generic crime prompts.
BIO_DOMAINS = {
    "protein_engineering", "virology", "pathogen_biology", "toxicology", "genomics",
    "synthetic_biology", "dual_use_research", "microbiology", "structural_biology",
    "biochemistry", "immunology", "cell_biology", "pathogen_enhancement",
    "chemical_synthesis", "microbiology_dual_use", "mixed_biology",
}


def qlen(r: dict) -> int:
    return len((r.get("query", "") or "").strip())


def is_template(r: dict) -> bool:
    return bool(TEMPLATE_RE.match((r.get("query", "") or "").strip()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--out", default=str(ROOT / "data" / "aug" / "ko_aug_source.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins", type=int, default=10, help="length quantile bins for matching")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = [json.loads(l) for l in open(args.train, encoding="utf-8")]

    # Benign pool: natural-language legitimate (drop protein template stubs).
    benign = [r for r in recs
              if r.get("binary_label") == "legitimate" and not is_template(r)]
    # Harmful pool: all negatives (none are templates).
    harmful_pool = [r for r in recs if r.get("binary_label") == "negative"]

    benign_lens = sorted(qlen(r) for r in benign)
    n_b = len(benign)

    # Build quantile bin edges from the benign length distribution.
    edges = [benign_lens[min(n_b - 1, (i * n_b) // args.n_bins)] for i in range(args.n_bins)]
    edges = sorted(set(edges)) + [float("inf")]

    def bin_of(L: int) -> int:
        for i in range(len(edges) - 1):
            if edges[i] <= L < edges[i + 1]:
                return i
        return len(edges) - 2

    benign_bin_counts = Counter(bin_of(qlen(r)) for r in benign)

    # Index harmful by bin; within bin, bio-domain first (shuffled), then others.
    harmful_by_bin: dict[int, list] = {}
    for r in harmful_pool:
        harmful_by_bin.setdefault(bin_of(qlen(r)), []).append(r)
    for b, lst in harmful_by_bin.items():
        bio = [r for r in lst if r.get("content_domain") in BIO_DOMAINS]
        non = [r for r in lst if r.get("content_domain") not in BIO_DOMAINS]
        rng.shuffle(bio); rng.shuffle(non)
        harmful_by_bin[b] = bio + non  # bio preferred

    # Length-matched harmful sample: take benign_bin_counts[b] harmful from each bin.
    harmful_sel, shortfall = [], 0
    for b, want in benign_bin_counts.items():
        avail = harmful_by_bin.get(b, [])
        take = avail[:want]
        harmful_sel.extend(take)
        if len(take) < want:
            shortfall += want - len(take)
    # Top up shortfall with longest remaining harmful (keeps KO harmful from being all-short).
    if shortfall:
        used = {id(r) for r in harmful_sel}
        remaining = sorted((r for r in harmful_pool if id(r) not in used),
                           key=qlen, reverse=True)
        harmful_sel.extend(remaining[:shortfall])

    def to_source(r: dict, idx: int) -> dict:
        out = dict(r)
        out["orig_id"] = r.get("id")
        out["id"] = f"{r.get('id', 'rec')}_koaug{idx}"
        out["augmentation"] = "ko_mt_nllb"
        out["lang"] = "en"  # flips to "ko" after translation
        out["query_en"] = (r.get("query", "") or "")
        out["response_en"] = (r.get("response", "") or "")
        return out

    selected = benign + harmful_sel
    src = [to_source(r, i) for i, r in enumerate(selected)]
    rng.shuffle(src)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in src:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Report ───────────────────────────────────────────────────────────────
    sel_b = [r for r in src if r["binary_label"] == "legitimate"]
    sel_h = [r for r in src if r["binary_label"] == "negative"]
    bl = sorted(qlen(r) for r in sel_b)
    hl = sorted(qlen(r) for r in sel_h)
    pct = lambda xs, p: xs[min(len(xs) - 1, p * len(xs) // 100)] if xs else 0
    print(f"benign pool (natural-language legit): {len(benign)}")
    print(f"harmful pool (all negatives):         {len(harmful_pool)}")
    print(f"\nSELECTED -> {args.out}")
    print(f"  legit (benign):  {len(sel_b)}")
    print(f"  negative (harm): {len(sel_h)}  (shortfall topped up: {shortfall})")
    print(f"  TOTAL:           {len(src)}")
    print(f"\nLength match check (chars)  [should be similar => no length->label shortcut]")
    print(f"  benign : p25={pct(bl,25)}  median={pct(bl,50)}  p75={pct(bl,75)}")
    print(f"  harmful: p25={pct(hl,25)}  median={pct(hl,50)}  p75={pct(hl,75)}")
    print(f"\nHarmful sample by content_domain (top 10):")
    for k, v in Counter(r.get("content_domain") for r in sel_h).most_common(10):
        tag = "bio" if k in BIO_DOMAINS else "   "
        print(f"  {v:4d} [{tag}] {k}")
    print(f"\nBenign sample by source:")
    for k, v in Counter(r.get("source") for r in sel_b).most_common():
        print(f"  {v:4d}  {k}")
    n_resp = sum(1 for r in src if (r.get("response_en") or "").strip())
    print(f"\nrecords with a response to also translate: {n_resp}/{len(src)}")


if __name__ == "__main__":
    main()
