#!/usr/bin/env python
"""Merge KO-translated augmentation into a new train set: train_ko_aug.jsonl.

train_ko_aug = data/splits/train.jsonl  +  data/aug/ko_aug_translated.jsonl (validated).
val / ood splits are NOT touched. Translation failures (empty or non-Hangul query) are
dropped so they never pollute training. Reports the final composition + KO fraction +
the legit/negative balance within the KO slice (must stay balanced — no "Korean=safe").
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "data" / "splits" / "train.jsonl"
TRANS = ROOT / "data" / "aug" / "ko_aug_translated.jsonl"
OUT = ROOT / "data" / "splits" / "train_ko_aug.jsonl"

HANGUL = re.compile(r"[가-힣]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--translated", nargs="+", default=[str(TRANS)],
                    help="one or more KO-translated aug files to merge in")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    train = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    trans = []
    for tf in args.translated:
        trans.extend(json.loads(l) for l in open(tf, encoding="utf-8"))

    valid, dropped = [], Counter()
    for r in trans:
        q = (r.get("query", "") or "").strip()
        if not q:
            dropped["empty_query"] += 1
            continue
        if not HANGUL.search(q):
            dropped["no_hangul"] += 1
            continue
        if r.get("binary_label") not in ("legitimate", "negative"):
            dropped["bad_label"] += 1
            continue
        valid.append(r)

    merged = train + valid
    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def lab_counts(recs):
        c = Counter(r.get("binary_label") for r in recs)
        return c["legitimate"], c["negative"]

    tl, tn = lab_counts(train)
    vl, vn = lab_counts(valid)
    ml, mn = lab_counts(merged)
    ko = sum(1 for r in merged if HANGUL.search((r.get("query", "") or "")))

    print(f"train (orig):    {len(train):5d}  (legit {tl} / neg {tn})")
    print(f"translated kept: {len(valid):5d}  (legit {vl} / neg {vn})")
    if dropped:
        print(f"  dropped: {dict(dropped)}")
    print(f"MERGED -> {args.out}")
    print(f"  total:  {len(merged):5d}  (legit {ml} / neg {mn}, "
          f"balance {ml/len(merged):.1%}/{mn/len(merged):.1%})")
    print(f"  Korean: {ko:5d}  ({ko/len(merged):.1%} of train)")
    print(f"  KO slice balance: legit {vl}/{len(valid)}={vl/max(1,len(valid)):.0%}  "
          f"neg {vn}/{len(valid)}={vn/max(1,len(valid)):.0%}  "
          f"[both classes present => no 'Korean=safe' shortcut]")
    print("  per-augmentation balance [each slice should be ~50/50]:")
    augs = sorted({r.get("augmentation", "?") for r in valid})
    for a in augs:
        sub = [r for r in valid if r.get("augmentation") == a]
        sl, sn = lab_counts(sub)
        print(f"    {a:28s} n={len(sub):4d}  legit {sl} / neg {sn}")


if __name__ == "__main__":
    main()
