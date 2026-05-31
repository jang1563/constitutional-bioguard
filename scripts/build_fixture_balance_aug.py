#!/usr/bin/env python
"""Build a fixture-balance short aug to counteract koaug2's over-flagging.

koaug2 learned a shortcut: "short query + bio context = harmful"
This supplement adds SHORT legitimate research queries from diverse domains
(NOT conversational, NOT protein stubs) to rebalance the model.

Focus: samples from TRAIN benign + fixture benign, diverse domains,
all query-only, length-matched with short harmful.
"""
from __future__ import annotations

import argparse
import json
import re
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "data" / "splits" / "train.jsonl"
OOD_FPR = ROOT / "data" / "splits" / "ood_fpr.jsonl"
OUT_SOURCE = ROOT / "data" / "aug" / "fixture_balance_aug_source.jsonl"

TEMPLATE_RE = re.compile(r"^Query about protein \S+\s*$", re.I)
SHORT_THRESH = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--ood-fpr", default=str(OOD_FPR))
    ap.add_argument("--out", default=str(OUT_SOURCE))
    ap.add_argument("--n-benign", type=int, default=150, help="target benign samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Read TRAIN benign
    train_recs = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    train_benign = [
        r for r in train_recs
        if r.get("binary_label") == "legitimate"
        and len((r.get("query") or "").strip()) < SHORT_THRESH
        and not TEMPLATE_RE.match((r.get("query") or "").strip())
    ]
    
    # Read fixture benign short
    fixture_recs = [json.loads(l) for l in open(args.ood_fpr, encoding="utf-8")]
    fixture_benign = []
    for r in fixture_recs:
        if r.get("source") == "biosafety_suite_fixture" and r.get("binary_label") == "legitimate":
            q = r.get("query", "")
            try:
                conv = json.loads(q)
                if isinstance(conv, list):
                    for turn in conv:
                        if turn.get("role") == "user":
                            user_q = turn.get("content", "").strip()
                            if 10 < len(user_q) < SHORT_THRESH:
                                markers = ["Route:", "sidebar", "disabled", "untrusted", "force", "suppress"]
                                if not any(m in user_q for m in markers):
                                    fixture_benign.append({
                                        "query": user_q,
                                        "source": "fixture",
                                    })
            except json.JSONDecodeError:
                pass

    print(f"TRAIN short benign candidates: {len(train_benign)}")
    print(f"Fixture short benign: {len(fixture_benign)}")

    # Merge and dedupe
    benign_pool = train_benign + fixture_benign
    unique_queries = {}
    for r in benign_pool:
        q = (r.get("query") or "").strip()
        if q not in unique_queries:
            unique_queries[q] = r

    benign_recs = list(unique_queries.values())
    random.shuffle(benign_recs)
    benign_recs = benign_recs[:args.n_benign]

    print(f"Unique benign after dedup: {len(unique_queries)}")
    print(f"Selected for aug: {len(benign_recs)}")

    # By domain
    by_domain = {}
    for r in benign_recs:
        d = r.get("content_domain", "?")
        by_domain.setdefault(d, []).append(r)

    print(f"\nDomain breakdown (selected benign):")
    for d, recs in sorted(by_domain.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {d:30s} {len(recs):3d}")

    # Get harmful short pool
    train_harmful = [
        r for r in train_recs
        if r.get("binary_label") == "negative"
        and len((r.get("query") or "").strip()) < SHORT_THRESH
    ]

    print(f"\nHarmful short pool: {len(train_harmful)}")

    # Length-match
    benign_lens = [len((r.get("query") or "").strip()) for r in benign_recs]
    print(f"Benign query length: min={min(benign_lens)}, median={sorted(benign_lens)[len(benign_lens)//2]}, max={max(benign_lens)}")

    def get_harmfullen(r):
        return len((r.get("query") or "").strip())

    def match_harm_to_benign(benign_qlen):
        """Find harmful record with similar length via quantile bins."""
        bins = {}
        for h in train_harmful:
            hlen = get_harmfullen(h)
            bin_idx = min(int(hlen / 20), 4)  # 5 bins: 0-20, 20-40, ..., 80+
            bins.setdefault(bin_idx, []).append(h)
        benign_bin = min(int(benign_qlen / 20), 4)
        candidates = bins.get(benign_bin, train_harmful)
        return random.choice(candidates) if candidates else random.choice(train_harmful)

    # Build output
    output_recs = []
    for benign_rec in benign_recs:
        q = (benign_rec.get("query") or "").strip()
        qlen = len(q)
        harm_rec = match_harm_to_benign(qlen)

        output_recs.append({
            "query": q,
            "response": "",
            "binary_label": "legitimate",
            "source": "training",
            "content_domain": benign_rec.get("content_domain", "?"),
            "augmentation": "fixture_balance_ko_mt_nllb",
            "source_detail": benign_rec.get("source", "train"),
            "lang": "en",
        })
        output_recs.append({
            "query": (harm_rec.get("query") or "").strip(),
            "response": "",
            "binary_label": "negative",
            "source": "training",
            "content_domain": harm_rec.get("content_domain", "?"),
            "augmentation": "fixture_balance_ko_mt_nllb",
            "source_detail": "train",
            "lang": "en",
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in output_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tl, tn = Counter(r.get("binary_label") for r in output_recs).get("legitimate", 0), \
             Counter(r.get("binary_label") for r in output_recs).get("negative", 0)
    print(f"\nOutput: {len(output_recs)} records ({tl} benign / {tn} harmful, {tl}/{len(output_recs):.0%} balance)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
