#!/usr/bin/env python
"""G.1: Train/eval overlap audit for v4 augmentation blocks.

Critical Goodhart question: does v4's 1.22% OR-Bench over-refusal reflect
genuine generalization or memorization of items also used in training?

For each augmentation block that draws from an externally-evaluated source,
compute set overlap between (B.1 training queries) and (evaluation queries).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL


def load_queries(path: Path) -> set:
    s = set()
    if not path.exists():
        return s
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            q = r.get("query") or r.get("prompt", "")
            if q.strip():
                s.add(q.strip())
    return s


def load_block_queries(aug_path: Path, block: str) -> set:
    s = set()
    if not aug_path.exists():
        return s
    with open(aug_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("block") == block:
                s.add(r["query"].strip())
    return s


def main():
    aug = DATA_EXTERNAL / "v4_splits" / "v4_augmentation.jsonl"
    print(f"v4 augmentation file: {aug}")
    print(f"  exists: {aug.exists()}")
    if aug.exists():
        n = sum(1 for _ in open(aug))
        print(f"  total items: {n}")
    print()

    audits = [
        ("B.1", "or_bench_health.jsonl", "OR-Bench health (over-refusal eval)"),
        ("B.2", "harmbench_bio.jsonl", "HarmBench bio (held-out eval source)"),
        ("B.2", "advbench_bio.jsonl", "AdvBench bio (held-out eval source)"),
        ("B.2", "jailbreakbench_bio.jsonl", "JailbreakBench bio"),
        ("B.2", "saladbench_cbrn.jsonl", "SaladBench CBRN (Phase 3 eval)"),
        ("B.3", "lab_bench.jsonl", "LAB-Bench (held-out eval source)"),
        ("B.4", "saladbench_cbrn.jsonl", "SaladBench CBRN (non-bio subset)"),
        ("B.4", "beavertails_subset.jsonl", "BeaverTails (Phase 2 eval)"),
    ]

    report = {}
    for block, fname, label in audits:
        eval_q = load_queries(DATA_EXTERNAL / fname)
        train_q = load_block_queries(aug, block)
        if not eval_q or not train_q:
            print(f"--- {block} vs {label} ---")
            print(f"  eval file items: {len(eval_q)}, train block items: {len(train_q)} -- SKIPPED")
            continue
        overlap = eval_q & train_q
        print(f"--- {block} vs {label} ({fname}) ---")
        print(f"  Eval file unique queries:   {len(eval_q):>6}")
        print(f"  {block} augmentation queries: {len(train_q):>6}")
        print(f"  Overlap:                    {len(overlap):>6} "
              f"({len(overlap)/len(eval_q)*100:.1f}% of eval, "
              f"{len(overlap)/len(train_q)*100:.1f}% of training block)")
        print(f"  Train-only (not in eval):   {len(train_q - eval_q):>6}")
        print(f"  Eval-only (not in train):   {len(eval_q - train_q):>6}")
        print()
        report[f"{block}_vs_{fname}"] = {
            "eval_n": len(eval_q),
            "train_block_n": len(train_q),
            "overlap_n": len(overlap),
            "overlap_pct_of_eval": round(100 * len(overlap) / max(len(eval_q), 1), 2),
            "overlap_pct_of_train": round(100 * len(overlap) / max(len(train_q), 1), 2),
            "train_only_n": len(train_q - eval_q),
            "eval_only_n": len(eval_q - train_q),
        }

    from constitutional_bioguard.config import METRICS_DIR
    out = METRICS_DIR / "v4_goodhart_g1_overlap_audit.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
