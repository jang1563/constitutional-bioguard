#!/usr/bin/env python
"""G.1 audit for v5: verify train/eval leakage is near zero."""
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
            q = (r.get("query") or r.get("prompt", "")).strip()
            if q:
                s.add(q)
    return s


def load_block_queries(aug_path: Path, block: str = None) -> set:
    s = set()
    if not aug_path.exists():
        return s
    with open(aug_path) as f:
        for line in f:
            r = json.loads(line)
            if block is None or r.get("block") == block:
                s.add(r["query"].strip())
    return s


def main():
    aug = DATA_EXTERNAL / "v5_splits" / "v5_augmentation.jsonl"
    print(f"v5 augmentation: {aug}")
    if not aug.exists():
        print("FILE MISSING")
        return
    n = sum(1 for _ in open(aug))
    print(f"  total items: {n}")
    aug_all = load_block_queries(aug)
    print(f"  unique queries: {len(aug_all)}")
    print()

    audits = [
        ("or_bench_hard_1k.jsonl",   "OR-Bench-Hard-1K (PRIMARY over-refusal eval)"),
        ("false_reject_test.jsonl",  "FalseReject test (PRIMARY over-refusal eval)"),
        ("xstest.jsonl",             "XSTest (SECONDARY over-refusal eval)"),
        ("or_bench_health.jsonl",    "or_bench_health (legacy v4 eval)"),
        ("simple_safety_bio.jsonl",  "SimpleSafetyTests bio (bio-recall eval)"),
        ("saladbench_cbrn.jsonl",    "SaladBench CBRN (broad OOD; partial B.2 source)"),
        ("alert_cbrn.jsonl",         "ALERT CBRN (broad OOD)"),
        ("wildguard_test.jsonl",     "WildGuard test (native bio eval)"),
        ("harmbench_full.jsonl",     "HarmBench full"),
        ("advbench_full.jsonl",      "AdvBench full"),
        ("beavertails_subset.jsonl", "BeaverTails (Phase 2 eval)"),
        ("lab_bench.jsonl",          "LAB-Bench (locked, never train)"),
    ]

    report = {}
    print(f"{'Eval set':<55} {'eval_n':>8} {'overlap':>10} {'overlap%':>10}")
    print("-" * 90)
    for fname, label in audits:
        eval_q = load_queries(DATA_EXTERNAL / fname)
        if not eval_q:
            continue
        overlap = aug_all & eval_q
        pct = 100 * len(overlap) / max(len(eval_q), 1)
        status = "CLEAN" if pct == 0 else ("PARTIAL" if pct < 25 else "LEAK")
        print(f"{label:<55} {len(eval_q):>8} {len(overlap):>10} {pct:>9.1f}% [{status}]")
        report[fname] = {
            "label": label,
            "eval_n": len(eval_q),
            "v5_aug_overlap_n": len(overlap),
            "overlap_pct_of_eval": round(pct, 2),
            "status": status,
        }

    # Verify all 8 acceptance gates' eval sets are clean
    print()
    print("=== Acceptance gate eval sets clean? ===")
    gates = {
        "OR-Bench-Hard-1K (gate 5)": "or_bench_hard_1k.jsonl",
        "XSTest (gate 6)": "xstest.jsonl",
        "WildGuard native (gate 7)": "wildguard_test.jsonl",
        "BioThreat-Eval (gate 8)": None,  # external corpus, not file-based
    }
    for gate, fname in gates.items():
        if fname is None:
            print(f"  {gate}: external corpus, structurally separate")
            continue
        d = report.get(fname)
        if d is None:
            print(f"  {gate}: file missing")
            continue
        status = "PASS" if d["overlap_pct_of_eval"] == 0 else "FAIL"
        print(f"  {gate}: {d['overlap_pct_of_eval']}% leak [{status}]")

    from constitutional_bioguard.config import METRICS_DIR
    out = METRICS_DIR / "v5_g1_overlap_audit.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
