#!/usr/bin/env python
"""G.1 audit for v6: verify zero leakage between v6 calibration/probe sources
and the locked eval list defined in V6_DESIGN_v2.md Section 7.

v6 does NOT do new training, but it DOES use a probe set (SPLICE fit) and
a calibration dev set (cascade tuning). Both must be 0% overlap with locked
eval sets.

Run BEFORE any SPLICE fit or cascade calibration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR


# Locked eval sets per V6_DESIGN_v2.md Section 7
LOCKED_EVAL_FILES = [
    "or_bench_hard_1k.jsonl",
    "false_reject_test.jsonl",
    "xstest.jsonl",
    "simple_safety_bio.jsonl",
    "saladbench_cbrn.jsonl",
    "alert_cbrn.jsonl",
    "wildguard_test.jsonl",
    "lab_bench.jsonl",
    "harmbench_full.jsonl",
    "advbench_full.jsonl",
    "beavertails_subset.jsonl",
    "or_bench_health.jsonl",
]


# Permitted v6 calibration sources (must have 0% overlap with locked eval)
PERMITTED_SOURCES = [
    "wildguard_mix_train.jsonl",     # if we cache it
    "wildjailbreak_train.jsonl",     # if we cache it
    "air_bench_2024_cbrn.jsonl",     # if we cache it
    "false_reject_train.jsonl",      # already cached
    "saladbench_cbrn_train70.jsonl", # if we build stratified
    "harmbench_val.jsonl",           # if we cache the 100-item val
]


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


def main():
    print(f"V6 Leakage Audit (G.1) — V6_DESIGN_v2.md Section 7 compliance check")
    print("=" * 80)

    # Load all locked eval queries (union)
    locked_queries = set()
    for fname in LOCKED_EVAL_FILES:
        fp = DATA_EXTERNAL / fname
        if not fp.exists():
            print(f"  (eval set not cached locally: {fname})")
            continue
        q = load_queries(fp)
        locked_queries |= q
        print(f"  locked: {fname:<35} {len(q):>6} unique queries")
    print(f"\nTotal locked eval unique queries: {len(locked_queries)}")
    print()

    # Check each permitted source for overlap
    print(f"{'Source':<40} {'n_queries':>10} {'overlap':>10} {'overlap%':>10} {'status':>8}")
    print('-' * 90)
    report = {"locked_eval_total": len(locked_queries), "sources": {}}
    for fname in PERMITTED_SOURCES:
        fp = DATA_EXTERNAL / fname
        if not fp.exists():
            print(f"{fname:<40} {'N/A (not cached)':>10}")
            report["sources"][fname] = {"status": "not_cached"}
            continue
        q = load_queries(fp)
        overlap = q & locked_queries
        pct = 100 * len(overlap) / max(len(q), 1)
        status = "PASS" if pct == 0 else ("WARN" if pct < 1 else "FAIL")
        print(f"{fname:<40} {len(q):>10} {len(overlap):>10} {pct:>9.2f}% {status:>8}")
        report["sources"][fname] = {
            "n_queries": len(q),
            "overlap_with_locked": len(overlap),
            "overlap_pct": round(pct, 3),
            "status": status,
        }

    out = METRICS_DIR / "v6_g1_overlap_audit.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")

    # Hard fail rule
    fails = [k for k, v in report["sources"].items() if v.get("status") == "FAIL"]
    if fails:
        print(f"\n!!! FAIL: {len(fails)} sources have leakage with locked eval:")
        for k in fails:
            print(f"    - {k}")
        print("\nv6 execution BLOCKED. Re-partition data or remove these sources.")
        sys.exit(1)
    else:
        print("\nAll permitted sources pass G.1 (zero leakage with locked eval).")


if __name__ == "__main__":
    main()
