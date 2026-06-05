#!/usr/bin/env python
"""G1 leakage audit for v7.B — train/eval query overlap (was never run for v7).

v7.B was trained on  data/processed/train.jsonl (v4 train, n=3062)  +
data/external/wildguard_mix_train_bio.jsonl (n=469), via build_v7b_cot_data.py.
It is EVALUATED on the data/external/*.jsonl benchmarks. If any eval query also
appears in the training set, the corresponding Phase-1 gate is contaminated.

Two overlap signals, both content-blind (only counts/hashes surfaced):
  - exact-normalized: lowercase + collapse-whitespace + strip, full-string match
  - prefix-64:        same normalization, first 64 chars (catches truncation /
                      trailing-template variants of the same prompt)

Special focus: wildguard_mix_train_bio vs wildguard_test (both derived from
WildGuardMix) — the split must be disjoint or the WildGuard F1 gate is invalid.

Read-only. No model. Surfaces only benchmark name, N, and overlap counts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from constitutional_bioguard.config import DATA_PROCESSED, DATA_EXTERNAL

TRAIN_FILES = [
    (DATA_PROCESSED / "train.jsonl", "query"),
    (DATA_EXTERNAL / "wildguard_mix_train_bio.jsonl", "query"),
]
# (filename, query_field)
EVAL_FILES = [
    ("wildguard_test.jsonl", "prompt"),
    ("saladbench_cbrn.jsonl", "query"),
    ("alert_cbrn.jsonl", "query"),
    ("xstest.jsonl", "query"),
    ("or_bench_hard_1k.jsonl", "query"),
    ("false_reject_test.jsonl", "query"),
    ("simple_safety_bio.jsonl", "query"),
]

_ws = re.compile(r"\s+")


def norm(s: str) -> str:
    return _ws.sub(" ", (s or "").strip().lower())


def load_queries(path: Path, field: str) -> list[str]:
    out = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        q = r.get(field) or r.get("query") or r.get("prompt") or ""
        if q:
            out.append(q)
    return out


def main():
    # Build training query sets.
    train_exact, train_pref = set(), set()
    total_train = 0
    for path, field in TRAIN_FILES:
        if not path.exists():
            print(f"WARN train file missing: {path}")
            continue
        qs = load_queries(path, field)
        total_train += len(qs)
        for q in qs:
            n = norm(q)
            train_exact.add(n)
            train_pref.add(n[:64])
        print(f"train source {path.name}: {len(qs)} queries")
    print(f"TOTAL train queries: {total_train}  "
          f"(unique exact-norm: {len(train_exact)})")
    print()

    hdr = "{:<26}{:>6}{:>10}{:>9}{:>10}{:>9}".format(
        "benchmark", "n", "exact", "exact%", "prefix64", "pref%")
    print(hdr)
    print("-" * len(hdr))
    flagged = []
    for fname, field in EVAL_FILES:
        path = DATA_EXTERNAL / fname
        if not path.exists():
            print(f"{fname:<26} MISSING")
            continue
        qs = load_queries(path, field)
        n = len(qs)
        ex = sum(1 for q in qs if norm(q) in train_exact)
        pf = sum(1 for q in qs if norm(q)[:64] in train_pref)
        print("{:<26}{:>6}{:>10}{:>8.1f}%{:>10}{:>8.1f}%".format(
            fname, n, ex, 100 * ex / n if n else 0, pf, 100 * pf / n if n else 0))
        if ex > 0 or pf > 0.02 * n:
            flagged.append((fname, n, ex, pf))
    print()

    # Dedicated WildGuard split check.
    wg_train = set()
    p = DATA_EXTERNAL / "wildguard_mix_train_bio.jsonl"
    if p.exists():
        for q in load_queries(p, "query"):
            wg_train.add(norm(q))
        wg_test = [norm(q) for q in load_queries(DATA_EXTERNAL / "wildguard_test.jsonl", "prompt")]
        inter = sum(1 for q in wg_test if q in wg_train)
        print(f"[WildGuard split] train_bio={len(wg_train)} test={len(wg_test)} "
              f"exact-overlap={inter} ({100*inter/len(wg_test):.1f}% of test)")
    print()

    print("=== VERDICT ===")
    if not flagged:
        print("No train/eval query overlap detected -> Phase-1 gates are leakage-clean.")
    else:
        print("LEAKAGE in:", ", ".join(f"{f}(exact={e},pref={p})" for f, n, e, p in flagged))
        print("-> affected gates are contaminated; recompute on the disjoint subset.")


if __name__ == "__main__":
    main()
