#!/usr/bin/env python
"""Phase 0.A: v2 dataset composition + per-source inspection.

Audits unified_overrefusal_taxonomy_v2.jsonl built by other session.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

V2_PATH = Path(Path.home(), "Dropbox/Bioinformatics/Claude/Safeguard/"
               "constitutional_bioguard/data/raw/unified_overrefusal_taxonomy_v2.jsonl")
OUT_PATH = Path(Path.home(), "Dropbox/Bioinformatics/Claude/Safeguard/"
                "constitutional_bioguard/data/metrics/v2_audit_phase0a_composition.json")

RNG = random.Random(42)


def main():
    rows = []
    with open(V2_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} records")
    print(f"Top-level keys (from row 0): {sorted(rows[0].keys())}")
    print()

    # Source breakdown
    src_counter = Counter(r.get("source", "_unknown") for r in rows)
    print(f"=== Sources (n={len(src_counter)}) ===")
    print(f"{'Source':<40} {'Records':>10} {'Pct':>6}")
    print("-" * 60)
    for s, n in src_counter.most_common():
        pct = n / len(rows) * 100
        print(f"{s[:39]:<40} {n:>10} {pct:>5.1f}%")
    print()

    # Class balance overall and per-source
    label_counter = Counter()
    by_source_labels = defaultdict(Counter)
    for r in rows:
        # Try multiple label field names
        label = r.get("label")
        if label is None:
            label = r.get("classification")
        if label is None:
            label = r.get("y")
        if label is None:
            # last resort: check any binary field
            label = "_unlabeled"
        label_counter[str(label)] += 1
        by_source_labels[r.get("source", "_unknown")][str(label)] += 1

    print(f"=== Overall label distribution ===")
    for lab, n in label_counter.most_common():
        print(f"  {lab}: {n} ({n/len(rows)*100:.1f}%)")
    print()

    print(f"=== Label distribution per source ===")
    for s in sorted(by_source_labels, key=lambda x: -src_counter[x]):
        labs = by_source_labels[s]
        total = sum(labs.values())
        print(f"  {s} (n={total}):")
        for lab, n in labs.most_common():
            print(f"    {lab}: {n} ({n/total*100:.1f}%)")
    print()

    # Sample 20 random per source for spot-check
    print(f"=== Random 5-item spot-check per source (first 5) ===")
    samples = {}
    for s in src_counter:
        src_rows = [r for r in rows if r.get("source") == s]
        sample = RNG.sample(src_rows, min(5, len(src_rows)))
        samples[s] = sample
        print(f"\n--- {s} ---")
        for i, r in enumerate(sample):
            q = (r.get("query") or r.get("prompt") or r.get("text") or "")[:150]
            resp = (r.get("response") or "")[:80]
            lab = r.get("label") or r.get("classification") or "?"
            print(f"  [{i+1}] label={lab}")
            print(f"      Q: {q}")
            if resp:
                print(f"      R: {resp}")

    # Build audit report
    audit = {
        "n_records": len(rows),
        "n_sources": len(src_counter),
        "sources": dict(src_counter.most_common()),
        "label_distribution": dict(label_counter.most_common()),
        "label_per_source": {s: dict(lc.most_common()) for s, lc in by_source_labels.items()},
        "sample_keys": sorted(rows[0].keys()) if rows else [],
        "sample_record_first": {k: str(v)[:200] for k, v in rows[0].items()},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
