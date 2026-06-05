#!/usr/bin/env python
"""Generate saladbench_cbrn_strict.jsonl — the honest O39 bio/chem subset.

The shipped saladbench_cbrn benchmark is a curation union (n=2268) where only
the O39 "Biological and Chemical Harms" category (n=52) is on-mission; the rest
(~2216) is non-bio CBRN (weapons, drugs, terrorism, infra, cyber). A single
union flag rate conflates bio recall with non-bio over-flagging (Section 6.21
audit: v7.C flags 92% of the union but that is selectivity loss, not recall).
This runner derives the strict O39 subset deterministically from the full cache
— no HF re-download. Content-blind: surfaces only counts and category labels.
"""
from __future__ import annotations

import collections
import json

from constitutional_bioguard.evaluation.extended_benchmarks import (
    cache_saladbench_cbrn_strict,
)


def main() -> None:
    p = cache_saladbench_cbrn_strict()
    rows = [json.loads(l) for l in open(p) if l.strip()]
    print("OUT  :", p)
    print("N    :", len(rows), "(expect 52)")
    c3 = collections.Counter(str(r.get("salad_category_3")) for r in rows)
    print("cat3 :", dict(c3.most_common(6)))
    assert all(r.get("is_bio_harm") and r["label"] == 1 for r in rows), \
        "tag/label invariant broken"
    assert len(rows) == 52, f"expected 52 O39 items, got {len(rows)}"
    print("OK   : all is_bio_harm, all label==1, n==52")


if __name__ == "__main__":
    main()
